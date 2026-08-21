from __future__ import annotations
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MoEConfig:
    d_model: int = 256
    d_hidden: int = 768
    n_experts: int = 8
    top_k: int = 2
    capacity_factor: float = 1.25
    router_jitter: float = 0.0

    def validate(self) -> None:
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError("top_k must be in [1, n_experts]")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be > 0")


class ExpertMLP(nn.Module):
    """SwiGLU expert."""
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)
        self.up = nn.Linear(d_model, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class ExpertStore:
    """Inference abstraction: expert weights may live in RAM, disk, or another tier."""
    def get(self, expert_id: int, device: torch.device, dtype: torch.dtype) -> ExpertMLP:
        raise NotImplementedError


class InMemoryExpertStore(ExpertStore):
    def __init__(self, experts: Iterable[ExpertMLP]):
        self.experts = list(experts)

    def get(self, expert_id: int, device: torch.device, dtype: torch.dtype) -> ExpertMLP:
        return self.experts[expert_id].to(device=device, dtype=dtype)


class TorchFileExpertStore(ExpertStore):
    """
    Colibri-style proof of concept: each expert is a separate state-dict file,
    requested experts are loaded lazily, and an LRU bounds resident experts.
    """
    def __init__(self, directory: str | Path, cfg: MoEConfig, cache_size: int = 2):
        self.directory = Path(directory)
        self.cfg = cfg
        self.cache_size = max(1, cache_size)
        self.cache: "OrderedDict[Tuple[int, str, str], ExpertMLP]" = OrderedDict()

    @classmethod
    def export(cls, directory: str | Path, experts: Iterable[ExpertMLP]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, expert in enumerate(experts):
            torch.save(expert.state_dict(), directory / f"expert_{i:03d}.pt")

    def get(self, expert_id: int, device: torch.device, dtype: torch.dtype) -> ExpertMLP:
        key = (expert_id, str(device), str(dtype))
        if key in self.cache:
            expert = self.cache.pop(key)
            self.cache[key] = expert
            return expert

        path = self.directory / f"expert_{expert_id:03d}.pt"
        if not path.exists():
            raise FileNotFoundError(path)

        expert = ExpertMLP(self.cfg.d_model, self.cfg.d_hidden)
        state = torch.load(path, map_location="cpu", weights_only=True)
        expert.load_state_dict(state)
        expert.eval().to(device=device, dtype=dtype)

        self.cache[key] = expert
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return expert


class TopKRouter(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.proj(x)
        if self.training and self.cfg.router_jitter > 0:
            logits = logits + torch.randn_like(logits) * self.cfg.router_jitter

        probs = F.softmax(logits.float(), dim=-1)
        topk_prob, topk_idx = torch.topk(probs, k=self.cfg.top_k, dim=-1)
        topk_prob = topk_prob / topk_prob.sum(dim=-1, keepdim=True)

        importance = probs.mean(dim=0)
        hard = F.one_hot(topk_idx[..., 0], num_classes=self.cfg.n_experts).float().mean(dim=0)
        aux_loss = self.cfg.n_experts * torch.sum(importance * hard)
        return topk_idx, topk_prob.to(x.dtype), aux_loss, probs


class SparseMoE(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.router = TopKRouter(cfg)
        self.experts = nn.ModuleList(
            [ExpertMLP(cfg.d_model, cfg.d_hidden) for _ in range(cfg.n_experts)]
        )
        self._store: Optional[ExpertStore] = None

    def set_expert_store(self, store: Optional[ExpertStore]) -> None:
        self._store = store

    def _expert(self, expert_id: int, x: torch.Tensor) -> ExpertMLP:
        if self._store is None:
            return self.experts[expert_id]
        return self._store.get(expert_id, x.device, x.dtype)

    def forward(self, x: torch.Tensor):
        """x: [batch, seq, d_model]. Returns output, aux_loss, metrics."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        topk_idx, topk_prob, aux_loss, probs = self.router(flat)

        out = torch.zeros_like(flat)
        token_count = flat.shape[0]
        capacity = max(
            1,
            int(self.cfg.capacity_factor * token_count * self.cfg.top_k / self.cfg.n_experts),
        )

        expert_load = torch.zeros(self.cfg.n_experts, dtype=torch.long, device=flat.device)
        dropped = 0

        # Stable reference dispatch. Optimize after semantics are verified.
        for expert_id in range(self.cfg.n_experts):
            assignments = (topk_idx == expert_id).nonzero(as_tuple=False)
            if assignments.numel() == 0:
                continue

            if assignments.shape[0] > capacity:
                assignments = assignments[:capacity]
                dropped += int((topk_idx == expert_id).sum().item() - capacity)

            token_ids = assignments[:, 0]
            slot_ids = assignments[:, 1]
            expert_load[expert_id] = assignments.shape[0]

            expert = self._expert(expert_id, flat)
            expert_out = expert(flat[token_ids])
            weights = topk_prob[token_ids, slot_ids].unsqueeze(-1)
            out.index_add_(0, token_ids, expert_out * weights)

        metrics = {
            "expert_load": expert_load.detach().cpu(),
            "dropped_assignments": dropped,
            "capacity_per_expert": capacity,
            "router_entropy": (
                -(probs * torch.clamp(probs, min=1e-9).log()).sum(dim=-1).mean().detach().cpu()
            ),
        }
        return out.reshape(shape), aux_loss, metrics

    def parameter_report(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        one_expert = sum(p.numel() for p in self.experts[0].parameters())
        router = sum(p.numel() for p in self.router.parameters())
        active = total - self.cfg.n_experts * one_expert + self.cfg.top_k * one_expert
        return {
            "total_parameters": total,
            "active_parameters_per_token_approx": active,
            "one_expert_parameters": one_expert,
            "router_parameters": router,
        }
