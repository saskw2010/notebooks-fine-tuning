from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MoEConfig, SparseMoE


@dataclass(frozen=True)
class LMConfig:
    vocab_size: int = 256
    max_seq_len: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    moe_every: int = 1
    n_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 768
    dense_hidden: int = 768
    capacity_factor: float = 1.25
    router_jitter: float = 0.01
    aux_loss_coef: float = 0.01
    dropout: float = 0.0

    @classmethod
    def preset(cls, name: str):
        presets = {
            "smoke": cls(max_seq_len=64, d_model=96, n_heads=4, n_layers=2,
                         n_experts=4, top_k=2, expert_hidden=192, dense_hidden=192),
            "research-30m": cls(max_seq_len=256, d_model=288, n_heads=6, n_layers=6,
                                n_experts=8, top_k=2, expert_hidden=640, dense_hidden=640),
        }
        if name not in presets:
            raise ValueError(f"unknown preset {name}; choose {list(presets)}")
        return presets[name]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(z):
            return z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = map(split, (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


class DenseFFN(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.dense_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.dense_hidden, bias=False)
        self.down = nn.Linear(cfg.dense_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: LMConfig, use_moe: bool):
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        if use_moe:
            self.ff = SparseMoE(MoEConfig(
                d_model=cfg.d_model,
                d_hidden=cfg.expert_hidden,
                n_experts=cfg.n_experts,
                top_k=cfg.top_k,
                capacity_factor=cfg.capacity_factor,
                router_jitter=cfg.router_jitter,
            ))
        else:
            self.ff = DenseFFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        if self.use_moe:
            y, aux, metrics = self.ff(self.norm2(x))
            return x + y, aux, metrics
        return x + self.ff(self.norm2(x)), x.new_zeros(()), None


class MK7CausalLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([
            Block(cfg, use_moe=((i + 1) % cfg.moe_every == 0))
            for i in range(cfg.n_layers)
        ])
        self.norm = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("sequence too long")
        pos = torch.arange(t, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos)[None, :, :]
        aux_total = x.new_zeros((), dtype=torch.float32)
        routing = []
        for block in self.blocks:
            x, aux, metrics = block(x)
            aux_total = aux_total + aux.float()
            if metrics is not None:
                routing.append(metrics)
        logits = self.lm_head(self.norm(x))
        lm_loss = loss = None
        aux_loss = aux_total / max(1, len(routing))
        if labels is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = lm_loss + self.cfg.aux_loss_coef * aux_loss
        return {
            "logits": logits,
            "loss": loss,
            "lm_loss": lm_loss,
            "aux_loss": aux_loss,
            "routing": routing,
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.cfg.max_seq_len:]
            logits = self(x)["logits"][:, -1, :] / max(temperature, 1e-5)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids

    def parameter_report(self):
        total = sum(p.numel() for p in self.parameters())
        expert_total = active_expert = 0
        for block in self.blocks:
            if block.use_moe:
                one = sum(p.numel() for p in block.ff.experts[0].parameters())
                expert_total += one * self.cfg.n_experts
                active_expert += one * self.cfg.top_k
        return {
            "total_parameters": total,
            "active_parameters_per_token_approx": total - expert_total + active_expert,
            "expert_parameters_total": expert_total,
            "active_expert_parameters": active_expert,
        }
