import tempfile
import torch

from model import MoEConfig, SparseMoE, TorchFileExpertStore


def run():
    torch.manual_seed(7)
    cfg = MoEConfig(
        d_model=128,
        d_hidden=384,
        n_experts=8,
        top_k=2,
        capacity_factor=1.5,
    )

    model = SparseMoE(cfg)
    model.train()
    x = torch.randn(2, 16, cfg.d_model)

    y, aux, metrics = model(x)
    loss = y.square().mean() + 0.01 * aux
    loss.backward()

    assert y.shape == x.shape
    assert torch.isfinite(loss)
    assert model.router.proj.weight.grad is not None
    assert int(metrics["expert_load"].sum()) <= x.shape[0] * x.shape[1] * cfg.top_k

    print("TRAINING SMOKE: PASS")
    print("parameter_report:", model.parameter_report())
    print("expert_load:", metrics["expert_load"].tolist())
    print("router_entropy:", float(metrics["router_entropy"]))
    print("dropped_assignments:", metrics["dropped_assignments"])

    with tempfile.TemporaryDirectory() as td:
        TorchFileExpertStore.export(td, model.experts)
        stream_store = TorchFileExpertStore(td, cfg, cache_size=2)

        model.eval()
        model.set_expert_store(stream_store)
        with torch.no_grad():
            y2, _, _ = model(x)

        assert y2.shape == x.shape
        assert torch.isfinite(y2).all()
        assert len(stream_store.cache) <= 2
        print("FILE-STREAMING SMOKE: PASS")
        print("resident_experts:", [k[0] for k in stream_store.cache.keys()])


if __name__ == "__main__":
    run()
