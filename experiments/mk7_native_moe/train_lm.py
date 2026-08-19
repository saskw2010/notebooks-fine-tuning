import argparse
from pathlib import Path
import torch
from lm import LMConfig, MK7CausalLM


def corpus_bytes(path=None):
    if path:
        return Path(path).read_bytes()
    seed = (
        "MK7 is a native sparse mixture of experts language model. "
        "The router selects two experts per token. "
        "Arabic: هذا نموذج تجريبي صغير لاختبار التوجيه والخبراء. "
        "ERP agents call tools, validate results, and keep an audit trail.\n"
    )
    return (seed * 200).encode("utf-8")


def batchify(data, batch_size, seq_len, device):
    starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in starts.tolist()]).to(device)
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in starts.tolist()]).to(device)
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="smoke", choices=["smoke", "research-30m"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--text", help="optional UTF-8 training text file")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = LMConfig.preset(args.preset)
    device = torch.device(args.device)
    model = MK7CausalLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    data = torch.tensor(list(corpus_bytes(args.text)), dtype=torch.long)

    print("device:", device)
    print("parameters:", model.parameter_report())

    for step in range(1, args.steps + 1):
        model.train()
        x, y = batchify(data, args.batch_size, cfg.max_seq_len, device)
        out = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % 5 == 0 or step == args.steps:
            loads = torch.stack([
                m["expert_load"].float().cpu() for m in out["routing"]
            ]).mean(0)
            drops = sum(m["dropped_assignments"] for m in out["routing"])
            print(
                f"step={step:04d} loss={out['loss'].item():.4f} "
                f"lm={out['lm_loss'].item():.4f} aux={out['aux_loss'].item():.4f} "
                f"drops={drops} mean_expert_load={loads.tolist()}"
            )

    prompt = torch.tensor([[ord(c) for c in "MK7 "]], dtype=torch.long, device=device)
    sample = model.generate(prompt, max_new_tokens=32)[0].cpu().tolist()
    print("sample:", bytes(sample).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
