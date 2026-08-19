# MK7 Native MoE Streaming v0

A minimal research prototype for a **native sparse MoE built with streaming in mind from day one**.

## What this proves

- 8 routed experts with **Top-2** token routing.
- SwiGLU experts.
- Router receives gradients during training.
- Auxiliary load-balancing loss and routing metrics.
- Capacity limit and dropped-assignment accounting.
- Explicit `ExpertStore` abstraction separating MoE semantics from weight residency.
- File-backed expert store with lazy loading and a bounded LRU resident cache.

This is deliberately a correctness-first reference implementation, not an optimized inference runtime.

## Smoke test

```bash
pip install torch
cd experiments/mk7_native_moe
python smoke.py
```

Expected result:

```text
TRAINING SMOKE: PASS
FILE-STREAMING SMOKE: PASS
```

The initial validated local run used 8 experts / Top-2 and reported approximately 1.18M total parameters vs 0.296M active parameters per token in the tiny smoke configuration. That small shape is only for fast correctness testing.

## Architecture boundary

```text
Transformer shared/resident trunk
          |
       Top-K router
          |
   +------+------+---- ...
   |      |      |
 Expert Expert Expert        <- separately addressable weights
   |      |      |
   +------+------+----------- weighted combine
          |
       next layer
```

`ExpertStore` is the key boundary. During training, experts remain normal PyTorch modules. During inference, the same routed MoE layer can obtain an expert from another residency policy. `TorchFileExpertStore` is the first proof: experts are stored as individual files and loaded only when routed, with an LRU cache limiting resident experts.

## Target direction — not instantiated yet

The research target is **not** a 300–400M-total dense-like model. The interesting regime is a larger total capacity with much smaller active compute, for example:

- total model capacity: ~1.5–3B parameters
- active parameters/token: ~300–500M
- experts: 8 initially, then 16
- routing: Top-2
- shared/resident attention/trunk
- routed FFN experts individually quantizable and streamable
- optional domain LoRA/adapters *inside or above experts*, without making adapters the primary MoE mechanism

The exact parameter budget must be derived from the full transformer shape; `SparseMoE.parameter_report()` currently reports the MoE block itself and should not be mistaken for a full-model active-parameter estimate.

## Next milestones

1. Wrap `SparseMoE` into a small causal Transformer LM and train it on a tiny synthetic/text corpus.
2. Add router diagnostics: expert utilization, imbalance, token drop rate, route stability and expert co-occurrence.
3. Add shared expert support and compare Top-1 vs Top-2.
4. Replace `.pt` proof files with safe tensor shards and explicit metadata.
5. Add residency tiers (VRAM / RAM / NVMe), asynchronous prefetch and cache-hit metrics.
6. Benchmark in-memory vs streamed experts and identify the I/O break-even point.
7. Only after correctness and routing stability: quantize routed experts to int4/int8 and build a Colibri-compatible/export path.

## Important caveat

Streaming is useful only when non-resident expert capacity is large enough to justify I/O. For a model that is only ~300–400M parameters total, keeping the whole model resident is normally simpler and faster. The design objective is therefore **small active compute with larger sparse total capacity**, not streaming for its own sake.
