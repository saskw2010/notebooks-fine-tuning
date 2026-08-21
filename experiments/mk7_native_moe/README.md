# MK7 Native MoE Streaming Prototype

This experiment explores a native sparse Mixture-of-Experts language model designed from the start around a streamable expert boundary.

> Full narrative and research notes: [`STORY.html`](./STORY.html)

## Current architecture

- Causal Transformer language model
- RMSNorm + causal self-attention
- SwiGLU feed-forward experts
- configurable sparse Top-k routing
- default research shape: 8 experts / Top-2
- router load-balancing auxiliary loss
- capacity/drop accounting and expert-load metrics
- `ExpertStore` abstraction for future tiered residency
- file-backed lazy expert loading with bounded LRU cache

## Presets

### `smoke`

Correctness-only configuration for local CPU/GPU validation.

Approximate parameter report:

- 548,064 total parameters
- 326,880 active parameters/token
- 4 experts / Top-2

Local validation: forward, backward, router gradients and language-model loss all pass. In a 3-step CPU smoke the loss moved from roughly 5.57 to 5.41. This is only a correctness signal, not a quality benchmark.

### `research-30m`

First real research-scale target suitable for a free/low-cost notebook GPU before scaling toward billion-parameter capacity.

Approximate parameter report:

- 28,697,760 total parameters
- 8,791,200 active parameters/token
- 8 experts / Top-2
- 6 Transformer layers
- d_model 288
- expert hidden size 640
- sequence length 256

### Kaggle T4 validation — 2026-08-19

A 100-step GPU run completed successfully on a Tesla T4 using the current toy corpus.

- loss: 5.6069 → 1.2943
- LM loss: 5.5932 → 1.2836
- router auxiliary loss: 1.3771 → 1.0656
- dropped assignments: 2391 → 440
- all 8 experts continued receiving traffic

Interpretation: this validates end-to-end learning and early router stability, not language quality. The toy corpus is intentionally tiny/repetitive and is unsuitable for a quality benchmark.

A Tesla P100 attempt failed because the current Kaggle PyTorch build did not include kernels for its `sm_60` compute capability; the T4 (`sm_75`) run succeeded. This was an environment compatibility issue rather than an MoE-model failure.

## Run locally

```bash
cd experiments/mk7_native_moe
python train_lm.py --preset smoke --steps 20 --batch-size 4
```

For a GPU:

```bash
python train_lm.py --preset research-30m --steps 100 --batch-size 2
```

A ready-to-run `colab_kaggle.ipynb` is included for Google Colab or Kaggle.

## Streaming boundary

Training currently uses resident PyTorch experts so gradients remain straightforward. The MoE layer is deliberately separated from expert residency through `ExpertStore`. The existing file-backed store proves that selected experts can be loaded lazily and bounded by an LRU cache.

The intended inference evolution is:

```text
router -> expert IDs -> tier manager -> VRAM / RAM / NVMe -> selected experts
```

This follows the same broad systems idea as Colibri-style expert streaming without coupling the model definition to one runtime.

## External research notes

The narrative page records two adjacent research directions:

- **PrismML Bonsai 27B** — binary/ternary low-bit weights as a complementary way to reduce local model footprint. The MK7 research question is whether routed experts can tolerate extreme low-bit representations while router/shared components stay at higher precision.
- **AllenAI olmOCR / olmOCR 2** — document linearization and verifiable-reward training as an upstream data-factory component for domain experts, especially document-heavy ERP and business datasets.

See [`STORY.html`](./STORY.html) for the verified-vs-hypothesis distinction, external-source links, and proposed experiments.

## Next gates

1. Replace the toy corpus with a real train/validation dataset.
2. Track per-layer expert utilization, entropy and dropped assignments over time.
3. Add validation loss and checkpoint/resume.
4. Add safer expert serialization (safetensors) and immutable expert shards.
5. Add VRAM/RAM/NVMe residency manager and async prefetch for inference.
6. Quantize experts independently (int8, then int4, later binary/ternary experiments).
7. Define/export a Colibri-compatible or Colibri-inspired expert manifest.
8. Add an olmOCR-based document-ingestion experiment for expert datasets.
9. Scale only after router stability is demonstrated; target architecture remains roughly 1.5-3B total with 300-500M active/token.

## Important distinction

Expert streaming is primarily an inference/runtime technique. Sparse MoE is the model architecture. We are designing the model so the two fit together from the beginning, but we should not train by repeatedly paging trainable experts from NVMe until the training semantics and optimizer-state strategy are explicitly designed for that case.
