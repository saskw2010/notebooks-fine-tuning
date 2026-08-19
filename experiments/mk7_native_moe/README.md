# MK7 Native MoE Streaming Prototype

This experiment explores a native sparse Mixture-of-Experts language model designed from the start around a streamable expert boundary.

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

The point of this preset is to validate stable routing and learning dynamics while keeping the full model small enough to iterate quickly.

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

## Next gates

1. Run `research-30m` for a meaningful token budget on GPU.
2. Track per-layer expert utilization, entropy and dropped assignments over time.
3. Add validation loss and checkpoint/resume.
4. Add safer expert serialization (safetensors) and immutable expert shards.
5. Add VRAM/RAM/NVMe residency manager and async prefetch for inference.
6. Quantize experts independently (int8, then int4 experiments).
7. Define/export a Colibri-compatible or Colibri-inspired expert manifest.
8. Scale only after router stability is demonstrated; target architecture remains roughly 1.5-3B total with 300-500M active/token.

## Important distinction

Expert streaming is primarily an inference/runtime technique. Sparse MoE is the model architecture. We are designing the model so the two fit together from the beginning, but we should not train by repeatedly paging trainable experts from NVMe until the training semantics and optimizer-state strategy are explicitly designed for that case.
