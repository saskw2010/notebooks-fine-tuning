# MK7NMoE Test, Escalation, and Data Protocol

This document defines the operating contract for MK7NMoE experiments. It is intentionally stricter than an ordinary notebook log: every run must end in a comparable report, every platform escalation must be justified, and toy data must not be mistaken for model-quality evidence.

## 1. Standard report required after every test

Every experiment must return a report with these sections, even if the run fails.

### A. Identity
- experiment name
- git commit / branch
- model preset
- router variant
- number of experts
- Top-K
- total parameters
- approximate active parameters/token
- dataset name/version
- tokenizer
- random seed

### B. Environment
- platform: local / Kaggle / Hugging Face / other
- OS / runtime image
- Python version
- PyTorch version
- CUDA version
- GPU model(s)
- compute capability
- VRAM per GPU
- RAM

### C. Training configuration
- steps / epochs
- batch size
- gradient accumulation
- sequence length
- optimizer
- learning rate / schedule
- weight decay
- precision
- gradient clipping
- capacity factor
- router auxiliary-loss coefficient
- z-loss coefficient if used

### D. Core metrics
- train loss: start / best / final
- validation loss: start / best / final
- perplexity when meaningful
- tokens processed
- tokens/sec
- wall-clock time
- peak VRAM
- checkpoint size

### E. Router / MoE metrics
For every MoE layer, or at minimum aggregate plus worst layer:
- expert utilization distribution
- expert load min / max / mean / coefficient of variation
- router entropy
- dropped assignments/tokens and drop rate
- Top-K routing weights
- auxiliary load-balancing loss
- z-loss if present
- dead/unused experts
- evidence of expert collapse
- routing stability over checkpoints

### F. Outcome classification
Use exactly one:
- PASS — correctness
- PASS — architecture viability
- PASS — training stability
- PASS — quality gate
- PARTIAL
- FAIL — code
- FAIL — environment
- FAIL — OOM
- FAIL — router instability
- FAIL — data

Then state what the result proves and, equally important, what it does NOT prove.

### G. Decision
Choose one:
- STOP
- FIX AND REPEAT SAME TEST
- CONTINUE SAME SCALE
- ESCALATE PLATFORM
- SCALE MODEL
- CHANGE ROUTER
- CHANGE DATA

No experiment should end with only raw console output.

## 2. Platform escalation ladder

MK7NMoE uses the cheapest environment that can answer the current question.

### Stage 0 — Local machine
Use first for:
- syntax/import tests
- CPU smoke
- tiny CUDA smoke where compatible
- unit tests
- router semantics
- serialization
- checkpoint load/save
- dataset validation

Do not use local hardware for a run that is obviously bottlenecked by insufficient VRAM if the same question can be answered cheaply elsewhere.

### Stage 1 — Kaggle
Use after local correctness passes for:
- 20M–100M class training experiments
- router stability tests
- dense-vs-MoE baselines
- short ablations
- first real-dataset runs

Known compatibility lesson: current Kaggle PyTorch images may not support Tesla P100 (sm_60). Prefer a supported T4/Ampere-class accelerator when the installed PyTorch build requires sm_70+.

### Stage 2 — Hugging Face Jobs
Use when available and economically justified for:
- reproducible scripted GPU runs
- remote benchmark jobs
- artifact persistence to the Hub
- larger/longer controlled experiments

If the account/plan does not permit Jobs, mark the platform BLOCKED and use the next available platform. Do not treat a billing restriction as a model failure.

### Stage 3 — Other GPU platforms
RunPod, cloud startup credits, or another provider may be used when:
- the experiment needs more VRAM
- persistent storage is required
- multi-GPU work is justified
- free notebook quotas are insufficient

The report must preserve the same schema regardless of platform so results remain comparable.

### Escalation rule
A successful test may automatically propose the next platform, but should only escalate if the next scientific/engineering question requires it. Do not move to expensive hardware merely because a smaller test passed.

## 3. Current validated baseline

The first MK7NMoE research-scale smoke used:
- ~28.7M total parameters
- ~8.79M active parameters/token
- 8 experts
- Top-2 routing
- Tesla T4
- 100 steps
- toy repeated byte-level corpus

Observed:
- loss approximately 5.6069 -> 1.2943
- router auxiliary loss approximately 1.377 -> 1.066
- dropped assignments approximately 2391 -> 440 by the final logged step
- all eight experts remained active
- no obvious expert collapse in this short run

Interpretation: architecture/routing viability PASS. This is NOT a language-model quality result because the corpus is tiny and highly repetitive.

## 4. Real-data transition: stop training on the toy corpus

The next milestone is not more toy-data steps. It is a versioned real dataset plus validation and checkpointing.

### Data sources we can exploit
MK7NMoE should build data in layers instead of trying to create one giant domain dump immediately.

#### General language/core
Use permissively licensed/open corpora suitable for language-model training. Maintain source/license metadata and deduplicate before training.

#### Arabic
Create a clean Arabic slice with:
- Modern Standard Arabic
- technical Arabic
- bilingual Arabic/English material
- code-switching only where it represents real target usage

Do not let noisy OCR Arabic dominate the core corpus.

#### Code / tools
Include high-quality code and structured tool-use examples separately from plain prose. Preserve repository/license provenance.

#### Enterprise / ERP specialization
Use domain data only when ownership/licensing permits it. Prefer:
- synthetic ERP workflows generated from schemas and business rules
- de-identified templates
- public accounting/ERP documentation
- structured SQL/tool traces
- validation rules
- function/tool call examples

Private client data must not silently enter the training corpus.

#### Documents via olmOCR-style pipeline
olmOCR is valuable as a document-to-training-data factory, not as part of the MoE router itself:

PDF/scans -> layout/OCR extraction -> reading-order text/Markdown -> cleanup -> dedup -> quality scoring -> chunks/examples -> dataset manifests.

Use document provenance, page references, extraction confidence/quality checks, and aggressive filtering before data enters training.

#### Synthetic expert data
A teacher model may expand high-quality seeds into:
- instruction/response pairs
- reasoning traces only where appropriate and legally usable
- tool-use trajectories
- classification/routing labels
- domain-specific QA

Synthetic examples must be tagged as synthetic and held out from evaluation contamination.

## 5. Dataset structure for MK7NMoE

Do not create eight isolated expert datasets first. The router needs overlapping evidence to learn specialization.

Recommended initial mixture:
- 50–60% general language/core
- 10–15% Arabic
- 10–15% code/tools
- 10–15% enterprise/ERP
- 5–10% document/OCR-derived material

These are starting ratios, not fixed truths. Adjust after router telemetry and validation results.

Each sample should carry metadata such as:
```json
{
  "text": "...",
  "source": "...",
  "license": "...",
  "language": "ar|en|mixed|code",
  "domain": "general|erp|code|documents|...",
  "quality": 0.0,
  "synthetic": false,
  "split": "train|validation|test"
}
```

Domain labels are metadata for analysis and controlled experiments; the native router should still learn from hidden states rather than receiving the label as an oracle during normal inference.

## 6. Dataset gates before real training

A dataset version may enter training only after:
- license/provenance audit
- exact and near-duplicate removal
- train/validation/test leakage check
- malformed/empty sample removal
- language distribution report
- length distribution report
- domain distribution report
- quality sample review
- tokenizer coverage / byte-fallback analysis
- contamination check against selected evaluation sets

Version each release: `mk7nmoe-data-v0.1`, `v0.2`, etc.

## 7. Immediate training roadmap

### Gate A — 30M real-data run
Add:
- real tokenizer (unless byte-level is an explicit ablation)
- real mixed dataset
- validation split
- checkpoint/resume
- tokens/sec and peak VRAM
- router telemetry per layer

Compare current MoE against a dense model with roughly similar active compute.

### Gate B — Router benchmark
On the SAME model/data/token budget compare:
- current MK7NMoE Top-K router
- OLMoE-equivalent baseline only if algorithmically distinct from current implementation
- MoST-style candidate only after its exact algorithm is verified

Do not add duplicate routers under different names.

### Gate C — 100M class
Only after Gate A/B show stable validation and routing behavior.

### Gate D — 300M–500M class
Use external GPU if required. At this point evaluate:
- two-GPU expert parallelism
- expert quantization
- streamable expert shards
- inference residency tiers

### Gate E — target architecture
Longer-term target remains approximately:
- 1.5–3B total capacity
- 300–500M active/token
- sparse native experts
- low-bit experts where quality permits
- streamable inference boundary

## 8. External-model research notes

### Bonsai 27B
Bonsai-style binary/ternary compression is a research direction for expert storage, not yet a validated MK7NMoE component. The useful hypothesis is:

higher-precision shared/router components + aggressively low-bit expert weights + sparse activation + optional streaming.

Any binary/ternary adoption requires independent quality and runtime benchmarks; compressed file size alone is not enough.

### olmOCR
olmOCR should be treated as a data-engineering reference for converting difficult documents into high-quality structured text. Its relevance to MK7NMoE is dataset construction, especially domain/document experts; it is not evidence about MoE routing quality.

## 9. Definition of 'real training has started'

MK7NMoE may claim real training only when all of these are true:
- non-toy, versioned dataset
- explicit train/validation split
- tokenizer frozen/versioned
- checkpoint/resume works
- validation metrics are logged
- router telemetry is logged
- run is reproducible from commit + config + dataset version

Until then, successful runs are architecture experiments, not model training milestones.
