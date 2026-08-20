# MK7NMoE Science Tree v0.1 — Cybersecurity

This is the first concrete science-tree template for MK7NMoE dataset construction.

## Why Cybersecurity first

Cybersecurity has a clean hierarchy and mixes declarative knowledge, causal reasoning, procedures, diagnostics, policy, tool semantics, scenarios, and edge cases. That makes it a useful stress test for whether one science tree can generate a rich curriculum instead of a flat instruction collection.

## Canonical hierarchy

```text
Science
└── Discipline
    └── Topic
        └── Concept
            ├── Learning objective
            ├── Atomic Knowledge Item (AKI)
            └── Task variants
```

The machine-readable source of truth is `cybersecurity.yaml`.

## Current disciplines

1. Security Foundations
2. Identity and Access Management
3. Network Security
4. Endpoint and Host Security
5. Application Security
6. Applied Cryptography
7. Cloud Security
8. Data Security and Privacy Engineering
9. Detection, Incident Response, and Forensics
10. Vulnerability and Exposure Management
11. Governance, Risk, and Compliance
12. Security Operations
13. Human and Organizational Security

## Example expansion

```text
Cybersecurity
└── Detection, Incident Response, and Forensics
    └── Incident Response
        └── Identification and Triage
            ├── definition / procedure
            ├── diagnostic signals
            ├── scenario
            ├── explain task
            ├── diagnosis task
            ├── sequencing task
            └── structured-output task
```

A leaf is not one training example. A leaf is a knowledge target from which multiple task types are generated.

## Atomic Knowledge Item types

The current tree supports:

- definition
- fact
- principle
- mechanism
- procedure
- comparison
- failure mode
- diagnostic signal
- mitigation
- policy rule
- tool semantics
- scenario

## Dataset generation contract

For each concept, build a source-backed packet first. Then extract or author AKIs. Only after that do we generate task variants.

```text
Sources
  ↓
Parse / OCR / normalize
  ↓
Science-tree mapping
  ↓
AKI extraction
  ↓
Teacher expansion
  ↓
Verification
  ↓
Deduplication
  ↓
JSONL examples
  ↓
Train / validation / test
```

The tree must preserve provenance all the way to generated examples. Synthetic examples must be tagged as synthetic. Evaluation examples require verification.

## Language and difficulty policy v0.1

Initial generation target:

- English: 55%
- Arabic: 25%
- bilingual: 20%

Difficulty:

- foundation: 35%
- intermediate: 40%
- advanced: 25%

These are starting priors, not fixed scientific truths. Coverage and validation results should be used to revise them.

## Important MoE rule

Science/domain labels are dataset metadata. They are **not** oracle routing labels.

We should not encode rules such as:

```text
cybersecurity -> expert 3
```

Instead, train on the mixed curriculum and record router telemetry by science, discipline, topic, task type, language, and difficulty. Then measure whether expert specialization emerges naturally.

This lets us answer useful questions later:

- Does one expert specialize in identity/security policy?
- Do network and cloud concepts share experts?
- Does Arabic trigger a language specialist or remain distributed?
- Does one router collapse on procedural tasks?
- Do different router algorithms form different specializations under identical data?

## What this tree is NOT

It is not yet a completed cybersecurity corpus. It is the coverage map and generation specification.

A real training dataset begins only after we attach source-backed AKIs and generate validated examples. Counting tree nodes or synthetic prompts alone would give a false sense of dataset completeness.

## Immediate next build

The next artifact should be a dataset builder that reads `cybersecurity.yaml` and emits a coverage manifest such as:

```json
{
  "science": "cybersecurity",
  "discipline": "identity_access",
  "topic": "authentication",
  "concept": "mfa",
  "aki_type": "mechanism",
  "task_type": "reason",
  "source_id": "...",
  "provenance": "...",
  "language": "en",
  "difficulty": "intermediate",
  "synthetic": false,
  "verified": true
}
```

That manifest becomes the bridge between the science tree and `mk7nmoe-data-v0.1`.
