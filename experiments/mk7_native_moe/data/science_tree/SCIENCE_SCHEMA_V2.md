# MK7NMoE Science Schema V2

## Core idea

Science Schema V2 is not a single taxonomy tree. It is a **polyhierarchical knowledge graph**.

A canonical knowledge entity exists once, but may appear in many classification systems simultaneously.

```text
Canonical Entity
    │
    ├── classified_as → Academic taxonomy
    ├── classified_as → Professional taxonomy
    ├── classified_as → Industry taxonomy
    ├── classified_as → Competency taxonomy
    ├── classified_as → Cognitive taxonomy
    ├── classified_as → Regulatory taxonomy
    └── classified_as → Custom/internal taxonomy
```

This prevents the false assumption that one science, discipline, topic or concept has exactly one parent.

## Why this matters

The same entity can legitimately live under different knowledge systems.

Example: `Public-key cryptography` can simultaneously be:

- Computer Science → Security → Cryptography
- Cybersecurity → Cryptographic Controls
- Security Engineering → Key Management
- Network Security → TLS Foundations
- Applied Mathematics → Number-theoretic Applications

These are not duplicate concepts. They are multiple views of the same canonical entity.

## Canonical graph model

```text
                       ┌───────────────────────┐
                       │   Classification      │
                       │      Systems          │
                       └──────────┬────────────┘
                                  │
                    classified_as │
                                  ▼
┌──────────────┐      ┌───────────────────────┐      ┌───────────────┐
│ Prerequisite │─────▶│   Canonical Entity    │─────▶│ Related Entity│
│   Entities   │      │ science/topic/concept │      │   Entities    │
└──────────────┘      └──────────┬────────────┘      └───────────────┘
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼              ▼
               Knowledge      Capabilities    Provenance
                 Atoms        & Reasoning     & Verification
```

## Two different kinds of structure

### 1. Classification structure

This answers:

> Where does this entity appear under a particular system of knowledge organization?

It is represented with taxonomy memberships.

A single entity may have zero, one or many memberships.

### 2. Semantic/dependency graph

This answers:

> What does this entity require, cause, enable, use, contradict, specialize or relate to?

Examples:

```text
Linear Algebra ──prerequisite_of──▶ Machine Learning
Cryptography ──used_by──▶ TLS
Authentication ──requires──▶ Identity
SQL Injection ──mitigated_by──▶ Parameterized Queries
Probability ──enables──▶ Bayesian Inference
```

Classification and semantic relations must not be confused.

## Main entity hierarchy

The familiar science hierarchy is retained as one useful view:

```text
Science
└── Discipline
    └── Subdiscipline
        └── Topic
            └── Concept
                ├── Knowledge Atoms
                ├── Learning Objectives
                ├── Skills
                ├── Reasoning Patterns
                └── Task Variants
```

But this is only one projection of the graph, not the canonical storage model.

## Multiple classification systems

Schema V2 supports independent systems such as:

- Academic classification
- University curriculum classification
- Professional certification classification
- Industry/application classification
- Competency/skills classification
- Standards-based classification
- Regulatory classification
- Cognitive/learning-objective classification
- Methodological classification
- Internal MK7NMoE classification

A taxonomy itself is versioned and has its own authority/source.

## Cross-science prerequisites

Prerequisites are first-class edges and may cross sciences completely.

Example:

```text
Machine Learning
├── requires → Linear Algebra
├── requires → Probability
├── requires → Optimization
└── recommended → Programming
```

This is critical because real curricula are graphs, not nested folders.

## Knowledge atoms

Every concept may decompose into atomic teachable/verifiable units:

- definition
- fact
- property
- component
- taxonomy statement
- relation
- mechanism
- process step
- rule/law
- constraint
- prerequisite
- exception
- failure mode
- example
- counterexample
- evidence
- misconception
- boundary condition

These are the units from which dataset items are generated.

## Capability layer

Knowledge alone is insufficient. Each concept is crossed with capabilities:

- recall
- explanation
- extraction
- classification
- comparison
- calculation
- diagnosis
- planning
- decision making
- problem solving
- tool use
- verification

And with reasoning patterns:

- deduction
- induction
- abduction
- causal
- analogical
- temporal
- spatial
- quantitative
- constraint
- multi-hop
- counterfactual
- uncertainty

This creates a `Knowledge × Capability × Reasoning` dataset rather than a bag of facts.

## Negative knowledge

Every mature concept should also include negative cases:

- false statement
- plausible-but-wrong answer
- common misconception
- invalid inference
- missing prerequisite
- insufficient information
- counterexample
- exception/boundary case

This is intended to reduce hallucination and improve refusal to infer unsupported conclusions.

## Coverage cube

Dataset coverage is measured across multiple axes:

```text
Science
× Classification membership
× Concept
× Capability
× Reasoning pattern
× Task type
× Difficulty
× Language
× Representation
× Verification method
```

This lets the builder detect holes such as:

`Cryptography × causal reasoning × advanced × Arabic = missing`

instead of merely reporting a raw sample count.

## MK7NMoE routing policy

Science/taxonomy labels are **not oracle routing labels**.

They are retained for telemetry so that after training we can ask:

- Did an expert specialize by science?
- Did an expert specialize by language?
- Did an expert specialize by reasoning type?
- Did specialization occur by skill rather than domain?
- Are some taxonomies correlated with particular experts?

The router must discover specialization unless a separate controlled oracle-routing experiment is explicitly enabled.

## Example: one entity in many places

```text
                     PUBLIC-KEY CRYPTOGRAPHY
                              │
       ┌──────────────────────┼───────────────────────┐
       │                      │                       │
       ▼                      ▼                       ▼
Academic CS            Cybersecurity          Security Engineering
Security               Cryptography           Key Management
Cryptography
       │
       ├── requires → Modular Arithmetic
       ├── related_to → Digital Signatures
       ├── used_by → TLS
       └── contrasts_with → Symmetric Cryptography
```

There remains exactly one canonical concept ID.

## Implementation files

Machine-readable schema:

`science_schema_v2.yaml`

Existing Cybersecurity tree should be migrated incrementally to V2 rather than duplicated wholesale.

## Next implementation gates

1. Add taxonomy registry with stable IDs and versions.
2. Migrate Cybersecurity entities to canonical IDs.
3. Add multi-taxonomy memberships to each migrated entity.
4. Add prerequisite and cross-domain relation edges.
5. Build a schema validator.
6. Build a graph compiler that produces:
   - taxonomy projections
   - prerequisite graph
   - concept graph
   - coverage manifest
   - dataset-generation queue
7. Generate train/validation/test items only after provenance and verification requirements pass.
