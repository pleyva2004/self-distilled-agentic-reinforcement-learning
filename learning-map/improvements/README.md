# SDAR improvements concept graph

Four improvement concepts split across two modes:
- **PROOFS** (101, 104) — formal mathematical extensions, fully written in `proofs/`.
- **MEASUREMENTS** (102, 103) — sandbox experiments measured by Agent C's scripts.

```mermaid
graph TD
    classDef proof fill:#d4edda,stroke:#155724,color:#000
    classDef measurement fill:#fff3cd,stroke:#856404,color:#000

    P12["paper.12 Gap gating"]
    P9["paper.9 Asymmetric trust"]
    P14["paper.14 SDAR objective"]

    I101["101. Bias bound (PROOF)"]:::proof
    I102["102. Adaptive gating (MEAS)"]:::measurement
    I103["103. Lambda sweep (MEAS)"]:::measurement
    I104["104. SDAR as constrained RL (PROOF)"]:::proof

    P12 --> I101
    P9 --> I101
    P12 --> I102
    P14 --> I103
    P14 --> I104

    click I101 "concepts/101-gating-bias-bound.md"
    click I102 "concepts/102-adaptive-gating-threshold.md"
    click I103 "concepts/103-lambda-sensitivity-sweep.md"
    click I104 "concepts/104-sdar-as-constrained-rl.md"
```

## Validation modes

| ID  | Mode        | Validation file                                      |
|-----|-------------|------------------------------------------------------|
| 101 | PROOF       | `proofs/gating-bias-bound.tex`                       |
| 102 | MEASUREMENT | `improvements/adaptive-gate.py` (sandbox)            |
| 103 | MEASUREMENT | `improvements/lambda-sweep.py` (sandbox)             |
| 104 | PROOF       | `proofs/sdar-as-constrained-rl.tex`                  |
