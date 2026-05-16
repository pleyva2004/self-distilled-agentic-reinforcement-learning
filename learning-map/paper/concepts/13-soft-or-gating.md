# Soft-OR combined gating strategy

**Level:** advanced
**Prerequisites:** [11-entropy-gating](11-entropy-gating.md), [12-gap-gating](12-gap-gating.md)
**Used by:** [14-sdar-combined-objective](14-sdar-combined-objective.md)

## Plain-English intro
Combine the two: gate fires if **either** the student is uncertain OR the teacher endorses the token. Implemented as a smooth analog of the boolean "OR" via De Morgan: $1 - (1-h_t)(1-\Delta_t)$.

## Formal definition
$$g_t^{\mathrm{or}} = \sigma\Big(\beta \cdot \big[1 - (1 - h_t)(1 - \Delta_t)\big]\Big).$$

Note $h_t \geq 0$ but $\Delta_t \in \mathbb{R}$ unbounded; in practice $\Delta_t$ is normalized (paper uses min-max or z-score over the rollout batch) before composing into the De Morgan form, so the product makes sense.

## Why this matters for the paper
Soft-OR is the most expressive of the three gates and is one of the configurations the paper benchmarks. It is also the gate where the gap-distribution sensitivity (cf. improvement [102](../../improvements/concepts/102-adaptive-gating-threshold.md)) bites hardest, because both inputs feed in.

## Code
See [`../code/13-soft-or-gating.py`](../code/13-soft-or-gating.py).

## Cross-link to the chain
Chain Ch 30 — soft-OR is structurally a soft maximum, the dual of the soft-min appearing in soft Q.
