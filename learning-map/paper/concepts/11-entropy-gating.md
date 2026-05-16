# Entropy gating strategy

**Level:** advanced
**Prerequisites:** [10-token-level-gate](10-token-level-gate.md)
**Used by:** [13-soft-or-gating](13-soft-or-gating.md)

## Plain-English intro
"Distill where the student is uncertain." When the student's entropy at $s_t$ is high, it has low confidence about the next token — there is more room for the teacher to help. When entropy is low, the student is decisive and distillation pressure is wasted (or worse, slows convergence to its own peak).

## Formal definition
Student entropy:
$$h_t = -\sum_{v \in \mathcal{V}} \pi_\theta(v \mid s_t) \log \pi_\theta(v \mid s_t).$$

Entropy gate:
$$g_t^{\mathrm{ent}} = \sigma(\beta \cdot h_t).$$

Note $h_t \geq 0$, so the gate is always above $1/2$ — entropy gating is *non-zero everywhere* and only modulates the strength of distillation.

## Why this matters for the paper
Entropy gating is one of three baselines benchmarked. It does *not* address asymmetric trust (high entropy at a state where retrieval failed is precisely where distillation hurts most). It is, however, computable purely from the student forward pass — no teacher required if you only want this signal.

## Code
See [`../code/11-entropy-gating.py`](../code/11-entropy-gating.py).

## Cross-link to the chain
Chain Ch 30 — student entropy is the same quantity used in soft Q-learning's entropy bonus, but used here as a *gate* on a different loss.
