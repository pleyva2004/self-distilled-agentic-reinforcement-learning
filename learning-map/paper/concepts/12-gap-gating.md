# Gap gating strategy (asymmetric trust)

**Level:** advanced
**Prerequisites:** [10-token-level-gate](10-token-level-gate.md), [09-asymmetric-trust-observation](09-asymmetric-trust-observation.md)
**Used by:** [13-soft-or-gating](13-soft-or-gating.md), [101-gating-bias-bound](../../improvements/concepts/101-gating-bias-bound.md), [102-adaptive-gating-threshold](../../improvements/concepts/102-adaptive-gating-threshold.md)

## Plain-English intro
The conceptual centerpiece of SDAR. Use the teacher-student gap $\Delta_t$ itself as the gating score. When the teacher endorses the student's sampled token (positive gap), distill heavily. When the teacher disagrees (negative gap, possibly due to a bad skill retrieval), attenuate.

## Formal definition
$$g_t^{\mathrm{gap}} = \sigma(\beta \cdot \Delta_t),\qquad \Delta_t = \log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t).$$

Resulting per-token loss term:
$$\ell_t^{\mathrm{SDAR,gap}} = \sigma(\beta \Delta_t) \cdot \Delta_t.$$

This is an **asymmetric** function of $\Delta_t$:
- $\Delta_t \to +\infty$: $\sigma(\beta\Delta_t) \to 1$, loss $\approx \Delta_t$ (full distillation).
- $\Delta_t \to -\infty$: $\sigma(\beta\Delta_t) \to 0$, loss $\to 0$ (no update).
- $\Delta_t = 0$: $\sigma(0) = 1/2$, loss $= 0$.

## Why this matters for the paper
Gap gating directly implements the §3 asymmetric-trust principle as a structural property of the activation function. No threshold, no clipping, no auxiliary network — just $\sigma$ composed with the gap.

## Code
See [`../code/12-gap-gating.py`](../code/12-gap-gating.py).

## Cross-link to the chain
Chain Ch 31 — view $\sigma(\beta \Delta_t)$ as a soft analog of the PPO clip's "trust region" indicator.
