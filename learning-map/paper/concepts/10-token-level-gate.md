# Sigmoid token-level gate (stop-gradient)

**Level:** advanced
**Prerequisites:** [07-single-sample-gap-estimator](07-single-sample-gap-estimator.md), [08-multi-turn-instability-observation](08-multi-turn-instability-observation.md)
**Used by:** [11-entropy-gating](11-entropy-gating.md), [12-gap-gating](12-gap-gating.md), [13-soft-or-gating](13-soft-or-gating.md), [14-sdar-combined-objective](14-sdar-combined-objective.md)

## Plain-English intro
The shared scaffolding of all SDAR gating strategies: a sigmoid of a per-token score, with **stop-gradient** applied to the gate so gradients flow only through the student log-probability inside the loss. The gate's job is to softly select *which tokens* should receive distillation pressure.

## Formal definition
Generic gate:
$$g_t = \mathrm{sg}\big(\sigma(\beta \cdot \mathrm{score}_t)\big), \qquad g_t \in (0, 1),\;\beta > 0,$$
where $\mathrm{sg}$ is the stop-gradient operator and $\sigma(z) = 1/(1+e^{-z})$.

Plug into the SDAR token loss:
$$\ell_t^{\mathrm{SDAR}} = g_t \cdot \big(\log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t)\big) = g_t \cdot \Delta_t.$$

When the gate and teacher are detached, the gradient is purely
$$\nabla_\theta \mathcal{L}_{\mathrm{SDAR}} = -\mathrm{Agg}\big[g_t \cdot \nabla_\theta \log \pi_\theta(y_t \mid s_t)\big]$$
— a gated REINFORCE-style update on the student.

## Why this matters for the paper
The stop-gradient is what keeps the dynamics clean. Without it, the gate would learn to game its own activation (a second-order coupling that destabilizes training). With it, SDAR is provably equivalent to a per-token reweighted policy gradient.

## Code
See [`../code/10-token-level-gate.py`](../code/10-token-level-gate.py).

## Cross-link to the chain
Chain Ch 30 (max-ent RL) — the gate is the per-token analog of an attention/Lagrange weight.
