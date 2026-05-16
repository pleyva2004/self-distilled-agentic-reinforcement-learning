# SDAR combined objective

**Level:** advanced
**Prerequisites:** [03-grpo-baseline](03-grpo-baseline.md), [10-token-level-gate](10-token-level-gate.md)
**Used by:** [103-lambda-sensitivity-sweep](../../improvements/concepts/103-lambda-sensitivity-sweep.md), [104-sdar-as-constrained-rl](../../improvements/concepts/104-sdar-as-constrained-rl.md)

## Plain-English intro
The full SDAR loss: GRPO carries the trajectory-level reward signal; the gated SDAR auxiliary injects per-token guidance from the privileged-context teacher branch. One scalar hyperparameter $\lambda_{\mathrm{SDAR}}$ controls the mixing.

## Formal definition
Per-token SDAR auxiliary:
$$\ell_t^{\mathrm{SDAR}} = g_t \cdot \Delta_t = g_t \cdot \big(\log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t)\big).$$

Aggregated:
$$\mathcal{L}_{\mathrm{SDAR}}(\theta) = \mathrm{Agg}(\ell_{1:T}^{\mathrm{SDAR}}) = \frac{\sum_t m_t g_t \Delta_t}{\sum_t m_t}.$$

Combined objective:
$$\boxed{\;\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{GRPO}}(\theta) + \lambda_{\mathrm{SDAR}} \cdot \mathcal{L}_{\mathrm{SDAR}}(\theta).\;}$$

Gradient (gate + teacher detached):
$$\nabla_\theta \mathcal{L}_{\mathrm{SDAR}} = -\mathrm{Agg}\big[g_t \cdot \nabla_\theta \log \pi_\theta(y_t \mid s_t)\big].$$

## Why this matters for the paper
This is *the* equation. Everything else — observations, gating strategies, sampled-token estimator — exists to make this single loss numerically well-behaved.

## Code
See [`../code/14-sdar-combined-objective.py`](../code/14-sdar-combined-objective.py).

## Cross-link to the chain
Chain Ch 30 — view $\lambda_{\mathrm{SDAR}}$ as a Lagrange multiplier on a per-token trust constraint (formalized in improvement [104](../../improvements/concepts/104-sdar-as-constrained-rl.md)).
