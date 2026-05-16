# Sensitivity sweep of lambda_SDAR

**Level:** intermediate
**Prerequisites:** [14-sdar-combined-objective](../../paper/concepts/14-sdar-combined-objective.md)
**Used by:** measurement file `improvements/lambda-sweep.py` (Agent C)

## Plain-English intro
The paper uses a single fixed $\lambda_{\mathrm{SDAR}}$ chosen by validation. We investigate two questions: (1) how flat is the loss landscape over $\lambda \in [0.01, 10]$? — i.e. how brittle is the choice — and (2) does an *annealing schedule* on $\lambda$ outperform a constant value? Anneal high early (when teacher signal is most informative) and decay as the student catches up.

## Formal definition
Let $\mathcal{L}(\theta; \lambda) = \mathcal{L}_{\mathrm{GRPO}}(\theta) + \lambda \cdot \mathcal{L}_{\mathrm{SDAR}}(\theta)$. Define the sensitivity at the optimum
$$S(\lambda^\star) = \frac{d\, \mathrm{val\_loss}(\theta^\star(\lambda))}{d \lambda}\bigg|_{\lambda = \lambda^\star},$$
where $\theta^\star(\lambda)$ is the parameter set that emerges from training with a given $\lambda$. By the envelope theorem, $S(\lambda^\star) = \mathcal{L}_{\mathrm{SDAR}}(\theta^\star)$ — i.e. the sensitivity is *identifiable from a single trained checkpoint*.

Annealing schedule (proposed):
$$\lambda(t) = \lambda_0 \cdot \exp(-t / \tau)$$
with $\lambda_0$ from the paper's value and $\tau$ chosen so $\lambda(t_{\mathrm{end}}) = \lambda_0 / 10$.

## Why this matters for the paper
The fixed-$\lambda$ choice is a hidden hyperparameter that drives training cost and final performance. Either showing the loss is flat (good — the paper's choice is robust) or showing annealing helps (good — a free win) is informative.

## Code
See [`../code/103-lambda-sensitivity-sweep.py`](../code/103-lambda-sensitivity-sweep.py) — measures the simulated training-loss curve over a $\lambda$ grid and contrasts a constant schedule with an exponential anneal.
