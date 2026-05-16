# Adaptive gating via running-quantile of gaps

**Level:** advanced
**Prerequisites:** [12-gap-gating](../../paper/concepts/12-gap-gating.md)
**Used by:** measurement file `improvements/adaptive-gate.py` (Agent C)

## Plain-English intro
Fixed sharpness $\beta$ in $g_t = \sigma(\beta \Delta_t)$ assumes the *scale* of $\Delta_t$ is constant across training. Empirically it is not — early in training $|\Delta_t|$ can be large; late in training it shrinks as student approaches teacher. We propose adapting $\beta$ (or shifting the sigmoid threshold) using a running estimate of the gap distribution's median and IQR.

## Formal definition
Let $\hat\mu_t$ and $\hat \tau_t$ be running median and IQR of recently observed gap values. The adaptive gate:
$$g_t^{\mathrm{adapt}} = \sigma\!\left(\frac{\Delta_t - \hat\mu_t}{\hat\tau_t / \log 3}\right),$$
where the $\log 3$ normalizer makes $g_t = 3/4$ at the upper-quartile of $\Delta_t$ regardless of the empirical scale. This guarantees a constant *fraction of tokens get strong distillation pressure* — the design knob shifts from "absolute sharpness" to "fraction of tokens to trust", a much more interpretable hyperparameter.

## Why this matters for the paper
The paper sweeps $\beta$ as a fixed hyperparameter; the right value depends on the model + task pair. Adaptive gating eliminates this sweep and (we conjecture) yields a Pareto improvement in the loss-vs-stability frontier.

## Code
See [`../code/102-adaptive-gating-threshold.py`](../code/102-adaptive-gating-threshold.py) — measures fraction-of-tokens-firing under fixed vs adaptive gating across a simulated training trajectory of shifting gap distributions.
