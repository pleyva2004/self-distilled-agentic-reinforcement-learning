# Single-sample teacher-student log-prob gap estimator

**Level:** advanced
**Prerequisites:** [06-per-token-reverse-kl](06-per-token-reverse-kl.md), Chain Ch 31 (PG theorem)
**Used by:** [09-asymmetric-trust-observation](09-asymmetric-trust-observation.md), [10-token-level-gate](10-token-level-gate.md), [12-gap-gating](12-gap-gating.md)

## Plain-English intro
We already drew $y_t \sim \pi_\theta(\cdot \mid s_t)$ for the rollout. So instead of summing over the whole vocabulary, evaluate the integrand at this one sample. That gives an **unbiased** single-sample estimate of the reverse KL. Negate it and you have the **teacher-student gap** $\Delta_t$ — the central scalar of SDAR.

## Formal definition
Single-sample reverse-KL estimate (one term in the expectation):
$$\hat D_{\mathrm{RKL}}^{(t)} = \log \pi_\theta(y_t \mid s_t) - \log \pi_T(y_t \mid s_t^+),\qquad \mathbb{E}_{y_t \sim \pi_\theta}[\hat D_{\mathrm{RKL}}^{(t)}] = D_{\mathrm{RKL}}^{(t)}.$$

Teacher-student gap (sign-flipped):
$$\boxed{\;\Delta_t = \log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t).\;}$$

Interpretation: $\Delta_t > 0$ ⇔ teacher endorses the student's sampled token; $\Delta_t < 0$ ⇔ teacher disagrees.

## Why this matters for the paper
$\Delta_t$ is *both* the cheap distillation signal *and* the natural input to the gate. One scalar serves two purposes — that economy is the mathematically clever core of SDAR.

## Code
See [`../code/07-single-sample-gap-estimator.py`](../code/07-single-sample-gap-estimator.py).

## Cross-link to the chain
Chain Ch 31 — same trick as the policy-gradient theorem proof: replace an expensive expectation with a single sample drawn from the distribution that's already being sampled anyway.
