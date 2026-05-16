# Observation 2: Asymmetric trust in privileged guidance

**Level:** advanced
**Prerequisites:** [05-teacher-branch-privileged-context](05-teacher-branch-privileged-context.md), [07-single-sample-gap-estimator](07-single-sample-gap-estimator.md)
**Used by:** [12-gap-gating](12-gap-gating.md)

## Plain-English intro
The teacher is the *same* model with privileged extra context. So:
- If $\Delta_t > 0$: the privileged context endorses the student's sampled token. High-quality distillation signal.
- If $\Delta_t < 0$: ambiguous. Either the token *should* be suppressed, *or* the retrieval was bad and the privileged context misled the teacher. Three failure modes: (i) skill quality, (ii) skill utilization, (iii) multi-turn drift.

The asymmetry is intrinsic — it follows from the teacher's parameter sharing — not a hyperparameter choice.

## Formal definition
Information-theoretic framing: privileged context can only *strictly* increase the log-likelihood of "good" continuations on average; it cannot reliably penalize "bad" ones (because the teacher has no access to the future reward). Formally, if $C^+$ is the privileged context and $G$ a good-completion event,
$$\log \pi_\theta(G \mid s_t, C^+) \geq \log \pi_\theta(G \mid s_t)$$
in expectation when retrieval is informative — but no analogous lower bound holds for arbitrary tokens.

## Why this matters for the paper
This is the design constraint that picks the sigmoid gate over a symmetric one (e.g. tanh). Sigmoid structurally up-weights $\Delta_t > 0$ and down-weights $\Delta_t < 0$.

## Code
See [`../code/09-asymmetric-trust-observation.py`](../code/09-asymmetric-trust-observation.py) — toy bandit showing positive-gap tokens are reliable, negative-gap ones are not.

## Cross-link to the chain
Chain Ch 28 — the asymmetry parallels the well-known "DPO is conservative on negatives" observation.
