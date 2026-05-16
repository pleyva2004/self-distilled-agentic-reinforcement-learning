# Observation 1: Multi-turn OPSD instability

**Level:** advanced
**Prerequisites:** [04-opsd-on-policy-self-distillation](04-opsd-on-policy-self-distillation.md), [05-teacher-branch-privileged-context](05-teacher-branch-privileged-context.md)
**Used by:** [10-token-level-gate](10-token-level-gate.md)

## Plain-English intro
Naive OPSD applied to a multi-turn agent destabilizes training: as the student drifts off the teacher-supported trajectory distribution, the per-turn KL grows superlinearly and the distillation signal becomes pure noise. The teacher distribution at $s_t$ was implicitly conditioned on a different (privileged) trajectory; once the student wanders far enough, the teacher's preferences are stale.

## Formal definition
Conceptually: let $\rho_T(s_t)$ be the state-visitation density induced by the teacher branch and $\rho_\theta(s_t)$ that of the student. The per-turn off-policy correction needed to make OPSD well-defined is the importance ratio $\rho_T(s_t)/\rho_\theta(s_t)$, which compounds across turns. Without correction, the OPSD gradient becomes
$$\nabla_\theta \mathcal{L}_{\mathrm{OPSD}} = -\mathbb{E}_{s_t \sim \rho_\theta}[\nabla_\theta D_{\mathrm{RKL}}^{(t)}],$$
but the *target* $\pi_T(\cdot \mid s_t^+)$ is reliable only on $\rho_T$ — a covariate shift that grows with horizon $H$.

## Why this matters for the paper
This is half of the motivation for gating. A token-level gate whose value depends on a *local* signal (the gap, or entropy) does not require global covariate-shift correction.

## Code
See [`../code/08-multi-turn-instability-observation.py`](../code/08-multi-turn-instability-observation.py) — finite-MDP demo where ungated OPSD's loss explodes with horizon.

## Cross-link to the chain
Chain Ch 31 — same compounding-error story as classical off-policy RL with importance sampling.
