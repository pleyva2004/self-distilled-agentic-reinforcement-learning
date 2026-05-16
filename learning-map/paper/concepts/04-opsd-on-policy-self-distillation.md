# On-Policy Self-Distillation (OPSD)

**Level:** intermediate
**Prerequisites:** [01-multi-turn-agentic-mdp](01-multi-turn-agentic-mdp.md), Chain Ch 28 (SFT/RLHF/DPO)
**Used by:** [05-teacher-branch-privileged-context](05-teacher-branch-privileged-context.md), [06-per-token-reverse-kl](06-per-token-reverse-kl.md), [08-multi-turn-instability-observation](08-multi-turn-instability-observation.md)

## Plain-English intro
On-policy self-distillation runs **two forward passes** of the same policy: one (the "student") on the normal context, one (the "teacher") on a richer context. We then ask the student to match the teacher token-by-token via a per-token KL. "Self" because both are the same parameters. "On-policy" because the trajectory is sampled by the student.

## Formal definition
Per-token reverse KL (used in OPSD):
$$D_{\mathrm{RKL}}^{(t)} = D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid s_t) \,\big\|\, \pi_T(\cdot \mid s_t^+)\big) = \sum_{v \in \mathcal{V}} \pi_\theta(v \mid s_t) \log \frac{\pi_\theta(v \mid s_t)}{\pi_T(v \mid s_t^+)}.$$

Standard OPSD auxiliary loss:
$$\mathcal{L}_{\mathrm{OPSD}}(\theta) = \mathrm{Agg}\big[D_{\mathrm{RKL}}^{(t)}\big].$$

## Why this matters for the paper
OPSD is the *workable* building block that breaks under multi-turn rollout (see [08](08-multi-turn-instability-observation.md), [09](09-asymmetric-trust-observation.md)). SDAR is, structurally, "OPSD + a token-level gate".

## Code
See [`../code/04-opsd-on-policy-self-distillation.py`](../code/04-opsd-on-policy-self-distillation.py).

## Cross-link to the chain
Chain Ch 28 (SFT/RLHF/DPO) — KL-to-reference is the same primitive as the RLHF KL regularizer to the SFT policy.
