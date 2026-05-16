# Per-token reverse KL between teacher and student

**Level:** intermediate
**Prerequisites:** [04-opsd-on-policy-self-distillation](04-opsd-on-policy-self-distillation.md), [05-teacher-branch-privileged-context](05-teacher-branch-privileged-context.md)
**Used by:** [07-single-sample-gap-estimator](07-single-sample-gap-estimator.md)

## Plain-English intro
At each token position $t$, OPSD measures how far the student distribution is from the teacher distribution using **reverse KL** (student is the sampling distribution, teacher is the reference). Why reverse and not forward? Reverse KL is *mode-seeking*: it forces the student to put mass only where the teacher has mass, but does not force the student to cover *all* teacher modes. For tool-use / agent tokens this is desirable — there is one "right" tool per state.

## Formal definition
$$D_{\mathrm{RKL}}^{(t)} = \sum_{v \in \mathcal{V}} \pi_\theta(v \mid s_t) \log \frac{\pi_\theta(v \mid s_t)}{\pi_T(v \mid s_t^+)}.$$

Equivalent expectation form (basis for the single-sample estimator):
$$D_{\mathrm{RKL}}^{(t)} = \mathbb{E}_{v \sim \pi_\theta(\cdot \mid s_t)}\Big[\log \pi_\theta(v \mid s_t) - \log \pi_T(v \mid s_t^+)\Big].$$

## Why this matters for the paper
This is the quantity the gating mechanism is built on. Computing it exactly is $O(|\mathcal{V}|)$ per token — for Qwen 2.5's vocab of $\approx 152$k, that is the dominant cost. The single-sample estimator ([07](07-single-sample-gap-estimator.md)) sidesteps this.

## Code
See [`../code/06-per-token-reverse-kl.py`](../code/06-per-token-reverse-kl.py).

## Cross-link to the chain
Chain Ch 30 (max-ent RL) — reverse KL is the divergence used in soft Q-learning and is mode-seeking; contrast with forward KL.
