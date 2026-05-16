# Teacher branch with privileged context (retrieved skills)

**Level:** intermediate
**Prerequisites:** [04-opsd-on-policy-self-distillation](04-opsd-on-policy-self-distillation.md)
**Used by:** [06-per-token-reverse-kl](06-per-token-reverse-kl.md), [08-multi-turn-instability-observation](08-multi-turn-instability-observation.md), [09-asymmetric-trust-observation](09-asymmetric-trust-observation.md)

## Plain-English intro
The teacher in SDAR is **not** a stronger model. It's the same policy with extra context appended — typically retrieved skill text (worked examples, documentation, hints) that's available only at training time, never at inference. This makes "self-distillation" literal: we distill from a privileged version of ourselves.

## Formal definition
$$s_t^+ = s_t \;\Vert\; \mathrm{retrieve}(\mathrm{skills} \mid x, o_{1:k(t)}),$$
where $\Vert$ denotes context concatenation. The teacher distribution is then
$$\pi_T(\cdot \mid s_t^+) := \pi_\theta(\cdot \mid s_t^+).$$

Importantly, $\pi_T$ shares parameters with $\pi_\theta$; the only difference is the conditioning context.

## Why this matters for the paper
Privileged context is what makes the teacher *useful* to distill from — it has information the student lacks. But it's also why the teacher is *not unconditionally trustworthy* (see [09](09-asymmetric-trust-observation.md)): the retrieval can fail, and the privileged tokens can mislead.

## Code
See [`../code/05-teacher-branch-privileged-context.py`](../code/05-teacher-branch-privileged-context.py).

## Cross-link to the chain
Chain Ch 28 — privileged context as a form of inference-time conditioning gap; closely related to the "expert iteration" pattern.
