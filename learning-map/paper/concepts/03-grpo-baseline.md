# GRPO baseline (per-token PPO clip + group-relative advantage)

**Level:** intermediate
**Prerequisites:** [02-rl-trajectory-reward](02-rl-trajectory-reward.md), Chain Ch 31 (PG/GRPO)
**Used by:** [14-sdar-combined-objective](14-sdar-combined-objective.md)

## Plain-English intro
GRPO (Group-Relative Policy Optimization) is the RL backbone SDAR keeps unchanged. For each prompt it samples $G$ responses, computes a group-relative advantage $A^{(i)} = (R^{(i)} - \bar R) / \mathrm{std}(R)$ (no learned value function), then applies a per-token PPO clipped objective + KL-to-reference regularizer.

## Formal definition
Per the paper (Equation 2):
$$\mathcal{L}_{\mathrm{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \mathrm{Agg}\Big[\min\big(r_t^{(i)} A^{(i)},\, \mathrm{clip}(r_t^{(i)}, 1{-}\epsilon, 1{+}\epsilon) A^{(i)}\big)\Big] + \beta \cdot \frac{1}{G}\sum_{i=1}^{G} \mathrm{Agg}\big[D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})\big]$$

where $r_t^{(i)} = \pi_\theta(y_t^{(i)} \mid s_t^{(i)}) / \pi_{\theta_{\mathrm{old}}}(y_t^{(i)} \mid s_t^{(i)})$ and $\mathrm{Agg}(z_{1:T}) = \sum_t m_t z_t / \sum_t m_t$ is the masked-token average.

## Why this matters for the paper
SDAR's combined loss is exactly $\mathcal{L}_{\mathrm{GRPO}} + \lambda_{\mathrm{SDAR}} \mathcal{L}_{\mathrm{SDAR}}$. The GRPO term carries the long-horizon outcome signal; the SDAR term injects per-token guidance.

## Code
See [`../code/03-grpo-baseline.py`](../code/03-grpo-baseline.py).

## Cross-link to the chain
Chain Ch 31 (PG/GRPO/RLHF-DPO bridge) derives this verbatim. The RLM sibling study uses the same root-loss form (its Section 2).
