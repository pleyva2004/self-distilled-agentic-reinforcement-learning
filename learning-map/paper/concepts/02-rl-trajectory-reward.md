# Trajectory-level RL reward (verifier-driven)

**Level:** intro
**Prerequisites:** [01-multi-turn-agentic-mdp](01-multi-turn-agentic-mdp.md), Chain Ch 31 (PG/GRPO)
**Used by:** [03-grpo-baseline](03-grpo-baseline.md)

## Plain-English intro
The supervision signal in agentic RL is **sparse**: one scalar reward at the end of a (potentially long) trajectory. There is no per-token target. This is the central pain SDAR addresses — it adds dense per-token signal via distillation.

## Formal definition
For trajectory $\tau = (s_0, y_1, s_1, \dots, y_T, s_T)$:

$$R(\tau) = \mathrm{verify}(\tau) \in \{0, 1\} \quad \text{(or scaled real value).}$$

The policy-gradient objective is
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)],$$
estimated via REINFORCE / PPO / GRPO. With *trajectory-level* reward only, the per-token gradient $\nabla_\theta \log \pi_\theta(y_t \mid s_t)$ is multiplied by the *same* advantage for every token $t$ in the trajectory — extremely high variance for long horizons.

## Why this matters for the paper
SDAR's auxiliary distillation loss assigns each token its own gated update. This converts trajectory-level supervision into token-level supervision *without* needing a process reward model.

## Code
See [`../code/02-rl-trajectory-reward.py`](../code/02-rl-trajectory-reward.py).

## Cross-link to the chain
Chain Ch 31 (PG/GRPO/RLHF-DPO bridge) — variance of the trajectory-level estimator and the role of per-step credit assignment.
