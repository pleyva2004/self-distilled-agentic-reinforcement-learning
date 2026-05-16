# Multi-turn agentic MDP

**Level:** intro
**Prerequisites:** Chain Ch 25 (causal LM as MDP)
**Used by:** [02-rl-trajectory-reward](02-rl-trajectory-reward.md), [04-opsd-on-policy-self-distillation](04-opsd-on-policy-self-distillation.md)

## Plain-English intro
An LLM agent solves tasks by alternating between *thinking/acting* and *observing the environment*. Mathematically this is a finite-horizon MDP: at turn $k$ the agent reads observation $o_k$, emits a token sequence $a_k$ (reasoning + tool calls), the environment returns $o_{k+1}$, and we eventually score the whole trajectory. SDAR's setting is exactly this multi-turn loop, with all response tokens flattened into one sequence $y = (y_1, \dots, y_T)$.

## Formal definition
State at token $t$ (turn index $k(t)$):
$s_t = (x, o_{1:k(t)}, y_{1:t-1})$ — task, observations so far, tokens emitted so far.

Policy: $y_t \sim \pi_\theta(\cdot \mid s_t)$.

Episode reward: $R(y, \text{traj})$ from a verifier (e.g. unit-test pass / final answer correct).

Horizon: number of turns $H$ (paper goes up to ~20).

## Why this matters for the paper
Every other piece of SDAR (gap, gate, distillation loss) is defined per-token over this MDP. The "multi-turn" qualifier is what breaks naive OPSD — the per-turn drift compounds as $H$ grows.

## Code
See [`../code/01-multi-turn-agentic-mdp.py`](../code/01-multi-turn-agentic-mdp.py).

## Cross-link to the chain
Chain Ch 25 (causal LM as MDP) — every per-token decision inside a turn IS the Ch 25 MDP. This concept is the multi-turn outer loop wrapped around it.
