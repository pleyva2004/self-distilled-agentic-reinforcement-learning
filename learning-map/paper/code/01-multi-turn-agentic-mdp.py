"""Concept 01: Multi-turn agentic MDP — finite-toy demonstration.

A 3-turn agent visits states drawn from a small grid. Each turn the agent
emits a 2-token "action" sampled from a tabular policy; the environment
returns a deterministic next observation; final reward = 1 iff the
flattened token sequence ends with the "correct" pair for the task.
We print the trajectory and the per-token state index s_t.
"""
import numpy as np

rng = np.random.default_rng(0)
NUM_TURNS = 3
TOKENS_PER_TURN = 2
VOCAB = 4
GOAL = (3, 1)  # last two tokens must be (3, 1)


def policy_logits(state_idx: int) -> np.ndarray:
    """Tabular logits: each state-index maps to a different soft preference."""
    base = np.array([0.1 * state_idx, 0.5, -0.2, 0.3 + 0.1 * state_idx])
    return base


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def env_step(prev_obs: int, action_tokens: tuple) -> int:
    return (prev_obs + action_tokens[-1]) % 5


def rollout(seed: int):
    rng_local = np.random.default_rng(seed)
    obs = 0
    flat_tokens, states = [], []
    for k in range(NUM_TURNS):
        for j in range(TOKENS_PER_TURN):
            s_t = (obs * 7 + len(flat_tokens)) % 11  # state index
            states.append(s_t)
            probs = softmax(policy_logits(s_t))
            tok = int(rng_local.choice(VOCAB, p=probs))
            flat_tokens.append(tok)
        obs = env_step(obs, tuple(flat_tokens[-TOKENS_PER_TURN:]))
    reward = 1.0 if tuple(flat_tokens[-2:]) == GOAL else 0.0
    return flat_tokens, states, reward


tokens, states, R = rollout(seed=0)
print(f"Trajectory tokens y_1..y_T = {tokens}")
print(f"Per-token state indices s_t = {states}")
print(f"Verifier reward R(tau) = {R}")
print(f"Horizon (turns) = {NUM_TURNS}, total response tokens T = {len(tokens)}")
