"""Adaptive-quantile gating for SDAR — toy MDP comparison.

Compares four variants on a 4-turn / 5-action toy MDP:
    1. GRPO baseline (no SDAR)
    2. GRPO + ungated OPSD (sanity check — should be unstable)
    3. SDAR with fixed-threshold gate g_t = sigma(2 * Delta_t)
    4. SDAR with adaptive-quantile gate g_t = sigma(2 * (Delta_t - q_hat_tau))
       where q_hat_tau is an EMA of the running tau-quantile of recent gaps.

CPU-runnable in <30s. Single-file, self-contained.

Maps to: 05-improvements.tex Section 2 (Code / Implementation Improvements).
"""
from __future__ import annotations
import json
import math
from typing import Callable, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Toy MDP: 4 turns, 5 actions. Reward = +1 if the chosen action equals the
# turn-specific "good" action, else 0. The "teacher" sees a privileged hint
# (the good action with probability p_hint, else uniform noise).
# ---------------------------------------------------------------------------
N_TURNS = 4
N_ACTIONS = 5
GOOD_ACTIONS = np.array([0, 2, 4, 1])  # one good action per turn
P_HINT = 0.7  # teacher hint reliability
TEMP = 1.0


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def sample_episode(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sample one episode under the student policy.

    Returns (actions[T], gaps[T] = log p_T(a) - log p_S(a), reward).
    """
    actions = np.empty(N_TURNS, dtype=int)
    gaps = np.empty(N_TURNS, dtype=float)
    reward = 0.0
    for t in range(N_TURNS):
        p_s = softmax(student_logits[t] / TEMP)
        p_t = softmax(teacher_logits[t] / TEMP)
        a = rng.choice(N_ACTIONS, p=p_s)
        actions[t] = a
        gaps[t] = math.log(p_t[a] + 1e-30) - math.log(p_s[a] + 1e-30)
        if a == GOOD_ACTIONS[t]:
            reward += 1.0
    return actions, gaps, reward / N_TURNS


def make_teacher_logits(rng: np.random.Generator) -> np.ndarray:
    """Per-turn teacher logits, biased toward the good action with prob P_HINT."""
    logits = rng.normal(0, 0.1, size=(N_TURNS, N_ACTIONS))
    for t in range(N_TURNS):
        if rng.random() < P_HINT:
            logits[t, GOOD_ACTIONS[t]] += 2.5  # endorse the right action
        else:
            # bad hint — push toward a random wrong action
            wrong = rng.choice([a for a in range(N_ACTIONS) if a != GOOD_ACTIONS[t]])
            logits[t, wrong] += 2.5
    return logits


def per_turn_kl(student_logits: np.ndarray, teacher_logits: np.ndarray) -> float:
    """Mean over turns of KL(student || teacher)."""
    kls = []
    for t in range(N_TURNS):
        p_s = softmax(student_logits[t] / TEMP)
        p_t = softmax(teacher_logits[t] / TEMP)
        kls.append(float(np.sum(p_s * (np.log(p_s + 1e-30) - np.log(p_t + 1e-30)))))
    return float(np.mean(kls))


# ---------------------------------------------------------------------------
# Training loop. We run a tabular surrogate: maintain student_logits as a
# free parameter; gradient ascent on R - lambda * sum_t g_t * (-Delta_t)
# (since L_SDAR = g_t * Delta_t and we MINIMIZE L => the policy pushes its
# log-prob toward the teacher's preferred token whenever g_t > 0).
# ---------------------------------------------------------------------------
def train(
    *,
    lambda_sdar: float,
    gate_fn: Callable[[float, float], float],  # (delta, q_hat) -> g
    use_adaptive_q: bool,
    n_steps: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    student_logits = rng.normal(0, 0.1, size=(N_TURNS, N_ACTIONS))
    lr = 0.4

    # EMA quantile tracker (rough running quantile via mean of order-statistic).
    # quantile_tau picks the cutoff of the empirical gap distribution: the gate
    # fires when Delta_t > q_hat_tau, so larger tau -> stricter gate -> lower
    # fire rate. tau=0.65 keeps the gate roughly as selective as the fixed
    # sigma(2*Delta_t) gate (which fires ~36% on this MDP) but recenters as
    # the gap distribution drifts during training.
    q_hat = 0.0
    eta = 0.05
    quantile_tau = 0.65

    gate_fires: List[float] = []
    rewards: List[float] = []

    for step in range(n_steps):
        teacher_logits = make_teacher_logits(rng)
        actions, gaps, reward = sample_episode(student_logits, teacher_logits, rng)
        rewards.append(reward)

        # GRPO surrogate: REINFORCE with reward as advantage (no baseline for simplicity)
        # plus SDAR gated term.
        grad = np.zeros_like(student_logits)
        for t in range(N_TURNS):
            a = actions[t]
            p_s = softmax(student_logits[t] / TEMP)

            # REINFORCE gradient on log pi(a|s)
            grad[t, a] += (1.0 - p_s[a]) * reward / TEMP
            for j in range(N_ACTIONS):
                if j != a:
                    grad[t, j] += (-p_s[j]) * reward / TEMP

            # SDAR gradient: maximize g_t * (log p_T(a) - log p_S(a))
            # equivalently, ascent on g_t * Delta_t means we INCREASE log p_S(a)
            # when teacher endorses (Delta_t > 0) — so gradient is +g_t on the
            # student log-prob of the sampled token.
            if lambda_sdar > 0:
                g = gate_fn(gaps[t], q_hat if use_adaptive_q else 0.0)
                gate_fires.append(g)
                grad[t, a] += lambda_sdar * g * (1.0 - p_s[a]) / TEMP
                for j in range(N_ACTIONS):
                    if j != a:
                        grad[t, j] += lambda_sdar * g * (-p_s[j]) / TEMP

        student_logits = student_logits + lr * grad

        # Update EMA quantile tracker.
        if use_adaptive_q and len(gaps) > 0:
            # Approximate the tau-quantile by taking the order statistic
            # of the current episode's gaps.
            sorted_gaps = np.sort(gaps)
            idx = int(np.floor(quantile_tau * (len(sorted_gaps) - 1)))
            q_target = float(sorted_gaps[idx])
            q_hat = (1 - eta) * q_hat + eta * q_target

    # Final evaluation: average reward over 50 episodes + per-turn KL on a
    # representative teacher.
    final_rewards = []
    for _ in range(50):
        teacher_logits = make_teacher_logits(rng)
        _, _, r = sample_episode(student_logits, teacher_logits, rng)
        final_rewards.append(r)
    final_reward = float(np.mean(final_rewards))

    eval_teacher = make_teacher_logits(rng)
    final_kl = per_turn_kl(student_logits, eval_teacher)

    fire_rate = float(np.mean(gate_fires)) if gate_fires else 0.0
    return {
        "final_reward": final_reward,
        "final_kl": final_kl,
        "gate_fire_rate": fire_rate,
    }


def fixed_gate(delta: float, _q: float, beta: float = 2.0) -> float:
    return float(1.0 / (1.0 + math.exp(-beta * delta)))


def adaptive_gate(delta: float, q_hat: float, beta: float = 2.0) -> float:
    return float(1.0 / (1.0 + math.exp(-beta * (delta - q_hat))))


def measure() -> Dict:
    np.random.seed(0)
    common = dict(n_steps=100, seed=0)

    grpo = train(
        lambda_sdar=0.0,
        gate_fn=fixed_gate,
        use_adaptive_q=False,
        **common,
    )
    fixed = train(
        lambda_sdar=1.0,
        gate_fn=fixed_gate,
        use_adaptive_q=False,
        **common,
    )
    adaptive = train(
        lambda_sdar=1.0,
        gate_fn=adaptive_gate,
        use_adaptive_q=True,
        **common,
    )
    ungated = train(
        lambda_sdar=1.0,
        gate_fn=lambda d, q: 1.0,  # gate always on -> ungated OPSD
        use_adaptive_q=False,
        **common,
    )

    return {
        "grpo_reward": grpo["final_reward"],
        "grpo_kl": grpo["final_kl"],
        "fixed_threshold_reward": fixed["final_reward"],
        "fixed_kl": fixed["final_kl"],
        "adaptive_threshold_reward": adaptive["final_reward"],
        "adaptive_kl": adaptive["final_kl"],
        "ungated_opsd_reward": ungated["final_reward"],
        "ungated_opsd_kl": ungated["final_kl"],
        "gate_fire_rate_fixed": fixed["gate_fire_rate"],
        "gate_fire_rate_adaptive": adaptive["gate_fire_rate"],
        "adaptive_vs_fixed_reward_delta": (
            adaptive["final_reward"] - fixed["final_reward"]
        ),
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
