"""Sensitivity sweep of lambda_SDAR on the toy MDP.

Sweeps lambda in {0.0, 0.1, 0.3, 1.0, 3.0} at fixed beta=2 with the standard
fixed-threshold gap gate g_t = sigma(2 * Delta_t). For each value, reports
final reward (averaged over 50 eval episodes) and final per-turn KL.

Identifies the constrained optimum
    lambda* = argmax_lambda reward(lambda)  s.t.  kl(lambda) <= 2 * kl(0).

CPU-runnable in <60s. Single-file, self-contained.

Maps to: 05-improvements.tex Section 3 (Experimental Extensions).
"""
from __future__ import annotations
import json
import math
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Toy MDP — same as adaptive-gate.py, kept inline so each prototype is
# self-contained (no cross-imports). 4 turns x 5 actions, teacher hints with
# probability P_HINT.
# ---------------------------------------------------------------------------
N_TURNS = 4
N_ACTIONS = 5
GOOD_ACTIONS = np.array([0, 2, 4, 1])
P_HINT = 0.7
TEMP = 1.0


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def make_teacher_logits(rng: np.random.Generator) -> np.ndarray:
    logits = rng.normal(0, 0.1, size=(N_TURNS, N_ACTIONS))
    for t in range(N_TURNS):
        if rng.random() < P_HINT:
            logits[t, GOOD_ACTIONS[t]] += 2.5
        else:
            wrong = rng.choice(
                [a for a in range(N_ACTIONS) if a != GOOD_ACTIONS[t]]
            )
            logits[t, wrong] += 2.5
    return logits


def sample_episode(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
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


def per_turn_kl(student_logits: np.ndarray, teacher_logits: np.ndarray) -> float:
    kls = []
    for t in range(N_TURNS):
        p_s = softmax(student_logits[t] / TEMP)
        p_t = softmax(teacher_logits[t] / TEMP)
        kls.append(float(np.sum(p_s * (np.log(p_s + 1e-30) - np.log(p_t + 1e-30)))))
    return float(np.mean(kls))


def train_for_lambda(lambda_sdar: float, n_steps: int = 100, seed: int = 0) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    student_logits = rng.normal(0, 0.1, size=(N_TURNS, N_ACTIONS))
    lr = 0.4
    beta = 2.0  # gate sharpness, fixed across the sweep

    for step in range(n_steps):
        teacher_logits = make_teacher_logits(rng)
        actions, gaps, reward = sample_episode(student_logits, teacher_logits, rng)

        grad = np.zeros_like(student_logits)
        for t in range(N_TURNS):
            a = actions[t]
            p_s = softmax(student_logits[t] / TEMP)

            # REINFORCE gradient (no baseline)
            grad[t, a] += (1.0 - p_s[a]) * reward / TEMP
            for j in range(N_ACTIONS):
                if j != a:
                    grad[t, j] += (-p_s[j]) * reward / TEMP

            # SDAR gated term
            if lambda_sdar > 0:
                g = 1.0 / (1.0 + math.exp(-beta * gaps[t]))
                grad[t, a] += lambda_sdar * g * (1.0 - p_s[a]) / TEMP
                for j in range(N_ACTIONS):
                    if j != a:
                        grad[t, j] += lambda_sdar * g * (-p_s[j]) / TEMP

        student_logits = student_logits + lr * grad

    # Eval
    final_rewards = []
    for _ in range(50):
        teacher_logits = make_teacher_logits(rng)
        _, _, r = sample_episode(student_logits, teacher_logits, rng)
        final_rewards.append(r)
    final_reward = float(np.mean(final_rewards))

    eval_teacher = make_teacher_logits(rng)
    final_kl = per_turn_kl(student_logits, eval_teacher)

    return {"reward": final_reward, "kl": final_kl}


def measure() -> Dict:
    np.random.seed(0)
    lambdas = [0.0, 0.1, 0.3, 1.0, 3.0]
    sweep: Dict[str, Dict[str, float]] = {}
    for lam in lambdas:
        sweep[f"{lam}"] = train_for_lambda(lambda_sdar=lam, n_steps=100, seed=0)

    kl_baseline = sweep["0.0"]["kl"]
    kl_budget = 2.0 * max(kl_baseline, 1e-6)

    # constrained argmax
    best_lambda = 0.0
    best_reward = -float("inf")
    for lam in lambdas:
        entry = sweep[f"{lam}"]
        if entry["kl"] <= kl_budget and entry["reward"] > best_reward:
            best_reward = entry["reward"]
            best_lambda = lam

    # If all candidates violate the budget (degenerate baseline KL), fall back
    # to the reward-maximiser without the constraint so the script still
    # returns a meaningful value rather than -inf.
    if best_reward == -float("inf"):
        best_lambda = max(lambdas, key=lambda l: sweep[f"{l}"]["reward"])
        best_reward = sweep[f"{best_lambda}"]["reward"]

    return {
        "sweep": sweep,
        "kl_baseline": kl_baseline,
        "kl_budget": kl_budget,
        "optimal_lambda": float(best_lambda),
        "optimal_reward": float(best_reward),
    }


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
