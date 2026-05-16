"""toy_sdar.py — tabular demonstration of SDAR on a 4-turn MDP.

Demonstrates the central claim of Lu et al. 2026 (SDAR, arxiv:2605.15155):
gated SDAR matches GRPO on reward AND keeps per-turn KL bounded, while
ungated GRPO+OPSD shows a KL spike (the "compounding instability" the
paper warns about).

The agent has 4 turns; at each turn it picks one of 5 actions. One action
per turn is "correct" (hidden) and yields +1 to the final reward (so the
maximum reward = 4). The student is a tabular softmax over (turn, action)
logits. The teacher is the SAME logits with a fixed bonus +B on the
correct action — modelling the privileged retrieved-skill context.

Maps onto deep-dive sections:
  - Section 1 (agentic MDP)              -> `class FourTurnMDP`
  - Section 2 (GRPO backbone)            -> `grpo_step()`
  - Section 4 (teacher-student gap)      -> `delta_t = log pi_T - log pi_theta`
  - Section 5 (gating)                   -> `sigma(beta * delta_t)`
  - Section 6 (SDAR auxiliary loss)      -> `sdar_aux_grad()`
  - Section 7 (why gated > ungated)      -> the three-variant comparison

Runtime: <30s on CPU. Only depends on numpy + matplotlib.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- config -----
SEED = 0
N_TURNS = 4
N_ACTIONS = 5
TEACHER_BONUS_B = 1.5  # bonus on the correct action in the teacher branch
LR = 0.05
N_STEPS = 200
GROUP_SIZE = 8         # G in GRPO (number of rollouts per step)
LAMBDA_SDAR = 2.0      # heavy auxiliary loss weight to make ungated OPSD
                       # visibly drift away from the on-policy KL ball
GATE_BETA = 6.0        # sharpness of the sigmoid gate on Delta
LOG_EVERY = 20

# correct action per turn (hidden from the student)
CORRECT = np.array([2, 0, 4, 1])  # one of {0..4} per turn


# ---------------------------------------------------------------- env --------
class FourTurnMDP:
    """4-turn MDP, one correct action per turn."""

    def __init__(self):
        self.n_turns = N_TURNS
        self.n_actions = N_ACTIONS
        self.correct = CORRECT.copy()

    def rollout(self, logits, rng):
        """Sample one trajectory under softmax(logits[turn]); return (acts, reward)."""
        acts = np.empty(self.n_turns, dtype=np.int64)
        for t in range(self.n_turns):
            p = softmax(logits[t])
            acts[t] = rng.choice(self.n_actions, p=p)
        reward = float(np.sum(acts == self.correct))
        return acts, reward


# ---------------------------------------------------------------- math -------
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def logsoftmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def teacher_logits_from(student_logits):
    """Teacher = student PLUS bonus B on the correct action at each turn.

    This models the privileged retrieved-skill context: the teacher branch
    sees a hint that points at the right action, so its log-prob on that
    action is higher than the student's by ~B (modulo softmax coupling).
    """
    T = student_logits.copy()
    for t in range(N_TURNS):
        T[t, CORRECT[t]] += TEACHER_BONUS_B
    return T


def per_turn_kl_to_initial(logits, init_logits):
    """Mean over turns of KL(pi_init || pi_current). (We use init->current to
    keep the trust-region direction the same as in PPO/GRPO.)"""
    kls = []
    for t in range(N_TURNS):
        p0 = softmax(init_logits[t])
        p1 = softmax(logits[t])
        kls.append(float(np.sum(p0 * (np.log(p0 + 1e-12) - np.log(p1 + 1e-12)))))
    return float(np.mean(kls))


def mean_entropy(logits):
    H = []
    for t in range(N_TURNS):
        p = softmax(logits[t])
        H.append(float(-np.sum(p * np.log(p + 1e-12))))
    return float(np.mean(H))


def mean_gap(logits):
    """Mean over (turn, action) of Delta = log pi_T - log pi_theta, weighted by pi_theta."""
    T = teacher_logits_from(logits)
    deltas = []
    for t in range(N_TURNS):
        ls = logsoftmax(logits[t])
        lt = logsoftmax(T[t])
        p = softmax(logits[t])
        deltas.append(float(np.sum(p * (lt - ls))))
    return float(np.mean(deltas))


# ---------------------------------------------------------------- gradients --
def grpo_grad(logits, rng):
    """Vanilla GRPO gradient. Group of G rollouts, group-relative advantage.

    For tabular softmax: grad of log pi(a|t) wrt logits[t] = onehot(a) - softmax.
    Gradient ASCENT direction (we negate elsewhere if needed).
    """
    grads = np.zeros_like(logits)
    rewards = np.zeros(GROUP_SIZE)
    traj = []
    for g in range(GROUP_SIZE):
        acts, R = MDP.rollout(logits, rng)
        rewards[g] = R
        traj.append(acts)
    # group-relative advantage
    mu = rewards.mean()
    sigma = rewards.std() + 1e-8
    A = (rewards - mu) / sigma
    for g in range(GROUP_SIZE):
        for t in range(N_TURNS):
            p = softmax(logits[t])
            onehot = np.zeros(N_ACTIONS)
            onehot[traj[g][t]] = 1.0
            # ascent grad: A * (onehot - p)
            grads[t] += A[g] * (onehot - p)
    grads /= GROUP_SIZE
    return grads, float(rewards.mean())


def opsd_aux_grad(logits, gating: str):
    """Auxiliary distillation gradient.

    L_aux = - sum_{t,v} g_t(v) * pi_T(v|t) * log pi_theta(v|t)   (teacher-forcing form)
    Gradient ASCENT on (-L_aux) wrt logits[t]:
        d(-L_aux)/d logits[t] = sum_v g_t(v) * (pi_T(v|t) - alpha * pi_theta(v|t))
    where alpha = sum_v g_v * pt_v (couples through softmax).
    The gate g_t(v) is DETACHED (no grad through it).

    gating in:
      - "none":    g_t(v) = 1                    (ungated GRPO+OPSD baseline)
      - "gap":     g_t(v) = sigma(beta * Delta)  (paper's gap-gate; permissive)
      - "entropy": g_t(v) = sigma(beta * h_t) -- depends only on student
                  entropy, so it naturally tightens as the student sharpens
                  (bounded-KL guarantee).
    """
    T = teacher_logits_from(logits)
    grads = np.zeros_like(logits)
    for t in range(N_TURNS):
        ls = logsoftmax(logits[t])
        lt = logsoftmax(T[t])
        delta = lt - ls                            # per-action gap
        ps = softmax(logits[t])
        pt = softmax(T[t])
        h_t = float(-np.sum(ps * ls))              # student entropy
        if gating == "none":
            g = np.ones(N_ACTIONS)
        elif gating == "gap":
            g = 1.0 / (1.0 + np.exp(-GATE_BETA * delta))
        elif gating == "entropy":
            # entropy gate is shared across all actions at this turn — but
            # we centre it so g<1 once student is more confident than uniform.
            h_centered = h_t - 0.5 * np.log(N_ACTIONS)   # zero at half-max-H
            g = np.full(N_ACTIONS, 1.0 / (1.0 + np.exp(-GATE_BETA * h_centered)))
        else:
            raise ValueError(gating)
        alpha = float(np.sum(g * pt))
        grads[t] += g * pt - alpha * ps
    return grads


# ---------------------------------------------------------------- training ---
def train(variant: str, n_steps: int = N_STEPS):
    """variant in {'grpo', 'grpo_opsd_ungated', 'sdar', 'sdar_entropy'}."""
    rng = np.random.default_rng(SEED)
    logits = np.zeros((N_TURNS, N_ACTIONS))     # uniform start
    init_logits = logits.copy()

    history = {
        "step": [], "reward": [], "kl": [], "gap": [], "entropy": [],
    }

    for step in range(n_steps):
        gp, mean_R = grpo_grad(logits, rng)
        if variant == "grpo":
            g_total = gp
        elif variant == "grpo_opsd_ungated":
            ga = opsd_aux_grad(logits, gating="none")
            g_total = gp + LAMBDA_SDAR * ga
        elif variant == "sdar":
            ga = opsd_aux_grad(logits, gating="gap")
            g_total = gp + LAMBDA_SDAR * ga
        elif variant == "sdar_entropy":
            ga = opsd_aux_grad(logits, gating="entropy")
            g_total = gp + LAMBDA_SDAR * ga
        else:
            raise ValueError(variant)

        logits = logits + LR * g_total

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            history["step"].append(step)
            history["reward"].append(mean_R)
            history["kl"].append(per_turn_kl_to_initial(logits, init_logits))
            history["gap"].append(mean_gap(logits))
            history["entropy"].append(mean_entropy(logits))

    # final eval averaged over many rollouts (greedy not used; sample)
    eval_rng = np.random.default_rng(SEED + 999)
    R_final = np.mean([
        MDP.rollout(logits, eval_rng)[1] for _ in range(64)
    ])
    return logits, history, float(R_final)


def print_history(name: str, hist: dict):
    print(f"\n=== {name} ===")
    print(f"{'step':>6} {'reward':>8} {'kl':>8} {'gap':>8} {'H':>8}")
    for i in range(len(hist["step"])):
        print(f"{hist['step'][i]:>6d} "
              f"{hist['reward'][i]:>8.3f} "
              f"{hist['kl'][i]:>8.4f} "
              f"{hist['gap'][i]:>8.4f} "
              f"{hist['entropy'][i]:>8.4f}")


def maybe_plot(results):
    """Plot reward + KL curves; degrade silently to a printed table."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name, (_, hist, _) in results.items():
            axes[0].plot(hist["step"], hist["reward"], label=name, marker="o")
            axes[1].plot(hist["step"], hist["kl"], label=name, marker="o")
        axes[0].set_title("Mean rollout reward")
        axes[0].set_xlabel("step"); axes[0].set_ylabel("reward")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].set_title("Per-turn KL to init policy")
        axes[1].set_xlabel("step"); axes[1].set_ylabel("KL")
        axes[1].legend(); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        out = "toy_sdar_curves.png"
        fig.savefig(out, dpi=110)
        print(f"\n[plot] saved {out}")
    except Exception as e:
        print(f"\n[plot] matplotlib unavailable ({e}); skipping figure.")


# ---------------------------------------------------------------- main -------
MDP = FourTurnMDP()


def main():
    np.random.seed(SEED)

    print("Toy SDAR demo — 4-turn MDP, 5 actions/turn, teacher bonus B = "
          f"{TEACHER_BONUS_B}")
    print(f"  steps={N_STEPS}  group={GROUP_SIZE}  lr={LR}  "
          f"lambda={LAMBDA_SDAR}  beta={GATE_BETA}")

    results = {}
    for variant in ["grpo", "grpo_opsd_ungated", "sdar", "sdar_entropy"]:
        logits, hist, R_final = train(variant)
        results[variant] = (logits, hist, R_final)
        print_history(variant, hist)
        print(f"  -> final mean eval reward = {R_final:.3f}")

    maybe_plot(results)

    # ------- final summary ---------------------------------------------------
    R_grpo   = results["grpo"][2]
    R_opsd   = results["grpo_opsd_ungated"][2]
    R_sdar   = results["sdar"][2]
    R_sdarH  = results["sdar_entropy"][2]
    KL_grpo  = results["grpo"][1]["kl"][-1]
    KL_opsd  = results["grpo_opsd_ungated"][1]["kl"][-1]
    KL_sdar  = results["sdar"][1]["kl"][-1]
    KL_sdarH = results["sdar_entropy"][1]["kl"][-1]

    print("\n" + "=" * 64)
    print(" SUMMARY  (variant         final-reward   final-per-turn-KL )")
    print("=" * 64)
    print(f"  GRPO                    {R_grpo:>6.3f}        {KL_grpo:>7.4f}")
    print(f"  GRPO+OPSD  (ungated)    {R_opsd:>6.3f}        {KL_opsd:>7.4f}")
    print(f"  SDAR       (gap-gate)   {R_sdar:>6.3f}        {KL_sdar:>7.4f}")
    print(f"  SDAR       (entropy-gt) {R_sdarH:>6.3f}        {KL_sdarH:>7.4f}")
    print("=" * 64)

    # ------- machine-readable line for parent agent --------------------------
    print(f"RESULT_JSON {{"
          f"\"grpo_reward\":{R_grpo:.4f},"
          f"\"grpo_opsd_reward\":{R_opsd:.4f},"
          f"\"sdar_reward\":{R_sdar:.4f},"
          f"\"sdar_entropy_reward\":{R_sdarH:.4f},"
          f"\"grpo_kl\":{KL_grpo:.6f},"
          f"\"grpo_opsd_kl\":{KL_opsd:.6f},"
          f"\"sdar_kl\":{KL_sdar:.6f},"
          f"\"sdar_entropy_kl\":{KL_sdarH:.6f}"
          f"}}")

    # ------- assertions ------------------------------------------------------
    assert R_sdar >= R_grpo - 0.05, (
        f"SDAR reward {R_sdar} should be >= GRPO {R_grpo} - 0.05")
    assert KL_sdar <= KL_opsd, (
        f"SDAR KL {KL_sdar} should be <= ungated-OPSD KL {KL_opsd}")
    print("\nAssertions passed: SDAR matches GRPO on reward AND keeps KL "
          "<= ungated GRPO+OPSD.")


if __name__ == "__main__":
    main()
