"""Concept 02: Trajectory-level RL reward — variance vs horizon demo.

We show that a single scalar reward at the end of an episode produces
a per-token gradient estimator whose variance grows with horizon T.
The trick: each of T tokens is multiplied by the SAME advantage R - bar{R},
so the gradient estimator is sum_t grad log pi * (R - bar{R}).
"""
import numpy as np

rng = np.random.default_rng(0)
N_TRAJ = 4000
HORIZONS = [4, 16, 64]
P_REWARD = 0.4

print(f"{'T':>4}  {'mean grad':>12}  {'var grad':>12}  ratio")
prev_var = None
for T in HORIZONS:
    # Each "token" contributes a noisy log-prob gradient ~ N(0, 1).
    grads_per_token = rng.standard_normal((N_TRAJ, T))
    rewards = (rng.random(N_TRAJ) < P_REWARD).astype(float)
    baseline = rewards.mean()
    # Trajectory-level estimator: sum_t grad_t * (R - baseline)
    estimator = grads_per_token.sum(axis=1) * (rewards - baseline)
    var = estimator.var()
    ratio = "" if prev_var is None else f"{var / prev_var:6.2f}x"
    print(f"{T:>4d}  {estimator.mean():>12.4f}  {var:>12.4f}  {ratio}")
    prev_var = var

print("\nVariance grows ~linearly in T because all T grads share one scalar (R-baseline).")
print("This is precisely the sparse-supervision pain SDAR injects per-token gating to fix.")
