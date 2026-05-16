"""Improvement 104: SDAR as constrained RL — primal-dual loop demo.

Constrained problem (toy 1-D):
  max_theta  -(theta - target_grpo)^2
  subject to (theta - target_sdar)^2 <= delta

Lagrangian: L(theta, mu) = (theta - target_grpo)^2 + mu * ((theta - target_sdar)^2 - delta)

Primal-dual updates:
  theta <- theta - lr_theta * d L / d theta
  mu    <- max(0, mu + lr_mu * (constraint - 0))   (dual ascent)

We show that mu* converges to a value reproducing the SDAR loss balance,
and that for large delta the constraint is slack -> mu* = 0 -> recovers GRPO.
"""
import numpy as np

target_grpo = 0.0
target_sdar = 1.5

print(f"{'delta':>8}  {'mu_final':>10}  {'theta_final':>12}  {'constraint slack?':>18}")
for delta in [0.1, 1.0, 2.5, 5.0]:
    theta = 0.0
    mu = 0.0
    lr_th, lr_mu = 0.05, 0.05
    for step in range(2000):
        # primal step (gradient descent on Lagrangian wrt theta)
        grad_th = 2 * (theta - target_grpo) + 2 * mu * (theta - target_sdar)
        theta -= lr_th * grad_th
        # dual step (gradient ascent on constraint)
        viol = (theta - target_sdar) ** 2 - delta
        mu = max(0.0, mu + lr_mu * viol)
    slack = (theta - target_sdar) ** 2 - delta
    print(f"{delta:>8.2f}  {mu:>10.4f}  {theta:>+12.4f}  "
          f"{'slack' if slack < -1e-3 else 'tight':>18}")

print("\nSmall delta (tight constraint) -> mu > 0, theta pulled toward target_sdar.")
print("Large delta (slack constraint) -> mu = 0, theta stays at target_grpo.")
print("Identifying mu with lambda_SDAR recovers the SDAR combined objective.")
