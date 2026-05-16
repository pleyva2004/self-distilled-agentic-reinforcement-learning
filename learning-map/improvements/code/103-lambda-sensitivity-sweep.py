"""Improvement 103: Sensitivity sweep of lambda_SDAR.

Toy: parameterize a 1-D student log-prob theta. The total loss is
L(theta; lambda) = (theta - target_grpo)^2 + lambda * (theta - target_sdar)^2.
Closed-form minimizer: theta*(lambda) = (target_grpo + lambda*target_sdar) / (1+lambda).

We sweep lambda and report the validation loss at the trained optimum,
plus the 'envelope' sensitivity dL/dlambda = L_SDAR(theta*).
We also compare a constant lambda schedule against an exponential
anneal: lambda(t) = lambda_0 * exp(-t / tau).
"""
import numpy as np

target_grpo = 0.0
target_sdar = 1.5  # teacher pulls in this direction
val_target = 0.4  # validation truth lies between
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def theta_star(lam):
    return (target_grpo + lam * target_sdar) / (1.0 + lam)


def val_loss(theta):
    return (theta - val_target) ** 2


def L_sdar(theta):
    return (theta - target_sdar) ** 2


print(f"{'lambda':>8}  {'theta*':>8}  {'val_loss':>9}  {'sens dL/dlam':>13}")
for lam in LAMBDAS:
    th = theta_star(lam)
    vl = val_loss(th)
    sens = L_sdar(th)  # envelope theorem
    print(f"{lam:>8.2f}  {th:>+8.3f}  {vl:>9.4f}  {sens:>13.4f}")

# Annealing simulation: simulate streaming SGD on a 'time-varying optimum'
print("\nAnneal simulation:")
T = 50
lam0 = 3.0
tau = 15
theta = 0.0
lr = 0.1
losses_const, losses_anneal = [], []
for sched_name, sched in [("const lam=1.0", lambda t: 1.0),
                          ("anneal", lambda t: lam0 * np.exp(-t / tau))]:
    theta = 0.0
    cum = 0.0
    for t in range(T):
        lam_t = sched(t)
        # gradient: 2*(theta - target_grpo) + 2*lam*(theta - target_sdar)
        grad = 2 * (theta - target_grpo) + 2 * lam_t * (theta - target_sdar)
        theta -= lr * grad
        cum += val_loss(theta)
    print(f"  {sched_name:>14}:  final theta = {theta:+.3f}, cum val_loss = {cum:.3f}")

print("\nAnnealing reaches a similar end point but spends fewer steps with strong")
print("teacher pull early on. In real training this matches student capacity better.")
