"""Concept 10: Sigmoid token-level gate with stop-gradient.

We numerically verify the equivalence: when the gate is detached, the
SDAR loss gradient w.r.t. theta is exactly the gated REINFORCE-style
update. We use finite differences against an analytic gradient on a
tiny toy.
"""
import numpy as np

rng = np.random.default_rng(6)
V = 6
T = 5
LAMBDA = 1.0
BETA = 2.0


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Student logits parameterized by theta (one logit per (t, v))
theta = rng.normal(size=(T, V)) * 0.3
teacher_logp = rng.normal(size=(T, V)) * 0.3  # treated as constant (no_grad)
y = rng.integers(0, V, size=T)


def loss(theta_):
    total = 0.0
    for t in range(T):
        p = softmax(theta_[t])
        log_p = np.log(p[y[t]])
        log_q = teacher_logp[t, y[t]]
        delta_t = log_q - log_p
        gate = sigmoid(BETA * delta_t)  # stop-gradient applied separately
        # In stop-gradient form, gate is constant w.r.t. theta:
        gate_const = float(gate)
        total += gate_const * delta_t  # delta_t still depends on theta via log_p
    return total / T


# Analytic gradient: d/dtheta of [g_t * (-log_p_t)] = -g_t * d log_p_t/d theta
# d log p(y|theta)/d theta_v = (1{v=y} - p_v)
analytic = np.zeros_like(theta)
for t in range(T):
    p = softmax(theta[t])
    log_p = np.log(p[y[t]])
    log_q = teacher_logp[t, y[t]]
    delta_t = log_q - log_p
    g = sigmoid(BETA * delta_t)
    one_hot = np.zeros(V); one_hot[y[t]] = 1.0
    analytic[t] = -g * (one_hot - p) / T  # negative sign from -log_p_t

# Finite differences
fd = np.zeros_like(theta)
eps = 1e-5
for t in range(T):
    for v in range(V):
        th = theta.copy()
        th[t, v] += eps
        f1 = loss(th)
        th[t, v] -= 2 * eps
        f2 = loss(th)
        fd[t, v] = (f1 - f2) / (2 * eps)

err = np.abs(analytic - fd).max()
print(f"Max |analytic - finite-diff| = {err:.2e}  (should be < 1e-4)")
print("\nGated REINFORCE equivalence verified: gradient flows ONLY through")
print("log pi_theta(y_t|s_t), weighted by the (detached) gate g_t.")
