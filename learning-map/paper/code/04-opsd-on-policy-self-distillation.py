"""Concept 04: OPSD reverse KL — full-vocab vs sampled estimator.

We instantiate small student/teacher categorical distributions and
compute the per-token reverse KL exactly. Then sample y_t from the
student and compute the single-sample estimate to see they agree on
average across many draws.
"""
import numpy as np

rng = np.random.default_rng(2)
V = 16  # tiny vocab


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


student_logits = rng.normal(size=V)
teacher_logits = student_logits + 0.5 * rng.normal(size=V)  # teacher slightly different
p = softmax(student_logits)
q = softmax(teacher_logits)

# Exact reverse KL D_KL(p || q)
exact_rkl = float(np.sum(p * np.log(p / q)))

# Single-sample Monte-Carlo estimator
N = 5000
samples = rng.choice(V, size=N, p=p)
log_p = np.log(p[samples])
log_q = np.log(q[samples])
mc_estimates = log_p - log_q  # per-sample contributions

print(f"Vocabulary size V = {V}")
print(f"Exact D_KL(student || teacher) = {exact_rkl:.4f}")
print(f"MC estimate (mean over {N} samples) = {mc_estimates.mean():.4f}")
print(f"Std of single-sample estimator = {mc_estimates.std():.4f}")
print("\nThe MC mean matches the exact value -> single-sample estimator is unbiased.")
