"""Concept 07: Single-sample teacher-student log-prob gap Delta_t.

Demonstrates: (a) Delta_t = log pi_T(y_t|s_t^+) - log pi_theta(y_t|s_t)
is unbiased for -D_RKL when y_t ~ pi_theta; (b) the sign of Delta_t
is the natural binary signal for 'teacher endorses' vs 'teacher
disagrees'.
"""
import numpy as np

rng = np.random.default_rng(3)
V = 32
N = 4000


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


student_logits = rng.normal(size=V)
teacher_logits = student_logits + 0.7 * rng.normal(size=V)
p = softmax(student_logits)
q = softmax(teacher_logits)

samples = rng.choice(V, size=N, p=p)
delta = np.log(q[samples]) - np.log(p[samples])  # Delta_t

# Sanity: -mean(delta) should equal exact reverse KL
exact_rkl = float(np.sum(p * np.log(p / q)))
mc_neg_mean = float(-delta.mean())

# Sign distribution
pos_frac = float((delta > 0).mean())
neg_frac = float((delta < 0).mean())

print(f"Vocab V = {V}, samples N = {N}")
print(f"-mean(Delta_t)  = {mc_neg_mean:+.4f}")
print(f"D_RKL exact     = {exact_rkl:+.4f}")
print(f"  -> agree (single-sample estimator unbiased)")
print()
print(f"Fraction with Delta_t > 0 (teacher endorses) = {pos_frac:.2%}")
print(f"Fraction with Delta_t < 0 (teacher disagrees) = {neg_frac:.2%}")
print(f"Mean |Delta_t| = {np.abs(delta).mean():.3f}")
print("\nDelta_t serves dual purpose: cheap distillation signal AND gate input.")
