"""Concept 06: Per-token reverse KL — exact full-vocab computation.

Reverse vs forward KL on a designed toy: teacher is bimodal, student
must choose where to put its mass. Reverse KL is mode-seeking; we
verify by minimizing each direction over a 1-D family of student
distributions and seeing where each one lands.
"""
import numpy as np

V = 5


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


teacher = np.array([0.45, 0.05, 0.05, 0.05, 0.40])  # bimodal


def student_for_alpha(alpha):
    """One-parameter family interpolating between mass on idx 0 vs idx 4."""
    logits = np.array([alpha, -2, -2, -2, -alpha])
    return softmax(logits)


def fwd_kl(p, q):  # D_KL(teacher || student) — mass-covering
    return float(np.sum(p * np.log(p / np.clip(q, 1e-12, 1))))


def rev_kl(p, q):  # D_KL(student || teacher) — mode-seeking
    return float(np.sum(q * np.log(q / np.clip(p, 1e-12, 1))))


alphas = np.linspace(-3, 3, 101)
fwd = [fwd_kl(teacher, student_for_alpha(a)) for a in alphas]
rev = [rev_kl(teacher, student_for_alpha(a)) for a in alphas]

a_fwd = alphas[int(np.argmin(fwd))]
a_rev = alphas[int(np.argmin(rev))]

print(f"Teacher distribution = {teacher.tolist()} (bimodal: mass at idx 0 and 4)")
print(f"Forward-KL minimizer alpha* = {a_fwd:+.2f} -> student {np.round(student_for_alpha(a_fwd),3)}")
print(f"  (covers both modes -> interior alpha)")
print(f"Reverse-KL minimizer alpha* = {a_rev:+.2f} -> student {np.round(student_for_alpha(a_rev),3)}")
print(f"  (snaps to ONE mode -> extreme alpha)")
print("\nSDAR uses reverse KL precisely because in agentic tool-use settings")
print("there is usually one 'right' next token, not several modes to cover.")
