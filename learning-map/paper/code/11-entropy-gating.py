"""Concept 11: Entropy gating — gate value vs student entropy.

Sweep student entropy from near-zero (sharp) to near-uniform; show
that g_t = sigma(beta * h_t) is monotone increasing in h_t and lies
in (1/2, 1).
"""
import numpy as np

V = 8
BETA = 1.0


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Sweep peakedness alpha; large alpha -> sharp dist, small -> uniform
print(f"{'alpha':>6}  {'entropy h_t':>12}  {'gate g_t':>10}")
for alpha in [10.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.0]:
    logits = np.zeros(V)
    logits[0] = alpha  # peak on token 0
    p = softmax(logits)
    h = float(-(p * np.log(p + 1e-12)).sum())
    g = sigmoid(BETA * h)
    print(f"{alpha:>6.2f}  {h:>12.4f}  {g:>10.4f}")

print(f"\nMax entropy of uniform on V={V}: log V = {np.log(V):.4f}")
print("Note: g_t > 1/2 always, since h_t >= 0. Entropy gating only modulates,")
print("never zeroes out, distillation pressure.")
