"""Concept 12: Gap gating — sigma(beta * Delta_t) vs Delta_t.

Plot the gate value and the per-token loss g_t * Delta_t as a function
of Delta_t for several beta. Show the asymmetry: for Delta_t -> +inf
the loss tracks Delta_t; for Delta_t -> -inf the loss vanishes.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


deltas = np.linspace(-5, 5, 21)
print(f"{'Delta_t':>8}  " + "  ".join(f"g_t(b={b})" for b in [0.5, 1.0, 4.0]) + "   loss(b=1)")
for d in deltas:
    gates = [sigmoid(b * d) for b in [0.5, 1.0, 4.0]]
    loss_b1 = gates[1] * d
    print(f"{d:>+8.2f}  " + "  ".join(f"{g:8.4f}" for g in gates) + f"   {loss_b1:+8.4f}")

print("\nAsymmetry check:")
print(f"  Delta -> +inf: g -> 1, loss -> +Delta (full distillation pressure)")
print(f"  Delta -> -inf: g -> 0, loss -> 0     (no update)")
print(f"  Delta = 0    : g = 1/2, loss = 0     (boundary)")
print("\nThis is the structural realization of 'asymmetric trust' from concept 09.")
