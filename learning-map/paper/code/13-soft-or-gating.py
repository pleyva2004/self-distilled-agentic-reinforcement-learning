"""Concept 13: Soft-OR gating — sigma(beta * [1 - (1-h_t)(1-Delta_t)]).

Demonstrates that the soft-OR gate fires when EITHER entropy is high
OR teacher endorses. We tabulate the gate over a grid of (h_t, Delta_t)
both in [0, 1] (Delta normalized).
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


BETA = 4.0
hs = np.linspace(0, 1, 5)
deltas_norm = np.linspace(0, 1, 5)

# Format: rows = h, cols = delta_norm
print(f"Soft-OR gate values: g = sigma({BETA} * [1 - (1-h)(1-d)])")
print(" " * 10 + "delta_norm:  " + "  ".join(f"{d:+.2f}" for d in deltas_norm))
for h in hs:
    row = []
    for d in deltas_norm:
        score = 1.0 - (1.0 - h) * (1.0 - d)
        row.append(sigmoid(BETA * score))
    print(f"  h = {h:+.2f}:           " + "  ".join(f"{g:5.3f}" for g in row))

print("\nObservations:")
print("  h=0, d=0 -> score = 0          -> g = sigma(0) = 0.5")
print("  h=1 OR d=1 -> score = 1        -> g = sigma(beta) ~ 1")
print("  Soft-OR fires under EITHER signal (high entropy OR teacher endorsement).")
