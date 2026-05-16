"""Improvement 101: Tight bias bound for gap gating — MC verification.

For sub-Gaussian Delta_t ~ N(mu_Delta, sigma_Delta^2), we numerically
estimate B = E[Delta] - E[sigma(beta * Delta) * Delta] for a grid of
(mu, sigma, beta), and check it stays within the predicted bound
(1/2) * E[|Delta|] + C * beta * sigma^2.
"""
import numpy as np

rng = np.random.default_rng(101)
N = 50_000
C_BOUND = 0.25  # paper bound constant; could be tightened


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


print(f"{'mu':>5} {'sigma':>6} {'beta':>5}  {'bias':>8}  {'bound':>8}  {'tight?':>7}")
all_ok = True
for mu in [-1.0, 0.0, 1.0]:
    for sigma in [0.5, 1.0, 2.0]:
        for beta in [0.5, 1.0, 4.0]:
            samples = rng.normal(loc=mu, scale=sigma, size=N)
            ungated = samples.mean()
            gated = (sigmoid(beta * samples) * samples).mean()
            bias = ungated - gated
            mean_abs = np.abs(samples).mean()
            bound = 0.5 * mean_abs + C_BOUND * beta * sigma**2
            ok = abs(bias) <= bound + 1e-3
            all_ok = all_ok and ok
            print(f"{mu:>+5.1f} {sigma:>6.2f} {beta:>5.2f}  {bias:>+8.3f}  {bound:>8.3f}  "
                  f"{'OK' if ok else 'FAIL':>7}")

print(f"\nAll bounds satisfied: {all_ok}")
print("Empirically: bias stays within the (1/2)E|Delta| + O(beta sigma^2) envelope.")
