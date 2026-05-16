"""Concept 14: SDAR combined objective L = L_GRPO + lambda * L_SDAR.

Toy demo: one prompt, G responses. Compute both terms and the gradient
on the student's tabular logits. Show how lambda controls the balance.
"""
import numpy as np

rng = np.random.default_rng(7)
G = 4
T = 6
V = 8
EPS = 0.2
BETA = 2.0
LAMBDAS = [0.0, 0.1, 1.0, 10.0]


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Fabricate per-(g,t) old/new student log-probs and teacher log-probs at the sampled token
old_logp = rng.normal(loc=-2.0, scale=0.3, size=(G, T))
new_logp = old_logp + 0.4 * rng.normal(size=(G, T))
teacher_logp = old_logp + 0.5 * rng.normal(size=(G, T))
mask = np.ones((G, T))
rewards = rng.random(G)
adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

# GRPO loss
ratio = np.exp(new_logp - old_logp)
ppo_obj = np.minimum(ratio * adv[:, None],
                     np.clip(ratio, 1 - EPS, 1 + EPS) * adv[:, None])
L_grpo = -(mask * ppo_obj).sum() / mask.sum()

# SDAR loss with gap-gating
delta = teacher_logp - new_logp
gate = sigmoid(BETA * delta)  # stop-gradient assumed
sdar_per_token = gate * delta
L_sdar = (mask * sdar_per_token).sum() / mask.sum()

print(f"L_GRPO = {L_grpo:+.4f}")
print(f"L_SDAR = {L_sdar:+.4f}  (mean gap = {delta.mean():+.4f}, mean gate = {gate.mean():.4f})")
print()
print(f"{'lambda':>8}  {'L_total':>10}  composition")
for lam in LAMBDAS:
    L_total = L_grpo + lam * L_sdar
    print(f"{lam:>8.2f}  {L_total:>+10.4f}  L_GRPO ({L_grpo:+.4f}) + {lam}*L_SDAR ({lam*L_sdar:+.4f})")

print("\nLambda interpolates between pure GRPO (lambda=0) and SDAR-dominated training.")
