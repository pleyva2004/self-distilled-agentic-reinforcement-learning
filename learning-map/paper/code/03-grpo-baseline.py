"""Concept 03: GRPO baseline — group-relative advantage + per-token clip.

For one prompt we sample G=8 fake responses, compute group-relative
advantages, and apply the PPO clipped per-token objective. We print the
loss with and without the clip to see when the clip activates.
"""
import numpy as np

rng = np.random.default_rng(1)
G = 8
T = 12
EPS = 0.2


def softmax_logprob(logits, idx):
    z = logits - logits.max()
    e = np.exp(z)
    return np.log(e[idx] / e.sum())


# Fabricate per-token log-probs for old (sampling) and new (current) policies.
old_logp = rng.normal(loc=-2.0, scale=0.3, size=(G, T))
new_logp = old_logp + rng.normal(loc=0.0, scale=0.4, size=(G, T))
mask = np.ones((G, T))  # all tokens valid

# Group-relative advantages
rewards = rng.random(G)  # in [0,1]
adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

ratio = np.exp(new_logp - old_logp)  # r_t = pi_new / pi_old
clipped = np.clip(ratio, 1 - EPS, 1 + EPS)

# PPO clipped objective (negated for loss)
unclipped_obj = ratio * adv[:, None]
clipped_obj = clipped * adv[:, None]
per_token = np.minimum(unclipped_obj, clipped_obj)

# Masked-token average (the "Agg" operator)
agg = (mask * per_token).sum() / mask.sum()
loss = -agg

# Diagnose how often clip fires
clip_fires = ((ratio < 1 - EPS) | (ratio > 1 + EPS)).mean()

print(f"Group size G = {G}, response length T = {T}, clip eps = {EPS}")
print(f"Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
print(f"Group-relative advantages (per response): {np.round(adv, 3).tolist()}")
print(f"GRPO loss (per-token PPO clip) = {loss:.4f}")
print(f"Fraction of tokens where clip fires = {clip_fires:.2%}")
