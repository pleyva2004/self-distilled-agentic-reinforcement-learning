# SDAR as a per-token trust-constraint Lagrangian

**Level:** advanced
**Prerequisites:** [14-sdar-combined-objective](../../paper/concepts/14-sdar-combined-objective.md), Chain Ch 30 (max-ent / dual)
**Used by:** [`proofs/sdar-as-constrained-rl.tex`](../../../proofs/sdar-as-constrained-rl.tex)

## Plain-English intro
The SDAR loss looks like an unrelated auxiliary, but it can be derived as the Lagrangian dual of a per-token *trust constraint*: "policy should not assign less log-probability than the teacher does to the sampled token, weighted by per-token trust $g_t$." Under this lens, $\lambda_{\mathrm{SDAR}}$ is the (fixed) Lagrange multiplier on this family of constraints, and the gating $g_t$ is a state-dependent constraint weight.

## Formal definition
Constrained problem:
$$\max_\theta J_{\mathrm{GRPO}}(\theta) \quad \text{s.t.} \quad \mathbb{E}_t\big[g_t (\log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t))\big] \leq \delta.$$

Lagrangian:
$$\mathcal{L}_{\mathrm{lagr}}(\theta, \mu) = -J_{\mathrm{GRPO}} + \mu \cdot \big(\mathbb{E}_t[g_t \Delta_t] - \delta\big).$$

Identifying $\mu = \lambda_{\mathrm{SDAR}}$ recovers SDAR's combined objective up to the slack constant $\delta$ (which doesn't affect $\nabla_\theta$). The paper trains with $\mu$ fixed; lifting to a primal-dual scheme that updates $\mu$ to satisfy the constraint exactly is a natural extension.

## Why this matters for the paper
The constrained-RL framing connects SDAR to a mature literature (CMDP, primal-dual RL) and immediately suggests:
- A principled annealing schedule for $\lambda$ (= dual ascent).
- Convergence guarantees (saddle-point conditions on $(\theta, \mu)$).
- Multiple constraints (one per gating strategy), with separate multipliers.

Full proof in [`proofs/sdar-as-constrained-rl.tex`](../../../proofs/sdar-as-constrained-rl.tex).

## Code
See [`../code/104-sdar-as-constrained-rl.py`](../code/104-sdar-as-constrained-rl.py) — toy primal-dual loop showing $\mu$ converges to a value reproducing the SDAR loss balance.
