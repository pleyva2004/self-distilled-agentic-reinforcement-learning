# SDAR study tour

## 1. Reader's contract

This learning map is for someone who wants to understand **Self-Distilled Agentic Reinforcement Learning** (Lu et al. 2026, arxiv:2605.15155) at the level needed to reproduce the math, critique the design, and propose extensions.

Three skill-level entry points (pick one and read the foundations walk to that depth):

- **Beginner** — comfortable with neural networks and LLM training but new to RL: budget 6-8 hours; read all of §2 below.
- **Intermediate** — comfortable with policy gradients and PPO but not GRPO: budget 3-4 hours; skim §2, focus on §3.
- **Advanced** — comfortable with GRPO, KL regularization, and trust-region methods: budget 1-2 hours; skip to §3 and §4.

The full chain to reach the paper from foundations is Chain Ch 25, 27, 28, 30, 31 (see foundations walk). The math in this paper itself is a few pages of the deep-dive document `02-math-deep-dive.md`; everything else is design rationale and ablations.

**Companion artifacts.** Each of the 14 paper concepts has a 1-page concept note in `paper/concepts/` and a runnable Python demo in `paper/code/`. Each of the 4 improvement concepts has the same in `improvements/`. The two formal proofs live in `proofs/`. The Jupyter notebook (`tour.ipynb`) walks through the same content with executable cells for the measurement-style improvements.

## 2. Foundations walk

These five chapters from the foundations chain are the prerequisite knowledge for SDAR. Read in order; later chapters assume earlier ones.

**Chain Ch 25 — Causal LM as MDP.** The per-token autoregressive decoding loop is a Markov decision process. State = prompt + tokens-so-far. Action = next token. Why this paper needs it: every per-token decision in SDAR is one transition of this MDP, and the multi-turn agentic setting wraps this MDP in an outer environment loop. *Pacing tag: 30 min, conceptual only.*

**Chain Ch 27 — Tiny GPT pre-training.** The decoder-only Transformer architecture. Why this paper needs it: the Qwen 2.5 backbone the paper uses follows this recipe; nothing about SDAR depends on the architecture *details*, but you need to be fluent enough to read the loss equations as gradients on the parameters of such a model. *Pacing tag: 60 min, mostly review for advanced readers.*

**Chain Ch 28 — SFT, RLHF, DPO.** Supervised fine-tuning, KL-regularized RL from human feedback, direct preference optimization. Why this paper needs it: SDAR's distillation loss is structurally a KL-to-reference, the same primitive as RLHF's KL-to-SFT regularizer. The asymmetric-trust observation parallels DPO's well-known conservatism on negatives. *Pacing tag: 90 min, central.*

**Chain Ch 30 — Maximum-entropy RL.** Soft Q-learning, the entropy bonus, the Lagrangian dual interpretation of constrained MDPs. Why this paper needs it: the gated SDAR loss IS a Lagrangian-style auxiliary, with $\lambda_{\mathrm{SDAR}}$ as the multiplier on a per-token trust constraint. Improvement 104 makes this formal. *Pacing tag: 60 min for the dual-form derivation alone.*

**Chain Ch 31 — Policy gradients, GRPO, RLHF-DPO bridge.** REINFORCE, PPO, GRPO, the policy-gradient theorem, single-sample estimators. Why this paper needs it: SDAR's RL backbone is GRPO from this chapter verbatim (Equation 2 of the paper). The single-sample teacher-student gap estimator uses the same expectation-replacement trick as the proof of the policy-gradient theorem. *Pacing tag: 90-120 min, core.*

## 3. Paper concepts walk

Read the 14 concepts in the order below. Edges in the DAG (`paper/README.md`) tell you what depends on what; the order here is one valid topological sort.

1. **Multi-turn agentic MDP** ([01](paper/concepts/01-multi-turn-agentic-mdp.md)) — set up the outer loop.
2. **Trajectory-level RL reward** ([02](paper/concepts/02-rl-trajectory-reward.md)) — sparse supervision is the pain.
3. **GRPO baseline** ([03](paper/concepts/03-grpo-baseline.md)) — RL backbone.
4. **OPSD** ([04](paper/concepts/04-opsd-on-policy-self-distillation.md)) — on-policy self-distillation primitive.
5. **Teacher branch with privileged context** ([05](paper/concepts/05-teacher-branch-privileged-context.md)) — what makes "self" non-trivial.
6. **Per-token reverse KL** ([06](paper/concepts/06-per-token-reverse-kl.md)) — why reverse KL (mode-seeking).
7. **Single-sample gap estimator** ([07](paper/concepts/07-single-sample-gap-estimator.md)) — the central estimator trick.
8. **Multi-turn instability observation** ([08](paper/concepts/08-multi-turn-instability-observation.md)) — first failure mode.
9. **Asymmetric trust observation** ([09](paper/concepts/09-asymmetric-trust-observation.md)) — second failure mode; key motivation.
10. **Sigmoid token-level gate** ([10](paper/concepts/10-token-level-gate.md)) — the scaffold.
11. **Entropy gating** ([11](paper/concepts/11-entropy-gating.md)) — first gating strategy.
12. **Gap gating** ([12](paper/concepts/12-gap-gating.md)) — second gating strategy; conceptual centerpiece.
13. **Soft-OR gating** ([13](paper/concepts/13-soft-or-gating.md)) — third gating strategy; combined.
14. **SDAR combined objective** ([14](paper/concepts/14-sdar-combined-objective.md)) — the full training loss.

## 4. Improvements walk

Each improvement is paired with a validation file. PROOF improvements have a typeset `.tex` file in `proofs/`; MEASUREMENT improvements have a sandbox script under `improvements/`.

**101 — Tight bias bound for gap gating** ([concept](improvements/concepts/101-gating-bias-bound.md)). The §7 sketch in the paper argues "lower bias" without a closed form. We state and prove a bound: for sub-Gaussian $\Delta_t$ with parameter $\sigma_\Delta$, the gating bias is at most $\tfrac{1}{2}\,\mathbb{E}[|\Delta_t|] + O(\beta\sigma_\Delta^2)$. **Validation mode: PROOF** — see [`proofs/gating-bias-bound.tex`](../proofs/gating-bias-bound.tex).

**102 — Adaptive gating threshold via running quantile** ([concept](improvements/concepts/102-adaptive-gating-threshold.md)). Replace fixed $\beta$ with a per-batch standardization of $\Delta_t$ using a running median + IQR. The hyperparameter shifts from "absolute gate sharpness" to "fraction of tokens to trust", which is more interpretable and stable across training phases. **Validation mode: MEASUREMENT** — see `improvements/adaptive-gate.py`.

**103 — Sensitivity sweep of $\lambda_{\mathrm{SDAR}}$** ([concept](improvements/concepts/103-lambda-sensitivity-sweep.md)). Two questions: how flat is the loss vs $\lambda$ landscape, and does an exponential anneal $\lambda(t) = \lambda_0 e^{-t/\tau}$ outperform a constant. The envelope theorem identifies $\partial L_{\mathrm{val}}/\partial \lambda$ from a single trained checkpoint, so this is cheap. **Validation mode: MEASUREMENT** — see `improvements/lambda-sweep.py`.

**104 — SDAR as a per-token trust-constraint Lagrangian** ([concept](improvements/concepts/104-sdar-as-constrained-rl.md)). Formal correspondence: the SDAR combined loss is the Lagrangian of the constrained problem "maximize GRPO objective subject to per-token trust constraint $\mathbb{E}_t[g_t \Delta_t] \leq \delta$". $\lambda_{\mathrm{SDAR}}$ is the (fixed) Lagrange multiplier. The connection imports primal-dual convergence theorems from the constrained-MDP literature. **Validation mode: PROOF** — see [`proofs/sdar-as-constrained-rl.tex`](../proofs/sdar-as-constrained-rl.tex).

## 5. What to do next

Three concrete actions, in increasing investment:

1. **(15 min)** Run all 14 paper concept Python scripts top-to-bottom: `for f in learning-map/paper/code/*.py; do python "$f"; done`. The printed outputs together form a guided narrative of the paper's mechanics.
2. **(2 hr)** Open `tour.ipynb`, execute every cell, and read the proofs. By the end you should be able to write down the SDAR loss from memory, derive the gradient, and explain why the gate is sigmoid (not tanh) without consulting any reference.
3. **(1 day)** Implement improvement 102 (adaptive gating) in the sandbox training script and run a short ablation: original gating vs adaptive on the smallest tractable training run. Even a negative result is publishable as an empirical observation.
