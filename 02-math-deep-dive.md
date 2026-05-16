# Math deep dive — Self-Distilled Agentic Reinforcement Learning

> Source: arxiv:2605.15155, Zhengxi Lu et al. (Zhejiang University REAL Lab + Meituan), 2026-05-14. Code: https://github.com/ZJU-REAL/SDAR. Where I cite foundational machinery I link to the canonical chapter in [`pleyva2004/first-principles-to-llms`](https://github.com/pleyva2004/first-principles-to-llms) (the **chain repo**) and to the sibling agentic-RL study [`pleyva2004/reinforcing-recursive-language-models`](https://github.com/pleyva2004/reinforcing-recursive-language-models) (the **RLM study**).

## Notation key

| Symbol | Meaning |
| --- | --- |
| $\pi_\theta(\cdot \mid s_t)$ | Student policy: probability distribution over the next token given student context $s_t$. |
| $\pi_\theta^+(\cdot \mid s_t^+) \equiv \pi_T(\cdot \mid s_t^+)$ | Teacher branch: same parameters $\theta$, but conditioned on $s_t^+ = s_t \cup \text{privileged context}$ (e.g. retrieved skill text the student does not see at inference). |
| $y_t$ | Student-sampled token at position $t$: $y_t \sim \pi_\theta(\cdot \mid s_t)$. |
| $m_t \in \{0, 1\}$ | Response mask (1 if token $t$ is a valid response token, 0 otherwise). |
| $A^{(i)}$ | Sequence-level GRPO advantage for the $i$-th sampled response. |
| $r_t^{(i)} = \pi_\theta(y_t^{(i)} \mid s_t^{(i)}) / \pi_{\theta_\mathrm{old}}(y_t^{(i)} \mid s_t^{(i)})$ | PPO importance ratio at token $t$. |
| $\epsilon$ | PPO clip radius (paper uses $\epsilon = 0.2$). |
| $\beta$ | KL-to-reference coefficient in GRPO; also reused as the gate sharpness in SDAR (paper notation overload). |
| $\Delta_t$ | Teacher-student log-prob gap at the student-sampled token: $\Delta_t = \log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t)$. Positive ⇒ teacher endorses; negative ⇒ teacher disagrees. |
| $h_t$ | Student entropy: $h_t = -\sum_v \pi_\theta(v \mid s_t) \log \pi_\theta(v \mid s_t)$. |
| $g_t \in [0, 1]$ | Token-level gate (sigmoid-bounded, stop-grad detached). |
| $\lambda_{\mathrm{SDAR}}$ | Auxiliary-loss weight (the main SDAR hyperparameter). |

Broader probability / RL notation (sample spaces, KL, expectations) follows the [math-foundations glossary](https://github.com/pleyva2004/math-foundations/blob/main/NOTATION.md) and chain Ch 8-11.

## 1. The agentic MDP

Following the paper's Section 2.1 ("Multi-Turn Agent Problem"), the setting is a finite-horizon MDP with episodic structure:

- Initial state: task description $x$.
- At turn $k$: agent receives observation $o_k$, generates response $a_k$ (containing reasoning + tool/action tokens), environment returns $o_{k+1}$.
- Flatten all valid response tokens into a single token sequence $y = (y_1, \ldots, y_T) \sim \pi_\theta(\cdot \mid x)$.
- Reward $R(y, \text{traj})$ from environment verifier (e.g. task success indicator).

Per-token, the **student context** $s_t$ contains the task description, all prior observations $o_{1:k}$, and all prior tokens $y_{1:t-1}$. The **privileged teacher context** $s_t^+$ adds retrieved-skill text (or other privileged data) that's available only during training.

Connect to chain Ch 25 (causal LM as MDP): every per-token decision is the Ch 25 MDP; the agentic wrapper is the multi-turn outer loop, identical in form to the trajectory tree formalism in the RLM study (Section 1 of its math deep dive).

## 2. GRPO backbone

SDAR keeps GRPO unchanged as the primary optimization signal. Per the paper (Equation 2):

$$\mathcal{L}_{\mathrm{GRPO}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \mathrm{Agg}\Big[\min\big(r_t^{(i)} A^{(i)}, \mathrm{clip}(r_t^{(i)}, 1-\epsilon, 1+\epsilon) A^{(i)}\big)\Big] \;+\; \beta \cdot \frac{1}{G}\sum_{i=1}^{G} \mathrm{Agg}\big[D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid s_t^{(i)}) \,\big\|\, \pi_{\mathrm{ref}}(\cdot \mid s_t^{(i)})\big)\big]$$

where $\mathrm{Agg}(z_{1:T}) = \frac{\sum_t m_t z_t}{\sum_t m_t}$ is the masked-token average. This is **identical** to the chain Ch 31 GRPO derivation and the RLM study's Section 2 root-loss formula. Group size $G$, importance ratio $r_t^{(i)}$, advantage $A^{(i)} = (R^{(i)} - \bar R)/\mathrm{std}(R)$ from the response group, KL-to-reference regularizer with coefficient $\beta$ — all standard.

## 3. OPSD background and its multi-turn failure modes

**On-Policy Self-Distillation (OPSD)** computes a per-token KL between the teacher branch and the student branch:

$$D_{\mathrm{RKL}}^{(t)} = D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid s_t) \,\big\|\, \pi_T(\cdot \mid s_t^+)\big) = \sum_{v \in \mathcal{V}} \pi_\theta(v \mid s_t) \log \frac{\pi_\theta(v \mid s_t)}{\pi_T(v \mid s_t^+)}.$$

This is a *reverse* KL (student is the sampling distribution, teacher is the "reference"). Summing across response tokens gives the standard OPSD auxiliary loss. The trouble in multi-turn agentic training (paper §1):

**Observation 1 — Multi-turn instability.** As the student drifts from teacher-supported trajectories, per-turn KL grows superlinearly. Concretely: if $s_t$ is reached via a low-probability student trajectory, the teacher distribution $\pi_T(\cdot \mid s_t^+)$ is no longer a reliable target — the privileged context was generated under the assumption of a different trajectory. The distillation signal becomes noise.

**Observation 2 — Asymmetric trust in privileged guidance.** The teacher is not an independently stronger model; it's the **same parameters** with extra privileged context (typically retrieved skill text). Consequences:
- If the teacher assigns *higher* probability than the student to $y_t$ (positive gap), the privileged context is endorsing on-policy student behavior. This is high-value distillation signal.
- If the teacher assigns *lower* probability (negative gap), the signal is ambiguous: it could mean "the token should be suppressed" OR "the skill retrieval was bad and the teacher's privileged context misled it". The paper documents three failure modes for the negative case: (i) skill quality (irrelevant or incomplete skills), (ii) skill utilization (teacher fails to ground skills into token-level preferences), (iii) multi-turn drift (teacher-student gap widens across turns).

The asymmetry is real and motivates the central design choice.

## 4. The teacher-student gap and its single-sample estimator

Computing the full-vocabulary reverse KL is expensive (vocab $\approx 152$k for Qwen 2.5). The paper introduces a **single-sample estimate** evaluated on the student-sampled token $y_t \sim \pi_\theta(\cdot \mid s_t)$:

$$\hat D_{\mathrm{RKL}}^{(t)} = \log \pi_\theta(y_t \mid s_t) - \log \pi_T(y_t \mid s_t^+).$$

This is an unbiased estimate of one term inside the expectation defining $D_{\mathrm{RKL}}$ (where the expectation is over $y_t \sim \pi_\theta$). Negate it to get the **teacher-student gap**:

$$\boxed{\quad \Delta_t = -\hat D_{\mathrm{RKL}}^{(t)} = \log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t) \quad}$$

A positive $\Delta_t$ means the teacher's privileged context endorses the student's sampled token; a negative $\Delta_t$ means the teacher would have preferred a different token.

This is the same gradient-estimator trick as the chain Ch 31 policy-gradient theorem — replace an expensive expectation with a sample drawn from the distribution that's already being sampled anyway.

## 5. Token-level gating

The central SDAR contribution. The paper defines three gating strategies, all of the form $g_t = \sigma(\beta \cdot \text{score}_t)$ where $\sigma$ is the sigmoid and $\beta > 0$ is sharpness:

1. **Entropy gating:** $g_t = \sigma(\beta h_t)$ — fires at high-entropy positions (student is uncertain; distillation has more room to help).
2. **Gap gating:** $g_t = \sigma(\beta \Delta_t)$ — fires when the teacher endorses the student's sampled token. The **asymmetric trust** of §3 made concrete: positive-gap tokens get weight close to 1; negative-gap tokens get weight close to 0; the sigmoid is smooth, so the transition is differentiable.
3. **Soft-OR gating:** $g_t = \sigma(\beta [1 - (1 - h_t)(1 - \Delta_t)])$ — combines both signals; gate fires if either student uncertainty is high OR teacher endorses.

The gate is **detached** via $\mathrm{sg}(\cdot)$ (stop-gradient), so gradients flow only through the student log-probability inside the loss term, not through the gate itself.

## 6. The SDAR auxiliary loss

The token-level SDAR loss:

$$\ell_t^{\mathrm{SDAR}} = g_t \cdot \big(\log \pi_T(y_t \mid s_t^+) - \log \pi_\theta(y_t \mid s_t)\big) = g_t \cdot \Delta_t.$$

Aggregated across response tokens via the masked-token average:

$$\mathcal{L}_{\mathrm{SDAR}} = \mathrm{Agg}(\ell_{1:T}^{\mathrm{SDAR}}) = \frac{\sum_t m_t g_t \Delta_t}{\sum_t m_t}.$$

Combined objective:

$$\boxed{\quad \mathcal{L}(\theta) = \mathcal{L}_{\mathrm{GRPO}}(\theta) + \lambda_{\mathrm{SDAR}} \cdot \mathcal{L}_{\mathrm{SDAR}}(\theta). \quad}$$

**Important subtlety.** When the gate is detached, the gradient of $\mathcal{L}_{\mathrm{SDAR}}$ w.r.t. $\theta$ is

$$\nabla_\theta \mathcal{L}_{\mathrm{SDAR}} = -\mathrm{Agg}\big[g_t \cdot \nabla_\theta \log \pi_\theta(y_t \mid s_t)\big]$$

(the teacher log-prob is also detached, since the teacher is the same parameters and computing its gradient would couple back; in practice the paper computes teacher logits with a separate forward pass under no-grad). So the SDAR loss is *exactly* a gated REINFORCE-style update on the student log-prob, with the gate $g_t$ playing the role of a per-token signed weight. Positive gate + positive gap = up-weight the student's probability for $y_t$; negative gate (close to 0 from sigmoid on negative $\Delta_t$) = essentially no update.

## 7. Why gated dominates ungated (theoretical sketch)

The paper's Appendix A provides a formal analysis; here's the intuition.

Consider the gap distribution $p(\Delta_t)$ over training tokens. Without gating, every token contributes $\Delta_t \cdot \nabla \log \pi_\theta(y_t \mid s_t)$ to the gradient — including tokens where the teacher's privileged context is unreliable (negative $\Delta_t$ caused by skill-retrieval failure). The expectation of these unreliable contributions is non-zero (the failure modes in §3 are not symmetric around zero), so the ungated estimator is **biased** in a direction that fights against the RL signal.

With gap gating $g_t = \sigma(\beta \Delta_t)$, the contribution from a token with $\Delta_t \to -\infty$ goes to $g_t \cdot \Delta_t \to 0 \cdot (-\infty) = 0$ (in the limit; $g_t$ goes to zero faster). Negative-gap tokens are silently dropped. Positive-gap tokens, where the privileged context is consistently endorsing, are kept at full weight. **Lower bias at the cost of dropped signal** — a textbook bias-variance tradeoff that the gating threshold $\beta$ controls.

The chain Ch 31 baseline-doesn't-bias proof applies here too: the gate $g_t$ is detached and is a function of the *current* student/teacher logits, which are conditional on the trajectory but not on the policy parameters being differentiated. So the gated estimator is still unbiased *for the gated objective*, just not for the original ungated OPSD objective.

## 8. Empirical results (Table 1)

On Qwen 2.5-3B-Instruct:
- ALFWorld average success: GRPO = 75.0%, SDAR = **84.4%** (+9.4 pp).
- WebShop Score: GRPO = 79.8%, SDAR = **85.0%** (+5.2 pp).
- Search-QA average: GRPO = 36.4%, SDAR = 43.4% (+7.0 pp).

On Qwen 2.5-7B-Instruct, similar gains; SDAR matches or beats GRPO and GRPO+OPSD on most benchmarks.

A key qualitative observation from the paper's Figure 2: **GRPO+OPSD without gating** shows catastrophic KL divergence growth over training, while SDAR keeps KL bounded — matching the §3 multi-turn-instability story.

## 9. Connections to the chain and the sibling RLM study

- **Chain Ch 25** (causal LM as MDP): the per-token decision inside each turn IS the Ch 25 MDP; the agentic wrapper is the multi-turn outer loop.
- **Chain Ch 27** (tiny GPT pre-training): the SFT-pretrained Qwen 2.5 backbone follows the Ch 27 recipe.
- **Chain Ch 28** (SFT, RLHF, DPO): SDAR's distillation loss is structurally a KL-to-reference, same primitive as RLHF's KL regularizer to the SFT policy.
- **Chain Ch 30** (max-entropy RL): the gated SDAR loss is a Lagrangian-style auxiliary — `lambda_SDAR` is the multiplier on a per-token-trust constraint.
- **Chain Ch 31** (PG/GRPO/RLHF-DPO bridge): SDAR's RL backbone is GRPO from Ch 31 verbatim; the single-sample estimator trick mirrors Ch 31's policy-gradient theorem proof.
- **Sibling RLM study**: both SDAR and the RLM blog are agentic-RL extensions of GRPO. **RLM blog** uses *shared-policy + advantage inheritance across a recursive call tree* — credit assignment by tree topology. **SDAR** uses *teacher-with-privileged-context + gated per-token distillation* — credit assignment by per-token reliability gating. Different angles on the same problem: how to give a base GRPO loop denser supervision without breaking stability.

## 10. Open mathematical questions

1. **Tight bias bound for gap gating.** The §7 sketch argues "lower bias", but no closed-form bound is given. A precise statement under explicit assumptions on the gap distribution would tighten the theory. This is the math improvement proposed in `05-improvements.tex`.
2. **Depth scaling of the KL bound.** The paper shows KL stays bounded at $\sim 20$ turns. Does the bound hold under increasing horizon $H$? Likely the bias contribution scales as $O(H)$ at worst — needs a proof.
3. **Adaptive gating.** Three fixed gating strategies + a sharpness $\beta$. Could a learned gate $g_\phi(s_t, \Delta_t, h_t)$ outperform? Closely connected to the meta-learning literature; would need its own analysis.
4. **Connection to constrained RL.** The gate behaves like a Lagrange multiplier on a per-token trust constraint. Formalizing this correspondence (mapping to Sutton-Precup-Singh-style constrained MDPs) is the theoretical improvement in `05-improvements.tex`.
