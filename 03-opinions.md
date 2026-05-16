# Opinions — Self-Distilled Agentic Reinforcement Learning

> **This file is a template for *you* (Pablo) to fill in. The skill never invents your opinions.**
> Read `01-interview-prep.md` and `02-math-deep-dive.md` first, then write the answers in your own voice. Stage 4+ will incorporate whatever you put here. Bullet points are fine; full sentences are better. If you skip a question, the skill proceeds with only the deep dive as input.

---

## 1. What did you find most surprising in the paper?

_(Could be a result, a method choice, an explicit limitation, an engineering detail, or a missing experiment. Examples: "the sigmoid gate alone handles two ostensibly different failure modes (multi-turn drift + asymmetric trust)", "OPSD alone catastrophically degrades performance on Search-QA (Table 1)", "the GRPO+OPSD baseline outperforms SDAR on some Search-QA sub-tasks".)_

> **Your answer:**

---

## 2. Where would this break in practice?

_(Specific deployment / scaling / generalisation scenarios where the method probably fails. Be concrete — name the failure mode, not just "it might not work". E.g.: "tasks where privileged context isn't retrieved skills but environment-side hints"; "horizons beyond 20 turns"; "when teacher is confidently wrong".)_

> **Your answer:**

---

## 3. What single experiment would falsify (or strongly support) the central claim?

_(The central claim: "a sigmoid-gated, per-token auxiliary distillation loss is strictly better than ungated OPSD in multi-turn agentic RL, because the gate handles asymmetric trust." What's the cleanest experiment that would settle it?)_

> **Your answer:**

---

## 4. If you were starting from this paper, what would you do differently — and why?

_(Different gating, different teacher, different RL backbone (PPO vs GRPO vs DPO), different task, learned vs hand-designed gate, etc.)_

> **Your answer:**

---

## 5. Free-form: anything else you want captured?

_(Connections to the RLM blog study (shared-policy + advantage inheritance — could SDAR's gating be lifted to recursive children?), philosophical reactions, hot takes, future-work ideas, math doubts. Anything that didn't fit above.)_

> **Your answer:**

---

When you're done, save the file and tell me to continue. Stages 4-7 will dispatch as 4 parallel subagents (sandbox, lit review, improvements, learning map), then Stage 8 will publish to GitHub.
