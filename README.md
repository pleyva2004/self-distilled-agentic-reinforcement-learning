# Self-Distilled Agentic Reinforcement Learning — study artifacts

[![Render LaTeX](https://github.com/pleyva2004/self-distilled-agentic-reinforcement-learning/actions/workflows/render.yml/badge.svg)](https://github.com/pleyva2004/self-distilled-agentic-reinforcement-learning/actions/workflows/render.yml)

> Layered study artifacts — talking points, math deep dive, opinion-capture template, LaTeX literature-review entry, proposed extensions with runnable prototypes, an interactive learning map, and a two-level sandbox (CPU baseline + hardware-upsized for `tier_mid_gpu`) — for [SDAR (Lu et al. 2026, arxiv:2605.15155)](https://arxiv.org/abs/2605.15155).

**Source:** https://arxiv.org/abs/2605.15155 · **Code:** https://github.com/ZJU-REAL/SDAR · **Authors:** Zhengxi Lu et al. (Zhejiang University REAL Lab + Meituan, 11 authors)

**Compiled PDFs:** [`pdfs/04-literature-review.pdf`](./pdfs/04-literature-review.pdf) · [`pdfs/05-improvements.pdf`](./pdfs/05-improvements.pdf) · [`learning-map/tour.pdf`](./learning-map/tour.pdf) · [`proofs/gating-bias-bound.pdf`](./proofs/gating-bias-bound.pdf) · [`proofs/sdar-as-constrained-rl.pdf`](./proofs/sdar-as-constrained-rl.pdf) — auto-built by GitHub Actions on every push to `.tex` / `.bib` files.

## 📍 Start here

New to this work? [Read the tour](./learning-map/tour.md) — a guided walk through the foundations you'll need (chain repo Ch 25 / 27 / 28 / 30 / 31), SDAR's paper concepts, and the four proposed improvements (each validated by a formal proof or a measured prototype).

Also available: [`tour.pdf`](./learning-map/tour.pdf) (CI-rendered) · [`tour.ipynb`](./learning-map/tour.ipynb) (interactive, runs the measurement prototypes inline).

Discovered via the [`study-paper`](https://github.com/pleyva2004/claude-skill-study-paper) Claude Code skill's auto-discovery feature (no-args invocation surfaced it as a "matched" candidate on 2026-05-16).

---

## The paper

**Title:** Self-Distilled Agentic Reinforcement Learning (SDAR)
**Authors:** Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu + 6 more (Zhejiang University REAL Lab + Meituan)
**Venue:** arXiv preprint, May 2026

**Headline claim.** Add a *gated, per-token* on-policy self-distillation auxiliary loss to GRPO. The teacher branch is the same policy conditioned on privileged context (retrieved skills). A sigmoid gate $g_t = \sigma(\beta \Delta_t)$ — where $\Delta_t$ is the teacher-student log-prob gap — handles two failure modes that break vanilla OPSD in multi-turn agentic training: (1) multi-turn instability caused by student drift and (2) asymmetric trust in privileged guidance.

**Key result.** On Qwen 2.5 3B Instruct: +9.4 pp on ALFWorld, +7.0 pp on Search-QA, +10.2 pp on WebShop-Acc over GRPO baseline. SDAR keeps per-turn KL bounded; vanilla GRPO+OPSD shows catastrophic KL divergence growth (Figure 2 of the paper, reproduced qualitatively in `sandbox/toy_sdar.py`).

## What's in this repo

| Path | Purpose |
|------|---------|
| [`source.pdf`](./source.pdf) | The paper |
| [`metadata.json`](./metadata.json) | Title, source URL, code URL, discovery provenance, hardware sizing tier |
| [`01-interview-prep.md`](./01-interview-prep.md) | ~500-word, opinionated talking points: novel / clever / push-back / open questions / one-sentence elevator |
| [`02-math-deep-dive.md`](./02-math-deep-dive.md) | Mathematician-grade walk-through (~2000 words): agentic MDP, GRPO backbone, OPSD failure modes, single-sample gap estimator, three gating strategies, full SDAR objective, why gated dominates ungated, connections to chain Ch 25-31 + sibling RLM study |
| [`findings.md`](./findings.md) | **Locally-verified empirical findings** — 3 paper-claim reproductions + 2 improvement measurements, with provenance (machine, numpy version, seed) and per-finding reproduce commands |
| [`03-opinions.md`](./03-opinions.md) | Opinion-capture template (filled in by hand, not by AI) |
| [`04-literature-review.tex`](./04-literature-review.tex) | Research-ready LaTeX literature-review entry, standalone-compilable |
| [`05-improvements.tex`](./05-improvements.tex) | Forward-looking proposals: math (tight bias bound for gap gating), code (adaptive-quantile gating), experimental ($\lambda$-sensitivity sweep), theoretical (SDAR as constrained-RL Lagrangian) |
| [`improvements/`](./improvements/) | Runnable Python prototypes: `adaptive-gate.py` + `lambda-sweep.py`, each with `measure() -> dict` |
| [`sandbox/`](./sandbox/) | **Two-level**: Level 1 = CPU baseline (`toy_sdar.py` + `tiny_sdar_lm.py`, pure numpy, ~5s total) + Level 2 = tier_mid_gpu-sized (`torch_sdar.py` ~30M GPT on MPS + `real_sdar_lora.py` LoRA on Qwen 2.5 1.5B). |
| [`learning-map/`](./learning-map/) | Three-graph interactive learning map: `paper/` (14 paper concepts) + `improvements/` (4 improvement concepts) + `tour.{md,tex,ipynb}` (cross-cutting tour) |
| [`proofs/`](./proofs/) | Per-improvement proof artifacts: `gating-bias-bound.tex` (math validation) + `sdar-as-constrained-rl.tex` (theoretical validation) |
| [`references.bib`](./references.bib) | BibTeX entries: SDAR + 14 prior works it engages with |

## Build the LaTeX artifacts (PDFs)

```bash
sudo apt install texlive-latex-base texlive-fonts-recommended texlive-publishers
pdflatex 04-literature-review.tex && bibtex 04-literature-review && pdflatex 04-literature-review.tex && pdflatex 04-literature-review.tex
pdflatex 05-improvements.tex
(cd learning-map && pdflatex tour.tex)
(cd proofs && pdflatex gating-bias-bound.tex && pdflatex sdar-as-constrained-rl.tex)
```

GitHub Actions does this automatically on every push that touches `.tex` or `.bib`.

## Run the sandbox

The sandbox has **two levels**. See [`sandbox/README.md`](./sandbox/README.md) for per-script details, expected wall-clock, and the mapping back to the math in `02-math-deep-dive.md`.

### Level 1 — CPU baseline (runs anywhere; ~5 seconds total)

```bash
cd sandbox && pip install numpy matplotlib
python3 toy_sdar.py        # ~1s on CPU; demonstrates SDAR reward ≥ GRPO reward, SDAR KL ≤ ungated-OPSD KL
python3 tiny_sdar_lm.py    # ~5s on CPU; reward 0.34 → 0.75 on toy "find substring" task
```

Level 1 verifies the *structural* claims of the paper (gating bounds KL, SDAR matches GRPO, ungated OPSD collapses). Measured numbers are recorded in [`findings.md`](./findings.md) § 1-2.

### Level 2 — Hardware-upsized for `tier_mid_gpu` (needs M-series Mac or NVIDIA GPU)

```bash
cd sandbox && pip install torch tiktoken transformers peft accelerate
# ~30M-param torch GPT with full SDAR training loop
python3 torch_sdar.py --steps 5            # smoke test (~2 min on CPU; <30s on MPS)
python3 torch_sdar.py --train              # full ~200-step training (~30-60 min on M4 Pro MPS)
# LoRA fine-tune of Qwen 2.5 1.5B Instruct with SDAR on a synthetic agentic task
python3 real_sdar_lora.py --steps 1        # smoke test (~5 min on M4 Pro)
python3 real_sdar_lora.py --train          # full LoRA fine-tune (~3-4 hr on M4 Pro)
```

On Linux without GPU + heavy deps, both Level 2 scripts gracefully print install messages and exit 0.

Level 2 results are **pending a Mac run** — see [`findings.md`](./findings.md) § "Level 2 results (pending Mac run)" for the placeholder template to fill in once you've run the scripts.

## Run the proposed-improvement prototypes

```bash
cd improvements && pip install -r requirements.txt
python3 adaptive-gate.py   # adaptive-quantile gating vs fixed-threshold; KL/reward tradeoff
python3 lambda-sweep.py    # sensitivity sweep of λ_SDAR ∈ {0, 0.1, 0.3, 1.0, 3.0}
```

Each `measure()` returns a JSON dict with the headline numbers.

## Chain-repo cross-links

This study sits on top of [`pleyva2004/first-principles-to-llms`](https://github.com/pleyva2004/first-principles-to-llms), the chain that walks from set theory through modern LLM training. Specifically:

- **Chain Ch 25** — causal LM as MDP (every per-token decision inside each turn IS the Ch 25 MDP)
- **Chain Ch 27** — tiny GPT pre-training pipeline (the backbone in `tiny_sdar_lm.py` is reused from here)
- **Chain Ch 28** — SFT, RLHF (PPO/GRPO), DPO (SDAR's distillation loss is a KL — same primitive as RLHF's KL-to-reference)
- **Chain Ch 30** — max-entropy RL (SDAR's gating is structurally a Lagrangian on a per-token trust constraint — see `proofs/sdar-as-constrained-rl.tex`)
- **Chain Ch 31** — policy gradient + GRPO derivation + RLHF/DPO bridge (SDAR's RL backbone is Ch 31 verbatim)

## Sibling study

[`pleyva2004/reinforcing-recursive-language-models`](https://github.com/pleyva2004/reinforcing-recursive-language-models) — different angle on agentic-RL post-training. **RLM blog** uses *shared-policy + advantage inheritance across a recursive call tree*; **SDAR** uses *teacher-with-privileged-context + gated per-token distillation*. Both extend GRPO; cross-reading recommended.

## License

- **Source paper (`source.pdf`):** © Lu et al. 2026 (arxiv CC license — see arxiv abs page).
- **All other files in this repo:** MIT.
