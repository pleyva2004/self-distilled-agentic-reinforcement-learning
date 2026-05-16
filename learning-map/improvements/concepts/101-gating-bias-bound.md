# Tight bias bound for gap gating

**Level:** advanced
**Prerequisites:** [12-gap-gating](../../paper/concepts/12-gap-gating.md), [09-asymmetric-trust-observation](../../paper/concepts/09-asymmetric-trust-observation.md)
**Used by:** [`proofs/gating-bias-bound.tex`](../../../proofs/gating-bias-bound.tex)

## Plain-English intro
The paper's §7 sketch argues "gating reduces bias" but gives no closed-form. We state and prove a quantitative bound: under a sub-Gaussian assumption on the gap distribution $p(\Delta_t)$, the bias of the gated update relative to the (idealized) ungated update is bounded by an explicit function of the sub-Gaussian parameter $\sigma_\Delta$ and the gate sharpness $\beta$.

## Formal definition
Let $\Delta_t \mid s_t$ be sub-Gaussian with parameter $\sigma_\Delta$. Let
$$U^{\mathrm{gated}}(s_t) = \mathbb{E}[\sigma(\beta \Delta_t) \cdot \Delta_t \mid s_t], \qquad U^{\mathrm{ungated}}(s_t) = \mathbb{E}[\Delta_t \mid s_t].$$
Define the gap-gating bias $B(s_t) = U^{\mathrm{ungated}}(s_t) - U^{\mathrm{gated}}(s_t)$. The full theorem and proof are in [`proofs/gating-bias-bound.tex`](../../../proofs/gating-bias-bound.tex). The headline bound is
$$|B(s_t)| \leq \tfrac{1}{2}\,\mathbb{E}[|\Delta_t|] + O(\beta \sigma_\Delta^2),$$
with explicit constants. The first term is the "negative-gap suppression" benefit (this is what gating *intends* to do); the second term is the residual bias from using a smooth sigmoid rather than a hard $0/1$ threshold.

## Why this matters for the paper
A formal bound turns "asymmetric treatment is good" from intuition into a falsifiable claim with measurable parameters. It also identifies which assumptions on the retrieval pipeline (Gaussian-tailed gap distribution) underwrite the analysis.

## Code
See [`../code/101-gating-bias-bound.py`](../code/101-gating-bias-bound.py) for a Monte-Carlo verification of the bound's tightness on synthetic gap distributions.
