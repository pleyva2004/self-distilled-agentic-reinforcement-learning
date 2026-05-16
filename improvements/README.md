# SDAR Improvements — Sandbox Prototypes

Numerical sandboxes for the proposals in
[`../05-improvements.tex`](../05-improvements.tex).

## Index

| Script | Maps to | What it measures |
| --- | --- | --- |
| `adaptive-gate.py` | `\section*{2. Code / Implementation Improvements}` | Compares fixed-threshold SDAR vs adaptive-quantile SDAR on a 4-turn / 5-action toy MDP. Reports reward, per-turn KL, and gate-fire rate for each variant (plus GRPO baseline and ungated OPSD sanity check). |
| `lambda-sweep.py` | `\section*{3. Experimental Extensions}` | Sweeps `lambda_SDAR` in {0.0, 0.1, 0.3, 1.0, 3.0} on the same toy MDP. Identifies the constrained argmax `lambda*` subject to a per-turn KL budget of `2 x kl(lambda=0)`. |

The two `\section*` blocks marked `PROOF` (1 and 4) are paper-side
contributions — the prototype pointers in the LaTeX file refer to
`proofs/gating-bias-bound.tex` and `proofs/sdar-as-constrained-rl.tex`, not
to scripts in this directory.

## Run

```bash
pip install -r requirements.txt   # numpy only
python3 adaptive-gate.py          # ~10s on CPU
python3 lambda-sweep.py           # ~30s on CPU
```

Each script prints a JSON summary on stdout; both expose a `measure()`
function returning the same dict.

## Reproducibility

Both scripts use `numpy` only and call `np.random.seed(0)` at the top of
`measure()`. The single-file design (each prototype inlines its own copy
of the toy MDP) means there are no cross-imports and no hidden dependencies
beyond numpy.
