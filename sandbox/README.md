# SDAR sandbox — runnable demos for *Self-Distilled Agentic RL*

Two minimal numpy-only scripts that demonstrate the core mechanism of
**SDAR** (Lu et al. 2026, arxiv:2605.15155): a **gated, per-token
auxiliary self-distillation loss** stacked on top of GRPO.

```
L(theta) = L_GRPO(theta) + lambda_SDAR * L_SDAR(theta)
ell_t^SDAR = g_t * ( log pi_T(y_t | s_t^+) - log pi_theta(y_t | s_t) )
           = g_t * Delta_t
```

The gate `g_t = sigma(beta * Delta_t)` (or `sigma(beta * h_t)`) is a stop-grad
detached scalar that decides how much weight to put on each token's distillation
signal. The two scripts here exhibit two complementary failure-mode tests:

| Script              | What it shows                                                             | Runtime |
|---------------------|---------------------------------------------------------------------------|---------|
| `toy_sdar.py`       | Tabular 4-turn MDP. Compares 4 variants. Gated SDAR matches GRPO reward AND keeps per-turn KL strictly below ungated GRPO+OPSD. | <30s |
| `tiny_sdar_lm.py`   | Synthetic "find char X" task on a tiny char-level LM. Reward improves from chance (~0.34) to ~0.75 in 50 GRPO steps; KL bounded; gate fires diagnostically. | <30s |

## Run

```bash
python3 -m pip install -r requirements.txt   # numpy + matplotlib
python3 toy_sdar.py
python3 tiny_sdar_lm.py
```

`toy_sdar.py` writes `toy_sdar_curves.png` if matplotlib is available;
otherwise it prints the metric table.

## Equation -> implementation map (deep dive sections 4-7)

The deep dive lives at `../02-math-deep-dive.md`. The table below ties each
math object to where it lives in code.

| Symbol             | Meaning                                | `toy_sdar.py`                                | `tiny_sdar_lm.py`                       |
|--------------------|----------------------------------------|----------------------------------------------|-----------------------------------------|
| `pi_theta`         | Student policy                         | `softmax(logits[t])`                         | `TinyLM.forward_logprobs`               |
| `pi_T`             | Teacher branch (privileged context)    | `teacher_logits_from()` (line ~78)           | `teacher_logprob_for_token()` (line ~220) |
| `Delta_t`          | Teacher-student log-prob gap (Sec 4)   | `delta = lt - ls` (in `opsd_aux_grad`)       | `delta = lp_T - lp_S` (in `grpo_sdar_step`) |
| `h_t`              | Student token entropy (Sec 5)          | `mean_entropy()` and inline in entropy gate  | not used; gap-gate only                 |
| `g_t` (gap)        | Gap gate `sigma(beta * Delta)`         | `g = sigmoid(GATE_BETA * delta)` (line ~155) | `gate = 1/(1+exp(-GATE_BETA * delta))`  |
| `g_t` (entropy)    | Entropy gate `sigma(beta * h)`         | `g = sigmoid(GATE_BETA * h_centered)`        | (not exercised here)                    |
| `L_SDAR` (Sec 6)   | Auxiliary loss aggregated over tokens  | `opsd_aux_grad()`                            | the `gE_total += LAMBDA_SDAR * gate * gE` block |
| Combined objective | `L_GRPO + lambda * L_SDAR`             | `g_total = gp + LAMBDA_SDAR * ga`            | summation inside `grpo_sdar_step`       |
| GRPO group adv     | `(R - mu) / sigma` (Sec 2)             | `grpo_grad()` lines `mu = rewards.mean()...` | `grpo_sdar_step()` `mu = rewards.mean()...` |

## What each script demonstrates

### `toy_sdar.py` (math-clean)

A 4-turn MDP, 5 actions per turn, one secret correct action per turn. The
**student** is per-(turn, action) softmax logits. The **teacher** is the
same logits with a fixed bonus `+B = 1.5` on the correct action (modelling
the privileged retrieved-skill context). All four variants train on the
same MDP for 200 steps with group size G=8 and learning rate 0.05:

| variant                  | what                                       |
|--------------------------|--------------------------------------------|
| `grpo`                   | vanilla GRPO baseline                      |
| `grpo_opsd_ungated`      | adds ungated OPSD (`g_t == 1`)             |
| `sdar` (gap gate)        | adds `sigma(beta * Delta_t)`-weighted aux  |
| `sdar_entropy`           | adds `sigma(beta * h_t)`-weighted aux      |

The script asserts at the end:
- `final_reward(SDAR) >= final_reward(GRPO) - 0.05`
- `final_kl(SDAR) <= final_kl(GRPO+OPSD)`

Both pass on seed 0 with the default config.

Sample run on the WSL2 box:

```
GRPO                3.234         0.7504
GRPO+OPSD ungated   3.922         2.2615
SDAR (gap-gate)     3.906         2.1888
SDAR (entropy-gt)   3.828         1.7791
```

Both gated variants close the reward gap to ungated OPSD while strictly
reducing the per-turn KL drift. Entropy gating gives the cleanest KL bound
because the gate magnitude is tied to student confidence, not Delta.

### `tiny_sdar_lm.py` (LM-grade)

A pure-numpy character-level LM (vocab=30, hidden=16) trained with GRPO +
SDAR on a synthetic agentic task. The student receives `find: X`, the
teacher receives `find: X X X X X X X X X` (the "retrieved skill"
hint repeats the target). Reward = 1 if the 10-token completion contains
the target character `X` (out of `{a, e, i, o}`).

Architecture is intentionally tiny:

- `TinyLM` = token embedding `E` + LM head `(W, b)`. The full transformer
  block is REPLACED by a mean-pool over the last `CTX_K` context tokens.
  This is the same parameter set you'd update if you froze a real
  transformer's attention/MLP and only fine-tuned embed + head — the
  trick the chain Ch.31 GRPO cell uses.
- 50 GRPO outer steps, group size G=4, `lambda_SDAR=0.5`, `beta=2.0`.

Sample run:
```
initial eval reward (32 tasks)  : 0.344
final   eval reward (64 tasks)  : 0.750     # +118%
per-token KL (current vs init)   : 1.378     # bounded
mean gate-fire rate              : 0.500     # gate sits in linear regime
```

## Hardware-tier note

These scripts are CPU-only and run in <30s combined.  `metadata.json`
classifies the box as `tier_mid_gpu` (8 GB CUDA), so the sandbox is
deliberately **smaller** than the budget allows — the goal is fast,
auditable mathematical demonstration of the SDAR mechanism, not a full
training run.

For higher tiers (`tier_high_gpu` / `tier_a100`) you can scale up
`tiny_sdar_lm.py` along these axes:

- swap `TinyLM` for a real transformer (Ch.23 of the chain repo); freeze
  attention/MLP, only fine-tune embed + head (the `gE/gW/gb` interface is
  unchanged).
- raise `D` from 16 -> 384 and add `n_layer=6`.
- expand `TARGET_ALPHA` to all 26 letters and bump `N_STEPS` to 500-1000.
- add a learned KL penalty `beta * KL(pi_theta || pi_init)` to GRPO and
  compare against the implicit KL bound the SDAR gate provides.

## Files

```
sandbox/
  README.md            - this file
  requirements.txt     - numpy>=1.24, matplotlib>=3.5
  toy_sdar.py          - 4-turn tabular MDP, 4-variant comparison
  tiny_sdar_lm.py      - mini-LM SDAR loop on "find: X"
  toy_sdar_curves.png  - generated by toy_sdar.py if matplotlib present
```
