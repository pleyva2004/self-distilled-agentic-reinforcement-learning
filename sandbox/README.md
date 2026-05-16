# Sandbox

This sandbox demonstrates SDAR at two levels.

## Level 1: CPU baseline (runs anywhere)

Pure numpy, no GPU required, fast (~3s total). Verifies the *structural* claims of the paper: gating bounds KL, SDAR matches GRPO on reward, ungated OPSD collapses.

| Script | Demonstrates | Runtime (CPU) |
|--------|--------------|---------------|
| `toy_sdar.py` | SDAR vs GRPO vs ungated-OPSD on 4-turn tabular MDP | ~1s |
| `tiny_sdar_lm.py` | SDAR on numpy tiny GPT, "find substring X" task | ~5s |

Install + run:
```bash
pip install numpy matplotlib
python3 toy_sdar.py
python3 tiny_sdar_lm.py
```

See [`findings.md`](../findings.md) for measured numbers from Level 1.

## Level 2: Hardware-upsized (tier_mid_gpu)

Sized to the detected hardware tier (`tier_mid_gpu` per `metadata.json`). Demonstrates SDAR at a scale where qualitative results match what would happen with a frontier model.

| Script | Demonstrates | Runtime (estimate) | Needs |
|--------|--------------|--------------------|-----|
| `torch_sdar.py` | ~30M-param torch GPT trained with SDAR, "find substring" task scaled up | ~30-60 min on M4 Pro MPS | torch + tiktoken |
| `real_sdar_lora.py` | LoRA fine-tune of Qwen 2.5 1.5B Instruct with SDAR on synthetic agentic task | ~3-4 hr on M4 Pro | torch + transformers + peft + accelerate; 48 GB unified RAM recommended |

Install + run (Mac with M-series silicon):
```bash
pip install torch tiktoken transformers peft accelerate
python3 torch_sdar.py --steps 5     # smoke test (~2 min on CPU; faster on MPS)
python3 torch_sdar.py --train       # full ~200-step training (~30-60 min on MPS)
python3 real_sdar_lora.py --steps 1 # smoke test (~5 min on Mac M4 Pro)
python3 real_sdar_lora.py --train   # full LoRA fine-tune (~3-4 hr on M4 Pro)
```

On Linux without GPU + heavy deps, both Level 2 scripts gracefully print install messages and exit 0.

See [`findings.md`](../findings.md) "## Level 2 results (pending Mac run)" for the placeholder where Level 2 numbers will be filled in once the user runs the scripts on their M4 Pro.

## What each level verifies

| Level | Verifies | Failure mode |
|-------|----------|--------------|
| 1 | Algorithm's *structural* properties — reward shape, KL bound, sensitivity curves. Reproduces deterministically with `np.random.seed(0)`. | Algorithm-level bug in the gating mechanism would show up here |
| 2 | Algorithm *transfers* to a setting closer to the paper's experimental regime (~30M GPT, or LoRA on 1.5B Qwen). Numbers differ from Level 1; the question is whether the qualitative shape holds at scale. | Scale-dependent failure modes only visible above ~1M params |

## Hardware tier note

This study's `metadata.json` lists `hardware_tier = "tier_mid_gpu"` (NVIDIA 8-23 GB OR Apple Silicon 16-32 GB unified) from the v4.0 study-paper hardware-detection feature. Re-run with `python3 ~/.claude/skills/study-paper/templates/detect-hardware.py --force --summary` to refresh.

User's primary device is a MacBook Pro M4 Pro (48 GB unified). On that machine, `tier_high_gpu` would apply — the Level 2 scripts are sized to fit `tier_mid_gpu` so they also run comfortably on M4 Pro.

## Equation -> implementation map (Level 1 scripts)

The deep dive lives at `../02-math-deep-dive.md`. The table below ties each math object to where it lives in the Level-1 numpy code.

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

The Level-2 scripts (`torch_sdar.py`, `real_sdar_lora.py`) share the same symbol-to-code structure — see their `sdar_grpo_loss()` function for the gate + Delta + PPO-clipped GRPO combination.

## Files

```
sandbox/
  README.md            - this file
  requirements.txt     - Level 1 + Level 2 deps
  toy_sdar.py          - LEVEL 1: 4-turn tabular MDP, 4-variant comparison
  tiny_sdar_lm.py      - LEVEL 1: mini-LM SDAR loop on "find: X"
  toy_sdar_curves.png  - generated by toy_sdar.py if matplotlib present
  torch_sdar.py        - LEVEL 2: ~30M-param torch GPT + SDAR on MPS/CUDA/CPU
  real_sdar_lora.py    - LEVEL 2: LoRA fine-tune of Qwen 2.5 1.5B Instruct
  torch_sdar.pt        - generated by torch_sdar.py after --train
  sdar_lora_adapter/   - generated by real_sdar_lora.py after --train
```
