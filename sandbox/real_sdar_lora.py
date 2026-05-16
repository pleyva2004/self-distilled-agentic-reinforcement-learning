#!/usr/bin/env python3
"""real_sdar_lora.py — Level-2 SDAR via LoRA on Qwen 2.5 1.5B Instruct.

This is the **Level 2** real-LM companion to the Level-1 numpy demo
(`tiny_sdar_lm.py`).  It wires the gap-gated SDAR objective into a Hugging
Face transformer (Qwen 2.5 1.5B Instruct) using LoRA so training fits in
~16-32 GB unified RAM on Apple Silicon (M4 Pro 48 GB recommended).

Math reference: see `../02-math-deep-dive.md`, Sections 4-6:

    L(theta)         = L_GRPO(theta) + lambda_SDAR * L_SDAR(theta)
    Delta_t           = log pi_T(y_t | s_t^+) - log pi_theta(y_t | s_t)
    g_t               = sigmoid(beta * Delta_t)        (gap gate)
    l_t^SDAR          = g_t * Delta_t

The TEACHER branch is the SAME LoRA-wrapped model queried with a privileged
prompt that exposes the gold paragraph index.  Teacher logprobs at the
student-sampled token positions feed Delta_t.

Task: a 5-paragraph context + a research question; the student must (a) pick
the right paragraph index and (b) quote the key sentence.  Reward is the sum
of two components (in [0, 2]):
  - +1 if the model's selected paragraph index matches the gold index
  - +(token-set-F1 of extracted sentence vs gold sentence) if > 0.3

CLI:
    python3 real_sdar_lora.py --steps 1     # smoke (~5 min on M4 Pro MPS)
    python3 real_sdar_lora.py --steps 5     # short loop
    python3 real_sdar_lora.py --train       # ~100 steps, ~3-4 hr on M4 Pro
    python3 real_sdar_lora.py --eval        # load saved adapter, 10 eval eps

This file is *importable* on any platform: if torch / transformers / peft
aren't installed it prints a clear install message and exits 0.

Hardware: training requires Apple Silicon MPS (M-series Mac) or a CUDA GPU
with >=12 GB VRAM.  48 GB unified RAM recommended for safety margin.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 1. Soft imports + device detection
# ---------------------------------------------------------------------------
_MISSING: list[str] = []
try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _MISSING.append("torch>=2.4")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _MISSING.append("transformers>=4.45")

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    _MISSING.append("peft>=0.12")


def _print_install_message_and_exit() -> None:
    msg = (
        "real_sdar_lora.py: required packages not installed: "
        + ", ".join(_MISSING)
        + "\n\n"
        "Install (Apple Silicon recommended):\n"
        "    pip install -r requirements.txt\n"
        "or minimally:\n"
        "    pip install torch>=2.4 transformers>=4.45 peft>=0.12 accelerate>=1.0\n"
        "\n"
        "Without these the algorithm is identical to tiny_sdar_lm.py "
        "(Level 1, numpy CPU).\n"
    )
    print(msg)
    sys.exit(0)


def detect_device() -> str:
    """Return 'mps' on Apple Silicon, 'cuda' on NVIDIA, else 'cpu'."""
    if torch is None:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# 2. Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# LoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# SDAR + GRPO hyperparameters
G_ROLLOUTS = 4
LAMBDA_SDAR = 0.3
GATE_BETA = 2.0
PPO_EPS = 0.2
LR = 1e-4
MAX_NEW_TOKENS = 80
LOG_EVERY = 5

ADAPTER_DIR = Path(__file__).parent / "sdar_lora_adapter"


# ---------------------------------------------------------------------------
# 3. Episode dataclass + inline synthetic corpus
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    question: str
    paragraphs: list[str]
    gold_paragraph_idx: int
    gold_sentence: str


# Hardcoded corpus (5 episodes) — no network, no datasets dep.
# Each episode is a small "research" Q + 3 candidate paragraphs.
RAW_EPISODES: list[dict[str, Any]] = [
    {
        "question": "Which paragraph explains why SDAR uses a sigmoid gate?",
        "paragraphs": [
            "GRPO is a policy-gradient algorithm that uses group-relative "
            "advantages computed across G rollouts of the same prompt.",
            "SDAR multiplies the per-token distillation loss by a sigmoid "
            "gate g_t = sigma(beta * Delta_t) to handle asymmetric trust "
            "in the privileged teacher branch and bound per-turn KL.",
            "Qwen 2.5 1.5B Instruct is a decoder-only transformer "
            "pre-trained on 18T tokens with grouped-query attention.",
        ],
        "gold_paragraph_idx": 1,
        "gold_sentence": (
            "SDAR multiplies the per-token distillation loss by a sigmoid "
            "gate g_t = sigma(beta * Delta_t) to handle asymmetric trust "
            "in the privileged teacher branch and bound per-turn KL."
        ),
    },
    {
        "question": "Which paragraph defines the teacher-student log-prob gap?",
        "paragraphs": [
            "The teacher-student gap Delta_t = log pi_T(y_t | s_t^+) - "
            "log pi_theta(y_t | s_t) measures how much extra signal the "
            "privileged-context teacher branch carries at token t.",
            "Reward in agentic RL is often sparse and trajectory-level, "
            "providing only coarse supervision for long-horizon tasks.",
            "On-Policy Self-Distillation (OPSD) augments RL with dense "
            "token-level guidance from a teacher branch.",
        ],
        "gold_paragraph_idx": 0,
        "gold_sentence": (
            "The teacher-student gap Delta_t = log pi_T(y_t | s_t^+) - "
            "log pi_theta(y_t | s_t) measures how much extra signal the "
            "privileged-context teacher branch carries at token t."
        ),
    },
    {
        "question": "Which paragraph states the SDAR combined objective?",
        "paragraphs": [
            "The reward model is trained on pairwise human preferences "
            "and outputs a scalar score per completion.",
            "LoRA freezes the base weights and inserts low-rank update "
            "matrices into attention projections, reducing trainable "
            "parameters by ~100x.",
            "The combined SDAR objective is L(theta) = L_GRPO(theta) + "
            "lambda_SDAR * L_SDAR(theta), where L_SDAR is the gated "
            "single-sample distillation auxiliary.",
        ],
        "gold_paragraph_idx": 2,
        "gold_sentence": (
            "The combined SDAR objective is L(theta) = L_GRPO(theta) + "
            "lambda_SDAR * L_SDAR(theta), where L_SDAR is the gated "
            "single-sample distillation auxiliary."
        ),
    },
    {
        "question": "Which paragraph shows ungated OPSD's failure mode?",
        "paragraphs": [
            "Multi-turn agentic training amplifies student drift; "
            "without gating, the OPSD KL term grows unboundedly and "
            "the policy collapses from chance-level reward to ~0.",
            "GRPO normalizes advantages within a group, removing the "
            "need for a learned value function or generalized advantage "
            "estimation.",
            "Qwen 2.5 supports a 128K context window via rope scaling, "
            "which is useful for long-horizon agentic rollouts.",
        ],
        "gold_paragraph_idx": 0,
        "gold_sentence": (
            "Multi-turn agentic training amplifies student drift; "
            "without gating, the OPSD KL term grows unboundedly and "
            "the policy collapses from chance-level reward to ~0."
        ),
    },
    {
        "question": "Which paragraph describes the PPO clipping mechanism?",
        "paragraphs": [
            "The vocabulary of Qwen 2.5 is 151,936 tokens including "
            "special control tokens for chat formatting.",
            "PPO uses a clipped surrogate min(rho * A, clip(rho, "
            "1 - eps, 1 + eps) * A) to bound the per-step policy "
            "change and prevent destructive updates.",
            "Adam optimizer maintains first and second moments of the "
            "gradient and is the default for transformer training.",
        ],
        "gold_paragraph_idx": 1,
        "gold_sentence": (
            "PPO uses a clipped surrogate min(rho * A, clip(rho, "
            "1 - eps, 1 + eps) * A) to bound the per-step policy "
            "change and prevent destructive updates."
        ),
    },
    {
        "question": "Which paragraph explains group-relative advantage?",
        "paragraphs": [
            "Apple Silicon MPS supports bfloat16 with stable kernels in "
            "PyTorch 2.4+; fp16 is less stable on MPS.",
            "Group-relative advantage A_g = (r_g - mu_r) / sigma_r "
            "computes each rollout's advantage relative to the mean and "
            "std of rewards across the group of G rollouts.",
            "The Hugging Face Transformers library auto-detects "
            "available devices and casts model weights accordingly.",
        ],
        "gold_paragraph_idx": 1,
        "gold_sentence": (
            "Group-relative advantage A_g = (r_g - mu_r) / sigma_r "
            "computes each rollout's advantage relative to the mean and "
            "std of rewards across the group of G rollouts."
        ),
    },
    {
        "question": "Which paragraph defines OPSD?",
        "paragraphs": [
            "OPSD is on-policy self-distillation: at every training "
            "step the student is distilled against a teacher branch of "
            "the same model conditioned on extra privileged context.",
            "DPO replaces the reward-model + PPO loop with a direct "
            "preference-optimization objective on preference pairs.",
            "Search-augmented agents use a retrieval step before "
            "generation; the retrieved chunks become the privileged "
            "context for the teacher branch.",
        ],
        "gold_paragraph_idx": 0,
        "gold_sentence": (
            "OPSD is on-policy self-distillation: at every training "
            "step the student is distilled against a teacher branch of "
            "the same model conditioned on extra privileged context."
        ),
    },
]


def build_episodes() -> list[Episode]:
    return [Episode(**raw) for raw in RAW_EPISODES]


# ---------------------------------------------------------------------------
# 4. Prompt templates
# ---------------------------------------------------------------------------
def _format_paragraphs(paragraphs: list[str]) -> str:
    return "\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs))


def student_prompt(ep: Episode) -> str:
    return (
        "You are a research assistant. Given a question and 3 candidate paragraphs, "
        "select the most relevant paragraph and quote its key sentence.\n\n"
        f"Question: {ep.question}\n\n"
        f"Paragraphs:\n{_format_paragraphs(ep.paragraphs)}\n\n"
        "Respond exactly in this format:\n"
        "Index: <number 0-2>\n"
        "Quote: <one sentence verbatim>\n\n"
        "Answer:\n"
    )


def teacher_prompt(ep: Episode) -> str:
    """Same prompt, but with a privileged hint exposing the gold index."""
    return (
        "You are a research assistant. Given a question and 3 candidate paragraphs, "
        "select the most relevant paragraph and quote its key sentence.\n\n"
        f"Hint: the correct paragraph is index {ep.gold_paragraph_idx}.\n\n"
        f"Question: {ep.question}\n\n"
        f"Paragraphs:\n{_format_paragraphs(ep.paragraphs)}\n\n"
        "Respond exactly in this format:\n"
        "Index: <number 0-2>\n"
        "Quote: <one sentence verbatim>\n\n"
        "Answer:\n"
    )


def parse_completion(text: str) -> dict:
    """Extract (paper_idx, extracted_quote) from a model response."""
    idx = -1
    quote = ""
    m = re.search(r"Index\s*:\s*([0-9]+)", text)
    if m:
        try:
            idx = int(m.group(1))
        except ValueError:
            pass
    if idx < 0:
        # Fallback: first standalone digit in 0-2.
        digits = re.findall(r"\b([0-2])\b", text)
        idx = int(digits[0]) if digits else 0
    m2 = re.search(r"Quote\s*:\s*(.+)", text, flags=re.DOTALL)
    if m2:
        quote = m2.group(1).strip().splitlines()[0].strip()
    else:
        # Fallback: longest sentence in the completion.
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        quote = max(sents, key=len, default="").strip() if sents else ""
    return {"paragraph_idx": idx, "quote": quote}


# ---------------------------------------------------------------------------
# 5. Reward
# ---------------------------------------------------------------------------
def _token_set_f1(a: str, b: str) -> float:
    """F1 over token sets — symmetric, in [0, 1]."""
    if not a or not b:
        return 0.0
    sa = set(re.findall(r"[a-zA-Z0-9_]+", a.lower()))
    sb = set(re.findall(r"[a-zA-Z0-9_]+", b.lower()))
    if not sa or not sb:
        return 0.0
    inter = sa & sb
    if not inter:
        return 0.0
    precision = len(inter) / len(sa)
    recall = len(inter) / len(sb)
    return 2 * precision * recall / (precision + recall)


def compute_reward(parsed: dict, ep: Episode) -> float:
    """+1 if paragraph index matches gold; +F1(extracted, gold_sentence) if >0.3."""
    r = 0.0
    if parsed["paragraph_idx"] == ep.gold_paragraph_idx:
        r += 1.0
    f1 = _token_set_f1(parsed["quote"], ep.gold_sentence)
    if f1 > 0.3:
        r += f1
    return r


# ---------------------------------------------------------------------------
# 6. Rollout: sample from student, cache old log-probs
# ---------------------------------------------------------------------------
@dataclass
class Rollout:
    episode: Episode
    full_ids: Any
    prompt_len: int
    gen_ids: Any
    old_logprobs: Any
    decoded: str
    parsed: dict = field(default_factory=dict)
    reward: float = 0.0


def generate_student_rollout(
    model, tokenizer, ep: Episode, device: str, max_new: int = MAX_NEW_TOKENS
) -> Rollout:
    """Sample a rollout from the student (no privileged hint)."""
    prompt = student_prompt(ep)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    full_ids = out.sequences  # [1, prompt_len + T_gen]
    gen_ids = full_ids[:, prompt_len:]
    # Re-run forward to cache old log-probs (under behaviour policy).
    with torch.no_grad():
        logits = model(full_ids).logits
    gen_len = gen_ids.shape[1]
    shift_logits = logits[:, prompt_len - 1: prompt_len - 1 + gen_len, :]
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    old_logprobs = log_probs_all.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    decoded = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    parsed = parse_completion(decoded)
    reward = compute_reward(parsed, ep)
    return Rollout(
        episode=ep,
        full_ids=full_ids.detach(),
        prompt_len=prompt_len,
        gen_ids=gen_ids.detach(),
        old_logprobs=old_logprobs.squeeze(0).detach(),
        decoded=decoded,
        parsed=parsed,
        reward=reward,
    )


def teacher_logprobs_at_student_tokens(
    model, tokenizer, rollout: Rollout, device: str
):
    """Compute log pi_T(y_t | s_t^+) where s_t^+ = teacher prompt + student's gen prefix.

    Same model parameters; only the prompt differs (privileged hint exposes gold idx).
    """
    ep = rollout.episode
    t_prompt = teacher_prompt(ep)
    t_prompt_ids = tokenizer(t_prompt, return_tensors="pt").to(device).input_ids
    t_prompt_len = t_prompt_ids.shape[1]
    # Concatenate teacher prompt + the same student-generated tokens.
    full = torch.cat([t_prompt_ids, rollout.gen_ids.to(device)], dim=1)
    with torch.no_grad():
        logits = model(full).logits
    gen_len = rollout.gen_ids.shape[1]
    shift = logits[:, t_prompt_len - 1: t_prompt_len - 1 + gen_len, :]
    log_probs_all = F.log_softmax(shift, dim=-1)
    teacher_lp = log_probs_all.gather(
        -1, rollout.gen_ids.to(device).unsqueeze(-1)
    ).squeeze(-1).squeeze(0)
    return teacher_lp.detach()


# ---------------------------------------------------------------------------
# 7. Combined SDAR + PPO-clipped GRPO loss
# ---------------------------------------------------------------------------
def sdar_grpo_loss(
    model,
    rollouts: list[Rollout],
    advantages: list[float],
    teacher_lps: list[Any],
) -> tuple[Any, dict]:
    """L = L_GRPO + LAMBDA_SDAR * L_SDAR over a group of rollouts."""
    grpo_terms = []
    sdar_terms = []
    gate_fires: list[float] = []
    delta_means: list[float] = []
    kl_estimates: list[float] = []

    for r, adv, teacher_lp in zip(rollouts, advantages, teacher_lps):
        logits = model(r.full_ids).logits
        gen_len = r.gen_ids.shape[1]
        shift_logits = logits[:, r.prompt_len - 1: r.prompt_len - 1 + gen_len, :]
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        new_lp = log_probs_all.gather(-1, r.gen_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)

        # --- PPO-clipped GRPO ---
        old_lp = r.old_logprobs.to(new_lp.device)
        ratio = torch.exp(new_lp - old_lp)
        adv_t = torch.tensor(adv, dtype=ratio.dtype, device=ratio.device)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - PPO_EPS, 1.0 + PPO_EPS) * adv_t
        grpo_term = -torch.minimum(unclipped, clipped).mean()
        grpo_terms.append(grpo_term)

        # --- Gap-gated SDAR ---
        delta = teacher_lp.to(new_lp.device) - new_lp.detach()
        gate = torch.sigmoid(GATE_BETA * delta).detach()
        # Loss = -mean(g_t * new_lp).  Gradient flows ONLY through student
        # log-prob (gate + teacher_lp detached) — see Sec 6 subtlety.
        sdar_term = -(gate * new_lp).mean()
        sdar_terms.append(sdar_term)

        with torch.no_grad():
            gate_fires.append(float(gate.mean().item()))
            delta_means.append(float(delta.mean().item()))
            kl_estimates.append(float((new_lp - old_lp).mean().item()))

    grpo_loss = torch.stack(grpo_terms).mean()
    sdar_loss = torch.stack(sdar_terms).mean()
    loss = grpo_loss + LAMBDA_SDAR * sdar_loss
    diag = {
        "grpo_loss": float(grpo_loss.item()),
        "sdar_loss": float(sdar_loss.item()),
        "mean_gate": float(sum(gate_fires) / max(1, len(gate_fires))),
        "mean_delta": float(sum(delta_means) / max(1, len(delta_means))),
        "kl_to_behaviour": float(sum(kl_estimates) / max(1, len(kl_estimates))),
    }
    return loss, diag


# ---------------------------------------------------------------------------
# 8. Model loading + LoRA setup
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(device: str):
    print(f"[load] device = {device}")
    print(f"[load] base model = {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # bf16 on MPS/CUDA — more stable than fp16 in PyTorch 2.4+.
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[lora] trainable params: {n_trainable:,} / {n_total:,} "
        f"({100 * n_trainable / n_total:.3f}%)"
    )
    return model, tokenizer


def _lora_grad_norm(model) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(sq)


# ---------------------------------------------------------------------------
# 9. Training loop
# ---------------------------------------------------------------------------
def train(steps: int) -> None:
    if _MISSING:
        _print_install_message_and_exit()

    device = detect_device()
    torch.manual_seed(0)
    random.seed(0)

    model, tokenizer = load_model_and_tokenizer(device)
    episodes = build_episodes()
    print(f"[data] {len(episodes)} inline episodes")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )

    t0 = time.time()
    reward_hist: list[float] = []
    for step in range(steps):
        ep = random.choice(episodes)
        rollouts: list[Rollout] = []
        teacher_lps: list[Any] = []
        for _ in range(G_ROLLOUTS):
            r = generate_student_rollout(model, tokenizer, ep, device)
            rollouts.append(r)
            t_lp = teacher_logprobs_at_student_tokens(model, tokenizer, r, device)
            teacher_lps.append(t_lp)

        # Group-relative advantage.
        rewards = [r.reward for r in rollouts]
        mu = sum(rewards) / len(rewards)
        var = sum((R - mu) ** 2 for R in rewards) / len(rewards)
        sigma = math.sqrt(var) + 1e-6
        advantages = [(R - mu) / sigma for R in rewards]

        model.train()
        loss, diag = sdar_grpo_loss(model, rollouts, advantages, teacher_lps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = _lora_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        reward_hist.append(mu)
        if step % LOG_EVERY == 0 or step == steps - 1:
            print(
                f"  step {step:4d}  reward {mu:.3f}  "
                f"gate {diag['mean_gate']:.3f}  "
                f"delta {diag['mean_delta']:+.3f}  "
                f"kl {diag['kl_to_behaviour']:+.4f}  "
                f"grad_norm {gnorm:.3f}  "
                f"loss {float(loss.item()):+.4f}"
            )

    elapsed = time.time() - t0
    print(f"[train] {steps} steps in {elapsed:.1f}s ({steps / max(elapsed, 1e-6):.2f} step/s)")
    if reward_hist:
        print(f"[train] reward init={reward_hist[0]:.3f}  final={reward_hist[-1]:.3f}")

    # Save adapter.
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_DIR))
    # Drop a small reward-history file alongside.
    (ADAPTER_DIR / "reward_hist.json").write_text(
        json.dumps({"rewards": reward_hist, "steps": steps}, indent=2)
    )
    print(f"[ckpt] saved adapter to {ADAPTER_DIR}")


# ---------------------------------------------------------------------------
# 10. Eval
# ---------------------------------------------------------------------------
def evaluate(n_episodes: int = 10) -> None:
    if _MISSING:
        _print_install_message_and_exit()
    device = detect_device()
    print(f"[eval] device = {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True
    )
    base.to(device)
    if (ADAPTER_DIR / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        model.to(device)
        print(f"[eval] loaded adapter from {ADAPTER_DIR}")
    else:
        model = base
        print(f"[eval] no adapter at {ADAPTER_DIR}; evaluating base model")

    eps_pool = build_episodes()
    correct = 0
    f1_sum = 0.0
    rewards = []
    torch.set_grad_enabled(False)
    for i in range(n_episodes):
        ep = eps_pool[i % len(eps_pool)]
        r = generate_student_rollout(model, tokenizer, ep, device)
        if r.parsed["paragraph_idx"] == ep.gold_paragraph_idx:
            correct += 1
        f1 = _token_set_f1(r.parsed["quote"], ep.gold_sentence)
        f1_sum += f1
        rewards.append(r.reward)
        print(
            f"  ep {i}  gold={ep.gold_paragraph_idx} pred={r.parsed['paragraph_idx']}  "
            f"f1={f1:.2f}  reward={r.reward:.2f}"
        )
    print(
        f"[eval] paragraph-pick {correct}/{n_episodes}  "
        f"mean_f1={f1_sum / n_episodes:.3f}  "
        f"mean_reward={sum(rewards) / n_episodes:.3f}"
    )


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=None,
        help="smoke-test step count (e.g. 1)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="full training run",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="load saved adapter and evaluate",
    )
    parser.add_argument(
        "--full-steps", type=int, default=100,
        help="step count for --train (default 100)",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="number of eval episodes (default 10)",
    )
    args = parser.parse_args()

    if _MISSING:
        _print_install_message_and_exit()

    if args.eval:
        evaluate(n_episodes=args.eval_episodes)
    elif args.train:
        train(steps=args.full_steps)
    elif args.steps is not None:
        train(steps=args.steps)
    else:
        parser.print_help()
        print("\nNo action specified. Try --steps 1 (smoke) or --train (full).")


if __name__ == "__main__":
    main()
