#!/usr/bin/env python3
"""torch_sdar.py — Level-2 SDAR on a ~30M-param torch GPT (MPS/CUDA/CPU).

This is the **Level 2** hardware-upsized companion to the Level-1 numpy demos
(`toy_sdar.py`, `tiny_sdar_lm.py`).  It scales the same SDAR algorithm to a
real ~30M-param decoder-only GPT (the chain-repo Ch27 backbone) trained with
GRPO + gap-gated SDAR auxiliary on a "find: X" task.

Math reference: see `../02-math-deep-dive.md`, especially Section 6:

    L(theta)        = L_GRPO(theta) + lambda_SDAR * L_SDAR(theta)
    L_SDAR per-token = g_t * Delta_t
    Delta_t          = log pi_T(y_t | s_t^+) - log pi_theta(y_t | s_t)
    g_t              = sigmoid(beta * Delta_t)            (gap gate)

The TEACHER branch is the **same model** queried with a privileged-context
prompt (it is given the position of the target character).  Re-running the
model with no_grad on the teacher prompt yields pi_T at the same token IDs
the student sampled.  Subtracting student log-probs gives Delta_t.

CLI:
    python3 torch_sdar.py --steps 5     # smoke test (~2 min on CPU)
    python3 torch_sdar.py --train       # ~200-step training (~30-60 min on MPS)
    python3 torch_sdar.py --sample      # generate from saved checkpoint

This file is *importable* on any platform: if torch / tiktoken aren't
installed it prints a clear install message and exits 0.  Full training
benefits from Apple Silicon MPS or NVIDIA CUDA.

Runtime targets:
  - --steps 1   :  <2  min on CPU  (smoke; smoke verification)
  - --steps 5   :  ~2  min on CPU  (smoke; smoke verification)
  - --train     :  ~30-60 min on M4 Pro MPS (200 steps, batch 4)
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Soft imports + device detection
# ---------------------------------------------------------------------------
_MISSING: list[str] = []
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - exercised only on hosts without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _MISSING.append("torch>=2.4")

try:
    import tiktoken
    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False
    # tiktoken is optional — we fall back to a small char tokenizer.


def _print_install_message_and_exit() -> None:
    msg = (
        "torch_sdar.py: required packages not installed: "
        + ", ".join(_MISSING)
        + "\n\n"
        "Install (Apple Silicon recommended):\n"
        "    pip install -r requirements.txt\n"
        "or minimally:\n"
        "    pip install torch>=2.4 tiktoken>=0.5\n"
        "\n"
        "Without these the algorithm is identical to tiny_sdar_lm.py (Level 1, "
        "numpy CPU).\n"
    )
    print(msg)
    sys.exit(0)


def pick_device() -> str:
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
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "torch_sdar.pt")

# Model size: ~30M params at d=384, 6 layers, 6 heads, ctx=256 (chain Ch27 ref)
@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    ctx_len: int = 256
    dropout: float = 0.0


# SDAR training hyperparameters (mirror tiny_sdar_lm.py at scale)
G_ROLLOUTS = 4              # group size for GRPO
GEN_LEN = 30                # tokens to generate per rollout
LAMBDA_SDAR = 0.3           # weight on SDAR auxiliary loss
GATE_BETA = 2.0             # sigmoid sharpness for gap gate
PPO_EPS = 0.2               # PPO ratio clip
LR = 3e-4                   # AdamW learning rate
LOG_EVERY = 5               # log frequency in --train


# ---------------------------------------------------------------------------
# 3. Tokenizer
# ---------------------------------------------------------------------------
class _CharTokenizer:
    """Tiny char-level fallback if tiktoken isn't installed.

    Covers ASCII letters, digits, and a few punctuation tokens — enough to
    encode the "find: a" / "hint: a at pos 3\nfind: a" prompts plus arbitrary
    short generations.
    """

    def __init__(self) -> None:
        chars = (
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789"
            " :,.\n"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"  # duplicate harmless; preserves V even
        )
        # Dedupe while preserving order.
        seen = []
        for c in chars:
            if c not in seen:
                seen.append(c)
        # Pad to nearest multiple of 8 for cleaner shapes.
        while len(seen) % 8 != 0:
            seen.append(chr(127 + len(seen)))
        self.itos = seen
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] if 0 <= i < len(self.itos) else "?" for i in ids)


def build_tokenizer():
    """Return (tokenizer, name).

    NOTE (v7.0.1 patch): we always use the char-level tokenizer for this task,
    even when tiktoken is installed.  With a 50k-vocab BPE the "find: X
    anywhere in the generation" task is trivial (the target character lands by
    chance in nearly every rollout, reward saturates at ~1.0 from step 0, and
    GRPO has no advantage variance to optimise against — observed empirically
    on M4 Pro MPS, 200 steps, reward 1.000 -> 0.750 with KL ~ 0).  Char-level
    (~30 vocab) plus the position-specific reward (see ``reward_for``) gives
    the student a real ~3% baseline that the teacher's privileged hint can
    actually push above.
    """
    return _CharTokenizer(), "char-fallback"


# ---------------------------------------------------------------------------
# 4. GPT model (Ch27 backbone — copy + adapt)
# ---------------------------------------------------------------------------
if torch is not None:

    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            assert cfg.d_model % cfg.n_head == 0
            self.n_head = cfg.n_head
            self.d_k = cfg.d_model // cfg.n_head
            self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
            self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
            self.drop = nn.Dropout(cfg.dropout)
            mask = torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len)).view(
                1, 1, cfg.ctx_len, cfg.ctx_len
            )
            self.register_buffer("mask", mask)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).split(C, dim=2)
            q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.drop(att)
            y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(y)

    class Block(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.d_model)
            self.attn = CausalSelfAttention(cfg)
            self.ln2 = nn.LayerNorm(cfg.d_model)
            self.ff = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff),
                nn.GELU(),
                nn.Linear(cfg.d_ff, cfg.d_model),
                nn.Dropout(cfg.dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self, cfg: GPTConfig):
            super().__init__()
            self.cfg = cfg
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight  # weight tying
            self.apply(self._init)

        @staticmethod
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        def num_params(self) -> int:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def forward(self, idx):
            B, T = idx.shape
            assert T <= self.cfg.ctx_len, f"T={T} > ctx={self.cfg.ctx_len}"
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            x = self.drop(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.ln_f(x)
            logits = self.head(x)  # [B, T, V]
            return logits


# ---------------------------------------------------------------------------
# 5. Task: synthetic "find: X" with privileged-context teacher branch
# ---------------------------------------------------------------------------
TARGET_ALPHA = "abcdefghij"   # 10-letter target alphabet (scaled up from L1)


def sample_task(rng) -> tuple[str, int]:
    char = TARGET_ALPHA[int(rng.integers(0, len(TARGET_ALPHA)))]
    pos = int(rng.integers(2, GEN_LEN - 2))
    return char, pos


def make_prompts(char: str, pos: int) -> tuple[str, str]:
    """Return (student_prompt, teacher_prompt).

    The teacher gets the privileged hint that exposes the target's *position*.
    The student knows only the target character.
    """
    student = f"place: {char} "
    teacher = f"hint: {char} at pos {pos}\nplace: {char} "
    return student, teacher


def reward_for(generated_text: str, target_char: str, pos: int) -> float:
    """Partial-credit reward, decaying with distance from the target position.

    (v7.0.1 patch.) The original reward ("+1 if target appears anywhere") let
    the 30M-param GPT pin reward at 1.0 from step 0 (every rollout had the
    target somewhere by chance), so GRPO had no advantage variance and
    couldn't learn.  A naive "+1 only if exactly at pos" overcorrects: at
    char-vocab=72 with G=4 rollouts per group, only ~5% of groups produce a
    non-zero reward, so most gradient steps are no-ops.

    Partial credit by distance gives a dense signal: every rollout where the
    target appears at all contributes something, but the magnitude shrinks
    as you move away from the hinted position.  Concretely:

        distance 0  ->  1.00
        distance 1  ->  0.67
        distance 2  ->  0.33
        distance 3+ ->  0.00  (and absent ->  0.00)

    The teacher's privileged hint ("X at pos P") tells it exactly where to
    place the character; the student starts uniform and must learn the
    placement from the gated distillation signal.
    """
    if not generated_text:
        return 0.0
    target_positions = [i for i, c in enumerate(generated_text) if c == target_char]
    if not target_positions:
        return 0.0
    dist = min(abs(i - pos) for i in target_positions)
    if dist > 2:
        return 0.0
    return max(0.0, 1.0 - dist / 3.0)


# ---------------------------------------------------------------------------
# 6. Rollout: sample from student, cache old log-probs
# ---------------------------------------------------------------------------
@dataclass
class Rollout:
    student_prompt: str
    teacher_prompt: str
    target_char: str
    full_ids: object       # tensor [1, prompt_len + GEN_LEN]
    prompt_len: int
    gen_ids: object        # tensor [1, GEN_LEN]
    old_logprobs: object   # tensor [GEN_LEN]
    decoded: str
    reward: float


def _truncate_to_ctx(ids: list[int], ctx: int) -> list[int]:
    if len(ids) <= ctx:
        return ids
    return ids[-ctx:]


def generate_rollout(model, tokenizer, char: str, pos: int, device: str) -> Rollout:
    """Sample one student rollout and cache its per-token logprobs."""
    student_prompt, teacher_prompt = make_prompts(char, pos)
    prompt_ids = tokenizer.encode(student_prompt)
    # Reserve room for GEN_LEN new tokens within ctx.
    ctx = model.cfg.ctx_len
    if len(prompt_ids) + GEN_LEN > ctx:
        prompt_ids = prompt_ids[-(ctx - GEN_LEN):]
    cur = list(prompt_ids)
    gen: list[int] = []
    model.eval()
    with torch.no_grad():
        for _ in range(GEN_LEN):
            x = torch.tensor(cur[-ctx:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)[0, -1, :]  # [V]
            # Temperature 1.0 sampling.
            probs = F.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, num_samples=1).item())
            cur.append(nxt)
            gen.append(nxt)
    full_ids = torch.tensor(cur, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = len(prompt_ids)
    gen_ids = torch.tensor(gen, dtype=torch.long, device=device).unsqueeze(0)
    # Re-run a forward pass on the full sequence to get old logprobs at the
    # generated positions.
    with torch.no_grad():
        logits = model(full_ids)  # [1, T_total, V]
    shift_logits = logits[:, prompt_len - 1: prompt_len - 1 + GEN_LEN, :]
    log_probs_all = F.log_softmax(shift_logits, dim=-1)
    old_logprobs = log_probs_all.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
    decoded = tokenizer.decode(gen)
    return Rollout(
        student_prompt=student_prompt,
        teacher_prompt=teacher_prompt,
        target_char=char,
        full_ids=full_ids.detach(),
        prompt_len=prompt_len,
        gen_ids=gen_ids.detach(),
        old_logprobs=old_logprobs.detach(),
        decoded=decoded,
        reward=reward_for(decoded, char, pos),
    )


# ---------------------------------------------------------------------------
# 7. Teacher log-probs at the student's sampled tokens
# ---------------------------------------------------------------------------
def teacher_logprobs_at_student_tokens(
    model, tokenizer, rollout: Rollout, device: str
):
    """Compute log pi_T(y_t | s_t^+) where s_t^+ is the privileged teacher prompt
    concatenated with the student's generated prefix.

    The same model parameters serve as teacher; only the prompt differs.
    """
    teacher_ids = tokenizer.encode(rollout.teacher_prompt)
    gen_list = rollout.gen_ids.squeeze(0).tolist()
    full = list(teacher_ids) + list(gen_list)
    ctx = model.cfg.ctx_len
    if len(full) > ctx:
        # Trim the prompt prefix to keep generated tokens intact.
        keep = ctx
        prompt_keep = max(1, keep - len(gen_list))
        teacher_ids = teacher_ids[-prompt_keep:]
        full = list(teacher_ids) + list(gen_list)
    teacher_prompt_len = len(teacher_ids)
    full_ids = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(full_ids)
    shift = logits[:, teacher_prompt_len - 1: teacher_prompt_len - 1 + len(gen_list), :]
    log_probs_all = F.log_softmax(shift, dim=-1)
    targets = torch.tensor(gen_list, dtype=torch.long, device=device).unsqueeze(0)
    teacher_lp = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
    return teacher_lp.detach()


# ---------------------------------------------------------------------------
# 8. Combined SDAR + PPO-clipped GRPO loss
# ---------------------------------------------------------------------------
def sdar_grpo_loss(
    model,
    rollouts: list[Rollout],
    advantages: list[float],
    teacher_lps: list[object],
) -> tuple[object, dict]:
    """Compute L = L_GRPO + LAMBDA_SDAR * L_SDAR over a group of rollouts.

    L_GRPO uses PPO-clipped ratio; L_SDAR uses gap-gated single-sample teacher.
    Returns (loss tensor, diagnostics dict).
    """
    G = len(rollouts)
    total_grpo = []
    total_sdar = []
    gate_fires: list[float] = []
    delta_means: list[float] = []
    kl_estimates: list[float] = []

    for r, adv, teacher_lp in zip(rollouts, advantages, teacher_lps):
        # Forward pass with gradients enabled.
        logits = model(r.full_ids)
        gen_len = r.gen_ids.shape[1]
        shift_logits = logits[:, r.prompt_len - 1: r.prompt_len - 1 + gen_len, :]
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        new_lp = log_probs_all.gather(-1, r.gen_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)

        # --- PPO-clipped GRPO term ---
        old_lp = r.old_logprobs.to(new_lp.device)
        ratio = torch.exp(new_lp - old_lp)
        adv_t = torch.tensor(adv, dtype=ratio.dtype, device=ratio.device)
        unclipped = ratio * adv_t
        clipped = torch.clamp(ratio, 1.0 - PPO_EPS, 1.0 + PPO_EPS) * adv_t
        grpo_term = -torch.minimum(unclipped, clipped).mean()
        total_grpo.append(grpo_term)

        # --- Gap-gated SDAR term ---
        # Delta_t = teacher_lp - student_lp_current
        delta = teacher_lp.to(new_lp.device) - new_lp.detach()
        # Detached gate (Sec 6 subtlety: gradient w.r.t. theta flows ONLY
        # through the student log-prob, not the gate or teacher branch).
        gate = torch.sigmoid(GATE_BETA * delta).detach()
        # The auxiliary loss is g_t * (teacher_lp - student_lp).  With the
        # teacher detached, gradient w.r.t. theta is -mean(g_t * d log pi_theta).
        # Equivalent loss (negative because we MAXIMIZE student log-prob):
        sdar_term = -(gate * new_lp).mean()
        total_sdar.append(sdar_term)

        with torch.no_grad():
            gate_fires.append(float(gate.mean().item()))
            delta_means.append(float(delta.mean().item()))
            # KL estimate (Sec 1.2 of the paper): mean(new_lp - old_lp) is the
            # per-token forward-KL approximation between current and behaviour.
            kl_estimates.append(float((new_lp - old_lp).mean().item()))

    grpo_loss = torch.stack(total_grpo).mean()
    sdar_loss = torch.stack(total_sdar).mean()
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
# 9. Training loop
# ---------------------------------------------------------------------------
def train(steps: int) -> None:
    if _MISSING:
        _print_install_message_and_exit()

    device = pick_device()
    print(f"[device] {device}")

    tokenizer, tok_name = build_tokenizer()
    print(f"[tok] {tok_name}  vocab={tokenizer.vocab_size}")

    cfg = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPT(cfg).to(device)
    n = model.num_params()
    print(
        f"[model] {n/1e6:.2f}M params  "
        f"layers={cfg.n_layer} d={cfg.d_model} h={cfg.n_head} ctx={cfg.ctx_len}"
    )

    torch.manual_seed(0)
    rng = _np_rng()
    opt = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01
    )

    t0 = time.time()
    reward_hist: list[float] = []
    for step in range(steps):
        char, pos = sample_task(rng)
        rollouts: list[Rollout] = []
        teacher_lps: list[object] = []
        # G rollouts at this prompt.
        for _ in range(G_ROLLOUTS):
            r = generate_rollout(model, tokenizer, char, pos, device)
            rollouts.append(r)
            t_lp = teacher_logprobs_at_student_tokens(model, tokenizer, r, device)
            teacher_lps.append(t_lp)

        # Group-relative advantage.
        rewards = [r.reward for r in rollouts]
        import numpy as _np
        mu = float(_np.mean(rewards))
        sigma = float(_np.std(rewards)) + 1e-6
        advantages = [(R - mu) / sigma for R in rewards]

        model.train()
        loss, diag = sdar_grpo_loss(model, rollouts, advantages, teacher_lps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        reward_hist.append(mu)
        if step % LOG_EVERY == 0 or step == steps - 1:
            print(
                f"  step {step:4d}  reward {mu:.3f}  "
                f"gate {diag['mean_gate']:.3f}  "
                f"delta {diag['mean_delta']:+.3f}  "
                f"kl {diag['kl_to_behaviour']:+.4f}  "
                f"loss {float(loss.item()):+.4f}  "
                f"char={char!r} pos={pos}"
            )

    elapsed = time.time() - t0
    print(f"[train] {steps} steps in {elapsed:.1f}s ({steps / max(elapsed, 1e-6):.2f} step/s)")
    if reward_hist:
        print(f"[train] reward init={reward_hist[0]:.3f}  final={reward_hist[-1]:.3f}")

    # Save checkpoint.
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg.__dict__,
            "tok_name": tok_name,
            "vocab_size": tokenizer.vocab_size,
            "reward_hist": reward_hist,
        },
        CKPT_PATH,
    )
    print(f"[ckpt] saved {CKPT_PATH}")


def _np_rng():
    import numpy as np
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 10. Sample from checkpoint
# ---------------------------------------------------------------------------
def sample_from_ckpt() -> None:
    if _MISSING:
        _print_install_message_and_exit()
    if not os.path.exists(CKPT_PATH):
        print(f"[sample] no checkpoint at {CKPT_PATH}; run --steps N or --train first")
        return
    device = pick_device()
    blob = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = GPTConfig(**blob["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    tokenizer, _ = build_tokenizer()
    rng = _np_rng()
    print("[sample] 5 examples post-training")
    for _ in range(5):
        char, pos = sample_task(rng)
        r = generate_rollout(model, tokenizer, char, pos, device)
        print(
            f"  prompt={r.student_prompt!r}  gen={r.decoded!r}  "
            f"reward={r.reward:.0f}  target={char!r}"
        )


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps", type=int, default=None,
        help="smoke-test step count (e.g. 5)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="full training run (200 steps)",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="load saved checkpoint and sample 5 generations",
    )
    parser.add_argument(
        "--full-steps", type=int, default=200,
        help="step count for --train (default 200)",
    )
    args = parser.parse_args()

    if _MISSING:
        _print_install_message_and_exit()

    if args.sample:
        sample_from_ckpt()
    elif args.train:
        train(steps=args.full_steps)
    elif args.steps is not None:
        train(steps=args.steps)
    else:
        parser.print_help()
        print("\nNo action specified. Try --steps 5 (smoke) or --train (full).")


if __name__ == "__main__":
    main()
