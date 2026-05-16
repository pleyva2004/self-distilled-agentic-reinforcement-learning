"""tiny_sdar_lm.py — SDAR on a tiny character-level LM (CPU, numpy only).

Synthetic agentic task: the model is asked to FIND a target character somewhere
in the alphabet.  It receives a prompt like ``find: a`` and must generate a
short completion that contains the target.  Reward = 1 if it does, else 0.

The TEACHER branch sees a privileged HINT: ``hint: a is at position 3\nfind: a``
which makes it much more likely to emit the target early.  This is a stand-in
for the SDAR paper's "retrieved-skill context".

We train a TINY transformer-less LM (token-embedding -> single linear LM head)
with GRPO + a SDAR auxiliary that uses GAP-GATING.  The token embedding and
LM head are the only trainable parameters, mirroring the chain Ch.31 trick of
freezing the transformer block during RL fine-tuning.

Maps onto deep-dive sections:
  - Section 1 (agentic MDP)        -> ``rollout()`` produces token trajectories
  - Section 2 (GRPO backbone)      -> ``grpo_step()`` group-relative PG
  - Section 4 (gap Delta)          -> ``compute_delta()`` log pi_T - log pi_theta
  - Section 5 (gating)             -> ``gate = sigmoid(beta * Delta)``
  - Section 6 (SDAR aux loss)      -> ``sdar_aux_grad()``  (token-weighted CE-ish)
  - Section 9 (chain Ch.31 trick)  -> only embed + lm_head trainable

Runtime target: <300s on CPU.
"""

from __future__ import annotations

import time
import numpy as np

# ---------------------------------------------------------------- config -----
SEED = 0
np.random.seed(SEED)

# vocab: 26 letters + 4 specials  (V=30 — keeps the spec)
LETTERS = list("abcdefghijklmnopqrstuvwxyz")
SPECIALS = [" ", ":", "\n", "."]
VOCAB = LETTERS + SPECIALS
V = len(VOCAB)                # 30
stoi = {c: i for i, c in enumerate(VOCAB)}
itos = {i: c for i, c in enumerate(VOCAB)}

# We restrict the *target* to a small task set so the LM can actually
# learn distinct conditional distributions in 50 steps with ~480 params.
# The vocab is still V=30; only the target alphabet is small.
TARGET_ALPHA = ["a", "e", "i", "o"]

D = 16              # hidden dim
T_GEN = 10          # generated tokens per rollout
G = 4               # GRPO group size
N_STEPS = 50        # GRPO outer steps
LR = 0.5
LAMBDA_SDAR = 0.5
GATE_BETA = 2.0
CTX_K = 4           # short recency window so the hint dominates the pool


def encode(s: str) -> np.ndarray:
    return np.array([stoi[c] for c in s], dtype=np.int64)


def decode(ids) -> str:
    return "".join(itos[int(i)] for i in ids)


# ---------------------------------------------------------------- model ------
class TinyLM:
    """Embed -> mean-pool over context -> Linear -> logits.

    No attention, no MLP.  This is the smallest causal LM that still has the
    SAME parameter set we want to study (token-embed + LM head).  The
    "context" enters via mean-pooling the embeddings of all prefix tokens.
    Crude but enough to make the hint affect next-token logits.
    """

    def __init__(self, V: int, D: int):
        self.V, self.D = V, D
        rng = np.random.default_rng(SEED)
        self.E  = rng.standard_normal((V, D)) * 0.1   # token embedding (trainable)
        self.W  = rng.standard_normal((D, V)) * 0.1   # LM head        (trainable)
        self.b  = np.zeros(V)                          # head bias      (trainable)

    def forward_logits(self, ctx_ids: np.ndarray) -> np.ndarray:
        """Return logits over V for the next token, given prefix ids.

        Mean-pool only the LAST CTX_K tokens to give a recency bias — this
        lets the teacher's hint near the prompt end actually influence the
        next-token distribution (otherwise it gets averaged out).
        """
        if len(ctx_ids) == 0:
            h = np.zeros(self.D)
        else:
            ctx_use = ctx_ids[-CTX_K:]
            h = self.E[ctx_use].mean(axis=0)
        return h @ self.W + self.b

    def forward_logprobs(self, ctx_ids):
        z = self.forward_logits(ctx_ids)
        z -= z.max()
        ez = np.exp(z); s = ez.sum()
        return z - np.log(s)        # log-softmax

    def grad_logp_token(self, ctx_ids, tok_id):
        """Return d log pi(tok_id | ctx) / d (E, W, b)."""
        if len(ctx_ids) == 0:
            h = np.zeros(self.D)
            ctx_use = np.array([], dtype=np.int64)
            inv_n = 0.0
        else:
            ctx_use = ctx_ids[-CTX_K:]
            h = self.E[ctx_use].mean(axis=0)
            inv_n = 1.0 / len(ctx_use)

        z = h @ self.W + self.b
        z = z - z.max()
        p = np.exp(z); p /= p.sum()

        onehot = np.zeros(self.V); onehot[tok_id] = 1.0
        dlogits = onehot - p

        gW = np.outer(h, dlogits)
        gb = dlogits.copy()
        dh = self.W @ dlogits
        gE = np.zeros_like(self.E)
        if inv_n > 0:
            for v in ctx_use:
                gE[v] += inv_n * dh
        return gE, gW, gb

    def add(self, gE, gW, gb, lr):
        self.E += lr * gE
        self.W += lr * gW
        self.b += lr * gb


# ---------------------------------------------------------------- task -------
def make_prompt(target_char: str, _unused_pos: int = 0):
    """Return (student_prompt, teacher_prompt) ids.

    The teacher hint comes AFTER `find: X` so it sits inside the recency
    window (CTX_K) — without that the mean-pool dominates over the hint
    and Delta collapses to zero.  The hint repeats the target character;
    via the mean-pooled embedding this biases next-token logits toward X.
    """
    student = f"find: {target_char} "
    # Teacher gets a very long suffix of the target character — with the
    # short recency window CTX_K, the teacher's mean-pool is essentially
    # the embedding of the target.  This guarantees a positive Delta.
    hint    = (target_char + " ") * 8
    teacher = f"find: {target_char}{hint}"
    return encode(student), encode(teacher)


def sample_task(rng):
    target_char = str(rng.choice(TARGET_ALPHA))
    target_pos = int(rng.integers(0, T_GEN))
    return target_char, target_pos


# ---------------------------------------------------------------- rollout ----
def rollout(model: TinyLM, prompt_ids: np.ndarray, target_char: str,
            rng: np.random.Generator):
    """Generate T_GEN tokens; return (gen_ids, reward, ctx_at_each_step)."""
    target_id = stoi[target_char]
    ctx = list(prompt_ids)
    gen = []
    contexts = []           # context BEFORE emitting each gen token
    for _ in range(T_GEN):
        contexts.append(np.array(ctx, dtype=np.int64))
        z = model.forward_logits(np.array(ctx, dtype=np.int64))
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        a = int(rng.choice(V, p=p))
        gen.append(a)
        ctx.append(a)
    reward = 1.0 if target_id in gen else 0.0
    return np.array(gen), reward, contexts


def teacher_logprob_for_token(model: TinyLM, teacher_prompt: np.ndarray,
                              gen_so_far: list, tok: int) -> float:
    """Eval teacher branch's log prob on token, given teacher prompt + same gen prefix."""
    ctx = np.concatenate([teacher_prompt, np.array(gen_so_far, dtype=np.int64)]) \
          if gen_so_far else teacher_prompt
    lp = model.forward_logprobs(ctx)
    return float(lp[tok])


def student_logprob_for_token(model: TinyLM, student_prompt, gen_so_far, tok):
    ctx = np.concatenate([student_prompt, np.array(gen_so_far, dtype=np.int64)]) \
          if gen_so_far else student_prompt
    lp = model.forward_logprobs(ctx)
    return float(lp[tok])


# ---------------------------------------------------------------- training ---
def grpo_sdar_step(model: TinyLM, rng):
    """One outer GRPO step: G rollouts, group-relative advantage, plus SDAR aux."""
    char, pos = sample_task(rng)
    student_prompt, teacher_prompt = make_prompt(char, pos)

    rewards = np.zeros(G)
    trajs = []
    for g in range(G):
        gen, R, ctxs = rollout(model, student_prompt, char, rng)
        rewards[g] = R
        trajs.append((gen, ctxs))

    mu = rewards.mean()
    sigma = rewards.std() + 1e-6
    A = (rewards - mu) / sigma   # group-relative advantage

    # Accumulate gradients
    gE_total = np.zeros_like(model.E)
    gW_total = np.zeros_like(model.W)
    gb_total = np.zeros_like(model.b)

    gate_fires = []         # mean-of-g across all tokens (for diagnostic)
    deltas     = []

    for g in range(G):
        gen, ctxs = trajs[g]
        for t in range(T_GEN):
            ctx_t = ctxs[t]
            tok   = int(gen[t])

            # ---- GRPO policy gradient: A * grad log pi(tok|ctx) -----------
            gE, gW, gb = model.grad_logp_token(ctx_t, tok)
            gE_total += A[g] * gE
            gW_total += A[g] * gW
            gb_total += A[g] * gb

            # ---- SDAR auxiliary on the SAMPLED token ---------------------
            # Use the token-level form: ell_t = g_t * (log pi_T - log pi_theta).
            # Maximizing ell_t over theta means MINIMIZING -log pi_theta(tok),
            # i.e. ADDING +g_t * grad log pi_theta(tok|ctx) (CE-ish).
            gen_so_far = list(gen[:t])
            lp_T = teacher_logprob_for_token(model, teacher_prompt, gen_so_far, tok)
            lp_S = student_logprob_for_token(model, student_prompt, gen_so_far, tok)
            delta = lp_T - lp_S
            gate  = 1.0 / (1.0 + np.exp(-GATE_BETA * delta))   # detached
            gate_fires.append(gate)
            deltas.append(delta)

            gE_total += LAMBDA_SDAR * gate * gE
            gW_total += LAMBDA_SDAR * gate * gW
            gb_total += LAMBDA_SDAR * gate * gb

    # average over rollouts only — keep per-token signal strong
    gE_total /= float(G)
    gW_total /= float(G)
    gb_total /= float(G)

    model.add(gE_total, gW_total, gb_total, LR)

    return {
        "mean_reward": float(rewards.mean()),
        "mean_gate":   float(np.mean(gate_fires)),
        "mean_delta":  float(np.mean(deltas)),
        "char": char,
    }


# ---------------------------------------------------------------- eval -------
def eval_reward(model: TinyLM, n: int, rng):
    R = 0.0
    for _ in range(n):
        char, pos = sample_task(rng)
        student_prompt, _ = make_prompt(char, pos)
        _, r, _ = rollout(model, student_prompt, char, rng)
        R += r
    return R / n


def kl_to_initial(model: TinyLM, init_E, init_W, init_b, rng, n=8):
    """Estimate per-token KL(pi_init || pi_current) on random eval prompts."""
    kls = []
    for _ in range(n):
        char, pos = sample_task(rng)
        sp, _ = make_prompt(char, pos)
        ctx = list(sp)
        for _ in range(T_GEN):
            arr = np.array(ctx, dtype=np.int64)
            # current
            z = model.forward_logits(arr)
            z -= z.max()
            p = np.exp(z); p /= p.sum()
            # init
            if len(arr) == 0:
                h0 = np.zeros(D)
            else:
                h0 = init_E[arr].mean(axis=0)
            z0 = h0 @ init_W + init_b
            z0 -= z0.max()
            p0 = np.exp(z0); p0 /= p0.sum()
            kls.append(float(np.sum(p0 * (np.log(p0 + 1e-12) - np.log(p + 1e-12)))))
            # advance ctx with greedy from CURRENT model (just to vary ctx)
            ctx.append(int(p.argmax()))
    return float(np.mean(kls))


# ---------------------------------------------------------------- main -------
def show_samples(model: TinyLM, label: str, rng):
    print(f"\n--- {label} sample completions ---")
    for char in TARGET_ALPHA:
        sp, _ = make_prompt(char, 5)
        gen, R, _ = rollout(model, sp, char, rng)
        print(f"  prompt={decode(sp)!r:35s}  gen={decode(gen)!r:15s}  R={R}")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    model = TinyLM(V, D)

    init_E = model.E.copy()
    init_W = model.W.copy()
    init_b = model.b.copy()

    R0 = eval_reward(model, 32, rng)
    print(f"Tiny SDAR-LM  V={V} D={D} T_gen={T_GEN}  G={G}  steps={N_STEPS}  "
          f"lambda={LAMBDA_SDAR} beta={GATE_BETA}")
    print(f"Initial eval reward (32 tasks): {R0:.3f}")

    show_samples(model, "BEFORE", np.random.default_rng(SEED + 1))

    R_curve = []
    gate_curve = []
    for step in range(N_STEPS):
        info = grpo_sdar_step(model, rng)
        R_curve.append(info["mean_reward"])
        gate_curve.append(info["mean_gate"])
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"  step {step:>3d}  rollout_R={info['mean_reward']:.2f}  "
                  f"mean_gate={info['mean_gate']:.3f}  "
                  f"mean_delta={info['mean_delta']:+.3f}  (target='{info['char']}')")

    R_final = eval_reward(model, 64, rng)
    KL_final = kl_to_initial(model, init_E, init_W, init_b, rng, n=8)
    gate_rate = float(np.mean(gate_curve))

    show_samples(model, "AFTER", np.random.default_rng(SEED + 1))

    runtime = time.time() - t0
    print("\n" + "=" * 60)
    print(" TINY SDAR-LM SUMMARY")
    print("=" * 60)
    print(f"  initial eval reward (32 tasks)  : {R0:.3f}")
    print(f"  final   eval reward (64 tasks)  : {R_final:.3f}")
    print(f"  mean gate-fire rate over training: {gate_rate:.3f}")
    print(f"  per-token KL (current vs init)   : {KL_final:.4f}")
    print(f"  runtime                          : {runtime:.1f}s")

    print(f"RESULT_JSON {{"
          f"\"initial_reward\":{R0:.4f},"
          f"\"final_reward\":{R_final:.4f},"
          f"\"gate_fire_rate\":{gate_rate:.4f},"
          f"\"kl_to_init\":{KL_final:.6f},"
          f"\"runtime_s\":{runtime:.2f}"
          f"}}")


if __name__ == "__main__":
    main()
