"""Concept 05: Teacher branch with privileged context.

Same shared 'parameters' (a tabular policy keyed by hash of context),
but the teacher receives extra retrieved-skill text concatenated. We
show that the teacher distribution differs from the student even though
the underlying mapping is identical.
"""
import numpy as np
import hashlib


def policy(context: str, vocab: int, rng_seed: int = 42) -> np.ndarray:
    """Shared 'policy parameters' = deterministic logit map from context hash."""
    h = hashlib.md5(context.encode()).digest()
    seed = int.from_bytes(h[:4], "big") ^ rng_seed
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=vocab)
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


V = 8
student_ctx = "task: solve. partial response: 'The answer is'"
skills = " | RETRIEVED_SKILL: arithmetic_reasoning_template_v3"
teacher_ctx = student_ctx + skills

p_student = policy(student_ctx, V)
p_teacher = policy(teacher_ctx, V)

print(f"Student context (len={len(student_ctx)}): {student_ctx!r}")
print(f"Teacher context (len={len(teacher_ctx)}): {teacher_ctx!r}\n")
print(f"pi_theta(. | s_t)   = {np.round(p_student, 3)}")
print(f"pi_T(. | s_t^+)     = {np.round(p_teacher, 3)}")
print(f"Total variation     = {0.5 * np.abs(p_student - p_teacher).sum():.3f}")
print("\nSame parameters, different conditioning -> different distributions.")
print("This is what makes 'self-distillation' literal in SDAR.")
