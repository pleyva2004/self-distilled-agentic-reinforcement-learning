"""Concept 09: Asymmetric trust — positive vs negative gaps.

Toy bandit with 'good tokens' (idx in correct_set) and 'bad tokens'.
We construct a teacher that is informative ON AVERAGE: when retrieval
succeeds (prob p_succeed), teacher up-weights correct tokens; when
retrieval fails (1 - p_succeed), teacher up-weights random tokens.
Then we measure: for tokens where Delta_t > 0 vs Delta_t < 0, what
fraction belong to the correct set?
"""
import numpy as np

rng = np.random.default_rng(5)
V = 16
N = 6000
P_RETRIEVAL_SUCCEEDS = 0.7
correct_set = {0, 1, 2, 3}


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


student_logits = rng.normal(size=V)
p = softmax(student_logits)
samples = rng.choice(V, size=N, p=p)

retrieval_ok = rng.random(N) < P_RETRIEVAL_SUCCEEDS
delta = np.empty(N)
for i in range(N):
    if retrieval_ok[i]:
        teacher_logits = student_logits.copy()
        for c in correct_set:
            teacher_logits[c] += 1.5  # informative shift
    else:
        # Bad retrieval: teacher gets random shift
        teacher_logits = student_logits + 1.5 * rng.normal(size=V)
    q = softmax(teacher_logits)
    delta[i] = np.log(q[samples[i]]) - np.log(p[samples[i]])

is_correct = np.array([s in correct_set for s in samples])

pos_mask = delta > 0
neg_mask = delta < 0
acc_pos = is_correct[pos_mask].mean()
acc_neg = is_correct[neg_mask].mean()
acc_overall = is_correct.mean()

print(f"Vocab V={V}, N={N}, P(retrieval ok) = {P_RETRIEVAL_SUCCEEDS}")
print(f"Correct token set = {sorted(correct_set)}")
print()
print(f"Overall fraction correct (under student) = {acc_overall:.3f}")
print(f"Fraction correct WHEN Delta_t > 0 (teacher endorses) = {acc_pos:.3f}  <- HIGH")
print(f"Fraction correct WHEN Delta_t < 0 (teacher disagrees) = {acc_neg:.3f}  <- AMBIGUOUS")
print()
print("Positive gaps reliably indicate good tokens; negative gaps are noisy.")
print("This asymmetry is intrinsic, motivating sigmoid (asymmetric) gating.")
