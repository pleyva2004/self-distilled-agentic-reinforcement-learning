# SDAR paper concept graph

The 14 concepts that constitute the SDAR paper's mathematical core, in a directed-acyclic dependency graph.

```mermaid
graph TD
    classDef intro fill:#cfe8ff,stroke:#2b6cb0,color:#000
    classDef intermediate fill:#ffeaa7,stroke:#b7791f,color:#000
    classDef advanced fill:#ffc1c1,stroke:#9b2c2c,color:#000

    C1["1. Multi-turn agentic MDP"]:::intro
    C2["2. Trajectory-level RL reward"]:::intro
    C3["3. GRPO baseline"]:::intermediate
    C4["4. OPSD (on-policy self-distillation)"]:::intermediate
    C5["5. Teacher branch + privileged context"]:::intermediate
    C6["6. Per-token reverse KL"]:::intermediate
    C7["7. Single-sample gap estimator"]:::advanced
    C8["8. Obs 1: multi-turn instability"]:::advanced
    C9["9. Obs 2: asymmetric trust"]:::advanced
    C10["10. Sigmoid token-level gate"]:::advanced
    C11["11. Entropy gating"]:::advanced
    C12["12. Gap gating"]:::advanced
    C13["13. Soft-OR gating"]:::advanced
    C14["14. SDAR combined objective"]:::advanced

    C1 --> C2
    C2 --> C3
    C1 --> C4
    C4 --> C5
    C4 --> C6
    C5 --> C6
    C6 --> C7
    C4 --> C8
    C5 --> C8
    C5 --> C9
    C7 --> C9
    C7 --> C10
    C8 --> C10
    C10 --> C11
    C10 --> C12
    C9 --> C12
    C11 --> C13
    C12 --> C13
    C3 --> C14
    C10 --> C14

    click C1 "concepts/01-multi-turn-agentic-mdp.md"
    click C2 "concepts/02-rl-trajectory-reward.md"
    click C3 "concepts/03-grpo-baseline.md"
    click C4 "concepts/04-opsd-on-policy-self-distillation.md"
    click C5 "concepts/05-teacher-branch-privileged-context.md"
    click C6 "concepts/06-per-token-reverse-kl.md"
    click C7 "concepts/07-single-sample-gap-estimator.md"
    click C8 "concepts/08-multi-turn-instability-observation.md"
    click C9 "concepts/09-asymmetric-trust-observation.md"
    click C10 "concepts/10-token-level-gate.md"
    click C11 "concepts/11-entropy-gating.md"
    click C12 "concepts/12-gap-gating.md"
    click C13 "concepts/13-soft-or-gating.md"
    click C14 "concepts/14-sdar-combined-objective.md"
```

## Color legend

- Blue (intro) — accessible from a working knowledge of LLM training.
- Yellow (intermediate) — needs background in policy-gradient RL and KL divergences (Chain Ch 28-31).
- Red (advanced) — paper-specific contributions; the gating mechanism + asymmetric-trust analysis.

## Suggested reading paths

- **Speed run (1 hr):** 1 → 2 → 3 → 4 → 7 → 10 → 12 → 14.
- **Full chain (3-4 hr):** depth-first along edges, in numerical order.
- **Math focus:** 6 → 7 → 9 → 10 → 12 then jump to `proofs/gating-bias-bound.tex`.
