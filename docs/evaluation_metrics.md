# Evaluation Metrics for the Learnt-Skill Controller

Why RMSE alone is insufficient, and the metric suite that actually evaluates the
contributions. Every metric is reported for **both** experiments (static and
moving material), as **mean ± std AND worst-case** over N randomized-physics
seeds, with paired significance tests.

## Why RMSE is not enough
- It is a **mean** — it hides the **tail** (a controller can have low RMSE and
  still spike dangerously once). Reliability is about the worst case.
- It says nothing about **continuity/smoothness** — a jerky controller and a
  smooth one can have identical RMSE.
- It is **meaningless for the released force channel** — we do not point-track
  force, so "force RMSE" measures the wrong thing.
- It captures nothing about **passivity** or **stability** (decay rate,
  boundedness, disturbance recovery).
- It is per-run — it ignores **reliability across conditions** (seed-to-seed
  consistency, completion rate, divergence rate).

---

## C1 — Precision vs. continuity trade-off

### Precision (on the REGULATED dimensions only, e.g. cut-depth Z)
- `e_rmse` — RMSE of tracking error (baseline).
- `e_p95`, `e_max` — 95th-percentile and maximum error → the **tail / worst
  case** (reliability, safety).
- `e_ss` — steady-state error (mean error over the last 10% of phase).

### Continuity / smoothness
- `jerk_rms`, `jerk_peak` — RMS and peak of the 3rd time-derivative of motion.
- `SPARC` — **spectral arc length** of the velocity profile: a dimensionless,
  duration- and amplitude-invariant smoothness measure (standard in motor-control
  literature; more robust than jerk). Less negative = smoother.
- `dK_rms`, `dtau_rms` — rate of change of commanded stiffness and torque →
  **control continuity** (no chattering / no gain jumps).
- `chatter_index` — number of control-sign reversals per second (contact
  chattering detector).

### Trade-off characterization
- **Pareto front**: sweep one control parameter (e.g. phase speed or stiffness
  scale), plot `e_rmse` vs `SPARC`. The contribution is that the learnt
  controller sits on a *favourable* knee of this curve (good precision AND good
  smoothness), not at an extreme.

---

## C2 — Passivity and (exponential) stability

### Passivity
- `passivity_violation_frac` — fraction of timesteps where the passivity residual
  `V̇ − fₑₓₜᵀẋ > 0` (energy generation). **Must be ≈ 0.**
- `tank_min` — minimum tank energy over the run (**must stay > 0**; if it hits
  the floor, the guarantee held by pausing, which is the designed behaviour).
- `E_inj_net` — net energy injected into the environment (bounded).

### Exponential stability  ← THE headline theoretical metric
- `lambda_hat` — identified exponential **decay rate**: fit
  `‖e(t)‖ ≤ c·e^(−λt)` to the error envelope; report **λ̂** and fit **R²**.
- `lambda_theory` — the rate predicted by the Lyapunov bound; report the ratio
  `λ̂ / λ_theory` (empirical confirms theory).
- `V_monotonic_frac` — fraction of time the Lyapunov function `V(t)` is
  non-increasing (should be ~1 outside the injection windows).
- **Disturbance recovery (ISS):** after an impulse perturbation —
  `t_recover` (time to return within ε of the reference), `overshoot`, and
  confirmation that the error returns to the bounded set (input-to-state
  stability). Report for static AND moving material.

---

## C3 — Task parametrization (phase-driven, adaptable)

### Temporal scale-invariance (adaptability across speeds)
- Run at phase speeds {0.5×, 1×, 2×, 4×}. Report each precision/stability metric
  **as a function of speed**. The contribution is **flatness**: metrics stay
  within a small band across speeds (`scale_invariance = max-min / mean` per
  metric; smaller = more scale-invariant).
- `completion_rate(speed)` — task completes (phase → 1) across the speed range.

### State-driven vs. time-driven phase
- `phase_track_err` — under disturbance, how well the phase variable reflects
  *actual* task progress (state-driven should keep φ aligned with the real
  state; time-driven cannot).
- `progress_monotonic` — phase advances monotonically and completes despite
  perturbation (state-driven should pause, not charge ahead).
- Side-by-side: state-driven vs time-driven on the SAME perturbed run.

---

## C4 — Reliability across conditions (what RMSE misses most)
Reported over N seeds of randomized physics, for both materials:
- `completion_rate` — fraction of runs that finish the cut.
- `divergence_rate` — fraction of runs that go non-physical (auto-flagged).
- For every quality metric: **mean ± std AND worst-case (max)** — reliability is
  the tail, not the average.
- `seed_cv` — coefficient of variation across seeds (low = consistent/reliable).
- Paired significance (Wilcoxon) for every claim vs. each baseline.

---

## Force channel — RELAXED, so NOT tracking accuracy
Because force is regulated as an envelope (it is not a reproducible target), the
correct force metrics are **safety and contact quality**, not tracking error:
- `force_envelope_coverage` — fraction of contact time the measured force stays
  **inside the demonstrated force band** (mean ± k·std from the demos). This is
  the right "did it respect the prior loosely" metric.
- `force_peak`, `force_overshoot` — safety (peak contact force; excess over a
  safety bound).
- `safety_violation_frac` — fraction of time force exceeds a hard safety limit.
- `contact_stable` — force does not chatter/bounce (contact maintained).

> Explicitly: we report **no** "force tracking RMSE", and we justify it — the
> demonstrations show force is not a reproducible target, so tracking it is not a
> meaningful objective. This pre-empts the obvious reviewer question.

---

## Reporting template (per claim)
| metric | static, learned | static, baseline | moving, learned | moving, baseline | p |
A claim is only "supported" if it holds — with significance — for **both**
static and moving material.

## Three main events to observe for showing results:
A. work piece moves? `MOBILE`
B. Robot tracks the workpiece motion ? `TRACK_WP_MOTION`
C. QP-based VIC controller use learned Gains(Q,R)? `USE_LEARNT_GAINS`

we build 6 scenarios of these 3 events:
S0. ~A and ~B and ~C
S1. ~A and ~B and  C
S2. ~A and  B and  ~C
S3. ~A and  B and   C
S4.  A and  ~B and  ~C
S5.  A and  ~B and   C
S6.  A and   B and  ~C
S7.  A and   B and   C

If the workpiece is static, then robot tracking its motion does not make sense therefore
we keep this

S0. ~A and  ~C
S1. ~A and   C
S2.  A and  ~B and   C
S3.  A and   B and  ~C
S4.  A and   B and   C

