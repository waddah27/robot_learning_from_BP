# The Dissertation Story (plain-language narrative)

A readable account of the contribution, the evidence, and the limits. Final numbers are filled
from the 12-seed experiment; preliminary values are marked *(prelim)*.

---

## 1. The problem statement to solve
Teach a robot a delicate, high-precision human skill — cutting (toward scalpel /
surgical tasks or soft skills tasks) — by learning from human demonstrations ("behaviour priors"),
rather than hand-programming it.
The hard part is *contact*: the blade must follow a path **and** interact with the material correctly.

## 2. The key insight (what data actually showed)
When the demonstrations are examined carefully, one fact dominates:

> **Humans were highly consistent in the cutting *path* (position), but highly
> *inconsistent* in the *force* they applied.**

Concretely, across repeated demonstrations:
- Position profiles agree trial-to-trial with correlation **~0.95**.
- Force profiles agree only **~0.10** — and the force signal is mostly a large
  (~900 N) near-constant offset (sensor/preload), not a reproducible skill.

This single fact reframes everything. The earlier struggle — *"the robot can't
track the demonstrated force"* — was **not a controller bug**. It is impossible
to faithfully reproduce a signal the humans themselves did not reproduce. That
impossibility is now a *proven finding*, not a failure.

## 3. The contribution (one sentence)
> **The variability of the demonstrations tells the robot what to control: track
> tightly where the humans were consistent, and stay compliant where they were
> not.**

This is an instance of the *minimal-intervention principle* (control only what
matters for the task). The novelty here is applying it to a **contact-rich force
task** and showing that the data itself **releases the force dimension** and
**keeps the cutting-depth dimension** — the robot is *told by the data* that
depth is the skill and lateral wander / absolute force are not.

## 4. The method (plain)
1. From the demonstrations, measure, at each point along the motion, **how
   consistent** the humans were in each direction (position X/Y/Z and force).
2. Turn that consistency into controller **stiffness**: very consistent → stiff
   (track it); inconsistent → soft (let it go). A minimum stiffness "floor"
   keeps the motion physically executable.
3. The result, *emerging from the data with no hand-tuning*: the robot becomes
   **stiff in cutting depth (Z)**, **soft laterally (X/Y)**, and **does not chase
   the unreliable force** — it lets the contact force settle naturally.

## 5. The evidence (the "killer experiment")
To prove the contribution is the *specific learned structure* (not just "the
robot has adjustable stiffness"), the same robot/task is run four ways:

| Controller | Idea |
|---|---|
| **Learned (true)** | Use the real demonstration-variability structure |
| **Inverted** | Flip it (control the inconsistent things, relax the consistent) |
| **Shuffled** | Randomly scramble which directions are controlled |
| **Naive force** | The old approach: just try to match demonstrated force |

**Result (cut-depth accuracy, lower is better; 12 seeds, 11 stable):**

| Controller | Cut-depth error (mm) | vs. learned |
|---|---|---|
| **Learned (true)** | **10.9 ± 2.5** | — (best) |
| Shuffled | 14.7 ± 2.9 | 26% worse |
| Inverted | 20.7 ± 4.0 | 47% worse |
| Naive force | 20.9 ± 5.9 | 48% worse |

The learned controller tracks **cutting depth best on all 11 of 11 seeds**.
Flipping or scrambling the learned structure makes depth tracking **26–48%
worse**. Paired Wilcoxon test: **p = 0.001** for every comparison (the strongest
the test allows at this sample size — every seed favours the learned controller).
This is the proof that the *specific learned variability structure*, not merely
"adjustable stiffness", is what carries the skill.

(Trade-off, stated openly: the learned controller's peak contact force is higher,
~137 N vs ~77–110 N for the baselines — the price of a stiff depth axis.)

## 6. Why "cut-depth" is the right way to judge it
A natural objection: *"averaged over X/Y/Z, the learned controller's position
error isn't the smallest."* Correct — and that is the point. It is **deliberately
soft** in the lateral directions the humans did not regulate, so it "loses" on a
metric that rewards stiffness everywhere. The task-relevant metric is **cutting
depth** — what governs cut quality and safety, and the one dimension the humans
*did* regulate. On that metric, the learned controller wins. Judging a
minimal-intervention controller by error in directions it intentionally releases
is the wrong lens.

## 7. limitations and open questions:
- **Force is not a trackable target in this data** — we regulate it as a loose
  envelope, not a trajectory. (This is a finding, not a weakness.)
- **Sim-only.** Compensated by randomizing the simulated physics and showing the
  result holds across that range. Real-robot validation is future work.
- **Higher peak contact force** for the learned controller (stiffer depth axis):
  a genuine stiffness/safety trade-off, reported openly.
- **The passivity/energy-tank stability mechanism** is currently not actually
  closing the loop in code (it uses a constant budget); the fix is specified.

## 8. Questions to address in the thesis
- **Ch. A — What transfers and what doesn't:** the variability analysis
  (position reproducible, force not). Hence, track position maintain force within the desired constraints.
- **Ch. B — Learned-variability control:** the method (stiffness = demonstration
  precision) and the experiment.
- **Ch. C — Robustness & generalization:** does the learned skill survive
  disturbances and transfer cork → PVC / penoplex (the remaining experiment).
- **Ch. D — Stability:** the passivity guarantee (theory + the tank fix + demo).

## 9. Summary
*"I learned a precision skill from human demonstrations by letting the
demonstrations' own variability decide what the robot controls. The data shows
position is a reliable skill and force is not, so the robot tracks the cut path
and depth tightly while staying compliant elsewhere. I prove this is the right
strategy: scrambling or inverting the learned structure doubles the cutting-depth
error. It is sim-only and trades higher peak force for depth accuracy, both
stated openly."*
