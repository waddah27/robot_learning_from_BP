# Cutting interaction: controller vs. material model (methods + results)

This documents, discuss precisely, how the simulated cutting force is
produced and what tracks what.

## 1. The two distinct roles (state them separately)

| Component | Role | What it achieves |
|---|---|---|
| **VIC controller** (learned-stiffness QP) | track the demonstrated **position/velocity** | accurate motion; cut-depth (Z) error ~2 mm |
| **Material model** (identified per material) | reproduce the **contact force** the real material exerts | force matches the demonstrated/desired force (≈2 N RMSE) |

The controller tracks **position** under demonstrated forces as constraints. The contact **force** matches because the **material model is
identified to the demonstration** — that is calibration.

## 2. Why a material model was needed
The original simulation modelled the blade–material contact as **rigid box-on-box
MuJoCo collision**. Two overlapping box geoms produce non-physical collision
forces (≈0 when grazing, spikes to 100s of N otherwise, traced to a single
`scalpel_geom↔material_geom` contact of 118 N). No controller can make such a
contact track a cutting-force constraint, because the force is a collision
artifact, not a cutting force. So the contact model — not the controller — was
the problem.

## 3. The identified per-material cutting law
Cutting force loads up elastically, then **plateaus** at the material's cutting
strength (the blade fractures/cuts). The model is the textbook saturating law,
applied along the demonstrated reaction direction:

    |F_react(depth)| = min( k_mat · depth , F_cut_mat )

- `depth` = penetration of the blade tip below the **live** material surface,
- `F_cut_mat` = cutting-force plateau, identified as a high percentile of the
  demonstrated reaction magnitude,
- `k_mat` = loading stiffness, set to reach `F_cut` within a small fraction of
  the typical demonstrated penetration.

Both parameters are **identified per material from that material's own
demonstration** (system identification). The geometric box contact is disabled
and this force is applied to the blade via `xfrc_applied`; the force estimator
reports it.

### Identified parameters and result (one run each, static)
| Material | F_cut (N) | k (N/m) | contact | actual vs desired (N) | force RMSE |
|---|---|---|---|---|---|
| penoplex (foam) | 63 | 9 104 | 100% | 63 vs 61 | 1.8 N |
| cork | 59 | 9 917 | 100% | 59 vs 58 | 2.5 N |
| PVC (hardest) | 86 | 13 855 | 100% | 86 vs 84 | 2.6 N |

The three materials yield **distinct** parameters (PVC is stiffest and strongest,
consistent with it being the hardest material), so the model **expresses
material-specific behaviour**, and the match is **approximate** (≈2 N), not an
exact replay.

## 4. Results summary
- Contact maintained **100%** of the cut (the earlier mid-cut force dropout was a
  stale-surface bug — the model now reads the live material surface each step).
- Force tracks the demonstrated/desired magnitude within **~2 N RMSE**, on
  **both** static and moving material.
- Cut-depth (Z) position error **~2 mm**; lateral (Y) error peaks ~50 mm during
  the large ±150 mm demonstrated Y swing (see limitations).

## 5. Assumptions and limitations
- The force "match" is the **identified material model** reproducing the
  demonstration, with the **VIC controller** providing position tracking — stated
  as such, not as controller force-tracking.
- **Lateral (Y) tracking** lags the large demonstrated Y swing (peak ~50 mm); the
  task-critical cut-depth (Z) is tight (~2 mm).
  Tightening Y is a task-parametrization (phase-speed) question, separate from the force model.
- **Sim-only**; the material parameters are identified from the demonstration
  force, which is the real material's measured interaction force.
