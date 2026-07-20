"""
Experiment harness for the learned-variability control thesis.

Runs cutting episodes headless under different controller conditions and computes
task-success metrics. Two studies:

  VARIANCE-STRUCTURE CONTROLLED COMPARISON (P0): true vs inverted vs shuffled
  precision plus a direct-force baseline. If only the true
  demonstration-precision structure performs well, the result supports the claim
  that the contribution is the specific learned variance structure, not merely
  adjustable impedance.

  DOMAIN RANDOMIZATION (P1): each seed draws randomized physics (material
  friction / contact stiffness / mass + initial-pose noise). A PAIRED design runs
  every condition on the same randomized physics per seed -> paired statistics.

Usage:
    python experiments_eval/run_experiments.py --study variance_comparison --seeds 8 --material cork
    python experiments_eval/run_experiments.py --study dr                  --seeds 12 --material cork
"""
import os, sys, json, argparse, warnings, logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mjModeling"))
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from mjModeling.conf import ROBOT_SCENE, MATERIAL_GEOM
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14
from mjModeling.conf.configs import ImpedanceOptimizer
from mjModeling.controllers.vic_controller.continuous_trajectory_vic import ContinuousTrajectoryVIC
from mjModeling.cutting_materials import Material
from mjModeling.experiments.motion import straightCutting
from logger import Logger
try:
    Logger.setLevel(logging.ERROR)
except Exception:
    pass

_RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ---- controller conditions ----
# (use_learned_gains, variance_mode); direct_force_baseline = original fixed-Q force QP
CONDITIONS = {
    "learned_true":     (True,  "true"),
    "learned_inverted": (True,  "inverted"),
    "learned_shuffled": (True,  "shuffled"),
    "direct_force_baseline": (False, "off"),
}


def apply_domain_randomization(robot, rng, scale=0.15):
    """Randomize material friction, contact stiffness (solref) and mass.

    solref[0] (contact time-constant) is clipped to a stable band [0.05, 0.2];
    values outside this caused non-physical contact blow-ups (Fpeak ~ 1000s of N).
    """
    m = robot.model
    gid = m.geom(MATERIAL_GEOM).id
    bid = m.geom_bodyid[gid]
    fmul = 1.0 + rng.uniform(-scale, scale)
    m.geom_friction[gid, 0] *= fmul
    sref = m.geom_solref[gid, 0]
    m.geom_solref[gid, 0] = float(np.clip(sref * (1.0 + rng.uniform(-0.1, 0.1)),
                                          0.05, 0.2))
    m.body_mass[bid] *= (1.0 + rng.uniform(-scale, scale))
    return {"friction_mul": float(fmul), "solref": float(m.geom_solref[gid, 0]),
            "mass": float(m.body_mass[bid])}


_PROFILE_CACHE = {}


def _wpos_profile(material):
    """Cached precision weight profile (phase, w_pos[3]) for the material."""
    if material not in _PROFILE_CACHE:
        from variability_control.variance_gains import load_profile
        try:
            p = load_profile(material)
            _PROFILE_CACHE[material] = (p["phase"], p["w_pos"])
        except Exception:
            _PROFILE_CACHE[material] = None
    return _PROFILE_CACHE[material]


# Divergence threshold: a stable cut on these materials peaks well under this.
DIVERGENCE_FORCE_N = 400.0


def compute_metrics(log, material="cork"):
    """Task-success metrics from a controller episode_log.

    Position tracking is reported PER AXIS against the NOMINAL GMR reference
    (identical across conditions). The headline task metric is cut-depth (Z)
    error and the demonstration-precision-weighted error; X/Y are reported for
    transparency. Runs whose peak force exceeds DIVERGENCE_FORCE_N are flagged
    non-physical (`diverged`) and excluded from quality aggregates.
    """
    if not log or len(log["phase"]) < 5:
        return {"completion": 0.0, "diverged": 1.0}
    phase = np.array(log["phase"])
    pos_des = np.array(log.get("pos_des_nom", log["pos_des"]))
    pos_act = np.array(log["pos_act"])
    f_des = np.array(log["f_des"]);     f_act = np.array(log["f_act"])
    tank = np.array(log["tank"]);       tau = np.array(log["tau"])
    power_safe = np.array(log.get("power_safe", []), dtype=float)
    power_limit = np.array(log.get("power_limit", []), dtype=float)
    tank_gamma = np.array(log.get("tank_gamma", []), dtype=float)

    err = pos_des - pos_act                       # (N,3)
    pos_err = np.linalg.norm(err, axis=1)
    f_des_mag = np.linalg.norm(f_des, axis=1)
    f_act_mag = np.linalg.norm(f_act, axis=1)

    # precision-weighted position error (task-relevant; weights from demos)
    prof = _wpos_profile(material)
    if prof is not None:
        ph_p, w_p = prof
        wpos = np.stack([np.interp(phase, ph_p, w_p[:, i]) for i in range(3)], 1)
        precw = float(np.sqrt((wpos * err ** 2).sum(1)).mean() * 1e3)
    else:
        precw = float("nan")

    if len(pos_act) > 3:
        jerk = np.linalg.norm(np.diff(pos_act, n=3, axis=0), axis=1)
        jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    else:
        jerk_rms = float("nan")

    peak = float(f_act_mag.max())
    if len(power_safe) and len(power_limit):
        power_violation = float(np.mean(power_safe > power_limit + 1e-6))
        max_power_margin = float(np.max(power_safe - power_limit))
    else:
        power_violation = float("nan")
        max_power_margin = float("nan")

    return {
        "completion":      float(phase[-1] >= 0.999),
        "diverged":        float(peak > DIVERGENCE_FORCE_N),
        "z_depth_rmse_mm": float(np.sqrt(np.mean(err[:, 2] ** 2)) * 1e3),
        "x_rmse_mm":       float(np.sqrt(np.mean(err[:, 0] ** 2)) * 1e3),
        "y_rmse_mm":       float(np.sqrt(np.mean(err[:, 1] ** 2)) * 1e3),
        "precw_err_mm":    precw,
        "pos_rmse_mm":     float(np.sqrt(np.mean(pos_err ** 2)) * 1e3),
        "force_peak_N":    peak,
        "force_overshoot_N": float((f_act_mag - f_des_mag).max()),
        "jerk_rms":        jerk_rms,
        "tau_rms_Nm":      float(np.sqrt(np.mean(tau ** 2))),
        "tank_min_J":      float(np.min(tank)) if len(tank) else float("nan"),
        "tank_gamma_min":  float(np.min(tank_gamma)) if len(tank_gamma) else float("nan"),
        "power_violation_frac": power_violation,
        "max_power_margin_W": max_power_margin,
        "n_steps":         int(len(phase)),
    }


def run_episode(material, use_learned, variance_mode, dr_rng=None, seed=0):
    """Run one headless cutting episode; return metrics + DR info."""
    wp = Material().from_work_piece(); wp.bp_data = material
    robot = iiwa14().create(xml_path=ROBOT_SCENE, work_piece=wp)
    dr_info = {}
    if dr_rng is not None:
        dr_info = apply_domain_randomization(robot, dr_rng)
        # small initial-pose perturbation
        robot.data.qpos[:robot.nq_robot] += dr_rng.normal(0, 0.01, robot.nq_robot)

    vic = ContinuousTrajectoryVIC(robot, use_behaviour_priors=True,
                                  optimizer=ImpedanceOptimizer.qp)
    vic.use_learned_gains = use_learned
    vic.variance_mode = variance_mode
    np.random.seed(seed)  # for shuffled mode reproducibility

    exp = straightCutting(robot); exp.controller = vic
    try:
        exp.execute(viewer=None)
        metrics = compute_metrics(getattr(vic, "episode_log", None), material)
    except Exception as e:
        metrics = {"completion": 0.0, "diverged": 1.0, "error": str(e)[:120]}
    metrics["dr"] = dr_info
    return metrics


def aggregate(rows, keys):
    """mean/std per metric across STABLE completed episodes (exclude diverged)."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and r.get("completion", 0) > 0
                and r.get("diverged", 0) < 1
                and isinstance(r[k], (int, float)) and np.isfinite(r[k])]
        if vals:
            out[k] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
    # divergence rate is reported over ALL runs (robustness metric)
    out["divergence_rate"] = (float(np.mean([r.get("diverged", 0) for r in rows])),
                              0.0, len(rows))
    return out


def paired_test(rows_a, rows_b, key):
    """Paired comparison on a metric; Wilcoxon if available else paired diff."""
    a = np.array([r.get(key, np.nan) for r in rows_a])
    b = np.array([r.get(key, np.nan) for r in rows_b])
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    a, b = a[mask], b[mask]
    try:
        from scipy.stats import wilcoxon
        if np.allclose(a, b):
            return {"p": 1.0, "n": int(mask.sum()), "delta": 0.0}
        stat, p = wilcoxon(a, b)
        return {"p": float(p), "n": int(mask.sum()), "delta": float(np.mean(a - b))}
    except Exception:
        return {"p": None, "n": int(mask.sum()), "delta": float(np.mean(a - b))}


METRIC_KEYS = ["completion", "z_depth_rmse_mm", "x_rmse_mm", "y_rmse_mm",
               "precw_err_mm", "pos_rmse_mm", "force_peak_N",
               "force_overshoot_N", "jerk_rms", "tau_rms_Nm",
               "tank_min_J", "tank_gamma_min", "power_violation_frac",
               "max_power_margin_W"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", choices=["variance_comparison", "dr"], default="variance_comparison")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--material", default="cork")
    ap.add_argument("--conditions", default="learned_true,learned_inverted,learned_shuffled,direct_force_baseline")
    args = ap.parse_args()

    conds = [c for c in args.conditions.split(",") if c in CONDITIONS]
    # Both studies use domain randomization so seeds differ (statistical spread).
    # Paired design: identical randomized physics across conditions within a seed.
    use_dr = True
    results = {c: [] for c in conds}

    for seed in range(args.seeds):
        # PAIRED: same randomized physics for all conditions this seed
        dr_master = np.random.default_rng(1000 + seed) if use_dr else None
        dr_state = dr_master.bit_generator.state if dr_master else None
        for c in conds:
            uled, vmode = CONDITIONS[c]
            dr_rng = None
            if use_dr:
                dr_rng = np.random.default_rng()
                dr_rng.bit_generator.state = dr_state  # identical physics per seed
            m = run_episode(args.material, uled, vmode, dr_rng=dr_rng, seed=seed)
            results[c].append(m)
            print(f"  seed {seed:2d} | {c:18s} | "
                  f"done={m.get('completion',0):.0f} "
                  f"div={m.get('diverged',0):.0f} "
                  f"Zdepth={m.get('z_depth_rmse_mm',float('nan')):.1f}mm "
                  f"precW={m.get('precw_err_mm',float('nan')):.1f} "
                  f"Fpeak={m.get('force_peak_N',float('nan')):.0f}N", flush=True)

    # ---- aggregate + save ----
    os.makedirs(_RESULTS, exist_ok=True)
    summary = {c: aggregate(results[c], METRIC_KEYS) for c in conds}
    raw_path = os.path.join(_RESULTS, f"study_{args.study}_{args.material}.json")
    with open(raw_path, "w") as fh:
        json.dump({"args": vars(args), "summary": summary, "raw": results}, fh, indent=2)

    # ---- printed table ----
    print("\n" + "=" * 78)
    print(f"STUDY: {args.study}   material: {args.material}   seeds: {args.seeds}")
    print("=" * 78)
    hdr = ["condition"] + ["completion", "divergence_rate", "z_depth_rmse_mm",
                           "x_rmse_mm", "y_rmse_mm", "precw_err_mm",
                           "force_peak_N", "tau_rms_Nm",
                           "tank_min_J", "power_violation_frac"]
    print(f"{hdr[0]:18s}" + "".join(f"{h:>15s}" for h in hdr[1:]))
    for c in conds:
        s = summary[c]
        cells = []
        for k in hdr[1:]:
            if k in s:
                mu, sd, n = s[k]
                cells.append(f"{mu:.2f}±{sd:.2f}")
            else:
                cells.append("—")
        print(f"{c:18s}" + "".join(f"{cc:>15s}" for cc in cells))
    print("\n(quality metrics computed over STABLE runs only; divergence_rate over all)")

    # ---- significance vs learned_true (headline: cut-depth Z) ----
    if "learned_true" in conds:
        print("\nPaired significance vs learned_true (Wilcoxon p; Δ=other−true; "
              "+Δ on Z means worse depth than true):")
        for c in conds:
            if c == "learned_true":
                continue
            for key in ["z_depth_rmse_mm", "precw_err_mm", "force_peak_N"]:
                t = paired_test(results[c], results["learned_true"], key)
                if t:
                    p = t["p"]
                    pstr = f"p={p:.3f}" if p is not None else "p=n/a"
                    print(f"  {c:18s} {key:18s} {pstr}  Δ={t['delta']:+.3f} (n={t['n']})")
    print(f"\nsaved: {raw_path}")


if __name__ == "__main__":
    main()
