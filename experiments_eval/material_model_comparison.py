"""
Material-model controlled comparison for the cutting report.

Runs the same learned-variability VIC cutting task twice:

  1. box_contact      : MuJoCo rigid box-on-box scalpel/material contact
  2. cutting_model    : identified saturating cutting-force law

The controller settings are otherwise kept identical. The resulting plot is used
as Figure 6 in docs/skill_learning_report.tex.
"""
import os
import sys
import warnings
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mjModeling"))
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mjModeling.conf import ROBOT_SCENE
from mjModeling.conf.configs import ImpedanceOptimizer
from mjModeling.cutting_materials import Material
from mjModeling.controllers.vic_controller.continuous_trajectory_vic import ContinuousTrajectoryVIC
from mjModeling.experiments.motion import straightCutting
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14
from logger import Logger

Logger.configure(log_file="/tmp/material_model_comparison.log", level=Logger.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figures")


CASES = [
    {
        "name": "box_contact",
        "label": "Rigid box-on-box contact",
        "use_cutting_model": False,
        "color": "tab:red",
    },
    {
        "name": "cutting_model",
        "label": "Identified material model",
        "use_cutting_model": True,
        "color": "tab:blue",
    },
]


def run_case(case, material="cork"):
    wp = Material().from_work_piece()
    wp.bp_data = material
    wp.is_movable = False

    robot = iiwa14().create(xml_path=ROBOT_SCENE, work_piece=wp)
    vic = ContinuousTrajectoryVIC(robot, use_behaviour_priors=True,
                                  optimizer=ImpedanceOptimizer.qp)
    vic.working_piece = wp
    vic.use_learned_gains = True
    vic.variance_mode = "true"
    vic.use_cutting_model = bool(case["use_cutting_model"])

    exp = straightCutting(robot)
    exp.controller = vic
    exp.execute(viewer=None)
    log = {k: np.asarray(v) for k, v in vic.episode_log.items()}
    try:
        robot.shutdown()
    except Exception:
        pass
    return log


def metrics(log):
    phase = np.asarray(log["phase"], dtype=float)
    pos_des = np.asarray(log.get("pos_des_nom", log["pos_des"]), dtype=float)
    pos_act = np.asarray(log["pos_act"], dtype=float)
    f_des = np.asarray(log["f_des"], dtype=float)
    f_act = np.asarray(log["f_act"], dtype=float)

    f_des_mag = np.linalg.norm(f_des, axis=1)
    f_act_mag = np.linalg.norm(f_act, axis=1)
    active = (phase > 0.05) & (phase < 0.98) & (f_des_mag > 0.25 * np.nanmax(f_des_mag))
    if active.sum() < 5:
        active = np.ones_like(phase, dtype=bool)

    ferr = f_act_mag - f_des_mag
    zerr = (pos_des[:, 2] - pos_act[:, 2]) * 1e3
    contact = f_act_mag > 2.0
    dropout = active & (f_act_mag < 2.0)

    return {
        "completion": float(phase[-1] >= 0.999),
        "force_rmse_N": float(np.sqrt(np.mean(ferr[active] ** 2))),
        "force_mae_N": float(np.mean(np.abs(ferr[active]))),
        "force_peak_N": float(np.max(f_act_mag)),
        "force_p95_N": float(np.percentile(f_act_mag[active], 95)),
        "contact_fraction": float(np.mean(contact[active])),
        "dropout_fraction": float(np.mean(dropout[active])),
        "z_rmse_mm": float(np.sqrt(np.mean(zerr[active] ** 2))),
        "n_steps": int(len(phase)),
    }


def _clip_for_plot(y, top):
    y = np.asarray(y, dtype=float)
    clipped = y > top
    return np.minimum(y, top), clipped


def plot(results, fname="cutting_model_fix.png"):
    os.makedirs(FIG_DIR, exist_ok=True)

    force_top = max(120.0, 1.45 * max(np.nanmax(r["f_des_mag"]) for r in results))
    err_top = force_top

    fig, ax = plt.subplots(2, 3, figsize=(15.8, 7.2), sharex="col")
    for row, result in enumerate(results):
        case = result["case"]
        log = result["log"]
        met = result["metrics"]
        phase = result["phase"]
        f_des_mag = result["f_des_mag"]
        f_act_mag = result["f_act_mag"]
        ferr_abs = np.abs(f_act_mag - f_des_mag)
        zerr = (log["pos_des_nom"][:, 2] - log["pos_act"][:, 2]) * 1e3
        contact = f_act_mag > 2.0

        y_force, clipped_force = _clip_for_plot(f_act_mag, force_top)
        ax[row, 0].plot(phase, f_des_mag, "k--", lw=1.25, label="desired")
        ax[row, 0].plot(phase, y_force, color=case["color"], lw=1.1, label="measured")
        if np.any(clipped_force):
            ax[row, 0].scatter(phase[clipped_force], np.full(clipped_force.sum(), force_top),
                               marker="v", s=12, color=case["color"], alpha=0.75,
                               label="clipped spike")
        ax[row, 0].set_ylim(0.0, force_top * 1.04)
        ax[row, 0].set_ylabel("|F| (N)")
        ax[row, 0].set_title(case["label"])
        ax[row, 0].legend(loc="upper right", fontsize=8)

        y_err, clipped_err = _clip_for_plot(ferr_abs, err_top)
        ax[row, 1].plot(phase, y_err, color=case["color"], lw=1.0)
        if np.any(clipped_err):
            ax[row, 1].scatter(phase[clipped_err], np.full(clipped_err.sum(), err_top),
                               marker="v", s=12, color=case["color"], alpha=0.75)
        ax[row, 1].set_ylim(0.0, err_top * 1.04)
        ax[row, 1].set_ylabel("|force error| (N)")
        ax[row, 1].set_title(
            f"RMSE={met['force_rmse_N']:.1f} N, peak={met['force_peak_N']:.1f} N"
        )

        ax[row, 2].plot(phase, zerr, color="tab:green", lw=1.0, label="depth error")
        ax[row, 2].fill_between(phase, -4.0, 4.0, color="tab:green", alpha=0.08,
                                label="+/-4 mm")
        ax[row, 2].set_ylabel("z error (mm)")
        ax[row, 2].set_title(
            f"contact={100*met['contact_fraction']:.1f}%, "
            f"dropout={100*met['dropout_fraction']:.1f}%"
        )
        ax2 = ax[row, 2].twinx()
        ax2.fill_between(phase, 0.0, contact.astype(float), step="mid",
                         color="0.2", alpha=0.12)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["free", "contact"], fontsize=8)
        ax2.set_ylim(-0.05, 1.05)

        text = (
            f"force MAE {met['force_mae_N']:.1f} N\n"
            f"p95 force {met['force_p95_N']:.1f} N\n"
            f"z RMSE {met['z_rmse_mm']:.1f} mm"
        )
        ax[row, 2].text(0.02, 0.97, text, transform=ax[row, 2].transAxes,
                        va="top", ha="left", fontsize=8,
                        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9))

    for col in range(3):
        ax[-1, col].set_xlabel("phase")
    for a in ax.ravel():
        a.grid(True, alpha=0.25, lw=0.6)

    fig.suptitle("Material-model comparison: same controller, different contact physics",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=160)
    extra = os.path.join(FIG_DIR, "material_model_comparison_cork.png")
    fig.savefig(extra, dpi=160)
    plt.close(fig)
    return out, extra


def main():
    results = []
    for case in CASES:
        log = run_case(case)
        phase = np.asarray(log["phase"], dtype=float)
        f_des_mag = np.linalg.norm(np.asarray(log["f_des"], dtype=float), axis=1)
        f_act_mag = np.linalg.norm(np.asarray(log["f_act"], dtype=float), axis=1)
        met = metrics(log)
        results.append({
            "case": case,
            "log": log,
            "phase": phase,
            "f_des_mag": f_des_mag,
            "f_act_mag": f_act_mag,
            "metrics": met,
        })
        print(
            f"{case['name']:13s} completion={met['completion']:.0f} "
            f"force_rmse={met['force_rmse_N']:.2f}N "
            f"force_peak={met['force_peak_N']:.2f}N "
            f"contact={100*met['contact_fraction']:.1f}% "
            f"dropout={100*met['dropout_fraction']:.1f}% "
            f"z_rmse={met['z_rmse_mm']:.2f}mm "
            f"steps={met['n_steps']}",
            flush=True,
        )

    out, extra = plot(results)
    print("saved figure:", out)
    print("saved copy:", extra)


if __name__ == "__main__":
    main()
