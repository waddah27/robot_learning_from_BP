"""Paired ablation of state-driven versus time-driven task parametrization.

All controller, material, passivity, and randomized-physics settings are held
fixed within a seed.  The only ablated term is the measured-state phase
correction; both conditions retain the energy-tank gate and torque projection.
"""
import argparse
import json
import logging
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "mjModeling"))
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from experiments_eval.run_experiments import apply_domain_randomization
from logger import Logger
from mjModeling.conf import ROBOT_SCENE
from mjModeling.conf.configs import ImpedanceOptimizer
from mjModeling.controllers.vic_controller.continuous_trajectory_vic import (
    ContinuousTrajectoryVIC,
)
from mjModeling.cutting_materials import Material
from mjModeling.experiments.motion import straightCutting
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14

_FIG_DIR = os.path.join(_ROOT, "results", "figures")
_RESULT_DIR = os.path.join(_ROOT, "results")

CASES = {
    "state_driven": True,
    "time_driven": False,
}


def _metrics(log, dt, nominal_phase_rate):
    phase = log["phase"]
    desired = log["pos_des_nom"]
    actual = log["pos_act"]
    error = desired - actual
    error_norm = np.linalg.norm(error, axis=1)
    force = np.linalg.norm(log["f_act"], axis=1)
    correction = log["phase_correction"]
    phase_dot = log["phase_dot"]
    phase_hat = log["phase_hat"]
    return {
        "completion": float(phase[-1] >= 0.999),
        "phase_final": float(phase[-1]),
        "duration_s": float(len(phase) * dt),
        "position_rmse_mm": float(np.sqrt(np.mean(error_norm ** 2)) * 1e3),
        "depth_rmse_mm": float(np.sqrt(np.mean(error[:, 2] ** 2)) * 1e3),
        "force_peak_N": float(np.max(force)),
        "force_rms_N": float(np.sqrt(np.mean(force ** 2))),
        "phase_observation_rmse": float(
            np.sqrt(np.mean((phase_hat - phase) ** 2))
        ),
        "negative_correction_fraction": float(np.mean(correction < -1e-12)),
        "retarded_phase_fraction": float(
            np.mean(phase_dot < nominal_phase_rate - 1e-12)
        ),
        "mean_phase_rate": float(np.mean(phase_dot)),
        "tank_min_J": float(np.min(log["tank"])),
        "power_violation_fraction": float(
            np.mean(log["power_safe"] > log["power_limit"] + 1e-6)
        ),
    }


def run_case(material, moving, seed, state_phase_enabled, dr_state,
             holdback_force, randomize=True):
    wp = Material().from_work_piece()
    wp.bp_data = material
    wp.is_movable = moving
    robot = iiwa14().create(xml_path=ROBOT_SCENE, work_piece=wp)

    dr = {}
    if randomize:
        rng = np.random.default_rng()
        rng.bit_generator.state = dr_state
        dr = apply_domain_randomization(robot, rng)
        robot.data.qpos[:robot.nq_robot] += rng.normal(
            0.0, 0.01, robot.nq_robot
        )

    vic = ContinuousTrajectoryVIC(
        robot, use_behaviour_priors=True, optimizer=ImpedanceOptimizer.qp
    )
    vic.use_learned_gains = True
    vic.variance_mode = "true"
    vic.state_phase_enabled = state_phase_enabled
    vic.holdback_disturbance_enabled = holdback_force > 0.0
    vic.holdback_force_N = holdback_force
    np.random.seed(seed)

    experiment = straightCutting(robot)
    experiment.controller = vic
    experiment.execute(viewer=None)

    log = {key: np.asarray(value) for key, value in vic.episode_log.items()}
    nominal_rate = 1.0 / vic.traj_duration
    metrics = _metrics(log, vic.dt, nominal_rate)
    metrics["domain_randomization"] = dr
    try:
        robot.shutdown()
    except Exception:
        pass
    return log, metrics, vic.dt


def _paired_summary(raw):
    summary = {}
    keys = [
        "completion", "duration_s", "position_rmse_mm", "depth_rmse_mm",
        "force_peak_N", "force_rms_N", "phase_observation_rmse",
        "negative_correction_fraction", "retarded_phase_fraction",
        "mean_phase_rate", "tank_min_J", "power_violation_fraction",
    ]
    for case in CASES:
        summary[case] = {}
        for key in keys:
            values = np.asarray([row[key] for row in raw[case]], dtype=float)
            summary[case][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": int(len(values)),
            }
    summary["paired_state_minus_time"] = {}
    for key in keys:
        state = np.asarray([row[key] for row in raw["state_driven"]])
        clock = np.asarray([row[key] for row in raw["time_driven"]])
        delta = state - clock
        entry = {
            "mean": float(np.mean(delta)),
            "std": float(np.std(delta)),
            "n": int(len(delta)),
        }
        try:
            from scipy.stats import wilcoxon
            entry["wilcoxon_p"] = (
                1.0 if np.allclose(delta, 0.0)
                else float(wilcoxon(state, clock).pvalue)
            )
        except Exception:
            entry["wilcoxon_p"] = None
        summary["paired_state_minus_time"][key] = entry
    return summary


def plot_comparison(representative, raw, material, moving, holdback_force, output):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    colors = {"state_driven": "tab:blue", "time_driven": "tab:orange"}
    labels = {"state_driven": "State-driven", "time_driven": "Time-driven"}

    for case, (log, dt) in representative.items():
        t = np.arange(len(log["phase"])) * dt
        color = colors[case]
        label = labels[case]
        axes[0, 0].plot(t, log["phase"], color=color, lw=1.3, label=label)
        axes[0, 0].plot(
            t, log["phase_hat"], color=color, lw=0.7, alpha=0.45, ls="--"
        )
        axes[0, 1].plot(t, log["phase_dot"], color=color, lw=1.0, label=label)
        axes[0, 2].plot(
            t,
            np.linalg.norm(log["pos_des_nom"] - log["pos_act"], axis=1) * 1e3,
            color=color,
            lw=1.0,
            label=label,
        )
        axes[1, 0].plot(
            t, np.linalg.norm(log["f_act"], axis=1),
            color=color, lw=1.0, label=label
        )
        disturbed = log["holdback_force"] > 0.0
        if np.any(disturbed):
            for ax in axes.flat[:4]:
                ax.axvspan(t[disturbed][0], t[disturbed][-1],
                           color="tab:red", alpha=0.08)

    axes[0, 0].set(title="Task progress and state observation",
                   xlabel="Time (s)", ylabel=r"$\phi,\ \hat{\phi}$")
    axes[0, 1].set(title="Executed phase rate",
                   xlabel="Time (s)", ylabel=r"$\dot{\phi}$ (s$^{-1}$)")
    axes[0, 2].set(title="Cartesian tracking error",
                   xlabel="Time (s)", ylabel="Error norm (mm)")
    axes[1, 0].set(title="Measured interaction force",
                   xlabel="Time (s)", ylabel="Force magnitude (N)")
    for ax in axes.flat[:4]:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    metric_specs = [
        ("position_rmse_mm", "Position RMSE (mm)", axes[1, 1]),
        ("duration_s", "Execution duration (s)", axes[1, 2]),
    ]
    for key, ylabel, ax in metric_specs:
        values = [
            [row[key] for row in raw["state_driven"]],
            [row[key] for row in raw["time_driven"]],
        ]
        bp = ax.boxplot(values, tick_labels=["State-driven", "Time-driven"],
                        patch_artist=True, widths=0.55)
        for patch, case in zip(bp["boxes"], CASES):
            patch.set_facecolor(colors[case])
            patch.set_alpha(0.45)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)

    motion = "moving workpiece" if moving else "static workpiece"
    fig.suptitle(
        f"Task-parametrization ablation: {material}, {motion}\n"
        f"Paired physics and controller; matched holdback={holdback_force:.0f} N",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="cork")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--moving", action="store_true")
    parser.add_argument("--holdback-force", type=float, default=120.0)
    args = parser.parse_args()

    try:
        Logger.setLevel(logging.ERROR)
    except Exception:
        pass

    raw = {case: [] for case in CASES}
    representative = {}
    for seed in range(args.seeds):
        master = np.random.default_rng(2400 + seed)
        state = master.bit_generator.state
        for case, enabled in CASES.items():
            log, metrics, dt = run_case(
                args.material, args.moving, seed, enabled, state,
                args.holdback_force
            )
            raw[case].append(metrics)
            if seed == 0:
                representative[case] = (log, dt)
            print(
                f"seed={seed:02d} {case:12s} completion={metrics['completion']:.0f} "
                f"phase={metrics['phase_final']:.3f} "
                f"duration={metrics['duration_s']:.2f}s "
                f"posRMSE={metrics['position_rmse_mm']:.3f}mm "
                f"depthRMSE={metrics['depth_rmse_mm']:.3f}mm "
                f"Fpeak={metrics['force_peak_N']:.2f}N",
                flush=True,
            )

    summary = _paired_summary(raw)
    os.makedirs(_RESULT_DIR, exist_ok=True)
    os.makedirs(_FIG_DIR, exist_ok=True)
    suffix = "moving" if args.moving else "static"
    json_path = os.path.join(
        _RESULT_DIR, f"task_parametrization_comparison_{args.material}_{suffix}.json"
    )
    figure_path = os.path.join(
        _FIG_DIR, f"task_parametrization_comparison_{args.material}_{suffix}.png"
    )
    with open(json_path, "w") as handle:
        json.dump(
            {"design": vars(args), "raw": raw, "summary": summary},
            handle, indent=2
        )
    plot_comparison(
        representative, raw, args.material, args.moving,
        args.holdback_force, figure_path
    )
    print(f"saved results: {json_path}")
    print(f"saved figure:  {figure_path}")


if __name__ == "__main__":
    main()
