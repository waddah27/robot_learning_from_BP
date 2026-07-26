"""
Generate the report monitor figure from a real headless cutting episode.

The live GUI monitor is useful during interactive runs, but a report figure
should be reproducible. This script reconstructs the same monitor channels from
the controller episode log: Cartesian position, contact force, stiffness and
tracking error.
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

Logger.configure(log_file="/tmp/monitor_snapshot.log", level=Logger.ERROR)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "results", "figures")


def run_episode(material="cork", moving=False, holdback_force=0.0):
    wp = Material().from_work_piece()
    wp.bp_data = material
    wp.is_movable = moving

    robot = iiwa14().create(xml_path=ROBOT_SCENE, work_piece=wp)
    vic = ContinuousTrajectoryVIC(robot, use_behaviour_priors=True,
                                  optimizer=ImpedanceOptimizer.qp)
    vic.working_piece = wp
    vic.use_learned_gains = True
    vic.variance_mode = "true"
    vic.use_cutting_model = True
    vic.state_phase_enabled = True
    vic.holdback_disturbance_enabled = holdback_force > 0.0
    vic.holdback_force_N = holdback_force

    exp = straightCutting(robot)
    exp.controller = vic
    exp.execute(viewer=None)
    log = {k: np.asarray(v) for k, v in vic.episode_log.items()}
    try:
        robot.shutdown()
    except Exception:
        pass
    return log, vic.dt


def plot_monitor_conditions(runs, fname="monitor_preview.png"):
    """Four-condition monitor: static/moving × holdback off/on."""
    fig, axes = plt.subplots(4, 4, figsize=(20, 12), squeeze=False)
    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = ["X", "Y", "Z"]

    for col, (title, (log, dt)) in enumerate(runs.items()):
        phase = np.asarray(log["phase"], dtype=float)
        time = np.arange(len(phase)) * dt
        pos_des = np.asarray(log["pos_des_nom"], dtype=float) * 1e3
        pos_act = np.asarray(log["pos_act"], dtype=float) * 1e3
        f_react_des = -np.asarray(log["f_des"], dtype=float)
        f_act = np.asarray(log["f_act"], dtype=float)
        kp = np.asarray(log["kp"], dtype=float) / 1000.0
        err = pos_des - pos_act
        tank = np.asarray(log["tank"], dtype=float)
        disturbed = np.asarray(log["holdback_force"], dtype=float) > 0.0

        axes[0, col].set_title(title, fontsize=11)
        for i, (color, label) in enumerate(zip(colors, labels)):
            axes[0, col].plot(time, pos_des[:, i], ls="--", lw=0.8,
                              color=color, alpha=0.55,
                              label=f"{label} ref")
            axes[0, col].plot(time, pos_act[:, i], lw=0.95,
                              color=color, label=f"{label} actual")
            axes[1, col].plot(time, f_react_des[:, i], ls="--", lw=0.8,
                              color=color, alpha=0.55,
                              label=f"R{label} ref")
            axes[1, col].plot(time, f_act[:, i], lw=0.95,
                              color=color, label=f"R{label} measured")
            axes[2, col].plot(time, kp[:, i], lw=1.0, color=color,
                              label=f"K{label}")
            axes[3, col].plot(time, np.abs(err[:, i]), lw=1.0,
                              color=color, label=f"|e_{label}|")

        tank_axis = axes[3, col].twinx()
        tank_axis.plot(time, tank, color="darkorange", lw=0.85,
                       alpha=0.7, label="tank")
        tank_axis.set_ylabel("T (J)", color="darkorange", fontsize=8)
        tank_axis.tick_params(axis="y", labelsize=7, colors="darkorange")

        if np.any(disturbed):
            t0, t1 = time[disturbed][0], time[disturbed][-1]
            for row in range(4):
                axes[row, col].axvspan(t0, t1, color="tab:red", alpha=0.08)

        for row in range(4):
            axes[row, col].grid(alpha=0.23)
            axes[row, col].tick_params(labelsize=8)
        axes[3, col].set_xlabel("runtime (s)")

        if col == 0:
            axes[0, col].set_ylabel("position (mm)")
            axes[1, col].set_ylabel("reaction force (N)")
            axes[2, col].set_ylabel("stiffness (kN/m)")
            axes[3, col].set_ylabel("|tracking error| (mm)")
            for row in range(4):
                axes[row, col].legend(fontsize=6, ncol=2,
                                      loc="upper left")

        metric = summarize(log)
        axes[3, col].text(
            0.98, 0.94,
            f"Z RMSE={metric['z_rmse_mm']:.2f} mm\n"
            f"$\\phi_f$={metric['phase_final']:.3f}",
            transform=axes[3, col].transAxes, ha="right", va="top",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    row_titles = ["Cartesian position", "Material reaction",
                  "Learned stiffness", "Tracking error and tank"]
    for row, title in enumerate(row_titles):
        axes[row, 0].annotate(
            title, xy=(-0.29, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center", fontsize=10,
            fontweight="bold",
        )

    fig.suptitle(
        "Runtime monitor: static/moving workpiece with holdback disabled/enabled\n"
        "The geometric path is common; runtime reference traces differ when "
        "state correction changes phase timing",
        fontsize=15,
    )
    fig.tight_layout(rect=[0.025, 0, 1, 0.965])
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_monitor(log, dt, fname="monitor_preview.png"):
    phase = np.asarray(log["phase"], dtype=float)
    time = np.arange(len(phase)) * dt
    pos_des = np.asarray(log["pos_des_nom"], dtype=float) * 1e3
    pos_act = np.asarray(log["pos_act"], dtype=float) * 1e3
    # f_des is the desired tool-side force used by the controller. f_act is the
    # material reaction on the blade, so compare it to the desired reaction.
    f_react_des = -np.asarray(log["f_des"], dtype=float)
    f_act = np.asarray(log["f_act"], dtype=float)
    kp = np.asarray(log["kp"], dtype=float) / 1000.0
    err = pos_des - pos_act
    tank = np.asarray(log["tank"], dtype=float)

    colors = ["tab:red", "tab:green", "tab:blue"]
    labels = ["X", "Y", "Z"]
    fig, ax = plt.subplots(4, 1, figsize=(13.5, 9.2), sharex=True)

    for i, (c, lab) in enumerate(zip(colors, labels)):
        ax[0].plot(time, pos_des[:, i], ls="--", lw=1.0, color=c, alpha=0.65,
                   label=f"{lab} desired")
        ax[0].plot(time, pos_act[:, i], lw=1.15, color=c, label=f"{lab} actual")
    ax[0].set_ylabel("position (mm)")
    ax[0].set_title("Cartesian position")
    ax[0].legend(ncol=3, fontsize=8, loc="upper left")

    for i, (c, lab) in enumerate(zip(colors, labels)):
        ax[1].plot(time, f_react_des[:, i], ls="--", lw=1.0, color=c, alpha=0.65,
                   label=f"R{lab} desired")
        ax[1].plot(time, f_act[:, i], lw=1.15, color=c, label=f"R{lab} measured")
    ax[1].set_ylabel("force (N)")
    ax[1].set_title("Material reaction on blade")
    ax[1].legend(ncol=3, fontsize=8, loc="upper left")

    for i, (c, lab) in enumerate(zip(colors, labels)):
        ax[2].plot(time, kp[:, i], lw=1.2, color=c, label=f"K{lab}")
    ax[2].set_ylabel("stiffness (kN/m)")
    ax[2].set_title("Commanded stiffness")
    ax[2].legend(ncol=3, fontsize=8, loc="upper left")

    for i, (c, lab) in enumerate(zip(colors, labels)):
        ax[3].plot(time, np.abs(err[:, i]), lw=1.15, color=c, label=f"|e_{lab}|")
    ax3 = ax[3].twinx()
    ax3.plot(time, tank, color="darkorange", lw=1.0, alpha=0.75, label="tank")
    ax3.set_ylabel("tank energy (J)")
    ax[3].set_ylabel("tracking error (mm)")
    ax[3].set_title("Position tracking error and tank state")
    h1, l1 = ax[3].get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax[3].legend(h1 + h2, l1 + l2, ncol=4, fontsize=8, loc="upper left")

    for a in ax:
        a.grid(True, which="major", alpha=0.28, lw=0.7)
        a.grid(True, which="minor", alpha=0.12, lw=0.4)
        a.minorticks_on()
        a_phase = a.twiny()
        a_phase.set_xlim(phase[0], phase[-1])
        if a is ax[0]:
            a_phase.set_xlabel("phase")
        else:
            a_phase.set_xticklabels([])

    ax[-1].set_xlabel("time (s)")
    fig.suptitle("Monitor snapshot from a cork cutting episode", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.965])

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def summarize(log):
    pos_des = np.asarray(log["pos_des_nom"], dtype=float)
    pos_act = np.asarray(log["pos_act"], dtype=float)
    f_react_des = -np.asarray(log["f_des"], dtype=float)
    f_act = np.asarray(log["f_act"], dtype=float)
    err_mm = (pos_des - pos_act) * 1e3
    f_vec_err = f_act - f_react_des
    f_mag_err = np.linalg.norm(f_act, axis=1) - np.linalg.norm(f_react_des, axis=1)
    return {
        "steps": int(len(log["phase"])),
        "phase_final": float(log["phase"][-1]),
        "z_rmse_mm": float(np.sqrt(np.mean(err_mm[:, 2] ** 2))),
        "force_mag_rmse_N": float(np.sqrt(np.mean(f_mag_err ** 2))),
        "force_vector_rmse_N": float(np.sqrt(np.mean(np.sum(f_vec_err ** 2, axis=1)))),
        "tank_min_J": float(np.min(log["tank"])),
    }


def main():
    conditions = [
        ("Static — nominal", False, 0.0),
        ("Static — 120 N holdback", False, 120.0),
        ("Moving — nominal", True, 0.0),
        ("Moving — 120 N holdback", True, 120.0),
    ]
    runs = {}
    for title, moving, holdback in conditions:
        log, dt = run_episode(moving=moving, holdback_force=holdback)
        runs[title] = (log, dt)
        m = summarize(log)
        print(
            f"{title}: steps={m['steps']} phase={m['phase_final']:.3f} "
            f"z_rmse={m['z_rmse_mm']:.2f}mm "
            f"tank_min={m['tank_min_J']:.3f}J",
            flush=True,
        )
    out = plot_monitor_conditions(runs)
    print(f"saved figure: {out}")


if __name__ == "__main__":
    main()
