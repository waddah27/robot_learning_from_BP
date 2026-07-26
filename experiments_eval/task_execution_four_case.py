"""Deterministic execution comparison for phase law × holdback.

Produces one four-row figure per workpiece condition.  No domain randomization
or initial-pose perturbation is applied, so these figures are accuracy
baselines rather than robustness summaries.
"""
import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from experiments_eval.task_parametrization_comparison import run_case
from logger import Logger

CASES = [
    ("State-driven — nominal", True, 0.0),
    ("Time-driven — nominal", False, 0.0),
    ("State-driven — 120 N holdback", True, 120.0),
    ("Time-driven — 120 N holdback", False, 120.0),
]


def plot_workpiece(workpiece_name, runs, output):
    fig = plt.figure(figsize=(18, 14))
    grid = fig.add_gridspec(4, 4)
    axis_colors = ("tab:red", "tab:green", "tab:blue")

    # Draw one canonical geometric reference in every 3-D panel.  Plotting the
    # condition's visited samples makes the same path look different when
    # state-driven phase skips or repeats samples.
    canonical_log = runs[1][1]  # deterministic time-driven nominal episode
    canonical_phase = np.asarray(canonical_log["phase"], dtype=float)
    canonical_desired = np.asarray(canonical_log["pos_des_nom"], dtype=float)
    order = np.argsort(canonical_phase)
    canonical_phase = canonical_phase[order]
    canonical_desired = canonical_desired[order]
    phase_grid = np.linspace(0.0, 1.0, 500)
    canonical_reference = np.column_stack([
        np.interp(phase_grid, canonical_phase, canonical_desired[:, i])
        for i in range(3)
    ])

    all_xyz = np.vstack(
        [canonical_reference]
        + [np.asarray(run[1]["pos_act"], dtype=float) for run in runs]
    )
    xyz_min = np.min(all_xyz, axis=0)
    xyz_max = np.max(all_xyz, axis=0)
    xyz_span = np.maximum(xyz_max - xyz_min, 1e-4)
    xyz_pad = 0.04 * xyz_span

    for row, (case_name, log, metrics, dt) in enumerate(runs):
        phase = log["phase"]
        time = np.arange(len(phase)) * dt
        desired = log["pos_des_nom"]
        actual = log["pos_act"]
        error_mm = (desired - actual) * 1e3
        disturbed = log["holdback_force"] > 0.0

        ax = fig.add_subplot(grid[row, 0], projection="3d")
        ax.plot(*canonical_reference.T, color="black", ls="--", lw=1.2,
                label="Canonical reference (identical)")
        ax.plot(*actual.T, color="tab:blue", lw=0.9, label="Actual")
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.set_xlim(xyz_min[0] - xyz_pad[0], xyz_max[0] + xyz_pad[0])
        ax.set_ylim(xyz_min[1] - xyz_pad[1], xyz_max[1] + xyz_pad[1])
        ax.set_zlim(xyz_min[2] - xyz_pad[2], xyz_max[2] + xyz_pad[2])
        # Common visual aspect (z expanded for legibility); numeric limits are
        # still identical in every row and retain physical units.
        ax.set_box_aspect((1.0, 1.0, 0.35))
        ax.set_title(f"{case_name}: Cartesian path", fontsize=10)
        ax.legend(fontsize=7)

        ax = fig.add_subplot(grid[row, 1])
        for index, label, color in zip(range(3), ("x", "y", "z"), axis_colors):
            ax.plot(phase, error_mm[:, index], color=color, lw=0.9,
                    label=label)
        ax.set(
            title=(
                f"Axis error; Z RMSE={metrics['depth_rmse_mm']:.2f} mm, "
                f"3-D RMSE={metrics['position_rmse_mm']:.2f} mm"
            ),
            xlabel=r"Phase $\phi$", ylabel="Error (mm)",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        ax = fig.add_subplot(grid[row, 2])
        ax.plot(time, phase, color="tab:blue", lw=1.1, label=r"$\phi$")
        if "State-driven" in case_name:
            ax.plot(time, log["phase_hat"], color="tab:blue", ls="--",
                    alpha=0.55, lw=0.8, label=r"$\hat{\phi}$")
        ax.plot(time, log["phase_dot"], color="tab:purple", lw=0.8,
                label=r"$\dot{\phi}$")
        ax.set(title=f"Phase dynamics; duration={metrics['duration_s']:.2f} s",
               xlabel="Runtime (s)", ylabel="Phase / phase rate")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        ax = fig.add_subplot(grid[row, 3])
        ax.plot(time, np.linalg.norm(log["f_des"], axis=1), color="black",
                ls="--", lw=1.0, label="Force reference")
        ax.plot(time, np.linalg.norm(log["f_act"], axis=1),
                color="tab:orange", lw=0.9, label="Material reaction")
        if np.any(disturbed):
            ax.axvspan(time[disturbed][0], time[disturbed][-1],
                       color="tab:red", alpha=0.09, label="Holdback interval")
        ax.set(title="Interaction-force magnitude",
               xlabel="Runtime (s)", ylabel="Force (N)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle(
        f"Deterministic {workpiece_name.lower()} execution: "
        "state/time-driven phase × holdback off/on\n"
        "All 3-D panels use the same canonical reference and identical limits",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    try:
        Logger.setLevel(logging.ERROR)
    except Exception:
        pass
    all_metrics = {}
    figure_dir = os.path.join(_ROOT, "results", "figures")
    os.makedirs(figure_dir, exist_ok=True)

    for workpiece_name, moving, suffix in (
        ("Static workpiece", False, "static"),
        ("Moving workpiece", True, "moving"),
    ):
        runs = []
        all_metrics[suffix] = {}
        for case_name, state_enabled, holdback in CASES:
            log, metrics, dt = run_case(
                material="cork",
                moving=moving,
                seed=0,
                state_phase_enabled=state_enabled,
                dr_state=None,
                holdback_force=holdback,
                randomize=False,
            )
            runs.append((case_name, log, metrics, dt))
            key = (
                ("state" if state_enabled else "time")
                + ("_holdback" if holdback else "_nominal")
            )
            all_metrics[suffix][key] = metrics
            print(
                f"{suffix:6s} {key:16s} done={metrics['completion']:.0f} "
                f"duration={metrics['duration_s']:.3f}s "
                f"depth={metrics['depth_rmse_mm']:.3f}mm "
                f"position={metrics['position_rmse_mm']:.3f}mm",
                flush=True,
            )

        output = os.path.join(
            figure_dir, f"task_execution_four_case_{suffix}.png"
        )
        plot_workpiece(workpiece_name, runs, output)
        print(f"saved figure: {output}")

    result_path = os.path.join(
        _ROOT, "results", "task_execution_four_case_deterministic.json"
    )
    with open(result_path, "w") as handle:
        json.dump(all_metrics, handle, indent=2)
    print(f"saved metrics: {result_path}")


if __name__ == "__main__":
    main()
