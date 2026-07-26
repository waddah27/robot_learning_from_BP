"""Generate matched nominal execution traces for static and moving workpieces."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from experiments_eval.task_parametrization_comparison import run_case


def main():
    dr_state = np.random.default_rng(3600).bit_generator.state
    runs = {}
    for name, moving in (("Static workpiece", False), ("Moving workpiece", True)):
        log, metrics, dt = run_case(
            "cork", moving, seed=0, state_phase_enabled=True,
            dr_state=dr_state, holdback_force=0.0,
        )
        runs[name] = (log, metrics, dt)
        print(
            f"{name}: completion={metrics['completion']:.0f} "
            f"depth={metrics['depth_rmse_mm']:.3f}mm "
            f"position={metrics['position_rmse_mm']:.3f}mm "
            f"duration={metrics['duration_s']:.3f}s",
            flush=True,
        )

    fig = plt.figure(figsize=(18, 8.5))
    grid = fig.add_gridspec(2, 4)
    for row, (name, (log, metrics, dt)) in enumerate(runs.items()):
        phase = log["phase"]
        t = np.arange(len(phase)) * dt
        desired = log["pos_des_nom"]
        actual = log["pos_act"]
        error = (desired - actual) * 1e3

        ax3d = fig.add_subplot(grid[row, 0], projection="3d")
        ax3d.plot(*desired.T, color="black", ls="--", lw=1.2, label="Reference")
        ax3d.plot(*actual.T, color="tab:blue", lw=1.0, label="Actual")
        ax3d.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
                 title=f"{name}: Cartesian execution")
        ax3d.legend(fontsize=7)

        ax = fig.add_subplot(grid[row, 1])
        for axis, label, color in zip(range(3), ("x", "y", "z"),
                                      ("tab:red", "tab:green", "tab:blue")):
            ax.plot(phase, error[:, axis], color=color, lw=0.9, label=label)
        ax.set(title=f"Tracking error; depth RMSE={metrics['depth_rmse_mm']:.2f} mm",
               xlabel=r"Phase $\phi$", ylabel="Error (mm)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        ax = fig.add_subplot(grid[row, 2])
        ax.plot(t, np.linalg.norm(log["f_des"], axis=1), color="black",
                ls="--", lw=1.1, label="Reference")
        ax.plot(t, np.linalg.norm(log["f_act"], axis=1), color="tab:orange",
                lw=1.0, label="Measured")
        ax.set(title="Interaction force", xlabel="Runtime (s)",
               ylabel="Magnitude (N)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

        ax = fig.add_subplot(grid[row, 3])
        ax.plot(t, phase, color="tab:blue", lw=1.1, label=r"$\phi$")
        ax.plot(t, log["phase_hat"], color="tab:blue", ls="--",
                alpha=0.55, lw=0.8, label=r"$\hat{\phi}$")
        ax.plot(t, log["tank_gamma"], color="tab:purple", lw=0.9,
                label=r"$\gamma_T$")
        ax.set(title=f"Timing and tank gate; duration={metrics['duration_s']:.2f} s",
               xlabel="Runtime (s)", ylabel="Normalized state")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Nominal state-driven cutting execution: static and moving workpieces",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output = os.path.join(
        _ROOT, "results", "figures", "task_execution_static_moving.png"
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
