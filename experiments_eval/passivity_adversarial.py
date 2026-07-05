"""
Adversarial passivity validation for the learned-variability cutting controller.

This script is deliberately stronger than a nominal stability run. It injects a
known positive joint-power burst into the nominal torque command during the cut,
then compares:

  1. no_tank      : no passivity projection, no phase gate
  2. tank_only    : live energy-tank torque projection, no phase gate
  3. tank_phase   : torque projection plus tank-gated phase advance

The proof artifact is the torque-port inequality:

    qdot.T @ tau_safe <= P_max = (T - T_min) / dt

The no_tank case should violate the counterfactual budget during the injected
burst. The tanked cases should keep the commanded torque power below the live
budget. The Cartesian residual is intentionally not used as the proof because it
depends on finite-difference velocities, force-estimator conventions, and an
approximate Cartesian storage model.
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

Logger.configure(log_file="/tmp/passivity_adversarial_controller.log",
                 level=Logger.ERROR)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIG = os.path.join(_ROOT, "results", "figures")


CASES = [
    {
        "name": "no_tank",
        "label": "No Tank",
        "passivity": False,
        "phase_gate": False,
    },
    {
        "name": "tank_only",
        "label": "Tank Projection",
        "passivity": True,
        "phase_gate": False,
    },
    {
        "name": "tank_phase",
        "label": "Tank + Phase Gate",
        "passivity": True,
        "phase_gate": True,
    },
]


def run_case(case, material="cork", moving=False):
    wp = Material().from_work_piece()
    wp.bp_data = material
    wp.is_movable = moving

    robot = iiwa14().create(xml_path=ROBOT_SCENE, work_piece=wp)
    vic = ContinuousTrajectoryVIC(robot, use_behaviour_priors=True,
                                  optimizer=ImpedanceOptimizer.qp)
    vic.working_piece = wp
    vic.use_learned_gains = True
    vic.variance_mode = "true"

    # Low-but-positive initial tank makes the budget visible on the plot and
    # ensures the injected burst is genuinely challenging.
    vic.tank_initial = 1.0
    vic.tank_energy = vic.tank_initial
    vic.tank_max = 3.0
    vic.T_max = vic.tank_max
    vic.tank_ref = 0.5
    vic.passivity_enabled = case["passivity"]
    vic.phase_gating_enabled = case["phase_gate"]

    # Positive joint power injected before _solve_passivity_qp. The tanked
    # controllers must project it back below P_max.
    vic.adversarial_power_enabled = True
    vic.adversarial_power_start = 0.25
    vic.adversarial_power_end = 0.75
    vic.adversarial_power_watts = 7000.0

    exp = straightCutting(robot)
    exp.controller = vic
    exp.execute(viewer=None)
    log = {k: np.asarray(v) for k, v in vic.episode_log.items()}
    dt = vic.dt
    try:
        robot.shutdown()
    except Exception:
        pass
    return log, dt


def summarize(log, tol=1e-3):
    power_safe = log["power_safe"]
    power_limit = log["power_limit"]
    margin = power_safe - power_limit
    finite = np.isfinite(margin)
    if not finite.any():
        violation_frac = float("nan")
        max_margin = float("nan")
    else:
        violation_frac = float(np.mean(margin[finite] > tol))
        max_margin = float(np.max(margin[finite]))

    balance = np.asarray(log.get("tank_balance_residual", []), dtype=float)
    return {
        "completion": float(log["phase"][-1] >= 0.999),
        "phase_final": float(log["phase"][-1]),
        "tank_min": float(np.min(log["tank"])),
        "gamma_min": float(np.min(log["tank_gamma"])),
        "violation_frac": violation_frac,
        "max_margin": max_margin,
        "max_adversarial_power": float(np.max(log["adversarial_power"])),
        "tank_balance_clip_max_abs": float(np.max(np.abs(balance))) if len(balance) else float("nan"),
    }


def plot(results, fname="passivity_adversarial_cork.png"):
    fig, ax = plt.subplots(len(results), 4, figsize=(20, 3.7 * len(results)),
                           sharex=False)
    if len(results) == 1:
        ax = ax[None, :]

    for row, (case, log, dt, metrics) in enumerate(results):
        t = np.arange(len(log["phase"])) * dt
        margin = log["power_safe"] - log["power_limit"]

        ax[row, 0].plot(t, log["tank"], color="darkorange", lw=1.1)
        ax[row, 0].axhline(0.001, color="black", ls=":", lw=0.8)
        ax[row, 0].set_title(f"{case['label']}: tank energy")
        ax[row, 0].set_ylabel("T (J)")

        inj = log["adversarial_power"] > 1.0
        if np.any(inj):
            t0, t1 = t[inj][0], t[inj][-1]
            for col in range(4):
                ax[row, col].axvspan(t0, t1, color="tab:red", alpha=0.08)

        ax[row, 1].plot(t, log["power_limit"], color="black", lw=1.2, label="P_max")
        ax[row, 1].plot(t, log["power_nominal"], color="tab:red", lw=0.8,
                        alpha=0.75, label="qdot.T tau_nom")
        ax[row, 1].plot(t, log["power_safe"], color="tab:green", lw=0.9,
                        label="qdot.T tau_safe")
        ax[row, 1].set_title("implemented torque-port inequality")
        ax[row, 1].set_ylabel("power (W)")
        ax[row, 1].legend(fontsize=7)

        ax[row, 2].plot(t, margin, color="tab:blue", lw=0.9)
        ax[row, 2].axhline(0.0, color="black", ls=":", lw=0.8)
        ax[row, 2].set_title(
            f"margin = safe - limit\n"
            f"viol={metrics['violation_frac']*100:.1f}%, max={metrics['max_margin']:.1f} W"
        )
        ax[row, 2].set_ylabel("W")

        ax[row, 3].plot(t, log["tank_gamma"], color="tab:purple", lw=1.0,
                        label="phase gate gamma")
        ax[row, 3].plot(t, log["phase"], color="tab:gray", lw=1.0, label="phase")
        ax[row, 3].set_ylim(-0.05, 1.05)
        ax[row, 3].set_title("phase response")
        ax[row, 3].legend(fontsize=7)

        for col in range(4):
            ax[row, col].set_xlabel("time (s)")

    fig.suptitle("Adversarial passivity validation at the joint torque port",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(_FIG, exist_ok=True)
    out = os.path.join(_FIG, fname)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main():
    results = []
    for case in CASES:
        log, dt = run_case(case)
        metrics = summarize(log)
        results.append((case, log, dt, metrics))
        print(
            f"{case['name']:10s} completion={metrics['completion']:.0f} "
            f"phase={metrics['phase_final']:.3f} "
            f"tank_min={metrics['tank_min']:.3f}J "
            f"gamma_min={metrics['gamma_min']:.3f} "
            f"viol={metrics['violation_frac']*100:.1f}% "
            f"max_margin={metrics['max_margin']:.1f}W "
            f"tank_clip_resid={metrics['tank_balance_clip_max_abs']:.2e}",
            flush=True,
        )
    out = plot(results)
    print("saved figure:", out)


if __name__ == "__main__":
    main()
