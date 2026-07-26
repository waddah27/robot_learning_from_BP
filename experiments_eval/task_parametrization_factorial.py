"""Full runtime validation of task parametrization.

Factorial design:
    workpiece: static / moving
    disturbance: disabled / path-opposing holdback
    phase law: state-driven / time-driven

Within each scenario and seed, both phase laws receive identical randomized
physics.  The experiment reports nominal performance separately from the
deliberate desynchronization validation.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from experiments_eval.task_parametrization_comparison import (
    _paired_summary,
    run_case,
)

_RESULT_DIR = os.path.join(_ROOT, "results")
_FIG_DIR = os.path.join(_RESULT_DIR, "figures")

SCENARIOS = [
    ("static_nominal", False, 0.0),
    ("static_perturbed", False, 120.0),
    ("moving_nominal", True, 0.0),
    ("moving_perturbed", True, 120.0),
]
PHASE_CASES = {"state_driven": True, "time_driven": False}
COLORS = {"state_driven": "tab:blue", "time_driven": "tab:orange"}
LABELS = {"state_driven": "State-driven", "time_driven": "Time-driven"}


def plot_runtime(representative, output):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2), sharey=True)
    titles = {
        "static_nominal": "Static workpiece — nominal",
        "static_perturbed": "Static workpiece — 120 N holdback",
        "moving_nominal": "Moving workpiece — nominal",
        "moving_perturbed": "Moving workpiece — 120 N holdback",
    }
    for ax, (scenario, _, _) in zip(axes.flat, SCENARIOS):
        for phase_case in PHASE_CASES:
            log, dt = representative[scenario][phase_case]
            t = np.arange(len(log["phase"])) * dt
            ax.plot(t, log["phase"], color=COLORS[phase_case], lw=1.4,
                    label=LABELS[phase_case])
            if phase_case == "state_driven":
                ax.plot(t, log["phase_hat"], color=COLORS[phase_case],
                        lw=0.8, ls="--", alpha=0.55,
                        label=r"Observed $\hat{\phi}$")
            disturbed = log["holdback_force"] > 0.0
            if np.any(disturbed):
                ax.axvspan(t[disturbed][0], t[disturbed][-1],
                           color="tab:red", alpha=0.09)
        ax.set_title(titles[scenario])
        ax.set_xlabel("Runtime (s)")
        ax.set_ylabel(r"Task phase $\phi$")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Runtime phase evolution under workpiece motion and controlled holdback",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_performance(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    metric_specs = [
        ("depth_rmse_mm", "Cut-depth RMSE (mm)"),
        ("position_rmse_mm", "Cartesian position RMSE (mm)"),
        ("duration_s", "Execution duration (s)"),
        ("mean_phase_rate", r"Mean phase rate (s$^{-1}$)"),
    ]
    scenario_labels = ["Static\nnominal", "Static\nholdback",
                       "Moving\nnominal", "Moving\nholdback"]
    x = np.arange(len(SCENARIOS))
    width = 0.35
    for ax, (metric, ylabel) in zip(axes.flat, metric_specs):
        for offset, phase_case in zip((-width / 2, width / 2), PHASE_CASES):
            means = []
            stds = []
            for scenario, _, _ in SCENARIOS:
                values = np.asarray(
                    [row[metric] for row in results[scenario][phase_case]],
                    dtype=float,
                )
                means.append(values.mean())
                stds.append(values.std())
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   color=COLORS[phase_case], alpha=0.75,
                   label=LABELS[phase_case])
        ax.set_xticks(x, scenario_labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Task-parametrization performance: paired five-seed factorial study",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--material", default="cork")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    results = {
        scenario: {case: [] for case in PHASE_CASES}
        for scenario, _, _ in SCENARIOS
    }
    representative = {scenario: {} for scenario, _, _ in SCENARIOS}

    for seed in range(args.seeds):
        master = np.random.default_rng(3600 + seed)
        dr_state = master.bit_generator.state
        for scenario, moving, holdback in SCENARIOS:
            for phase_case, enabled in PHASE_CASES.items():
                log, metrics, dt = run_case(
                    args.material, moving, seed, enabled, dr_state, holdback
                )
                results[scenario][phase_case].append(metrics)
                if seed == 0:
                    representative[scenario][phase_case] = (log, dt)
                print(
                    f"seed={seed:02d} {scenario:18s} {phase_case:12s} "
                    f"done={metrics['completion']:.0f} "
                    f"duration={metrics['duration_s']:.3f}s "
                    f"depth={metrics['depth_rmse_mm']:.3f}mm "
                    f"position={metrics['position_rmse_mm']:.3f}mm",
                    flush=True,
                )

    summary = {
        scenario: _paired_summary(results[scenario])
        for scenario, _, _ in SCENARIOS
    }
    os.makedirs(_RESULT_DIR, exist_ok=True)
    os.makedirs(_FIG_DIR, exist_ok=True)
    result_path = os.path.join(
        _RESULT_DIR, f"task_parametrization_factorial_{args.material}.json"
    )
    runtime_path = os.path.join(
        _FIG_DIR, f"task_parametrization_runtime_{args.material}.png"
    )
    performance_path = os.path.join(
        _FIG_DIR, f"task_parametrization_performance_{args.material}.png"
    )
    with open(result_path, "w") as handle:
        json.dump(
            {"design": vars(args), "raw": results, "summary": summary},
            handle, indent=2,
        )
    plot_runtime(representative, runtime_path)
    plot_performance(results, performance_path)
    print(f"saved results:     {result_path}")
    print(f"saved runtime:     {runtime_path}")
    print(f"saved performance: {performance_path}")


if __name__ == "__main__":
    main()
