"""
Turn a saved study JSON (from run_experiments.py) into a thesis comparison figure:
grouped bar charts with error bars across controller conditions.

Usage:
    python experiments_eval/plot_results.py results/study_killer_cork.json
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# metrics to show and whether lower-is-better (for annotation)
PANELS = [
    ("z_depth_rmse_mm",   "Cut-depth Z RMSE (mm)\n[HEADLINE; lower better]"),
    ("precw_err_mm",      "Precision-weighted err (mm)\n[task-relevant; lower better]"),
    ("y_rmse_mm",         "Lateral Y RMSE (mm)\n[lower better]"),
    ("force_peak_N",      "Peak contact force (N)\n[lower better, safety]"),
    ("tau_rms_Nm",        "Torque RMS (N·m)\n[lower better, effort]"),
    ("divergence_rate",   "Divergence rate\n[lower better, robustness]"),
]
ORDER = ["learned_true", "learned_inverted", "learned_shuffled", "naive_force"]
COLORS = {"learned_true": "#2c7fb8", "learned_inverted": "#d95f02",
          "learned_shuffled": "#7570b3", "naive_force": "#969696"}


def main(path):
    with open(path) as fh:
        data = json.load(fh)
    summary, raw = data["summary"], data["raw"]
    conds = [c for c in ORDER if c in summary] + \
            [c for c in summary if c not in ORDER]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (key, title) in zip(axes.ravel(), PANELS):
        mus, sds, labels, colors = [], [], [], []
        for c in conds:
            if key in summary[c]:
                mu, sd, n = summary[c][key]
                mus.append(mu); sds.append(sd)
                labels.append(c.replace("learned_", "")); colors.append(COLORS.get(c, "#444"))
        x = np.arange(len(mus))
        ax.bar(x, mus, yerr=sds, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    mat = data["args"].get("material", "?")
    seeds = data["args"].get("seeds", "?")
    fig.suptitle(f"Learned-variability killer experiment — {mat} "
                 f"(N={seeds} randomized-physics seeds, paired)\n"
                 f"True precision vs inverted / shuffled / naive-force baseline",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = path.replace("results/", "results/figures/").replace(".json", ".png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved figure:", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/study_killer_cork.json")
