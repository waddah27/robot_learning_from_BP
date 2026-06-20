"""
Learned-variability control: derive per-phase, per-axis controller gains directly
from the demonstration variability (minimal-intervention principle).

Core idea (Calinon, task-parametrized GMM): the controller stiffness in a given
direction should be the PRECISION (inverse variance) of the demonstrations in
that direction. Directions the human reproduced consistently (low variance) are
tracked stiffly; directions they did not (high variance) are released to
compliance. We apply this to a contact-rich cutting task and show that the data
itself releases the FORCE dimension while keeping POSITION stiff.

All channels are range-normalized to a common dimensionless scale BEFORE computing
precision, and a SINGLE shared precision normalization is used across all six
channels. This is what lets "position tight / force loose" *emerge* from the data
rather than being imposed: force precision is genuinely tiny on the shared scale.

Outputs (per material):
  - results/profiles/variance_gains_<mat>.npz   (phase, K_pos[φ,3], q_force[φ,3], ...)
  - results/figures/variance_gains_<mat>.png     (thesis figure)
"""
import os
import glob
import numpy as np

# --- channel layout in the raw CSVs: idx,X,Y,Z,Rx,Ry,Rz,Fx,Fy,Fz,Tx,Ty,Tz ---
_POS_FORCE_COLS = [1, 2, 3, 7, 8, 9]            # X Y Z Fx Fy Fz
CHANNELS = ["X", "Y", "Z", "Fx", "Fy", "Fz"]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEMO_DIR = os.path.join(_ROOT, "data", "CleanData", "grouped", "cleanWPprofiles")
_PROF_DIR = os.path.join(_ROOT, "results", "profiles")
_FIG_DIR = os.path.join(_ROOT, "results", "figures")

# Material name -> demo file prefix
MATERIALS = {"cork": "corck_parallel", "pvc": "pvc_parallel", "peno": "peno_parallel"}

# Stiffness bounds (match mjModeling/conf/configs.py paramVIC)
K_MIN, K_MAX = 100.0, 2000.0
# Task-feasibility stiffness FLOOR for position axes. The raw precision can drive
# weakly-regulated axes toward K_MIN, leaving the tool unable to hold a path under
# lateral contact forces. A floor guarantees the motion is executable while the
# precision still modulates stiffness ABOVE it (regulated axes get more). This is
# a documented regularization, not per-result tuning: the emergent ordering of
# axes (which is stiffer) is unchanged by the floor.
K_FLOOR = 600.0

# Number of phase samples in the published profile
N_PHASE = 300
# pre-contact window used to de-bias force (samples)
_BASELINE_W = 20
_EPS = 1e-9


def load_demos(material: str, n_phase: int = N_PHASE) -> np.ndarray:
    """Return (n_trials, n_phase, 6) array of [X,Y,Z,Fx,Fy,Fz], force de-biased."""
    prefix = MATERIALS[material]
    files = sorted(glob.glob(os.path.join(_DEMO_DIR, f"{prefix}_iter*.csv")))
    trials = []
    for f in files:
        d = np.genfromtxt(f, delimiter=",", skip_header=1)
        if d.shape[0] >= n_phase:
            trials.append(d[:n_phase, _POS_FORCE_COLS])
    T = np.asarray(trials)                                  # (n,phase,6)
    # de-bias force: subtract each trial's pre-contact baseline (force cols only)
    base = T[:, :_BASELINE_W, 3:].mean(axis=1, keepdims=True)
    T[:, :, 3:] = T[:, :, 3:] - base
    return T


def compute_precision_weights(T: np.ndarray):
    """
    Derive dimensionless precision weights w[φ,channel] in [0,1] on a shared scale.

    Returns dict with: phase, mean_traj, std_traj (cross-trial), w (precision
    weight), and the channel dynamic ranges used for normalization.
    """
    n_phase = T.shape[1]
    phase = np.linspace(0.0, 1.0, n_phase)

    mean_traj = T.mean(axis=0)                              # (phase,6)
    std_traj = T.std(axis=0)                                # (phase,6) cross-trial

    # range-normalize each channel to a common O(1) dimensionless scale
    rng = mean_traj.max(0) - mean_traj.min(0)               # (6,)
    rng = np.where(rng < _EPS, _EPS, rng)
    std_norm = std_traj / rng                               # dimensionless inconsistency

    # precision on the shared dimensionless scale
    precision = 1.0 / (std_norm ** 2 + _EPS)                # (phase,6)

    # SINGLE shared normalization across ALL channels -> force's genuinely low
    # precision stays low (this is what makes "force loose" emerge from data)
    p_ref = np.percentile(precision, 95)                    # robust upper scale
    w = np.clip(precision / (p_ref + _EPS), 0.0, 1.0)       # (phase,6) in [0,1]

    return {
        "phase": phase,
        "mean_traj": mean_traj,
        "std_traj": std_traj,
        "std_norm": std_norm,
        "precision": precision,
        "w": w,
        "range": rng,
    }


def derive_gains(material: str, n_phase: int = N_PHASE):
    """
    Full pipeline: demos -> precision weights -> position stiffness K_pos(φ) and
    force-tracking weights q_force(φ). Saves an .npz profile and returns the dict.
    """
    T = load_demos(material, n_phase)
    res = compute_precision_weights(T)
    w = res["w"]                                            # (phase,6)
    w_pos = w[:, 0:3]                                       # position weights
    w_force = w[:, 3:6]                                     # force weights

    # Position stiffness scales with precision weight ABOVE a task-feasibility
    # floor (tight where consistent; never below K_FLOOR so the path is holdable)
    K_pos = K_FLOOR + (K_MAX - K_FLOOR) * w_pos             # (phase,3) N/m
    # Force-tracking weight is the (low) force precision weight (loose envelope)
    q_force = w_force                                       # (phase,3) in [0,1]

    res.update({"material": material, "n_trials": T.shape[0],
                "K_pos": K_pos, "q_force": q_force})

    os.makedirs(_PROF_DIR, exist_ok=True)
    out = os.path.join(_PROF_DIR, f"variance_gains_{material}.npz")
    np.savez(out, phase=res["phase"], K_pos=K_pos, q_force=q_force,
             w_pos=w_pos, w_force=w_force, std_norm=res["std_norm"],
             mean_traj=res["mean_traj"])
    res["_path"] = out
    return res


def load_profile(material: str):
    """Load a previously derived gain profile for use by the controller."""
    path = os.path.join(_PROF_DIR, f"variance_gains_{material}.npz")
    return np.load(path)


def plot_profiles(res: dict, show: bool = False):
    """Thesis figure: cross-trial consistency and the emergent gain split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phase = res["phase"]
    w_pos, w_force = res["K_pos"], res["q_force"]
    mat = res["material"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    axes_lbl = ["X", "Y", "Z"]

    # (0,0) normalized cross-trial inconsistency: position vs force
    for i, l in enumerate(axes_lbl):
        ax[0, 0].plot(phase, res["std_norm"][:, i], label=f"pos {l}")
    for i, l in enumerate(axes_lbl):
        ax[0, 0].plot(phase, res["std_norm"][:, i + 3], "--", label=f"force {l}")
    ax[0, 0].set_title("Cross-trial inconsistency (normalized std)")
    ax[0, 0].set_xlabel("phase φ"); ax[0, 0].set_ylabel("σ̃ (dimensionless)")
    ax[0, 0].legend(fontsize=7, ncol=2); ax[0, 0].set_yscale("log")

    # (0,1) emergent precision weights w in [0,1]
    for i, l in enumerate(axes_lbl):
        ax[0, 1].plot(phase, res["w"][:, i], label=f"pos {l}")
    for i, l in enumerate(axes_lbl):
        ax[0, 1].plot(phase, res["w"][:, i + 3], "--", label=f"force {l}")
    ax[0, 1].set_title("Emergent precision weight w(φ)  [shared scale]")
    ax[0, 1].set_xlabel("phase φ"); ax[0, 1].set_ylabel("w ∈ [0,1]")
    ax[0, 1].legend(fontsize=7, ncol=2)

    # (1,0) derived position stiffness
    for i, l in enumerate(axes_lbl):
        ax[1, 0].plot(phase, res["K_pos"][:, i], label=f"K_{l}")
    ax[1, 0].set_title("Derived position stiffness K_pos(φ)")
    ax[1, 0].set_xlabel("phase φ"); ax[1, 0].set_ylabel("N/m")
    ax[1, 0].legend(fontsize=8)

    # (1,1) derived force-tracking weight
    for i, l in enumerate(axes_lbl):
        ax[1, 1].plot(phase, res["q_force"][:, i], label=f"q_F{l}")
    ax[1, 1].set_title("Derived force-tracking weight q_force(φ)  (≈0 ⇒ released)")
    ax[1, 1].set_xlabel("phase φ"); ax[1, 1].set_ylabel("weight ∈ [0,1]")
    ax[1, 1].legend(fontsize=8); ax[1, 1].set_ylim(-0.02, 1.02)

    fig.suptitle(f"Learned-variability gains — {mat} "
                 f"(n={res['n_trials']} demos)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(_FIG_DIR, exist_ok=True)
    out = os.path.join(_FIG_DIR, f"variance_gains_{mat}.png")
    fig.savefig(out, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return out


if __name__ == "__main__":
    for mat in MATERIALS:
        res = derive_gains(mat)
        figp = plot_profiles(res)
        wpos = res["w"][:, 0:3].mean()
        wforce = res["w"][:, 3:6].mean()
        print(f"[{mat}] n={res['n_trials']}  "
              f"mean w_pos={wpos:.3f}  mean w_force={wforce:.4f}  "
              f"-> K_pos≈{res['K_pos'].mean():.0f} N/m, q_force≈{res['q_force'].mean():.4f}")
        print(f"        profile: {res['_path']}")
        print(f"        figure : {figp}")
