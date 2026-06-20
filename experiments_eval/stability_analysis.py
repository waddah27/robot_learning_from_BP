"""
Stability & passivity visualization for the learnt-skill controller.

Runs the cutting task (static AND moving material) headless, then computes from
the controller's own logged signals:
  - tracking error norm ||e(t)|| and cut-depth error e_z(t)
  - an exponential fit  ||e|| ~ c*exp(-lambda*t)  on the post-contact settling
    window  ->  identified decay rate  lambda_hat   (exponential-stability metric)
  - Lyapunov function V(t) = 1/2 m||e_dot||^2 + 1/2 e^T diag(K) e   (impedance
    energy; scalar effective mass) -> should decay/stay bounded
  - passivity residual  Vdot - f_ext . x_dot  -> must stay <= 0

Produces results/figures/stability_<material>.png for static and moving.

Usage:
    python experiments_eval/stability_analysis.py
"""
import os, sys, warnings, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mjModeling"))
warnings.filterwarnings("ignore"); logging.getLogger().setLevel(logging.ERROR)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mjModeling.conf import ROBOT_SCENE
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

_FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "figures")
M_EFF = 2.0   # effective mass used in the controller's damping design


def run(material="cork", moving=False):
    wp = Material().from_work_piece(); wp.bp_data = material
    wp.is_movable = moving
    robot = iiwa14().create(xml_path=ROBOT_SCENE, work_piece=wp)
    vic = ContinuousTrajectoryVIC(robot, use_behaviour_priors=True,
                                  optimizer=ImpedanceOptimizer.qp)
    vic.use_learned_gains = True; vic.variance_mode = "true"
    exp = straightCutting(robot); exp.controller = vic
    exp.execute(viewer=None)
    L = vic.episode_log
    return {k: np.array(v) for k, v in L.items()}, vic.dt


def fit_exponential(t, y):
    """Fit y ~ c*exp(-lambda t) on the segment after the peak (settling)."""
    i0 = int(np.argmax(y))                      # start at the error peak
    tt, yy = t[i0:], y[i0:]
    if len(tt) < 10:
        return None
    yy = np.maximum(yy, 1e-9)
    # robust linear fit of log(y) vs t over the decaying window
    A = np.vstack([tt - tt[0], np.ones_like(tt)]).T
    slope, intercept = np.linalg.lstsq(A, np.log(yy), rcond=None)[0]
    lam = -slope
    fit = np.exp(intercept) * np.exp(-lam * (tt - tt[0]))
    ss_res = np.sum((np.log(yy) - (slope * (tt - tt[0]) + intercept)) ** 2)
    ss_tot = np.sum((np.log(yy) - np.log(yy).mean()) ** 2) + 1e-12
    r2 = 1 - ss_res / ss_tot
    return dict(t0=tt[0], tt=tt, fit=fit, lam=lam, r2=r2)


def analyze(log, dt, label):
    n = len(log["phase"]); t = np.arange(n) * dt
    e = log["pos_des_nom"] - log["pos_act"]            # (n,3)
    e_norm = np.linalg.norm(e, axis=1)
    e_z = np.abs(e[:, 2])
    K = log["kp"]                                      # (n,3)
    fext = log["f_act"]                                # (n,3) contact reaction
    # velocities by finite difference
    xdot = np.gradient(log["pos_act"], dt, axis=0)
    edot = np.gradient(e, dt, axis=0)
    # Lyapunov (impedance energy, scalar effective mass)
    V = 0.5 * M_EFF * np.sum(edot ** 2, axis=1) + 0.5 * np.sum(K * e ** 2, axis=1)
    Vdot = np.gradient(V, dt)
    passivity_resid = Vdot - np.sum(fext * xdot, axis=1)
    fit = fit_exponential(t, e_norm)
    return dict(t=t, e_norm=e_norm, e_z=e_z, V=V, resid=passivity_resid, fit=fit,
                label=label)


def plot(panels, fname):
    fig, ax = plt.subplots(len(panels), 3, figsize=(15, 4.2 * len(panels)))
    if len(panels) == 1:
        ax = ax[None, :]
    for r, P in enumerate(panels):
        t = P["t"]
        # (r,0) error + exponential fit
        ax[r, 0].plot(t, P["e_norm"] * 1e3, lw=1, label="‖e(t)‖")
        ax[r, 0].plot(t, P["e_z"] * 1e3, lw=1, alpha=0.7, label="cut-depth |e_z|")
        if P["fit"]:
            f = P["fit"]
            ax[r, 0].plot(f["tt"], f["fit"] * 1e3, "r--", lw=2,
                          label=f"exp fit  λ̂={f['lam']:.2f} 1/s (R²={f['r2']:.2f})")
        ax[r, 0].set_title(f"[{P['label']}] tracking error + exponential fit")
        ax[r, 0].set_xlabel("time (s)"); ax[r, 0].set_ylabel("error (mm)")
        ax[r, 0].legend(fontsize=8)
        # (r,1) Lyapunov V(t)
        ax[r, 1].plot(t, P["V"], lw=1, color="purple")
        ax[r, 1].set_title(f"[{P['label']}] Lyapunov V(t)  (impedance energy)")
        ax[r, 1].set_xlabel("time (s)"); ax[r, 1].set_ylabel("V")
        # (r,2) passivity residual
        ax[r, 2].plot(t, P["resid"], lw=0.8, color="teal")
        ax[r, 2].axhline(0, color="k", lw=0.8, ls=":")
        viol = float(np.mean(P["resid"] > 1e-6) * 100)
        ax[r, 2].set_title(f"[{P['label']}] passivity residual V̇−fₑₓₜ·ẋ\n"
                           f"(>0 only {viol:.1f}% of time)")
        ax[r, 2].set_xlabel("time (s)"); ax[r, 2].set_ylabel("residual")
    fig.tight_layout()
    os.makedirs(_FIG, exist_ok=True)
    out = os.path.join(_FIG, fname)
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    panels = []
    for moving, lab in [(False, "STATIC material"), (True, "MOVING material")]:
        log, dt = run("cork", moving=moving)
        P = analyze(log, dt, lab)
        panels.append(P)
        f = P["fit"]
        print(f"[{lab}] mean ‖e‖={P['e_norm'].mean()*1e3:.1f}mm  "
              f"cut-depth={P['e_z'].mean()*1e3:.1f}mm  "
              f"λ̂={f['lam']:.2f}/s (R²={f['r2']:.2f})  "
              f"passivity-violation={np.mean(P['resid']>1e-6)*100:.1f}%", flush=True)
    out = plot(panels, "stability_cork.png")
    print("saved figure:", out)
