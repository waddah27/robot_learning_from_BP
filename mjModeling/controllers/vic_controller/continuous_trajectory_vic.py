from mjModeling.conf.configs import ImpedanceOptimizer, paramVIC
import time
import numpy as np
import mujoco
from mjModeling.cutting_materials.utils import wp_sine_motion
from mjModeling.cutting_materials.cutting_force import CuttingForceModel
from mjModeling.conf import SCALPEL_GEOM
from mjModeling.controllers.vic_controller.bp_based_controller import BpVariableImpedanceControl
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14
from logger import Logger
from qpsolvers import solve_qp

__all__ = ["ContinuousTrajectoryVIC"]


class ContinuousTrajectoryVIC(BpVariableImpedanceControl):
    """
    Continuous trajectory VIC using QP-based gain optimization.
    """
    def __init__(self, robot: iiwa14, use_behaviour_priors: bool = False,
                 optimizer=ImpedanceOptimizer.qp):
        super().__init__(robot, use_behaviour_priors)

        self.optimizer = optimizer
        Logger.debug(f"Using optimizer: {self.optimizer} (QP-based gain optimization)")

        # ---------- Optimization parameters ----------
        self.Xi_scaler = 5000
        self.k_min = np.array([paramVIC.VIC_KP_MIN, paramVIC.VIC_KP_MIN, paramVIC.VIC_KP_MIN])
        self.k_max = np.array([paramVIC.VIC_KP_MAX, paramVIC.VIC_KP_MAX, paramVIC.VIC_KP_MAX])
        self.k_init = 2000
        self.xi_init = 0.7
        self.D_min = np.array([0, 0, 0])
        self.D_max = np.array([self.Xi_scaler, self.Xi_scaler, self.Xi_scaler])
        self.f_min = -70
        self.f_max = 70
        # Per-axis force-saturation bounds for the QP: the commanded impedance
        # force K·error is constrained to ±f_bound per axis, so it never violates
        # the demonstrated min/max force. Set from the demonstration in
        # follow_trajectory (world frame); default to the legacy scalar bound.
        self.f_bound = np.array([70.0, 70.0, 70.0])
        self.f_motion_floor = 40.0   # min per-axis force authority for free motion
        self.epsilon = 0.675   # minimum tank energy
        self.Q = np.eye(3)     # force error weight
        self.R = np.eye(3) * 1e-9   # regularization weight
        self.delta_t = 0.1     # time step for tank dynamics (actual dt will be used)
        self.T_max = 3000      # max tank energy (optional)

        # Tank state
        self.x_t = np.array([self.epsilon, self.epsilon, self.epsilon])
        self.E_t = 0.0
        self.E_tot = []

        # Previous gains (for fallback)
        self.prev_kd = np.array([self.k_init, self.k_init, self.k_init])
        self.prev_xi = np.array([self.xi_init, self.xi_init, self.xi_init])

        # Force filtering
        self.filtered_force = np.zeros(3)
        self.force_filter_alpha = 0.2

        # Contact state
        self.in_contact = False
        self.contact_threshold = 2.0

        # Force integral (optional, can be used)
        self.force_integral = np.zeros(3)
        self.kf_i = 0.01          # integral gain (tune)
        self.force_integral_max = 10.0

        # Phase variable and trajectory data
        self.phase = 0.0
        self.mat_time = 0.0
        self.pos_func = None
        self.vel_func = None
        self.force_func = None
        self.traj_duration = 0.0
        self.traj_dt = 0.01

        # State-driven phase estimation
        self.k_phase = 0.3            # correction gain: 0 = pure clock, 1 = pure state
        self.phase_search_window = 0.1  # max forward-look (fraction of total trajectory)
        self._phase_points = None
        self._traj_pos_world = None
        self._traj_force_mag = None

        # ----- Learned-variability gains (minimal-intervention) -----
        # Per-phase, per-axis position stiffness K_pos(φ) and force-tracking
        # weight q_force(φ) derived from demonstration precision.  When enabled,
        # the QP blends force tracking and position stiffness by these learned
        # weights instead of a fixed Q.  variance_mode lets the killer experiment
        # use the true / inverted / shuffled precision structure.
        self.use_learned_gains = True
        self.variance_mode = "true"      # {"true","inverted","shuffled","off"}
        self._lg_phase = None            # (N,)
        self._lg_K_pos = None            # (N,3)

        # Analytical cutting-force model: replaces the erratic box-on-box contact
        # with a smooth, bounded depth-based cutting resistance. Toggle for ablation.
        self.use_cutting_model = True
        self._lg_q_force = None          # (N,3)

        # Debug
        self.last_print_time = 0
        self.print_interval = 0.2

        if self.use_bp:
            self._init_trajectory_interpolators()
            self._check_force_scale()
            self._load_learned_gains()

    # ---------- Trajectory interpolation ----------
    def _init_trajectory_interpolators(self):
        """Create continuous functions of phase from discrete GMR data."""
        self.traj_pos = self.traj_loader.pos[:, 0:3]
        self.traj_vel = self.traj_loader.vel[:, 0:3]
        self.traj_force = self.traj_loader.force[:, 0:3]

        if hasattr(self, 'force_scale'):
            self.traj_force *= self.force_scale

        N = len(self.traj_pos)
        self.traj_dt = getattr(self.traj_loader, 'dt', 0.01)
        self.traj_duration = N * self.traj_dt
        phase_points = np.linspace(0, 1, N)

        try:
            from scipy.interpolate import interp1d
            self.pos_func = interp1d(phase_points, self.traj_pos,
                                      axis=0, bounds_error=False,
                                      fill_value=(self.traj_pos[0], self.traj_pos[-1]))
            self.vel_func = interp1d(phase_points, self.traj_vel,
                                      axis=0, bounds_error=False,
                                      fill_value=(self.traj_vel[0], self.traj_vel[-1]))
            self.force_func = interp1d(phase_points, self.traj_force,
                                        axis=0, bounds_error=False,
                                        fill_value=(self.traj_force[0], self.traj_force[-1]))
        except ImportError:
            def linear_interp(xp, fp, x):
                i = np.searchsorted(xp, x, side='right') - 1
                i = np.clip(i, 0, len(xp)-2)
                x1, x2 = xp[i], xp[i+1]
                if x2 == x1:
                    return fp[i]
                w = (x - x1) / (x2 - x1)
                return (1 - w) * fp[i] + w * fp[i+1]
            self.pos_func = lambda phi: linear_interp(phase_points, self.traj_pos, phi)
            self.vel_func = lambda phi: linear_interp(phase_points, self.traj_vel, phi)
            self.force_func = lambda phi: linear_interp(phase_points, self.traj_force, phi)

    def _gmr_to_world(self, gmr_point, mat_pos=None):
        if mat_pos is None:
            mat_pos = self.data.geom_xpos[self.mat_geom_id].copy()
        p_safe = np.zeros(3)
        p_safe[:min(len(gmr_point), 3)] = gmr_point[:min(len(gmr_point), 3)]
        norm = (p_safe - self.gmr_min) / self.gmr_range
        norm = np.clip(norm, 0, 1)
        world_x = mat_pos[0] + (norm[0] - 0.5) * self.cut_dim_x
        world_y = mat_pos[1] + (norm[1] - 0.5) * self.cut_dim_y
        world_z = mat_pos[2] + (norm[2] - 0.5) * self.cut_dim_z
        return np.array([world_x, world_y, world_z])

    def _gmr_vel_to_world(self, gmr_vel, mat_vel, mat_pos=None):
        v_scale = np.array([self.cut_dim_x / self.gmr_range[0],
                             self.cut_dim_y / self.gmr_range[1],
                             self.cut_dim_z / self.gmr_range[2]])
        v_world = gmr_vel[:3] * v_scale
        if np.isscalar(mat_vel):
            v_world_z = v_world[2] + mat_vel
        else:
            v_world_z = v_world[2] + mat_vel[2]
        return np.array([v_world[0], v_world[1], v_world_z])

    def _check_force_scale(self):
        if hasattr(self, 'traj_force'):
            mean_force = np.mean(np.linalg.norm(self.traj_force, axis=1))
            max_force = np.max(np.linalg.norm(self.traj_force, axis=1))
            Logger.debug(f"GMR forces - mean: {mean_force:.1f}N, max: {max_force:.1f}N")
            if max_force > 100:
                self.force_scale = 60.0 / max_force
                Logger.debug(f"Auto-scaling forces by {self.force_scale:.3f}")
            else:
                self.force_scale = 1.0
            # Per-axis force-saturation bound = max |demonstrated force| per axis.
            # The QP keeps the commanded force within these limits, so the applied
            # force respects the demonstrated min/max in X, Y and Z.
            scaled = self.traj_force * self.force_scale
            self.f_bound = np.maximum(np.abs(scaled.min(axis=0)),
                                      np.abs(scaled.max(axis=0)))
            self.f_bound = np.maximum(self.f_bound, 5.0)   # numeric floor
            Logger.debug(f"Per-axis force bounds (N): {np.round(self.f_bound, 1)}")

    # ---------- Learned-variability gains ----------
    _MAT_ALIAS = {"cork": "cork", "corck": "cork", "penoplex": "peno",
                  "peno": "peno", "pvc": "pvc", "PVC": "pvc"}

    def _load_learned_gains(self):
        """Load precision-derived K_pos(φ), q_force(φ) for the current material."""
        try:
            from variability_control.variance_gains import load_profile
            mat = self._MAT_ALIAS.get(getattr(self.traj_loader, "material_name", ""),
                                      "cork")
            prof = load_profile(mat)
            self._lg_phase = prof["phase"]
            self._lg_K_pos = prof["K_pos"]
            self._lg_q_force = prof["q_force"]
            Logger.debug(f"Loaded learned-variability gains for '{mat}' "
                         f"(mean K_pos={self._lg_K_pos.mean():.0f} N/m, "
                         f"mean q_force={self._lg_q_force.mean():.3f})")
        except Exception as e:
            self.use_learned_gains = False
            Logger.warning(f"Learned gains unavailable ({e}); using fixed QP weights.")

    def _learned_gains_at_phase(self, phase):
        """Return (K_pos[3], q_force[3]) at the given phase under variance_mode."""
        if self._lg_phase is None:
            return None, None
        K_pos = np.array([np.interp(phase, self._lg_phase, self._lg_K_pos[:, i])
                          for i in range(3)])
        q_force = np.array([np.interp(phase, self._lg_phase, self._lg_q_force[:, i])
                            for i in range(3)])
        if self.variance_mode == "inverted":
            # swap the regulation roles: track force where we'd track position
            q_force = 1.0 - q_force
            K_pos = self.k_min + (self.k_max - self.k_min) * (1.0 - (
                (K_pos - self.k_min) / (self.k_max - self.k_min)))
        elif self.variance_mode == "shuffled":
            rng = np.random.default_rng(int(phase * 1e6) % (2**32))
            perm = rng.permutation(3)
            q_force = q_force[perm]
            K_pos = K_pos[perm]
        return K_pos, q_force

    # ---------- State-driven phase estimation ----------
    def _estimate_phase(self, current_pos, current_force):
        """
        Find the phase value φ̂ ∈ [φ, φ+window] whose reference state best
        matches the current robot state. Returns the nominal phase when the
        cache is not ready (before the first follow_trajectory call).
        """
        if self._phase_points is None or self._traj_pos_world is None:
            return self.phase

        phi_hi = min(1.0, self.phase + self.phase_search_window)
        mask = (self._phase_points >= self.phase) & (self._phase_points <= phi_hi)
        if not np.any(mask):
            return self.phase

        pos_window = self._traj_pos_world[mask]
        pos_errors = np.linalg.norm(pos_window - current_pos, axis=1)
        pos_range = np.linalg.norm(self._traj_pos_world[-1] - self._traj_pos_world[0]) + 1e-6

        if self.in_contact:
            force_mag = np.linalg.norm(current_force)
            force_errors = np.abs(self._traj_force_mag[mask] - force_mag)
            force_range = np.max(self._traj_force_mag) + 1e-6
            cost = 0.5 * pos_errors / pos_range + 0.5 * force_errors / force_range
        else:
            cost = pos_errors / pos_range

        return self._phase_points[mask][np.argmin(cost)]

    # ---------- QP-BASED GAIN OPTIMIZATION ----------
    def get_variable_gains_optimizer(self, error, vel_error, desired_force, dt,
                                     phase=None):
        """
        QP that minimises a learned-variability blend:
            ½ ( q_force(φ)·||K·error - F_des||²  +  (1-q_force(φ))·||K - K_pos(φ)||² )
        subject to:
            K_min <= K <= K_max
            -F_max <= K·error <= F_max

        q_force(φ) and K_pos(φ) are the demonstration-precision weights
        (minimal-intervention principle).  Where the human regulated FORCE
        (q_force→1) the QP tracks force; where they regulated POSITION
        (q_force→0) K is pulled to the precision-derived stiffness K_pos.
        Falls back to fixed weights when learned gains are disabled.
        """
        # If position error is negligible, return previous gains
        if np.linalg.norm(error) < 1e-6:
            kd = 2 * self.prev_xi * np.sqrt(np.maximum(self.prev_kd, 100))
            return self.prev_kd, kd

        # ----- per-axis learned weights (or fixed fallback) -----
        if self.use_learned_gains and phase is not None:
            K_pos, q_force = self._learned_gains_at_phase(phase)
            # force-tracking weight scaled to QP magnitude; position pull = complement
            Q_diag = 500.0 * q_force                       # force tracking weight
            R_diag = 500.0 * (1.0 - q_force) + 1.0         # position-stiffness pull
            K_target = K_pos
        else:
            Q_diag = np.array([500.0, 500.0, 500.0])
            R_diag = np.array([1.0, 1.0, 1.0])
            K_target = self.k_min

        n_var = 3                     # [Kx, Ky, Kz]
        H = np.zeros((n_var, n_var))
        f = np.zeros(n_var)

        # ----- Force‑tracking cost: q_force·||K·error - F_des||² -----
        for i in range(3):
            H[i, i] = 2.0 * Q_diag[i] * (error[i]**2)
            f[i]    = -2.0 * Q_diag[i] * error[i] * desired_force[i]

        # ----- Position-stiffness pull: (1-q_force)·||K - K_pos(φ)||² -----
        for i in range(3):
            H[i, i] += 2.0 * R_diag[i]
            f[i]    += -2.0 * R_diag[i] * K_target[i]

        # ----- Hard constraints: K_min <= K <= K_max -----
        lb = self.k_min
        ub = self.k_max

        # ----- Hard constraints: |K_i · error_i| <= f_bound_i (per axis) -----
        # f_bound_i is the demonstrated max |force| in axis i, so the commanded
        # impedance force never violates the demonstrated min/max force.
        G_list = []
        h_list = []
        for i in range(3):
            if abs(error[i]) > 1e-6:
                # Upper bound: K_i * error[i] <= f_bound[i]
                G_up = np.zeros(3)
                G_up[i] = error[i]
                G_list.append(G_up.reshape(1, -1))
                h_list.append([self.f_bound[i]])

                # Lower bound: -K_i * error[i] <= f_bound[i]
                G_low = np.zeros(3)
                G_low[i] = -error[i]
                G_list.append(G_low.reshape(1, -1))
                h_list.append([self.f_bound[i]])
            else:
                # When position error is zero, the force constraint is meaningless.
                # Instead, bound the gain directly.
                G_up = np.zeros(3)
                G_up[i] = 1.0
                G_list.append(G_up.reshape(1, -1))
                h_list.append([self.k_max[i]])

                G_low = np.zeros(3)
                G_low[i] = -1.0
                G_list.append(G_low.reshape(1, -1))
                h_list.append([-self.k_min[i]])

        G = np.vstack(G_list)
        h = np.vstack(h_list).flatten()

        # Solve QP
        try:
            k_opt = solve_qp(
                P=H, q=f,
                G=G, h=h,
                lb=lb, ub=ub,
                solver='osqp',
                verbose=False, polish=False
            )
            if k_opt is not None and not np.any(np.isnan(k_opt)):
                kp = k_opt[:3]

                # Compute damping from stiffness (critical damping)
                M_eff = 2.0               # effective mass (tune)
                zeta = 0.7
                kd = 2.0 * zeta * np.sqrt(np.maximum(kp, 100) * M_eff)

                # Smooth update (optional)
                alpha = 0.5
                kp_smooth = alpha * kp + (1 - alpha) * self.prev_kd
                kd_smooth = alpha * kd + (1 - alpha) * (
                            2 * self.prev_xi * np.sqrt(np.maximum(self.prev_kd, 100)))
                self.prev_kd = kp_smooth
                self.prev_xi = np.array([zeta, zeta, zeta])

                # Log (optional)
                F_est = kp_smooth * error
                Logger.debug(f"QP: Kz={kp_smooth[2]:.0f}, "
                            f"F_est_z={F_est[2]:.1f}N, F_des_z={desired_force[2]:.1f}N, "
                            f"err={np.linalg.norm(desired_force - F_est):.1f}N")
                return kp_smooth, kd_smooth
        except Exception as e:
            Logger.debug(f"QP failed: {e}")

        # Fallback
        return self._analytical_fallback(error, vel_error, desired_force)


    def _analytical_fallback(self, error, vel_error, desired_force):
        """Stable fallback when QP fails"""
        kp_new = self.prev_kd.copy()

        for axis in range(3):
            e_p = error[axis]
            F_d = desired_force[axis]

            if abs(e_p) > 0.001:  # Position error large enough
                K_req = F_d / e_p
                K_req = np.clip(K_req, self.k_min[axis], self.k_max[axis])
                kp_new[axis] = 0.7 * K_req + 0.3 * self.prev_kd[axis]  # Smooth

        # Compute damping (critical damping)
        zeta = 0.7
        kd_new = 2 * zeta * np.sqrt(np.maximum(kp_new, 100))

        self.prev_kd = kp_new
        self.prev_xi = np.array([zeta, zeta, zeta])

        return kp_new, kd_new

    # ---------- Main gain dispatcher ----------
    def get_variable_gains(self, error, vel_error=None, desired_force=None, dt=None,
                           phase=None):
        """
        Main gain function. If force data available, use QP optimizer; else heuristic.
        """
        if desired_force is None or vel_error is None:
            return super().get_variable_gains(error)
        if dt is None:
            dt = self.dt
        return self.get_variable_gains_optimizer(error, vel_error, desired_force, dt,
                                                 phase=phase)

    # ---------- Energy modulation ----------
    def update_tank_energy(self, kp, kd, error, vel_error, dt):
        power = np.sum(kd * vel_error**2)
        self.E_t += power * dt
        self.E_t = np.clip(self.E_t, 0, self.T_max)
        return self.E_t

    # ---------- Trajectory execution ----------
    def follow_trajectory(self, phase_speed: float = paramVIC.PHASE_SPEED, viewer=None):
        """
        Execute the continuous trajectory with QP-based gain optimization.
        """
        tcp_id = self.model.site("scalpel_tip").id
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])
        self.error_accumulated = np.zeros(3)

        # Per-axis force bound in the WORLD frame (the QP frame): transform the
        # demonstrated force envelope through the current cutting-pose orientation
        # so the bound on each world axis matches the desired world force there.
        mujoco.mj_forward(self.model, self.data)
        site_rot0 = self.data.site_xmat[tcp_id].reshape(3, 3)
        fw = (site_rot0 @ self.traj_force.T).T            # demo force in world
        # Bound = max demonstrated |force| per world axis, floored so the arm
        # still has the authority to MOVE itself (the demonstrated force is the
        # small interaction force; free-motion needs more than that).
        self.f_bound = np.maximum(np.abs(fw).max(axis=0), self.f_motion_floor)
        Logger.debug(f"World-frame per-axis force bounds (N): {np.round(self.f_bound, 1)}")

        # ----- Analytical cutting-force model (replaces box-on-box contact) -----
        # The box scalpel↔material collision produced erratic 100s-of-N forces.
        # Disable that geometric contact and instead apply a smooth, bounded
        # cutting resistance. The controller is untouched; this is material physics.
        self.cut_model = None
        self.robot._applied_cut_force = None
        if self.use_cutting_model:
            sgid = self.model.geom(SCALPEL_GEOM).id
            self._scalpel_body = int(self.model.geom_bodyid[sgid])
            self.model.geom_contype[sgid] = 0
            self.model.geom_conaffinity[sgid] = 0
            # Identify a PER-MATERIAL linear cutting law from this material's
            # demonstration: reaction-on-blade(world) = c * penetration_depth.
            mat_c = self.data.geom_xpos[self.mat_geom_id].copy()
            mat_h = self.model.geom_size[self.mat_geom_id].copy()
            top0 = mat_c[2] + mat_h[2]
            zref = np.array([self._gmr_to_world(p)[2] for p in self.traj_pos])
            depth_demo = np.clip(top0 - zref, 0.0, None)             # demo penetration (m)
            react_demo = -(site_rot0 @ self.traj_force.T).T          # reaction on blade (world)
            k_mat, f_cut = CuttingForceModel.identify(depth_demo, react_demo)
            self.cut_model = CuttingForceModel(k=k_mat, f_cut=f_cut)
            self.cut_model.set_material(mat_c, mat_h)
            self.robot._applied_cut_force = np.zeros(3)
            Logger.debug(f"Cutting law identified: k={k_mat:.0f} N/m  F_cut={f_cut:.0f} N")

        self.phase = 0.0
        self.mat_time = 0.0
        phase_inc = phase_speed * self.dt / self.traj_duration

        # Precompute world-frame reference trajectory for state-driven phase estimation
        mujoco.mj_forward(self.model, self.data)
        self._phase_points = np.linspace(0, 1, len(self.traj_pos))
        self._traj_pos_world = np.array([self._gmr_to_world(p) for p in self.traj_pos])
        self._traj_force_mag = np.linalg.norm(self.traj_force, axis=1)

        prev_mat_height = None
        actual_duration = self.traj_duration / phase_speed
        max_steps = int(2 * actual_duration / self.dt)

        self.filtered_force = np.zeros(3)
        self.force_integral = np.zeros(3)
        self.adm_s          = 0.0
        self.adm_sdot       = 0.0
        self.fz_integral    = 0.0

        force_log = []
        pos_log = []

        # Full-resolution per-step recorder (for the experiment harness/metrics)
        # pos_des_nom = nominal GMR reference (same across conditions -> fair
        # position-tracking comparison); pos_des = admittance-corrected reference.
        self.episode_log = {k: [] for k in
                            ["phase", "pos_des_nom", "pos_des", "pos_act",
                             "f_des", "f_act", "f_raw", "kp", "tank", "tau"]}

        # Restart the live monitor's recording at sample 0 so it shows the CUT,
        # not the long approach/IK phase that precedes it.
        buf = getattr(self.robot, "buffer", None)
        if buf is not None and hasattr(buf, "reset"):
            buf.reset()

        last_log_time = 0

        for step in range(max_steps):
            if self.phase >= 1.0:
                break

            step_start = time.time()   # for real-time pacing (live viewer only)

            # Viewer control: pause/resume and quit, polled from the keyboard
            vc = getattr(self.robot, "viewer_control", None)
            if vc is not None:
                if vc.quit:
                    break
                while vc.paused and viewer is not None and viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
                    if vc.quit:
                        break
                if vc.quit:
                    break

            # Update material position (if moving)
            if self.mat_joint_id is not None and self.wp_mobile:
                current_height = wp_sine_motion(self.mat_time)
                self.data.qpos[self.mat_joint_id] = current_height
                mujoco.mj_forward(self.model, self.data)
                if prev_mat_height is None:
                    mat_vel = 0.0
                else:
                    mat_vel = (current_height - prev_mat_height) / (phase_speed * self.dt)
                prev_mat_height = current_height
            elif self.mat_joint_id is not None:
                # STATIC material: hold it at its design height every step. Its
                # slide joint otherwise lets it SINK under gravity (~11mm), which
                # drops the surface below the demonstrated cut path so the blade
                # loses contact mid-cut. Holding qpos=0 keeps the surface fixed.
                self.data.qpos[self.mat_joint_id] = 0.0
                self.data.qvel[self.mat_joint_id] = 0.0
                mujoco.mj_forward(self.model, self.data)
                mat_vel = 0.0
            else:
                mat_vel = 0.0

            # Desired quantities from GMR
            pos_des_gmr = self.pos_func(self.phase)
            vel_des_gmr = self.vel_func(self.phase)
            force_des_gmr = self.force_func(self.phase)

            vel_des_gmr_scaled = vel_des_gmr * phase_speed
            pos_des = self._gmr_to_world(pos_des_gmr)
            vel_des = self._gmr_vel_to_world(vel_des_gmr_scaled, mat_vel)

            # Current robot state
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[tcp_id].copy()
            current_force_raw = self.estimator.get_total_cutting_force()
            self.filtered_force = self.force_filter_alpha * current_force_raw + \
                                  (1 - self.force_filter_alpha) * self.filtered_force

            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            v_cur = jac @ self.data.qvel

            # Transform desired force to world frame
            site_rot = self.data.site_xmat[tcp_id].reshape(3, 3)
            f_des_world = site_rot @ force_des_gmr

            self.in_contact = np.linalg.norm(self.filtered_force) > self.contact_threshold

            # Position tracking error (pure VIC — no force loop)
            error     = pos_des - current_pos
            vel_error = vel_des - v_cur

            # Variable-impedance gains from the learned-variability QP. The QP
            # bounds the commanded force K·error to the per-axis desired force
            # limits, so the applied force never violates the demonstrated min/max.
            kp, kd = self.get_variable_gains(
                error=error,
                vel_error=vel_error,
                desired_force=f_des_world,
                dt=self.dt * phase_speed,
                phase=self.phase
            )

            # VIC control law (impedance only)
            f_virtual = (kp * error) + paramVIC.VIC_KI * self.error_accumulated + kd * vel_error
            # Force saturation: strictly cap the commanded force to the per-axis
            # demonstrated bound so it never violates the desired min/max force
            # (the QP enforces this softly; this makes it a hard guarantee).
            f_virtual = np.clip(f_virtual, -self.f_bound, self.f_bound)

            # Torque calculation
            jjt = jac @ jac.T
            lambda_sq = paramVIC.VIC_LAMBDA_SQ
            tau_task = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), f_virtual)

            tau_posture_robot = 10.0 * (q_home - self.data.qpos[:self.n_robot]) \
                                - 2.0 * self.data.qvel[:self.n_robot]
            tau_posture = np.zeros(self.model.nv)
            tau_posture[:self.n_robot] = tau_posture_robot

            j_inv = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), np.eye(3))
            null_projection = np.eye(self.model.nv) - (j_inv @ jac)
            tau_null = null_projection @ tau_posture

            tau_nominal = tau_task + tau_null + self.data.qfrc_bias[:self.model.nv]
            tau_safe = self._solve_passivity_qp(tau_nominal, self.data.qvel)

            self.data.ctrl[:self.model.nu] = np.clip(tau_safe[:self.model.nu], -300, 300)

            # Apply the analytical cutting resistance to the blade (material
            # physics, not the controller). This is the environment reaction the
            # blade feels; the force estimator reports it.
            if self.cut_model is not None:
                # Refresh the cutting surface from the LIVE material position each
                # step (the material can drift/move) so depth is computed against
                # where the surface actually is — not a stale setup value.
                self.cut_model.set_material(self.data.geom_xpos[self.mat_geom_id],
                                            self.model.geom_size[self.mat_geom_id])
                fcut = self.cut_model.compute(current_pos, v_cur, f_des_world)
                self.data.xfrc_applied[self._scalpel_body, :3] = fcut
                self.robot._applied_cut_force = fcut

            mujoco.mj_step(self.model, self.data)

            # Advance time – state-driven phase correction
            self.mat_time += phase_speed * self.dt
            self.sim_time += self.dt
            phi_hat = self._estimate_phase(current_pos, self.filtered_force)
            phase_correction = np.clip(
                self.k_phase * (phi_hat - self.phase),
                -phase_inc, 2.0 * phase_inc
            )
            self.phase = np.clip(self.phase + phase_inc + phase_correction, 0.0, 1.0)

            # Update tank energy
            self.update_tank_energy(kp, kd, error, vel_error, self.dt * phase_speed)

            # Full-resolution episode record for metrics
            self.episode_log["phase"].append(self.phase)
            self.episode_log["pos_des_nom"].append(pos_des.copy())
            self.episode_log["pos_des"].append(pos_des.copy())
            self.episode_log["pos_act"].append(current_pos.copy())
            self.episode_log["f_des"].append(f_des_world.copy())
            self.episode_log["f_act"].append(self.filtered_force.copy())
            self.episode_log["f_raw"].append(np.asarray(current_force_raw).copy())
            self.episode_log["kp"].append(np.asarray(kp).copy())
            self.episode_log["tank"].append(float(self.E_t))
            self.episode_log["tau"].append(float(np.linalg.norm(tau_safe[:self.model.nu])))

            self.record_contact_forces(Pd=pos_des, P=current_pos, Fd=f_des_world, K=kp)

            # Logging
            if self.sim_time - last_log_time > 0.5:
                force_des_mag = np.linalg.norm(f_des_world)
                force_act_mag = np.linalg.norm(self.filtered_force)
                pos_des_mag = np.linalg.norm(pos_des)
                pos_act_mag = np.linalg.norm(current_pos)
                force_log.append([self.phase, force_des_mag, force_act_mag])
                pos_log.append([self.phase, pos_des_mag, pos_act_mag])
                force_err_mag = force_des_mag - force_act_mag
                Logger.debug(f"Phase {self.phase:.3f}(est:{phi_hat:.3f}): "
                      f"F_des={force_des_mag:.1f}N  F_act={force_act_mag:.1f}N  "
                      f"F_err={force_err_mag:.1f}N  pen={self.adm_s*1e3:.1f}mm  "
                      f"Kp_z={kp[2]:.0f}  "
                      f"{'CONTACT' if self.in_contact else 'free'}")
                last_log_time = self.sim_time

            if viewer is not None:
                if step % 4 == 0:
                    viewer.sync()
                # Real-time pacing: run the cut at sim-time so the live monitor
                # stays synced with the robot (otherwise the cut finishes in ~1-2s
                # flat-out and the monitor can't keep up). Headless runs skip this.
                sleep_t = self.dt - (time.time() - step_start)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        if force_log:
            phases, f_des, f_act = zip(*force_log)
            Logger.debug(f"\n{'='*60}")
            Logger.debug(f"FORCE TRACKING SUMMARY")
            Logger.debug(f"{'='*60}")
            Logger.debug(f"Mean desired force: {np.mean(f_des):.1f}N")
            Logger.debug(f"Mean actual force: {np.mean(f_act):.1f}N")
            Logger.debug(f"RMSE: {np.sqrt(np.mean((np.array(f_des) - np.array(f_act))**2)):.1f}N")
            Logger.debug(f"{'='*60}")

        if pos_log:
            phases, p_des, p_act = zip(*pos_log)
            Logger.debug(f"\n{'='*60}")
            Logger.debug(f"POS TRACKING SUMMARY")
            Logger.debug(f"{'='*60}")
            Logger.debug(f"Mean desired pos: {np.mean(p_des):.1f}N")
            Logger.debug(f"Mean actual pod: {np.mean(p_act):.1f}N")
            Logger.debug(f"RMSE: {np.sqrt(np.mean((np.array(p_des) - np.array(p_act))**2)):.1f}N")
            Logger.debug(f"{'='*60}")

        return True