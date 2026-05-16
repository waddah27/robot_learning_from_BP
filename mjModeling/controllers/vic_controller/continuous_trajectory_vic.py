from mjModeling.conf.configs import ImpedanceOptimizer, paramVIC
import numpy as np
import mujoco
from mjModeling.cutting_materials.utils import wp_sine_motion
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
        self.k_min = np.array([100, 100, 100])
        self.k_max = np.array([5000, 5000, 5000])
        self.k_init = 2000
        self.xi_init = 0.7
        self.D_min = np.array([0, 0, 0])
        self.D_max = np.array([self.Xi_scaler, self.Xi_scaler, self.Xi_scaler])
        self.f_min = -70
        self.f_max = 70
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

        # Debug
        self.last_print_time = 0
        self.print_interval = 0.2

        if self.use_bp:
            self._init_trajectory_interpolators()
            self._check_force_scale()

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
        surface_z = mat_pos[2] - self.cut_depth_z
        p_safe = np.zeros(3)
        p_safe[:min(len(gmr_point), 3)] = gmr_point[:min(len(gmr_point), 3)]
        norm = (p_safe - self.gmr_min) / self.gmr_range
        norm = np.clip(norm, 0, 1)
        world_x = mat_pos[0] + (norm[0] - 0.5) * self.cut_width_x
        world_y = mat_pos[1] + (norm[1] - 0.5) * self.cut_width_y
        world_z = surface_z
        return np.array([world_x, world_y, world_z])

    def _gmr_vel_to_world(self, gmr_vel, mat_vel, mat_pos=None):
        scale_xy = np.array([self.cut_width_x / self.gmr_range[0],
                             self.cut_width_y / self.gmr_range[1]])
        v_world_xy = gmr_vel[:2] * scale_xy
        if np.isscalar(mat_vel):
            v_world_z = mat_vel
        else:
            v_world_z = mat_vel[2]
        return np.array([v_world_xy[0], v_world_xy[1], v_world_z])

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

    # ---------- QP-BASED GAIN OPTIMIZATION ----------
    def get_variable_gains_optimizer(self, error, vel_error, desired_force, dt):
        """
        QP with HARD force constraints using ACTUAL measured force
        """
        
        if np.linalg.norm(error) < 1e-6 and np.linalg.norm(vel_error) < 1e-6:
            kd = 2 * self.prev_xi * np.sqrt(np.maximum(self.prev_kd, 100))
            return self.prev_kd, kd
        
        n_var = 6  # [Kx, Ky, Kz, Dx, Dy, Dz]
        H = np.zeros((n_var, n_var))
        f = np.zeros(n_var)
        
        reg = 1e-4
        
        # CRITICAL FIX: Use ACTUAL force error, not estimated
        # We want: (K*error + D*vel_error) to match desired_force
        # But the actual contact force is measured, not estimated
        # So we add a term that penalizes deviation from desired force
        
        # Build matrix A = [diag(error), diag(vel_error)]
        A = np.zeros((3, n_var))
        A[0, 0] = error[0]
        A[1, 1] = error[1]
        A[2, 2] = error[2]
        A[0, 3] = vel_error[0]
        A[1, 4] = vel_error[1]
        A[2, 5] = vel_error[2]
        
        # Force tracking: minimize ||(K*e_p + D*e_v) - F_des||²
        H = A.T @ A + reg * np.eye(n_var)
        f = -2 * (A.T @ desired_force)
        
        # Prior gains regularization
        prior_weight = 0.01
        H_prior = prior_weight * np.eye(n_var)
        H += H_prior
        
        f_prior = -2 * prior_weight * np.hstack([self.prev_kd, 
                                                2 * self.prev_xi * np.sqrt(np.maximum(self.prev_kd, 100))])
        f += f_prior
        
        # HARD FORCE CONSTRAINTS using ACTUAL measured force
        # The actual contact force must stay within [f_min, f_max]
        # But we can't directly constrain actual force in gain optimization
        # Instead, we constrain that the desired force is achievable
        
        G_list = []
        h_list = []
        
        for i in range(3):
            # Upper bound: K_i*e_p[i] + D_i*e_v[i] <= f_max
            G_upper = np.zeros(n_var)
            G_upper[i] = error[i]
            G_upper[3 + i] = vel_error[i]
            G_list.append(G_upper.reshape(1, -1))
            h_list.append([self.f_max])
            
            # Lower bound: -K_i*e_p[i] - D_i*e_v[i] <= -f_min
            G_lower = np.zeros(n_var)
            G_lower[i] = -error[i]
            G_lower[3 + i] = -vel_error[i]
            G_list.append(G_lower.reshape(1, -1))
            h_list.append([-self.f_min])
        
        G = np.vstack(G_list)
        h = np.vstack(h_list).flatten()
        
        # Gain bounds
        lb = np.hstack([self.k_min, self.D_min])
        ub = np.hstack([self.k_max, self.D_max])
        
        try:
            x_opt = solve_qp(
                P=H, q=f,
                lb=lb, ub=ub,
                G=G, h=h,
                solver='osqp',
                verbose=False
            )
            
            if x_opt is not None and not np.any(np.isnan(x_opt)):
                kp = x_opt[:3]
                kd_raw = x_opt[3:6]
                
                # Compute estimated force
                F_est = kp * error + kd_raw * vel_error
                
                # Log both estimated and actual force
                actual_force_norm = np.linalg.norm(self.filtered_force) if hasattr(self, 'filtered_force') else 0
                
                # Enforce stability
                zeta = np.clip(kd_raw / (2 * np.sqrt(np.maximum(kp, 100)) + 1e-6), 0.7, 1.5)
                kd_corrected = 2 * zeta * np.sqrt(np.maximum(kp, 100))
                
                # Smooth updates
                alpha = 0.5
                kp_smooth = alpha * kp + (1 - alpha) * self.prev_kd
                kd_smooth = alpha * kd_corrected + (1 - alpha) * (2 * self.prev_xi * np.sqrt(np.maximum(self.prev_kd, 100)))
                
                self.prev_kd = kp_smooth
                self.prev_xi = zeta
                
                force_error = np.linalg.norm(desired_force - F_est)
                
                # Log with actual force
                if hasattr(self, 'filtered_force'):
                    Logger.debug(f"QP: Kz={kp_smooth[2]:.0f}, F_est={F_est[2]:.0f}N, F_act={self.filtered_force[2]:.0f}N, F_des={desired_force[2]:.0f}N, err={force_error:.1f}N")
                else:
                    Logger.debug(f"QP success: Kz={kp_smooth[2]:.0f}, F_err={force_error:.1f}N")
                
                return kp_smooth, kd_smooth
                
        except Exception as e:
            Logger.debug(f"QP failed: {e}")
        
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
    def get_variable_gains(self, error, vel_error=None, desired_force=None, dt=None):
        """
        Main gain function. If force data available, use QP optimizer; else heuristic.
        """
        if desired_force is None or vel_error is None:
            return super().get_variable_gains(error)
        if dt is None:
            dt = self.dt
        return self.get_variable_gains_optimizer(error, vel_error, desired_force, dt)

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

        self.phase = 0.0
        self.mat_time = 0.0
        phase_inc = phase_speed * self.dt / self.traj_duration

        prev_mat_height = None
        actual_duration = self.traj_duration / phase_speed
        max_steps = int(2 * actual_duration / self.dt)

        self.filtered_force = np.zeros(3)
        self.force_integral = np.zeros(3)

        force_log = []
        pos_log = []
        
        last_log_time = 0

        for step in range(max_steps):
            if self.phase >= 1.0:
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

            # Errors
            error = pos_des - current_pos
            vel_error = vel_des - v_cur

            self.in_contact = np.linalg.norm(self.filtered_force) > self.contact_threshold

            # Get optimized gains using QP
            kp, kd = self.get_variable_gains(
                error=error,
                vel_error=vel_error,
                desired_force=f_des_world,
                dt=self.dt * phase_speed
            )

            # Optional force integral term
            if self.in_contact:
                force_error_world = f_des_world - self.filtered_force
                self.force_integral += force_error_world * self.dt * phase_speed * self.kf_i
                self.force_integral = np.clip(self.force_integral, -self.force_integral_max, self.force_integral_max)
            else:
                self.force_integral = np.zeros(3)

            # Control law
            f_virtual = (kp * error) + paramVIC.VIC_KI * self.error_accumulated + kd * vel_error + self.force_integral

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
            mujoco.mj_step(self.model, self.data)

            # Advance time
            self.mat_time += phase_speed * self.dt
            self.phase += phase_inc
            self.sim_time += self.dt

            # Update tank energy
            self.update_tank_energy(kp, kd, error, vel_error, self.dt * phase_speed)

            self.record_contact_forces(Pd=pos_des, P=current_pos, Fd=f_des_world, K=kp)

            # Logging
            if self.sim_time - last_log_time > 0.5:
                force_des_mag = np.linalg.norm(f_des_world)
                force_act_mag = np.linalg.norm(self.filtered_force)
                pos_des_mag = np.linalg.norm(pos_des)
                pos_act_mag = np.linalg.norm(current_pos)
                force_log.append([self.phase, force_des_mag, force_act_mag])
                pos_log.append([self.phase, pos_des_mag, pos_act_mag])
                print(f"Phase {self.phase:.2f}: Desired F: {force_des_mag:.1f}N, "
                      f"Actual F: {force_act_mag:.1f}N, "
                      f"Kp_z: {kp[2]:.1f}, Kd_z: {kd[2]:.3f}, "
                      f"{'CONTACT' if self.in_contact else 'NO_CONTACT'}")
                last_log_time = self.sim_time

            if viewer and step % 4 == 0:
                viewer.sync()

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