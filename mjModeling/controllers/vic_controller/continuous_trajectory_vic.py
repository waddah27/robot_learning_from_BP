from mjModeling.conf.configs import ImpedanceOptimizer, paramVIC
import numpy as np
import mujoco
from mjModeling.cutting_materials.utils import wp_sine_motion
from mjModeling.controllers.vic_controller.bp_based_controller import BpVariableImpedanceControl
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14

__all__ = ["ContinuousTrajectoryVIC"]

# Try to import optional solvers
try:
    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    CVXOPT_AVAILABLE = True
except ImportError:
    CVXOPT_AVAILABLE = False
    print("cvxopt not available, using alternative solvers")

try:
    from scipy.optimize import lsq_linear
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy.optimize not available, using gradient descent")


class ContinuousTrajectoryVIC(BpVariableImpedanceControl):
    """
    Variable Impedance Controller with multiple gain optimization methods.
    Choose which method to use via the 'optimizer' parameter.
    """
    def __init__(self, robot: iiwa14, use_behaviour_priors: bool = False, optimizer=ImpedanceOptimizer.qp):
        """
        Args:
            optimizer: 'qp', 'lsq', or 'gd' (gradient descent)
        """
        super().__init__(robot, use_behaviour_priors)

        # Optimizer selection
        self.optimizer = optimizer
        print(f"Using optimizer: {self.optimizer}")

        # Store previous gains for constraints
        self.last_kp = np.ones(3) * paramVIC.VIC_KP_MIN
        self.last_kd = np.ones(3) * 0.001 * np.sqrt(paramVIC.VIC_KP_MIN)

        # Force filtering
        self.filtered_force = np.zeros(3)
        self.force_filter_alpha = 0.2

        # Contact state
        self.in_contact = False
        self.contact_threshold = 2.0  # N

        # Feedforward force scaling
        # self.ff_z_scale = 0.1
        # self.ff_xy_scale = 0.3

        # Gradient descent parameters
        self.gd_learning_rate = 1e-6
        self.gd_iterations = 10

        # Debug
        self.last_print_time = 0
        self.print_interval = 0.2

        if self.use_bp:
            self._init_trajectory_interpolators()
            self._check_force_scale()

    def _check_force_scale(self):
        """Check if GMR forces need scaling"""
        if hasattr(self, 'traj_force'):
            mean_force = np.mean(np.linalg.norm(self.traj_force, axis=1))
            max_force = np.max(np.linalg.norm(self.traj_force, axis=1))
            print(f"GMR forces - mean: {mean_force:.1f}N, max: {max_force:.1f}N")

            # Scale forces to reasonable range (20-80N)
            if max_force > 100:
                self.force_scale = 60.0 / max_force
                print(f"Auto-scaling forces by {self.force_scale:.3f}")
            else:
                self.force_scale = 1.0

    def _init_trajectory_interpolators(self):
        """Create continuous functions of phase from discrete GMR data."""
        # Extract data arrays (assumed shape Nx3)
        self.traj_pos = self.traj_loader.pos[:, 0:3]
        self.traj_vel = self.traj_loader.vel[:, 0:3]
        self.traj_force = self.traj_loader.force[:, 0:3]

        # Scale forces if needed
        if hasattr(self, 'force_scale'):
            self.traj_force *= self.force_scale

        N = len(self.traj_pos)
        # Sampling period of original recording
        self.traj_dt = getattr(self.traj_loader, 'dt', 0.01)   # default 100 Hz
        self.traj_duration = N * self.traj_dt

        # Phase values at original waypoints (0 to 1)
        phase_points = np.linspace(0, 1, N)

        try:
            from scipy.interpolate import interp1d
            # Smooth interpolation
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
            # Fallback: linear interpolation using numpy
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
        """
        Transform GMR position to world coordinates.
        """
        if mat_pos is None:
            mat_pos = self.data.geom_xpos[self.mat_geom_id].copy()
        surface_z = mat_pos[2] + self.cut_depth_z   # top of material

        p_safe = np.zeros(3)
        p_safe[:min(len(gmr_point), 3)] = gmr_point[:min(len(gmr_point), 3)]

        norm = (p_safe - self.gmr_min) / self.gmr_range
        norm = np.clip(norm, 0, 1)

        world_x = mat_pos[0] + (norm[0] - 0.5) * self.cut_width_x
        world_y = mat_pos[1] + (norm[1] - 0.5) * self.cut_width_y
        world_z = surface_z   # desired Z is always at the surface

        return np.array([world_x, world_y, world_z])

    def _gmr_vel_to_world(self, gmr_vel, mat_vel, mat_pos=None):
        """
        Transform GMR velocity to world frame.
        gmr_vel : (3,) velocity from GMR (usually in normalized coordinates)
        mat_vel : scalar (if material only moves vertically) or (3,) array.
        """
        # XY components: scale by material size / GMR range
        scale_xy = np.array([self.cut_width_x / self.gmr_range[0],
                             self.cut_width_y / self.gmr_range[1]])
        v_world_xy = gmr_vel[:2] * scale_xy

        # Z component: from material motion (GMR Z velocity is ignored)
        if np.isscalar(mat_vel):
            v_world_z = mat_vel
        else:
            v_world_z = mat_vel[2]   # full 3D material velocity

        return np.array([v_world_xy[0], v_world_xy[1], v_world_z])

    # ==================== OPTIMIZER 1: QUADRATIC PROGRAMMING ====================

    def get_variable_gains_qp(self, error, vel_error, desired_force, current_force):
        """
        Optimize gains using Quadratic Programming with stability fixes for CVXOPT.
        """
        if not CVXOPT_AVAILABLE:
            return super().get_variable_gains(error)

        n_dims = 3
        n_vars = 2 * n_dims

        # Pre-processing and safety clips
        error = np.clip(np.nan_to_num(error), -0.1, 0.1)
        vel_error = np.clip(np.nan_to_num(vel_error), -1.0, 1.0)
        desired_force = np.nan_to_num(desired_force)

        # Build regressor matrix R
        R = np.zeros((n_dims, n_vars))
        for i in range(n_dims):
            R[i, i] = error[i]
            R[i, i + n_dims] = vel_error[i]

        # 1. FIX: Stronger Regularization to prevent "Domain Error" from singular Q
        # 1e-4 is often too small for real-time sensor data scales.
        Q = 2.0 * (R.T @ R) + np.eye(n_vars) * 1e-2
        p = -2.0 * R.T @ desired_force

        # Constraint Parameters
        kp_min, kp_max = float(max(paramVIC.VIC_KP_MIN, 1.0)), float(paramVIC.VIC_KP_MAX)
        # 2. FIX: kd_max of 0.1 was likely too small, conflicting with passivity K_d >= 2*zeta*sqrt(K_p)
        kd_min, kd_max = 0.01, 50.0

        G_list = []
        h_list = []

        # Lower and Upper Bounds (Gx <= h format)
        for i in range(n_vars):
            lb, ub = (kp_min, kp_max) if i < n_dims else (kd_min, kd_max)
            # -x <= -lb
            row_l = np.zeros(n_vars); row_l[i] = -1.0
            G_list.append(row_l); h_list.append(-float(lb))
            # x <= ub
            row_u = np.zeros(n_vars); row_u[i] = 1.0
            G_list.append(row_u); h_list.append(float(ub))

        # 3. FIX: Simplified Passivity Linearization (Kd >= slope * Kp + intercept)
        zeta = 0.5
        for i in range(n_dims):
            kp_prev = max(self.last_kp[i], kp_min)
            # Linearized constraint: Kd >= zeta * sqrt(Kp_prev)
            # Rearranged for CVXOPT (Gx <= h): -Kd <= -zeta * sqrt(Kp_prev)
            row = np.zeros(n_vars)
            row[i + n_dims] = -1.0
            G_list.append(row)
            h_list.append(-float(zeta * np.sqrt(kp_prev)))

        try:
            # 4. FIX: Ensure all inputs are explicit float64 (np.double)
            # CVXOPT is sensitive to matrix types and scaling.
            Q_mat = matrix(Q.astype(np.double))
            p_mat = matrix(p.astype(np.double))
            G_mat = matrix(np.array(G_list).astype(np.double))
            h_mat = matrix(np.array(h_list).astype(np.double))

            solvers.options['show_progress'] = False
            solvers.options['reltol'] = 1e-5 # Slightly looser tolerance for robustness

            sol = solvers.qp(Q_mat, p_mat, G_mat, h_mat)

            if sol['status'] == 'optimal':
                x = np.array(sol['x']).flatten()
                self.last_kp = np.clip(x[:n_dims], kp_min, kp_max)
                self.last_kd = np.clip(x[n_dims:], kd_min, kd_max)
                return self.last_kp, self.last_kd

        except (ValueError, ArithmeticError) as e:
            # Explicitly catching domain/singular matrix errors
            print(f"QP Numerical Failure: {e}. Using heuristic.")

        return super().get_variable_gains(error)



    # ==================== OPTIMIZER 2: LEAST SQUARES ====================
    def get_variable_gains_lsq(self, error, vel_error, desired_force, current_force):
        """
        Optimize gains using least squares with bounds (most robust).
        """
        if not SCIPY_AVAILABLE:
            print("scipy.optimize not available, falling back to gradient descent")
            return self.get_variable_gains_gd(error, vel_error, desired_force, current_force)

        n_dims = 3

        try:
            kp = np.zeros(n_dims)
            kd = np.zeros(n_dims)

            for i in range(n_dims):
                # Build system: [error_i, vel_error_i] * [kp_i; kd_i] = desired_force_i
                A = np.array([[error[i], vel_error[i]]])
                b = np.array([desired_force[i]])

                # Check for degenerate case
                if np.linalg.norm(A) < 1e-6:
                    kp[i] = paramVIC.VIC_KP_MIN
                    kd[i] = 0.001
                    continue

                # Solve with bounds
                result = lsq_linear(A, b,
                                   bounds=([paramVIC.VIC_KP_MIN, 0.001],
                                          [paramVIC.VIC_KP_MAX, 0.1]),
                                   method='trf',
                                   max_iter=100,
                                   tol=1e-4)

                kp[i] = result.x[0]
                kd[i] = result.x[1]

            # Apply passivity constraint
            zeta = 0.5
            for i in range(n_dims):
                min_kd = 2 * zeta * np.sqrt(kp[i])
                kd[i] = max(kd[i], min_kd)

            self.last_kp = kp.copy()
            self.last_kd = kd.copy()

            return kp, kd

        except Exception as e:
            print(f"LSQ solver failed: {e}, using heuristic gains")
            return super().get_variable_gains(error)

    # ==================== OPTIMIZER 3: GRADIENT DESCENT ====================
    def get_variable_gains_gd(self, error, vel_error, desired_force, current_force):
        """
        Optimize gains using simple gradient descent.
        """
        n_dims = 3

        # Initial gains from last solution or heuristic
        if hasattr(self, 'last_kp'):
            kp = self.last_kp.copy()
            kd = self.last_kd.copy()
        else:
            kp, kd = super().get_variable_gains(error)

        # Force tracking error
        force_error = desired_force - current_force

        # Gradient descent
        for _ in range(self.gd_iterations):
            # Current force from impedance
            f_imp = kp * error + kd * vel_error

            # Error gradient (simplified)
            grad_kp = -2 * (desired_force - f_imp) * error
            grad_kd = -2 * (desired_force - f_imp) * vel_error

            # Update with clipping to prevent explosions
            grad_kp = np.clip(grad_kp, -1e4, 1e4)
            grad_kd = np.clip(grad_kd, -1e4, 1e4)

            kp = kp - self.gd_learning_rate * grad_kp
            kd = kd - self.gd_learning_rate * grad_kd

            # Apply bounds
            kp = np.clip(kp, paramVIC.VIC_KP_MIN, paramVIC.VIC_KP_MAX)
            kd = np.clip(kd, 0.001, 0.1)

        # Apply passivity constraint
        zeta = 0.5
        for i in range(n_dims):
            min_kd = 2 * zeta * np.sqrt(kp[i])
            kd[i] = max(kd[i], min_kd)

        self.last_kp = kp.copy()
        self.last_kd = kd.copy()

        return kp, kd

    # ==================== MAIN GAIN FUNCTION ====================
    def get_variable_gains(self, error, vel_error=None, desired_force=None, current_force=None):
        """
        Main gain function that dispatches to the selected optimizer.
        """
        # If no force data, use heuristic
        if desired_force is None or current_force is None or vel_error is None:
            return super().get_variable_gains(error)

        # Dispatch to selected optimizer
        if self.optimizer == ImpedanceOptimizer.qp:
            return self.get_variable_gains_qp(error, vel_error, desired_force, current_force)
        elif self.optimizer == ImpedanceOptimizer.lsq:
            return self.get_variable_gains_lsq(error, vel_error, desired_force, current_force)
        elif self.optimizer == ImpedanceOptimizer.gd:
            return self.get_variable_gains_gd(error, vel_error, desired_force, current_force)
        else:
            print(f"Unknown optimizer {self.optimizer}, using heuristic")
            return super().get_variable_gains(error)

    # ==================== TRAJECTORY EXECUTION ====================
    def follow_trajectory(self, phase_speed: float = paramVIC.PHASE_SPEED, viewer=None):
        """
        Execute the entire GMR trajectory as one continuous motion with optimized gains.
        """
        tcp_id = self.model.site("scalpel_tip").id
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])
        self.error_accumulated = np.zeros(3)

        # Phase variable and scaled time for material motion
        self.phase = 0.0
        self.mat_time = 0.0
        phase_inc = phase_speed * self.dt / self.traj_duration

        # For material velocity computation (finite difference)
        prev_mat_height = None
        actual_duration = self.traj_duration / phase_speed
        max_steps = int(2 * actual_duration / self.dt)

        # Reset force filter
        self.filtered_force = np.zeros(3)

        # For debugging
        force_log = []
        last_log_time = 0

        for step in range(max_steps):
            if self.phase >= 1.0:
                break

            # --- Update material position using SCALED TIME ---
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

            # --- Get desired quantities at current phase ---
            pos_des_gmr = self.pos_func(self.phase)
            vel_des_gmr = self.vel_func(self.phase)
            force_des_gmr = self.force_func(self.phase)

            # Scale velocity to match phase speed
            vel_des_gmr_scaled = vel_des_gmr * phase_speed

            # Transform to world coordinates
            pos_des = self._gmr_to_world(pos_des_gmr)
            vel_des = self._gmr_vel_to_world(vel_des_gmr_scaled, mat_vel)

            # --- Current robot state ---
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[tcp_id].copy()
            current_force_raw = self.estimator.get_total_cutting_force()

            # Filter force measurement
            self.filtered_force = self.force_filter_alpha * current_force_raw + \
                                  (1 - self.force_filter_alpha) * self.filtered_force

            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            v_cur = jac @ self.data.qvel

            # Transform feedforward force (tool frame to world)
            site_rot = self.data.site_xmat[tcp_id].reshape(3, 3)
            f_ff_world = site_rot @ force_des_gmr
            # f_ff_world[2] *= self.ff_z_scale
            # f_ff_world[:2] *= self.ff_xy_scale

            # --- Error and contact detection ---
            error = pos_des - current_pos
            vel_error = vel_des - v_cur

            # mat_pos = self.data.geom_xpos[self.mat_geom_id].copy()
            # surface_z = mat_pos[2] + self.cut_depth_z
            # penetration_depth = max(0.0, surface_z - current_pos[2])

            # Contact detection
            self.in_contact = np.linalg.norm(self.filtered_force) > self.contact_threshold

            # --- Get optimized gains ---
            kp, kd = self.get_variable_gains(
                error=error,
                vel_error=vel_error,
                desired_force=force_des_gmr,
                current_force=self.filtered_force
            )

            # if penetration_depth > 0.001:
            #     kp[2] *= 5.0
            #     kd[2] *= 3.0
            #     kp = np.clip(kp, paramVIC.VIC_KP_MIN, paramVIC.VIC_KP_MAX)

            # --- Control law with velocity tracking ---
            f_virtual = (kp * error) + paramVIC.VIC_KI * self.error_accumulated \
                        + kd * vel_error + f_ff_world

            # Update position integral (with phase speed scaling)
            # if penetration_depth < 0.005 and np.linalg.norm(error) < 0.05:
            #     self.error_accumulated += error * self.dt * phase_speed
            #     self.error_accumulated = np.clip(self.error_accumulated, -0.05, 0.05)

            # --- Torque calculation (unchanged from your code) ---
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

            # Passivity QP
            tau_safe = self._solve_passivity_qp(tau_nominal, self.data.qvel)

            # --- Apply and step ---
            self.data.ctrl[:self.model.nu] = np.clip(tau_safe[:self.model.nu], -300, 300)
            mujoco.mj_step(self.model, self.data)

            # Advance scaled time and phase
            self.mat_time += phase_speed * self.dt
            self.phase += phase_inc
            self.sim_time += self.dt

            # Record forces
            self.record_contact_forces(Fd=f_ff_world, K=kp)

            # Log periodically
            if self.sim_time - last_log_time > 0.5:
                force_des_mag = np.linalg.norm(force_des_gmr)
                force_act_mag = np.linalg.norm(self.filtered_force)
                force_log.append([self.phase, force_des_mag, force_act_mag])

                print(f"Phase {self.phase:.2f}: Desired F: {force_des_mag:.1f}N, "
                      f"Actual F: {force_act_mag:.1f}N, "
                      f"Kp_z: {kp[2]:.1f}, Kd_z: {kd[2]:.3f}, "
                      f"{'CONTACT' if self.in_contact else 'NO_CONTACT'}")

                last_log_time = self.sim_time

            if viewer and step % 4 == 0:
                viewer.sync()

        # Print force tracking summary
        if force_log:
            phases, f_des, f_act = zip(*force_log)
            print(f"\n{'='*60}")
            print(f"FORCE TRACKING SUMMARY")
            print(f"{'='*60}")
            print(f"Mean desired force: {np.mean(f_des):.1f}N")
            print(f"Mean actual force: {np.mean(f_act):.1f}N")
            print(f"RMSE: {np.sqrt(np.mean((np.array(f_des) - np.array(f_act))**2)):.1f}N")
            print(f"{'='*60}")

        return True