from mjModeling.conf.configs import paramVIC
import numpy as np
import mujoco
from mjModeling.cutting_materials.utils import wp_sine_motion
from mjModeling.controllers.vic_controller.bp_based_controller import BpVariableImpedanceControl
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14

__all__ = ["ContinuousTrajectoryVIC"]


class ContinuousTrajectoryVIC(BpVariableImpedanceControl):
    """
    Variable Impedance Controller with continuous trajectory tracking.
    Inherits from VariableImpedanceControl and replaces the per‑waypoint execution
    with a phase‑based continuous interpolation of the GMR data.
    """
    def __init__(self, robot: iiwa14, use_behaviour_priors: bool = False):
        super().__init__(robot, use_behaviour_priors)
        if self.use_bp:
            self._init_trajectory_interpolators()

    def _init_trajectory_interpolators(self):
        """Create continuous functions of phase from discrete GMR data."""
        # Extract data arrays (assumed shape Nx3)
        self.traj_pos = self.traj_loader.pos[:, 0:3]
        self.traj_vel = self.traj_loader.vel[:, 0:3]
        self.traj_force = self.traj_loader.force[:, 0:3]

        N = len(self.traj_pos)
        # Sampling period of original recording (adjust as needed)
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

    def _gmr_vel_to_world(self, gmr_vel, mat_vel, mat_pos=None):
        """
        Transform GMR velocity to world frame.
        gmr_vel : (3,) velocity from GMR (usually in normalized coordinates)
        mat_vel : scalar (if material only moves vertically) or (3,) array.
        mat_pos : current material position (unused here, kept for compatibility)
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

    def follow_trajectory(self, phase_speed: float=paramVIC.PHASE_SPEED, viewer=None):
        """
        Execute the entire GMR trajectory as one continuous motion.
        phase_speed : float > 0, scaling factor for execution speed.
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

        max_steps = int(2 * self.traj_duration / self.dt)

        for step in range(max_steps):
            if self.phase >= 1.0:
                break

            # --- Update material position using SCALED TIME ---
            if self.mat_joint_id is not None and self.wp_mobile:
                current_height = wp_sine_motion(self.mat_time)
                self.data.qpos[self.mat_joint_id] = current_height
                mujoco.mj_forward(self.model, self.data)

                # Compute material velocity via finite difference
                if prev_mat_height is None:
                    mat_vel = 0.0
                else:
                    # Actual time step for material is phase_speed * dt
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

            # Transform to world coordinates – DO NOT pass mat_pos to _gmr_to_world
            pos_des = self._gmr_to_world(pos_des_gmr)
            vel_des = self._gmr_vel_to_world(vel_des_gmr_scaled, mat_vel)

            # --- Current robot state ---
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[tcp_id].copy()
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            v_cur = jac @ self.data.qvel

            # Transform feedforward force (tool frame to world)
            site_rot = self.data.site_xmat[tcp_id].reshape(3, 3)
            f_ff_world = site_rot @ force_des_gmr
            f_ff_world[2] *= 0.1
            f_ff_world[:2] *= 0.3

            # --- Error and variable gains ---
            error = pos_des - current_pos
            mat_pos = self.data.geom_xpos[self.mat_geom_id].copy()
            surface_z = mat_pos[2] + self.cut_depth_z
            penetration_depth = max(0.0, surface_z - current_pos[2])
            kp, kd = self.get_variable_gains(error)

            if penetration_depth > 0.001:
                kp[2] *= 5.0
                kd[2] *= 3.0
                kp = np.clip(kp, paramVIC.VIC_KP_MIN, paramVIC.VIC_KP_MAX)

            # --- Control law with velocity tracking ---
            f_virtual = (kp * error) + paramVIC.VIC_KI * self.error_accumulated \
                        + kd * (vel_des - v_cur) + f_ff_world

            if penetration_depth < 0.005 and np.linalg.norm(error) < 0.05:
                self.error_accumulated += error * self.dt
                self.error_accumulated = np.clip(self.error_accumulated, -0.05, 0.05)

            # --- Torque calculation (unchanged) ---
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

            # Optional passivity QP
            tau_safe = self._solve_passivity_qp(tau_nominal, self.data.qvel)

            # --- Apply and step ---
            self.data.ctrl[:self.model.nu] = np.clip(tau_safe[:self.model.nu], -300, 300)
            mujoco.mj_step(self.model, self.data)

            # Advance scaled time and phase
            self.mat_time += phase_speed * self.dt
            self.phase += phase_inc
            self.sim_time += self.dt

            # Record forces
            self.record_contact_forces(kp)

            if viewer and step % 4 == 0:
                viewer.sync()

        return True