from logger import Logger
from mjModeling.cutting_materials import Material
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14
import numpy as np
import mujoco
from mjModeling.conf import paramVIC, workPiece
from mjModeling.controllers.controller_api import Controller
from mjModeling.estimators import ImpedanceEstimator
from mjModeling.mjRobot import Robot
import sys
__all__ = ["BasicVariableImpedanceControl"]


class BasicVariableImpedanceControl(Controller): # Removed parent for standalone clarity
    def __init__(self, robot: iiwa14):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.error_accumulated = np.zeros(3) # For Integral term
        self.estimator = ImpedanceEstimator(robot)
        self._working_piece: Material = None
        # Number of robot joints (always 7 for iiwa)
        self.n_robot = robot.nq_robot

    def get_variable_gains(self, error):
        """
        error : np.ndarray of shape (3,) – position error.
        Returns:
            kp : np.ndarray (3,) – proportional gains for each axis.
            kd : np.ndarray (3,) – derivative gains for each axis.
        """
        # These can later be made per‑axis arrays (e.g., self.kp_min = [x_min, y_min, z_min])
        k_min = paramVIC.VIC_KP_MIN
        k_max = paramVIC.VIC_KP_MAX
        alpha = 0.02  # tune this – error scale at which stiffness reaches halfway to max

        # Per‑axis proportional gain using saturating function
        abs_e = np.abs(error)
        kp = k_min + (k_max - k_min) * (abs_e / (alpha + abs_e))

        # Derivative gain: critically damped would be 2*sqrt(kp), but we keep your heuristic
        kd = 0.5 * np.sqrt(kp)

        return kp, kd

    def compensate_cutting_resistance(self, current_pos, v_tip):
        """OBSOLETE to be deleted later"""
        if not self.working_piece:
            print("No material was set to working piece or No working piece was added!")
            return np.zeros(3)
        # Material surface is at center_z + size_z = 0.04
        f_res = np.zeros(3)
        surface_z = self.working_piece.surface_height
        magnitude = self.working_piece.cut_resistance

        depth = surface_z - current_pos[2]

        if depth > 0:
            # 1. Damping (The 'v_tip' part - only active while moving)
            f_damping = -magnitude * v_tip

            # 2. Stiffness (The 'depth' part - active even when stopped)
            # Use a constant like 500 N/m to simulate material pushing back up
            f_stiffness = np.array([0, 0, 500.0 * depth])
            f_res = f_damping + f_stiffness

            # here estimated contact force is registered to shared memory if the material is not solid
            # this whole function should be refactored to consider estimating forces for 3d directions
            self.robot.robot_state["shared_array"][-1] = np.linalg.norm(f_res)
            return f_res

        return f_res


    def move_to_position(self, target_pos, viewer=None):
        tcp_id = self.model.site("scalpel_tip").id
        # Define 'home' posture to keep the elbow up (joint angles in radians)
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])

        self.error_accumulated = np.zeros(3)
        # Use a small epsilon for Damped Least Squares stability
        lambda_sq = paramVIC.VIC_LAMBDA_SQ

        for step in range(self.opt_max_steps):
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[tcp_id].copy()
            error = target_pos - current_pos
            dist = np.linalg.norm(error)

            if not step % int(self.opt_max_steps/10):
                Logger.debug(f"default VIC: opt_step {step}: target = {target_pos} -- current = {current_pos} --err = {dist}")

            # 2mm tolerance for 2026 surgical/precision tasks
            if dist < paramVIC.VIC_TOL:
                return True

            # 1. VARIABLE GAIN SCHEDULING
            # High stiffness far away, lower stiffness for delicate contact
            kp_val, kd_val = self.get_variable_gains(dist)

            # 2. INTEGRAL TERM (The "Closer")
            # Only accumulate when within 5cm to prevent huge overshoots
            if dist < 0.05:
                # ki=200 is strong enough to compensate for steady-state error
                self.error_accumulated += error * self.model.opt.timestep
            ki_val = paramVIC.VIC_KI

            # 3. TASK SPACE FORCE
            v_tip = (np.zeros(3))
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            v_tip = jac @ self.data.qvel
            # f_res = self.compensate_cutting_resistance(current_pos, v_tip)

            # F = Kp*e + Ki*∫e - Kd*v
            f_virtual = (kp_val * error) + (ki_val * self.error_accumulated) - (kd_val * v_tip)
            # f_virtual +=  f_res #  self.sim_cutting_resistance(current_pos, v_tip)
            # 4. STABLE MAPPING (Damped Least Squares)
            # Solves: tau = J^T * inv(JJ^T + λ^2I) * F
            jjt = jac @ jac.T
            tau_task = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), f_virtual)

            # 5. NULL-SPACE POSTURE CONTROL (Fixes "lying on material")
            # Keeps the robot elbow up while the tip follows target_pos
            k_posture, d_posture = 10.0, 2.0


            # Compute posture torque only for robot joints
            tau_posture_robot = k_posture * (q_home - self.data.qpos[:self.n_robot]) - d_posture * self.data.qvel[:self.n_robot]

            # Pad with zeros for extra DOFs (material joint)
            tau_posture = np.zeros(self.model.nv)
            tau_posture[:self.n_robot] = tau_posture_robot
            # tau_posture = k_posture * (q_home[:self.model.nv] - self.data.qpos[:self.model.nv]) - d_posture * self.data.qvel

            # Project posture into null-space: P = (I - J_pinv * J)
            j_inv = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), np.eye(3))
            null_projection = np.eye(self.model.nv) - (j_inv @ jac)
            tau_null = null_projection @ tau_posture

            # 6. FINAL TORQUE + BIAS COMPENSATION
            # qfrc_bias handles Gravity and Coriolis automatically
            tau_total = tau_task + tau_null + self.data.qfrc_bias[:self.model.nv]

            # Apply to actuators within hardware limits
            self.data.ctrl[:self.model.nu] = np.clip(tau_total[:self.model.nu], -300, 300)

            # 7. STEP PHYSICS
            mujoco.mj_step(self.model, self.data)
            K = np.array([kp_val, kp_val, kp_val])
            self.record_contact_forces(K)

            if viewer and step % 4 == 0:
                viewer.sync()

        return False

    def record_contact_forces(self, K:np.ndarray = None):
        if self.estimator:
            force = self.estimator.get_total_cutting_force()   # assume returns [fx, fy, fz]
            if isinstance(K, np.ndarray):
                sample = np.append(force, K.flatten())
            else:
                residual =np.zeros(self.robot.buffer.num_signals - len(sample))
                sample = np.append(force, residual)

            self.robot.buffer.write_samples(sample)

    @property
    def working_piece(self):
        return self._working_piece

    @working_piece.setter
    def working_piece(self, material: Material):
        self._working_piece = material
