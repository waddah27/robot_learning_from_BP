from logger import Logger
from mjModeling.conf.configs import TCP_POS
from mjModeling.cutting_materials import Material
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14
import numpy as np
import mujoco
from mjModeling.conf import paramVIC
from mjModeling.controllers.controller_api import Controller
from mjModeling.estimators import ImpedanceEstimator
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

    def get_variable_gains(self, error, adaptive:bool = paramVIC.ADAPTIVE):
        """
        error : np.ndarray of shape (3,) – position error.
        adaptive: decide the behaviour of calculating gains as function of error
        Returns:
            kp : np.ndarray (3,) – proportional gains for each axis.
            kd : np.ndarray (3,) – derivative gains for each axis.
        """
        if paramVIC.DISABLE_PTP_VIC:
            kp = np.array([2000]*3)
            kd = 0.5 * np.sqrt(kp)
        else:
            # These can later be made per‑axis arrays (e.g., self.kp_min = [x_min, y_min, z_min])
            k_min = paramVIC.VIC_KP_MIN
            k_max = paramVIC.VIC_KP_MAX
            if adaptive:
                alpha = 0.02  # tune this – error scale at which stiffness reaches halfway to max

                # Per‑axis proportional gain using saturating function
                abs_e = np.abs(error)
                kp = k_min + (k_max - k_min) * (abs_e / (alpha + abs_e))
            else:
                error_norm = np.linalg.norm(error)
                kp = np.clip(k_max * (error_norm / 0.2), k_min, k_max) * np.ones(len(error))

            # Derivative gain
            kd = 2* 0.7 * np.sqrt(kp)

        return kp, kd


    def move_to_position(self, target_pos, viewer=None):
        tcp_id = self.model.site("scalpel_tip").id
        # Define 'home' posture to keep the elbow up (joint angles in radians)
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])

        self.error_accumulated = np.zeros(3)
        # Use a small epsilon for Damped Least Squares stability
        lambda_sq = paramVIC.VIC_LAMBDA_SQ

        for step in range(self.opt_max_steps):
            mujoco.mj_forward(self.model, self.data)
            self.robot.state[TCP_POS] = self.data.site_xpos[tcp_id].copy()
            error = target_pos - self.robot.state.get(TCP_POS)
            dist = np.linalg.norm(error)

            if not step % int(self.opt_max_steps/10):
                Logger.debug(f"default VIC: opt_step {step}: target = {target_pos} -- current = {self.robot.state.get(TCP_POS)} --err = {dist}")

            # 2mm tolerance for 2026 surgical/precision tasks
            if dist < paramVIC.VIC_TOL:
                return True

            # 1. VARIABLE GAIN SCHEDULING
            # High stiffness far away, lower stiffness for delicate contact
            kp_val, kd_val = self.get_variable_gains(error)

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

            # F = Kp*e + Ki*∫e - Kd*v
            f_virtual = (kp_val * error) + (ki_val * self.error_accumulated) - (kd_val * v_tip)
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
            self.record_contact_forces( K=kp_val)

            if viewer and step % 4 == 0:
                viewer.sync()

        return False

    def record_contact_forces(self, Pd: np.ndarray=None, P: np.ndarray=None, Fd:np.ndarray = None, K:np.ndarray = None):
        if self.estimator:
            force = self.estimator.get_total_cutting_force()   # assume returns [fx, fy, fz]
            if isinstance(P, np.ndarray) and isinstance(Pd, np.ndarray) and isinstance(Fd, np.ndarray) and isinstance(K, np.ndarray):
                sample = np.append(Pd.flatten(), P.flatten())
                sample = np.append(sample, Fd.flatten())
                sample = np.append(sample, force)
                sample = np.append(sample, K.flatten())
                residual = np.zeros(self.robot.buffer.num_signals - len(sample))
                sample = np.append(sample, residual)

            elif isinstance(Pd, np.ndarray) and isinstance(Fd, np.ndarray) and isinstance(K, np.ndarray):
                sample = np.append(Pd.flatten(), Fd.flatten())
                sample = np.append(sample, force)
                sample = np.append(sample, K.flatten())
                residual = np.zeros(self.robot.buffer.num_signals - len(sample))
                sample = np.append(sample, residual)

            elif isinstance(Fd, np.ndarray) and isinstance(K, np.ndarray):
                sample = np.append(Fd.flatten(), force)
                sample = np.append(sample, K.flatten())
                residual = np.zeros(self.robot.buffer.num_signals - len(sample))
                sample = np.append(sample, residual)

            elif isinstance(K, np.ndarray):
                sample = np.append(force, K.flatten())
                residual =np.zeros(self.robot.buffer.num_signals - len(sample))
                sample = np.append(residual, sample)
            elif isinstance(Fd, np.ndarray):
                sample = np.append(Fd.flatten(), force)
                residual =np.zeros(self.robot.buffer.num_signals - len(sample))
                sample = np.append(sample, residual)
            else:
                residual = np.zeros(self.robot.buffer.num_signals - len(sample))
                sample = np.append(force, residual)

            self.robot.buffer.write_samples(sample)

    @property
    def working_piece(self):
        return self._working_piece

    @working_piece.setter
    def working_piece(self, material: Material):
        self._working_piece = material
