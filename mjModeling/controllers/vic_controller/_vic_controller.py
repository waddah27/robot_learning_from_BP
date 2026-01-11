import numpy as np
import mujoco
from mjModeling.conf import paramVIC
from mjModeling.controllers.controller_api import Controller
from mjModeling.mjRobot import Robot
__all__ = ["VariableImpedanceControl"]


class VariableImpedanceControl(Controller): # Removed parent for standalone clarity
    def __init__(self, robot):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.error_accumulated = np.zeros(3) # For Integral term

    def get_variable_gains(self, error_norm):
        # STABILITY: Lower the max stiffness. 
        # Most MuJoCo robots explode above 2000-5000 if timestep is 0.002
        k_min, k_max = paramVIC.VIC_KP_MIN.value, paramVIC.VIC_KP_MAX.value
        kp = np.clip(k_max * (error_norm / 0.2), k_min, k_max)
        # DAMPING: Critically damped is 2 * sqrt(K). 
        # Over-damp slightly (1.2 multiplier) to stop the shaking.
        kd = 0.5 * np.sqrt(kp) 
        return kp, kd

    def sim_cutting_resistance(self, current_pos, v_tip, magnitude=100):
        """simulate cutting resistance: here we can test different 
        force reactions from different materials to validate the research 
        results"""
        # Check if tip is inside the box (z < 0.04)
        if current_pos[2] < 0.04:
            # Simulate resistance: add an upward force proportional to velocity
            return -magnitude * v_tip[2] 
        return 0.0

    def move_to_position(self, target_pos, viewer=None, max_steps=8000):
        tcp_id = self.model.site("scalpel_tip").id
        # Define 'home' posture to keep the elbow up (joint angles in radians)
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 0.0]) 
        
        self.error_accumulated = np.zeros(3)
        # Use a small epsilon for Damped Least Squares stability
        lambda_sq = paramVIC.VIC_LAMBDA_SQ.value 

        for step in range(max_steps):
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.site_xpos[tcp_id].copy()
            error = target_pos - current_pos
            dist = np.linalg.norm(error)

            # 2mm tolerance for 2026 surgical/precision tasks
            if dist < paramVIC.VIC_TOL.value: 
                return True 

            # 1. VARIABLE GAIN SCHEDULING
            # High stiffness far away, lower stiffness for delicate contact
            kp_val, kd_val = self.get_variable_gains(dist)
            # 2. INTEGRAL TERM (The "Closer")
            # Only accumulate when within 5cm to prevent huge overshoots
            if dist < 0.05:
                # ki=200 is strong enough to compensate for steady-state error
                self.error_accumulated += error * self.model.opt.timestep
            ki_val = paramVIC.VIC_KI.value

            # 3. TASK SPACE FORCE
            v_tip = (np.zeros(3))
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            v_tip = jac @ self.data.qvel
            
            # F = Kp*e + Ki*∫e - Kd*v
            f_virtual = (kp_val * error) + (ki_val * self.error_accumulated) - (kd_val * v_tip)
            f_virtual += self.sim_cutting_resistance(current_pos, v_tip)
            # 4. STABLE MAPPING (Damped Least Squares)
            # Solves: tau = J^T * inv(JJ^T + λ^2I) * F
            jjt = jac @ jac.T
            tau_task = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), f_virtual)

            # 5. NULL-SPACE POSTURE CONTROL (Fixes "lying on material")
            # Keeps the robot elbow up while the tip follows target_pos
            k_posture, d_posture = 10.0, 2.0
            tau_posture = k_posture * (q_home[:self.model.nv] - self.data.qpos[:self.model.nv]) - d_posture * self.data.qvel
            
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

            if viewer and step % 4 == 0:
                viewer.sync()
                
        return False
 