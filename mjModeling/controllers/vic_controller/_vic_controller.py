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
        k_min, k_max = 400.0, 1500.0
        kp = np.clip(k_max * (error_norm / 0.2), k_min, k_max)
        
        # DAMPING: Critically damped is 2 * sqrt(K). 
        # Over-damp slightly (1.2 multiplier) to stop the shaking.
        kd = 0.5 * np.sqrt(kp) 
        return kp, kd

    def move_to_position(self, target_pos, viewer=None, max_steps=6000):
        tcp_id = self.model.site("scalpel_tip").id
        # Define a 'home' or 'elbow-up' posture (adjust these to your robot's neutral pos)
        q_home = np.zeros(self.model.nq) 
        q_home[1] = -0.5 # Example: slight bend in elbow
        
        self.error_accumulated = np.zeros(3)

        for step in range(max_steps):
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.site_xpos[tcp_id].copy()
            error = target_pos - current_pos
            dist = np.linalg.norm(error)

            if dist < 0.0015: return True # Target reached!

            # 1. Gains: Slightly higher Ki to solve that 2cm gap
            kp, kd = 2000.0, 80.0
            ki = 150.0 
            
            if dist < 0.05:
                self.error_accumulated += error * self.model.opt.timestep

            # 2. Jacobian & Task Space Force
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            
            v_tip = jac @ self.data.qvel
            f_virtual = (kp * error) + (ki * self.error_accumulated) - (kd * v_tip)

            # 3. Solve for Torque using Damped Least Squares (Prevents "Going Crazy" near table)
            # tau = J^T * (JJ^T + lambda^2 * I)^-1 * F
            diag = 0.01 * np.eye(3)
            tau_task = jac.T @ np.linalg.solve(jac @ jac.T + diag, f_virtual)

            # 4. Null-Space: Push the elbow UP to avoid "lying on material"
            # This uses joints that aren't busy moving the tip to maintain posture
            k_posture = 20.0
            d_posture = 2.0
            tau_posture = k_posture * (q_home[:self.model.nv] - self.data.qpos[:self.model.nv]) - d_posture * self.data.qvel
            
            # Project posture torque into the null-space of the main task
            identity = np.eye(self.model.nv)
            # Null space projection matrix: P = I - J_pinv * J
            j_inv = jac.T @ np.linalg.solve(jac @ jac.T + diag, np.eye(3))
            tau_null = (identity - j_inv @ jac) @ tau_posture

            # 5. Combine everything
            self.data.ctrl[:self.model.nu] = tau_task + tau_null + self.data.qfrc_bias[:self.model.nu]

            mujoco.mj_step(self.model, self.data)
            if viewer and step % 30 == 0: viewer.sync()
            
        return False
 