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

    def move_to_position(self, target_pos, viewer=None, max_steps=8000):
        tcp_id = self.model.site("scalpel_tip").id
        # Use a lower control frequency (e.g., 500Hz if timestep is 0.001)
        control_decimation = 5 
        
        self.error_accumulated = np.zeros(3)
        
        for step in range(max_steps):
            # 1. Update State
            mujoco.mj_forward(self.model, self.data)
            
            # Only update control signal every few steps to stop fast shaking
            if step % control_decimation == 0:
                current_pos = self.data.site_xpos[tcp_id].copy()
                error = target_pos - current_pos
                dist = np.linalg.norm(error)
                
                if dist < 0.002: return True

                # 2. Variable Gains (Reduced for stability)
                # High-frequency shake usually means KD is too high for the simulation
                kp = 1200.0 if dist > 0.05 else 2500.0
                kd = 2 * 0.5 * np.sqrt(kp) # Reduced damping ratio to 0.5

                # 3. Task Space Mapping
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
                
                # Integral term (Solves the "High Error" steady-state issue)
                if dist < 0.1:
                    self.error_accumulated += error * (self.model.opt.timestep * control_decimation)
                
                f_virtual = (kp * error) + (40.0 * self.error_accumulated) - (kd * (jac @ self.data.qvel))

                # 4. Torque Calculation + Bias
                # Transpose is safer than Inverse for high-error starts
                tau = jac.T @ f_virtual
                
                # Apply ONLY to controlled joints
                self.data.ctrl[:self.model.nu] = (tau + self.data.qfrc_bias)[:self.model.nu]

            # 5. Physics Step
            mujoco.mj_step(self.model, self.data)

            if viewer and step % 50 == 0:
                viewer.sync()
                
        return False
 