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

    def get_const_gains(self, kp, m):
        kd = 2 * 0.7 * np.sqrt(kp*m)
        return (kp, kd, m)

    def get_variable_gains(self, error_norm):
        # Scale these down significantly for stability
        k_min, k_max = paramVIC.VIC_KP_MIN.value, paramVIC.VIC_KP_MAX.value
        # Smoothly interpolate stiffness
        kp = np.clip(k_max * (error_norm / 0.2), k_min, k_max)
        # Damping ratio (0.7 to 1.0 is "critical" and smooth)
        kd = 2 * 0.7 * np.sqrt(kp) 
        return kp, kd

    def move_to_position(self, target_pos, viewer=None, max_steps=paramVIC.VIC_MAX_STEPS.value,
                         tolerance=paramVIC.VIC_TOL.value):
        tcp_id = self.model.site("scalpel_tip").id
        nv = self.model.nv
        
        for step in range(max_steps):
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[tcp_id].copy()
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                return True

            # 1. Jacobian for Transpose mapping
            jac_pos = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None, tcp_id)
            
            # 2. Linear velocity in Task Space
            lin_vel = jac_pos @ self.data.qvel

            # 3. Virtual Force Calculation
            # kp_val, kd_val = self.get_variable_gains(error_norm)
            kp_val, kd_val, _ = self.get_const_gains(paramVIC.VIC_KP_MAX.value,
                                                     paramVIC.VIC_M.value)
            f_virtual = (kp_val * error) - (kd_val * lin_vel)

            # 4. Jacobian Transpose: tau = J^T * F
            # This is significantly more stable than the DLS solve for VIC
            tau = jac_pos.T @ f_virtual

            # 5. Gravity/Coriolis Compensation
            # Map full bias forces to only the actuated degrees of freedom
            tau_total = tau + self.data.qfrc_bias
            
            # 6. Apply to actuators (ensuring index match)
            self.data.ctrl[:nv] = tau_total[:nv]
            
            mujoco.mj_step(self.model, self.data)
            sync_rate = max_steps/10

            if viewer and step % sync_rate == 0:
                viewer.sync()
        return False
 