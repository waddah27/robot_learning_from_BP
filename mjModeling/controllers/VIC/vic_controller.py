import numpy as np
import mujoco

from mjModeling.mjRobot import Robot


class VariableImpedanceControl:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data

    def get_variable_gains(self, error_norm):
        """
        Logic for 'Variable' impedance: 
        Higher stiffness when far, lower/compliant when close.
        """
        # Example: Linear scaling of stiffness based on distance
        k_min, k_max = 50.0, 500.0
        # High stiffness far away, lower stiffness as we approach
        kp = np.clip(k_max * (error_norm / 0.1), k_min, k_max)
        # Critical damping: d = 2 * sqrt(m * k). Simplified here as a ratio.
        kd = 2 * np.sqrt(kp)
        return kp, kd

    def move_to_position(self, target_pos, viewer=None, max_steps=1000):
        tcp_id = self.model.site("scalpel_tip").id
        nv = self.model.nv
        
        for step in range(max_steps):
            # 1. Get current state in Operational Space
            current_pos = self.data.site_xpos[tcp_id].copy()
            current_vel = np.zeros(6)  # Site velocity
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, tcp_id, current_vel, 0)
            lin_vel = current_vel[3:]  # Linear part

            # 2. Calculate Errors
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < 0.001:  # Tolerance
                return True

            # 3. Dynamic Gain Scheduling (The "Variable" part)
            kp_val, kd_val = self.get_variable_gains(error_norm)
            
            # 4. Define Virtual Force (Impedance Law: F = K*e - B*v)
            # This makes the TCP behave like a spring-damper system
            f_virtual = kp_val * error - kd_val * lin_vel

            # 5. Map Virtual Force to Joint Torques (Generalized Forces)
            # Use Jacobian Transpose: tau = J^T * F
            jac_pos = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None, tcp_id)
            tau = jac_pos.T @ f_virtual

            # 6. Gravity Compensation (Optional but recommended)
            # mj_forward must be called to update qfrc_bias (gravity/coriolis)
            mujoco.mj_forward(self.model, self.data)
            tau += self.data.qfrc_bias[:nv]

            # 7. Apply to Robot
            self.data.ctrl[:nv] = tau
            
            # Step the actual physics engine
            mujoco.mj_step(self.model, self.data)

            if viewer and step % 10 == 0:
                viewer.sync()
                
        return False
