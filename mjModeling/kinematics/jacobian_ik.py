import numpy as np
from mjModeling import *
class JacobianIK:
    def __init__(self, robot):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        
    def move_to_position(self, target_pos, max_steps=500, tolerance=0.001, kp=5.0):
        """
        Move TCP to target position using Jacobian IK
        target_pos: [x, y, z] in world coordinates
        """
        
        tcp_id = self.model.site("scalpel_tip").id
        nv = self.model.nv  # number of degrees of freedom
        
        print(f"\nJacobian IK: Moving to {target_pos}")
        
        for step in range(max_steps):
            # Current TCP position
            current_pos = self.data.site_xpos[tcp_id].copy()
            
            # Position error
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}: error = {error_norm:.6f}, pos = {current_pos}")
            
            # Check if converged
            if error_norm < tolerance:
                print(f"✓ Converged in {step} steps, final error: {error_norm:.6f}")
                return True
            
            # Get position Jacobian (3 x nv)
            jac_pos = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, None, tcp_id)
            
            # Pseudo-inverse of Jacobian
            try:
                jac_pinv = np.linalg.pinv(jac_pos)
            except np.linalg.LinAlgError:
                print("  Warning: Singular Jacobian, using damped least squares")
                damping = 0.01
                jac_pinv = jac_pos.T @ np.linalg.inv(jac_pos @ jac_pos.T + damping * np.eye(3))
            
            # Desired velocity: P-control on position error
            desired_velocity = kp * error
            
            # Joint velocity: dq = J⁺ * v_desired
            dq = jac_pinv @ desired_velocity
            
            # Update joint positions: q = q + dq * dt
            dt = self.model.opt.timestep
            self.data.qpos[:nv] += dq * dt
            
            # Optional: Clamp to joint limits
            # self._clamp_joint_limits()
            
            # Forward kinematics to update everything
            mujoco.mj_forward(self.model, self.data)
        
        print(f"✗ Failed to converge after {max_steps} steps")
        print(f"  Final error: {error_norm:.6f}")
        return False
    
    def move_with_orientation(self, target_pos, target_quat=None, max_steps=500, kp_pos=5.0, kp_ori=1.0):
        """
        Move to position AND orientation (if you need to control rotation)
        target_quat: [w, x, y, z] quaternion
        """
        
        tcp_id = self.model.site("scalpel_tip").id
        nv = self.model.nv
        
        for step in range(max_steps):
            # Current position
            current_pos = self.data.site_xpos[tcp_id].copy()
            pos_error = target_pos - current_pos
            
            if target_quat is not None:
                # Current orientation (from site xmat)
                current_mat = self.data.site_xmat[tcp_id].reshape(3, 3)
                target_mat = quat_to_mat(target_quat)
                
                # Orientation error as axis-angle
                error_mat = target_mat @ current_mat.T
                ori_error = mat_to_axisangle(error_mat)
            else:
                ori_error = np.zeros(3)
            
            # Stack position and orientation errors
            error = np.concatenate([pos_error, ori_error])
            
            # Get 6D Jacobian (position + orientation)
            jac_pos = np.zeros((3, nv))
            jac_ori = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_ori, tcp_id)
            jac = np.vstack([jac_pos, jac_ori])
            
            # Gains for position vs orientation
            gains = np.array([kp_pos, kp_pos, kp_pos, kp_ori, kp_ori, kp_ori])
            desired_twist = gains * error
            
            # Solve: dq = J⁺ * twist
            jac_pinv = np.linalg.pinv(jac)
            dq = jac_pinv @ desired_twist
            
            # Update
            dt = self.model.opt.timestep
            self.data.qpos[:nv] += dq * dt
            mujoco.mj_forward(self.model, self.data)
            
            if np.linalg.norm(pos_error) < 0.001 and np.linalg.norm(ori_error) < 0.01:
                print(f"Converged in {step} steps")
                break