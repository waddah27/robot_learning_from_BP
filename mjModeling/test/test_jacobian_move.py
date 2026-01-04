from mjModeling.kuka_iiwa_14 import iiwa14
from mjModeling import *
from mjModeling.kinematics import JacobianIK

def test_jacobian_movement():
    """Test Jacobian IK by moving TCP along X, Y, Z axes"""
    
    robot = iiwa14().create(robot_scene_xml)
    
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        print("Testing Jacobian IK movement...")
        
        ik = JacobianIK(robot)
        tcp_id = robot.model.site("scalpel_tip").id
        
        # Get start position
        start_pos = robot.data.site_xpos[tcp_id].copy()
        print(f"Start TCP: {start_pos}")
        
        # TEST 1: Move +10cm in X
        print("\n=== TEST 1: Move +0.1m in X ===")
        target_x = start_pos.copy()
        target_x[0] += 0.1
        ik.move_to_position(target_x, viewer=viewer)
        
        # TEST 2: Move -5cm in Y  
        print("\n=== TEST 2: Move -0.05m in Y ===")
        target_y = robot.data.site_xpos[tcp_id].copy()
        target_y[1] -= 0.05
        ik.move_to_position(target_y, viewer=viewer)
        
        # TEST 3: Move -20cm in Z
        print("\n=== TEST 3: Move -0.2m in Z ===")
        target_z = robot.data.site_xpos[tcp_id].copy()
        target_z[2] -= 0.2
        ik.move_to_position(target_z, viewer=viewer)
        
        print("\n✓ All tests done")
        while viewer.is_running():
            viewer.sync()

# MODIFIED JacobianIK class:
class JacobianIK:
    def __init__(self, robot):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
    
    def move_to_position(self, target_pos, viewer=None, max_steps=500, kp=10.0):
        """Move TCP to target using Jacobian IK"""
        
        tcp_id = self.model.site("scalpel_tip").id
        nv = self.model.nv
        
        print(f"Moving TCP from {self.data.site_xpos[tcp_id]} to {target_pos}")
        
        for step in range(max_steps):
            # Current position
            current = self.data.site_xpos[tcp_id].copy()
            error = target_pos - current
            error_norm = np.linalg.norm(error)
            
            if step % 10 == 0:
                print(f"  Step {step}: error={error_norm:.6f}")
            
            if error_norm < 0.001:
                print(f"  ✓ Converged in {step} steps")
                break
            
            # Get Jacobian
            jac = np.zeros((3, nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            
            # Damped pseudo-inverse
            damping = 0.01
            jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping * np.eye(3))
            
            # Desired velocity
            v_desired = kp * error
            
            # Joint velocity
            dq = jac_pinv @ v_desired
            
            # Update: q = q + dq * dt
            dt = self.model.opt.timestep
            self.data.qpos[:nv] += dq * dt
            
            # Update
            mujoco.mj_forward(self.model, self.data)
            
            # Sync viewer if provided
            if viewer:
                viewer.sync()
        
        return self.data.qpos[:nv].copy()

# Run the test
if __name__ == "__main__":
    test_jacobian_movement()