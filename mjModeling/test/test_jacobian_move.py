from mjModeling.kuka_iiwa_14 import iiwa14
import mujoco
from mjModeling.kinematics import JacobianIK
from mjModeling import robot_scene_xml


robot = iiwa14().create(robot_scene_xml)


def test_jacobian_movement():
    """Test Jacobian IK by moving TCP along X, Y, Z axes"""
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
        ik.move_to_position(target_x, viewer=viewer, kp=10, kd=0.001)
        # TEST 2: Move -5cm in Y  
        print("\n=== TEST 2: Move -0.05m in Y ===")
        target_y = robot.data.site_xpos[tcp_id].copy()
        target_y[1] -= 0.05
        ik.move_to_position(target_y, viewer=viewer, kp=10, kd=0.001)
        # TEST 3: Move -20cm in Z
        print("\n=== TEST 3: Move -0.2m in Z ===")
        target_z = robot.data.site_xpos[tcp_id].copy()
        target_z[2] -= 0.2
        ik.move_to_position(target_z, viewer=viewer, kp=10, kd=0.001)
        print("\n✓ All tests done")
        while viewer.is_running():
            viewer.sync()


# Run the test
if __name__ == "__main__":
    test_jacobian_movement()
