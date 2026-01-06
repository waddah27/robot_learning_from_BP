import numpy as np
import mujoco
import mujoco.viewer
from mjModeling.conf import MATERIAL_GEOM, ROBOT_SCENE
from mjModeling.kuka_iiwa_14 import iiwa14
from mjModeling.kinematics import JacobianIK
robot = iiwa14().create(ROBOT_SCENE)


def test_positioning():
    """Test Jacobian IK positioning"""
    # Create IK solver
    ik = JacobianIK(robot)

    # Open viewer
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:

        print("Initial TCP position:")
        tcp_id = robot.model.site("scalpel_tip").id
        start_pos = robot.data.site_xpos[tcp_id].copy()
        print(f"  {start_pos}")

        # Test 1: Move above material
        print("\n=== Test 1: Move above material ===")
        target1 = np.array([0.5, 0.0, 0.3])  # X=0.5, Y=0, Z=0.3
        success = ik.move_to_position(target1, max_steps=300)

        if success:
            current = robot.data.site_xpos[tcp_id]
            print(f"✓ Moved to: {current}")
            print(f"  Error: {np.linalg.norm(current - target1):.6f}")

        # Test 2: Move down closer
        print("\n=== Test 2: Move closer to material ===")
        target2 = np.array([0.5, 0.0, 0.1])  # 10cm above
        ik.move_to_position(target2, max_steps=200)

        # Test 3: Small circle to test IK
        print("\n=== Test 3: Circle motion ===")
        center = robot.data.site_xpos[tcp_id].copy()
        radius = 0.05

        for i in range(50):
            angle = 2 * np.pi * i / 50
            target_circle = center.copy()
            target_circle[0] += radius * np.cos(angle)
            target_circle[1] += radius * np.sin(angle)

            # Fast move - few steps per waypoint
            ik.move_to_position(target_circle, max_steps=10, kp=10.0)
            viewer.sync()

        print("\n✓ All IK tests complete")

        while viewer.is_running():
            viewer.sync()


def position_for_cutting_no_viewer():
    """One function to position robot for cutting"""

    ik = JacobianIK(robot)

    # Get material position
    mat_id = robot.model.geom(MATERIAL_GEOM).id
    mat_center = robot.model.geom_pos[mat_id].copy()
    mat_size = robot.model.geom_size[mat_id]

    print(f"Material: center={mat_center}, size={mat_size}")

    # Position 1: Safe approach (30cm above)
    approach_pos = mat_center.copy()
    approach_pos[2] = mat_center[2] + mat_size[2] + 0.3  # Top + 30cm
    print(f"\n1. Moving to approach position: {approach_pos}")
    ik.move_to_position(approach_pos, max_steps=400)

    # Position 2: Cutting height (5cm above)
    cut_pos = mat_center.copy()
    cut_pos[2] = mat_center[2] + mat_size[2] + 0.05  # Top + 5cm
    print(f"\n2. Moving to cutting height: {cut_pos}")
    ik.move_to_position(cut_pos, max_steps=200)

    # Verify
    tcp_id = robot.model.site("scalpel_tip").id
    final_pos = robot.data.site_xpos[tcp_id]
    print(f"\n✓ Final TCP: {final_pos}")
    print(f"  Desired: {cut_pos}")
    print(f"  Error: {np.linalg.norm(final_pos - cut_pos):.6f}m")

    return robot


def position_for_cutting_with_viewer():
    """Position robot for cutting WITH VISUALIZATION"""

    # Open viewer FIRST
    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        print("Viewer opened. Robot should appear.")

        # Wait a moment to see initial state
        for _ in range(100):
            viewer.sync()

        ik = JacobianIK(robot)

        # Get material position
        mat_id = robot.model.geom(MATERIAL_GEOM).id
        mat_center = robot.model.geom_pos[mat_id].copy()
        mat_size = robot.model.geom_size[mat_id]

        print(f"Material: center={mat_center}, size={mat_size}")

        # Position 1: Safe approach (30cm above)
        approach_pos = mat_center.copy()
        approach_pos[2] = mat_center[2] + mat_size[2] + 0.3  # Top + 30cm
        print(f"\n1. Moving to approach position: {approach_pos}")

        # Visualized move
        success1 = ik.move_to_position(approach_pos, viewer, max_steps=400)

        if success1:
            print("✓ Approach position reached")
        else:
            print("✗ Failed to reach approach position")

        # Position 2: Cutting height (5cm above)
        cut_pos = mat_center.copy()
        cut_pos[2] = mat_center[2] + mat_size[2] + 0.05  # Top + 5cm
        print(f"\n2. Moving to cutting height: {cut_pos}")

        # Visualized move
        success2 = ik.move_to_position(cut_pos, viewer, max_steps=200)

        if success2:
            print("✓ Cutting height reached")
        else:
            print("✗ Failed to reach cutting height")

        # Final verification
        tcp_id = robot.model.site("scalpel_tip").id
        final_pos = robot.data.site_xpos[tcp_id]
        print(f"\n✓ Final TCP: {final_pos}")
        print(f"  Desired: {cut_pos}")
        print(f"  Error: {np.linalg.norm(final_pos - cut_pos):.6f}m")

        # Keep viewer open
        print("\nPositioning complete. Close viewer or press ESC.")
        while viewer.is_running():
            viewer.sync()

    return robot


# SIMPLER VERSION - Just step manually with viewer:
def position_for_cutting_manual_viewer():
    """Even simpler - manual stepping with viewer"""

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        print("Viewer opened. Close with ESC.")

        ik = JacobianIK(robot)

        # Get positions
        mat_id = robot.model.geom(MATERIAL_GEOM).id
        mat_center = robot.model.geom_pos[mat_id].copy()
        mat_size = robot.model.geom_size[mat_id]

        approach_pos = mat_center.copy()
        approach_pos[2] = mat_center[2] + mat_size[2] + 0.3

        cut_pos = mat_center.copy()
        cut_pos[2] = mat_center[2] + mat_size[2] + 0.05

        print(f"\nPress '1' to move to approach position")
        print(f"Press '2' to move to cutting height")
        print(f"Press ESC to exit")

        step_count = 0

        while viewer.is_running():
            # Get current TCP
            tcp_id = robot.model.site("scalpel_tip").id
            current_pos = robot.data.site_xpos[tcp_id]

            # Display every 100 steps
            if step_count % 100 == 0:
                print(
                    f"\rTCP: [{
                        current_pos[0]:.3f}, {
                        current_pos[1]:.3f}, {
                        current_pos[2]:.3f}]",
                    end="")

            # Just step simulation (no movement unless if add control)
            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            step_count += 1

        print("\nViewer closed.")

# QUICK TEST - Run this:


def quick_visual_test():
    """Just open viewer and show robot"""

    print("Opening viewer...")
    print("We should see:")
    print("1. Robot with scalpel")
    print("2. Green material block")
    print("3. Red TCP site at scalpel tip")

    with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
        # Show TCP and other visual aids
        with viewer.lock():
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE  # Show site names
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE  # Show site frames
            viewer.opt.sitegroup = 3  # Show all sites

        print("\nViewer controls:")
        print("- Mouse drag: rotate view")
        print("- Right click + drag: pan")
        print("- Scroll: zoom")
        print("- ESC: close")

        step = 0
        while viewer.is_running():
            # Optional: Add small movement to test
            if step < 100:
                # Try moving joint 5 a tiny bit
                if robot.model.nq > 5:
                    robot.data.qpos[5] += 0.001

            mujoco.mj_step(robot.model, robot.data)
            viewer.sync()

            step += 1

        print("\nViewer closed.")


# Run the quick test FIRST:
if __name__ == "__main__":
    position_for_cutting_with_viewer()
    # test_positioning()
    # quick_visual_test()
    # position_for_cutting_no_viewer()
