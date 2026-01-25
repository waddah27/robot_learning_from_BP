import numpy as np
from mjModeling.experiments import Experiment
from mjModeling.experiments.motion import InitPos
from mjModeling.mjRobot import Robot


class straightCutting(InitPos):
    def __init__(self, robot: Robot):
        super().__init__(robot) # Ensure parent InitPos is initialized
        self.robot = robot
        self.controller = None

    def _execute_straight_cut(self, viewer, length_m=0.3, num_waypoints=10):
        """Executes a straight line cut with real-time force reporting"""
        if not self.controller:
            return

        tcp_id = self.robot.model.site("scalpel_tip").id
        start_pos = self.robot.data.site_xpos[tcp_id].copy()

        print(f"\n3. Starting Monitored Cut: {start_pos}")
        # Updated header to reflect Magnitude
        print(f"{'Step':<10} | {'Force Mag (N)':<15} | {'Z-Pos (m)':<15}")
        print("-" * 45)

        for i in range(1, num_waypoints + 1):
            fraction = i / num_waypoints
            target_waypoint = start_pos.copy()
            target_waypoint[0] += length_m * fraction

            success = self.controller.move_to_position(target_pos=target_waypoint, viewer=viewer)

            # FIX: Convert the numpy array to a scalar magnitude
            raw_force = self.robot.state["shared_array"][-1]
            if isinstance(raw_force, np.ndarray):
                force_val = np.linalg.norm(raw_force)
            else:
                force_val = raw_force

            z_height = self.robot.data.site_xpos[tcp_id][2]

            # Use force_val (the float) for the format string
            print(f"{i:<10} | {force_val:<15.4f} | {z_height:<15.6f}")

            if not success:
                print("✗ Cut interrupted.")
                return False

        print(f"✓ Cut completed: {length_m*100:.1f}cm path executed.")
        return True


    def execute(self, viewer):
        # 1. Position the robot
        status = self._init_position_for_cutting(viewer)

        # 2. FORCE INJECTION: Manually nudge the TCP deeper into the material
        # If surface is 0.04, moving to 0.03 will trigger the resistance logic
        print("\n--- Lowering scalpel into material for depth ---")
        tcp_id = self.robot.model.site("scalpel_tip").id
        target_deep = self.robot.data.site_xpos[tcp_id].copy()
        target_deep[2] = 0.035  # 5mm below the surface
        self.controller.move_to_position(target_deep, viewer=viewer)

        # 3. Perform the straight cut at this new depth
        if status == 0:
            self._execute_straight_cut(viewer, length_m=0.15, num_waypoints=10)

        return 0

