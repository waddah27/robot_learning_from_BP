from mjModeling.controllers import VariableImpedanceControl
import numpy as np
from mjModeling.experiments import Experiment
from mjModeling.experiments.motion import InitPos
from mjModeling.mjRobot import Robot


class straightCutting(InitPos):
    def __init__(self, robot: Robot):
        super().__init__(robot) # Ensure parent InitPos is initialized
        self.robot = robot
        self.controller = None

    def _execute_straight_cut(self, viewer, length_m=0.3, num_waypoints=1):
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
        print("=================================DONE CUTTING EPISODE===========================")
        return True


    def execute(self, viewer):
        # 1. Approach
        status = self._init_position_for_cutting(viewer)
        if status != 0: return status

        if isinstance(self.controller, VariableImpedanceControl) and self.controller.use_gmr:
            print("\n--- Starting Raw Trajectory Cut via iter/next ---")

            # Align to first point of raw data
            start_p = self.controller.traj_loader.pos[0, 1:4]
            start_world = self.controller._gmr_to_world(start_p)
            self.controller.move_to_position(target_pos=start_world, viewer=viewer)

            # TRIGGER CUT: Call without target_pos
            self.controller.move_to_position(viewer=viewer)
        else:
            self._execute_straight_cut(viewer, length_m=0.15, num_waypoints=1)

        return 0
