from mjModeling.controllers import VariableImpedanceControl
import numpy as np
from mjModeling.experiments import Experiment
from mjModeling.experiments.motion import InitPos
from mjModeling.mjRobot import Robot
from logger import Logger

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
        desired_z = 0.02
        start_pos = self.robot.data.site_xpos[tcp_id].copy()
        start_pos[2] = desired_z   # override Z with the target depth

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
        # Phase 1: Approach to generic height
        status = self._init_position_for_cutting(viewer)
        if status != 0: return status
        Logger.info(f"\n{'*'*100}\nCUTTING PHASE\n{'*'*100}\n")

        if isinstance(self.controller, VariableImpedanceControl) and self.controller.use_gmr:
            # Phase 2: Perfect Alignment to GMR Start
            p_start = self.controller.traj_loader.pos[0, 0:3]
            start_world = self.controller._gmr_to_world(p_start)
            Logger.debug(f"\n2.1. Aligning to Start: {start_world} ---")
            self.controller.move_to_position(target_pos=start_world, viewer=viewer)

            # Phase 3: Execute Cut (No target_pos provided)
            if hasattr(self.controller, "traj_loader"):
                Logger.debug("\n 2.2 Executing Real-Time GMR Cut ---")
                traj_iter = iter(self.controller.traj_loader)
                for i, move in enumerate(traj_iter):
                    p_raw, v_raw, f_raw = move
                    # Logger.debug(f"move {i}: to {p_raw} -- Fx = {f_raw[0]}, Fy = {f_raw[1]}, Fz = {f_raw[2]}")
                    success = self.controller.move_to_position(use_default=False, target_pos=p_raw, v_raw=v_raw, f_raw=f_raw, viewer=viewer)
                    Logger.debug(f"move is done!" if success else f"move {i} failed!")
                Logger.info(f"\n{'*'*100}\nEND CUTTING\n{'*'*100}\n")
            else:
                raise AttributeError(self.controller, "traj_loader")
        else:
            self._execute_straight_cut(viewer, length_m=0.15, num_waypoints=1)
        return 0
