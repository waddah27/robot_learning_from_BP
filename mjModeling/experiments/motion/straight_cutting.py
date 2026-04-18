from typing import Union

from mjModeling.conf.configs import paramVIC
from mjModeling.controllers import BpVariableImpedanceControl, ContinuousTrajectoryVIC
import numpy as np
from mjModeling.experiments import Experiment
from mjModeling.experiments.motion import InitPos
from mjModeling.mjRobot import Robot
from logger import Logger

class straightCutting(InitPos):
    def __init__(self, robot: Robot, use_continuous=paramVIC.CONTINUOUS_TRAJ):
        super().__init__(robot)
        self.robot = robot
        self.use_continuous = use_continuous
        self.controller: Union[BpVariableImpedanceControl, ContinuousTrajectoryVIC] = None

    def execute(self, viewer):
        # Phase 1: Approach
        if not self._init_position_for_cutting(viewer):
            return False
        Logger.info(f"\n{'*'*100}\nCUTTING PHASE\n{'*'*100}\n")

        if isinstance(self.controller, BpVariableImpedanceControl) and self.controller.use_bp:
            # Align to start
            p_start = self.controller.traj_loader.pos[0, 0:3]
            start_world = self.controller._gmr_to_world(p_start)
            Logger.debug(f"\n2.1. Aligning to Start: {start_world} ---\n")
            if self.controller.move_to_position(target_pos=start_world, viewer=viewer):
                Logger.debug("\n 2.1 SUCESS: Reached pos successfully .. !\n")
            else:
                Logger.error("\n 2.1 FAILURE: failed to reach commanded position!\n")
            if self.use_continuous and hasattr(self.controller, 'follow_trajectory'):
                # Use continuous trajectory tracking
                Logger.debug("\n 2.2 Executing Continuous GMR Trajectory ---\n")
                if self.controller.follow_trajectory(phase_speed=paramVIC.PHASE_SPEED, viewer=viewer):
                    Logger.info(f"\n{'*'*100}\nEND CUTTING\n{'*'*100}\n")
                    return True
                else:
                    Logger.error("\n Trajectory execution did not complete.\n")
                    return False
            else:
                # Fallback to per‑waypoint (original behaviour)
                Logger.debug("\n 2.2 Executing Per‑Waypoint GMR Cut ---\n")
                traj_iter = iter(self.controller.traj_loader)
                for i, move in enumerate(traj_iter):
                    p_raw, v_raw, f_raw = move
                    success = self.controller.move_to_position(use_default=False,
                                                                target_pos=p_raw,
                                                                v_raw=v_raw,
                                                                f_raw=f_raw,
                                                                viewer=viewer)
                    Logger.debug(f"\n move {i} {'done' if success else 'failed'}!\n")
                Logger.info(f"\n{'*'*100}\nEND CUTTING\n{'*'*100}\n")
        else:
            self._execute_default_straight_cut(viewer, length_m=0.15, num_waypoints=1)
        return 0