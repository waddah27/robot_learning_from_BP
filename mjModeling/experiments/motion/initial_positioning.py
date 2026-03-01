from logger import Logger
from mjModeling.controllers.vic_controller import BasicVariableImpedanceControl
import numpy as np
from mjModeling.controllers.controller_api import Controller
from mjModeling.experiments import Experiment
from mjModeling.controllers import JacobianIK
from mjModeling import Robot
from mjModeling.conf import MATERIAL_GEOM


class InitPos(Experiment):
    def __init__(self, robot: Robot):
        self.robot = robot
        self.controller : BasicVariableImpedanceControl = None

    def _init_position_for_cutting(self, viewer):
        """Position robot for cutting WITH VISUALIZATION"""
        if not self.controller:
            Logger.error("self.controller was not set ... No controller was found!")
            return
        Logger.info(f" \n {'*'*100} \n APPROACHING PHASE \n {'*'*100} \n")

        # Get material position
        mat_id = self.robot.model.geom(MATERIAL_GEOM).id
        mat_center = self.robot.model.geom_pos[mat_id].copy()
        mat_size = self.robot.model.geom_size[mat_id]
        Logger.debug(f"Material: center={mat_center}, size={mat_size}")
        # Position 1: Safe approach (30cm above)
        approach_pos = mat_center.copy()
        approach_pos[2] = mat_center[2] + mat_size[2] + 0.3  # Top + 30cm
        Logger.debug(f"\n1.1. Moving to approach position: {approach_pos}")
        # Visualized move
        success1 = self.controller.move_to_position(target_pos=approach_pos, viewer=viewer)
        if success1:
            Logger.debug("✓ Approach position reached")
        else:
            Logger.debug("✗ Failed to reach approach position")
        # Position 2: Cutting height (5cm above)
        cut_pos = mat_center.copy()
        cut_pos[2] = mat_center[2] + 0.02 #+ mat_size[2] + 0.05  # Top + 5cm
        Logger.debug(f"\n1.2. Moving to cutting height: {cut_pos}")
        # Visualized move
        success2 = self.controller.move_to_position(target_pos=cut_pos, viewer=viewer)
        if success2:
            Logger.debug("✓ Cutting height reached")
        else:
            Logger.debug("✗ Failed to reach cutting height")
        # Final verification
        tcp_id = self.robot.model.site("scalpel_tip").id
        final_pos = self.robot.data.site_xpos[tcp_id]
        err = np.linalg.norm(final_pos - cut_pos)
        Logger.debug(f"\n✓ reached Final TCP: actual {final_pos} -- desired {cut_pos} -- err = {err} ")
        Logger.debug(f"\n {'*'*100} \n Robot tcp at final pos .. start cutting phase... \n {'*'*100} \n")
        return 0

    def execute(self, viewer):
        return self._init_position_for_cutting(viewer)




