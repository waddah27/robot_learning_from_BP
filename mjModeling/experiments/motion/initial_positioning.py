from logger import Logger
from mjModeling.conf.configs import TCP_POS
from mjModeling.controllers.vic_controller import BasicVariableImpedanceControl
from mjModeling.experiments import Experiment
from mjModeling import iiwa14
from mjModeling.conf import MATERIAL_GEOM


class InitPos(Experiment):
    def __init__(self, robot: iiwa14):
        self.robot = robot
        self.controller : BasicVariableImpedanceControl = None

    def _init_position_for_cutting(self, viewer):
        """Position robot for cutting WITH VISUALIZATION"""
        if not self.controller:
            Logger.error("self.controller was not set ... No controller was found!")
            return False
        Logger.info(f" \n {'*'*100} \n APPROACHING PHASE \n {'*'*100} \n")

        # Get material position
        mat_id = self.robot.model.geom(MATERIAL_GEOM).id
        mat_center = self.robot.model.geom_pos[mat_id].copy()
        mat_size = self.robot.model.geom_size[mat_id]
        Logger.debug(f"\n Material: center={mat_center}, size={mat_size}\n")
        # Position 1: Safe approach (30cm above)
        approach_pos = mat_center.copy()
        approach_pos[2] = mat_center[2] + mat_size[2] + 0.3  # Top + 30cm
        Logger.debug(f"\n1.1. Moving to approach position: {approach_pos}\n")
        # Visualized move
        if self.controller.move_to_position(target_pos=approach_pos, viewer=viewer):
            self.robot.state[TCP_POS] = approach_pos
            Logger.debug("\n 1.1 SUCCESS: Approach position reached \n")
        else:
            Logger.error("\n 1.1 FAILURE: Failed to reach approach position\n")

        return True

    def execute(self, viewer):
        return self._init_position_for_cutting(viewer)




