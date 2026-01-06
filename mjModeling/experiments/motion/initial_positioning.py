import numpy as np
from mjModeling.experiments import Experiment
from mjModeling.kinematics import JacobianIK
from mjModeling import Robot
from mjModeling.conf import MATERIAL_GEOM


class InitPos(Experiment):
    def __init__(self, robot: Robot):
        self.robot = robot

    def execute(self, viewer):
        return self._init_position_for_cutting(viewer)

    def _init_position_for_cutting(self, viewer):
        """Position robot for cutting WITH VISUALIZATION"""

        ik = JacobianIK(self.robot)
        # Get material position
        mat_id = self.robot.model.geom(MATERIAL_GEOM).id
        mat_center = self.robot.model.geom_pos[mat_id].copy()
        mat_size = self.robot.model.geom_size[mat_id]
        print(f"Material: center={mat_center}, size={mat_size}")
        # Position 1: Safe approach (30cm above)
        approach_pos = mat_center.copy()
        approach_pos[2] = mat_center[2] + mat_size[2] + 0.3  # Top + 30cm
        print(f"\n1. Moving to approach position: {approach_pos}")
        # Visualized move
        success1 = ik.move_to_position(target_pos=approach_pos, viewer=viewer)
        if success1:
            print("✓ Approach position reached")
        else:
            print("✗ Failed to reach approach position")
        # Position 2: Cutting height (5cm above)
        cut_pos = mat_center.copy()
        cut_pos[2] = mat_center[2] #+ mat_size[2] + 0.05  # Top + 5cm
        print(f"\n2. Moving to cutting height: {cut_pos}")
        # Visualized move
        success2 = ik.move_to_position(target_pos=cut_pos, viewer=viewer)
        if success2:
            print("✓ Cutting height reached")
        else:
            print("✗ Failed to reach cutting height")
        # Final verification
        tcp_id = self.robot.model.site("scalpel_tip").id
        final_pos = self.robot.data.site_xpos[tcp_id]
        print(f"\n✓ Final TCP: {final_pos}")
        print(f"  Desired: {cut_pos}")
        print(f"  Error: {np.linalg.norm(final_pos - cut_pos):.6f}m")
        return 0
