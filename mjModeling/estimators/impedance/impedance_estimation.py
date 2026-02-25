import mujoco
import numpy as np
from mjModeling.estimators import Estimator
from mjModeling.conf import FORCE_HISTORY, SCALPEL_GEOM

class ImpedanceEstimator(Estimator):
    def __init__(self, robot):
        self.robot = robot

    def get_total_cutting_force(self):
        """Standardizes force estimation by combining contact and constraints."""
        total_force = np.zeros(3)
        scalpel_geom_id = self.robot.model.geom(SCALPEL_GEOM).id

        # 1. Geometric Contact (Hard Impacts)
        for i in range(self.robot.data.ncon):
            contact = self.robot.data.contact[i]
            if contact.geom1 == scalpel_geom_id or contact.geom2 == scalpel_geom_id:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.robot.model, self.robot.data, i, force)
                total_force += force[:3]

        # 2. Constraint Forces (Essential for sliding/penetration)
        # If geometric contacts are zero, read the solver's internal constraint forces
        if np.linalg.norm(total_force) < 0.01:
            jac = np.zeros((3, self.robot.model.nv))
            tcp_id = self.robot.model.site("scalpel_tip").id
            mujoco.mj_jacSite(self.robot.model, self.robot.data, jac, None, tcp_id)
            # F = (J^T)^+ * qfrc_constraint
            q_force = self.robot.data.qfrc_constraint[:self.robot.model.nv]
            total_force = np.linalg.pinv(jac.T) @ q_force

        return total_force

    def get_force_magnitude(self):
        return np.linalg.norm(self.get_total_cutting_force())

    def estimate(self, *args, **kwargs):
        return super().estimate(*args, **kwargs)