import mujoco
import numpy as np
from mjModeling.estimators import Estimator
from mjModeling.conf import FORCE_HISTORY, SCALPEL_GEOM

class ImpedanceEstimator(Estimator):
    def __init__(self, robot):
        self.robot = robot

    def get_total_cutting_force(self):
        """Standardizes force estimation by combining contact and constraints."""
        # If the analytical cutting-force model is active, it IS the cutting force
        # (the box-on-box geometric contact has been disabled).
        f_cut = getattr(self.robot, "_applied_cut_force", None)
        if f_cut is not None:
            return np.asarray(f_cut, dtype=float).copy()

        total_force = np.zeros(3)
        scalpel_geom_id = self.robot.model.geom(SCALPEL_GEOM).id

        # 1. Geometric Contact (Hard Impacts)
        for i in range(self.robot.data.ncon):
            contact = self.robot.data.contact[i]
            if contact.geom1 == scalpel_geom_id or contact.geom2 == scalpel_geom_id:
                force = np.zeros(6)
                mujoco.mj_contactForce(self.robot.model, self.robot.data, i, force)
                total_force += force[:3]

        # When there is NO geometric scalpel↔material contact the true cutting
        # force is ~0 (the blade is in free space / the open groove). The old
        # fallback mapped the FULL joint constraint vector (joint limits, the
        # material slide-joint limit, posture/null-space constraints) to the tip
        # via pinv(J^T)·qfrc_constraint — i.e. it reported non-cutting constraint
        # reactions as "cutting force" (spikes to ~118N when the real contact is
        # zero). That is a measurement artifact, so we report 0 instead.
        return total_force

    def get_force_magnitude(self):
        return np.linalg.norm(self.get_total_cutting_force())

    def estimate(self, *args, **kwargs):
        return super().estimate(*args, **kwargs)