from mjModeling.estimators import Estimator
import mujoco
import numpy as np
from mjModeling import Robot
from mjModeling.conf import (
    FORCE_HISTORY,
    SCALPEL_GEOM
    )
__all__ = ["ImpedanceEstimator"]

class ImpedanceEstimator(Estimator):
    def __init__(self, robot: Robot):
        self.robot = robot

    def get_scalpel_contact_forces(self):
        """Get all contact forces on the scalpel"""
        scalpel_forces = []
        scalpel_geom_id = self.robot.model.geom(SCALPEL_GEOM).id

        for i in range(self.robot.data.ncon):
            contact = self.robot.data.contact[i]

            # Check if scalpel is involved in this contact
            if contact.geom1 == scalpel_geom_id or contact.geom2 == scalpel_geom_id:
                # Get contact force in global frame
                force = np.zeros(6)
                mujoco.mj_contactForce(self.robot.model, self.robot.data, i, force)

                # Extract force vector (first 3 elements are force)
                force_vector = force[:3]
                scalpel_forces.append({
                    'force': force_vector.copy(),
                    'position': contact.pos.copy(),
                    'frame': contact.frame.copy()
                })

                # Debug print
                # print(f"Contact {i}: force = {np.linalg.norm(force_vector):.4f} N")

        return scalpel_forces

    def get_total_cutting_force(self):
        """Get total force vector on scalpel (sum of all contacts)"""
        contacts = self.get_scalpel_contact_forces()
        total_force = np.zeros(3)

        for i,contact in enumerate(contacts):
            total_force += contact['force']
        return total_force

    def record_force_step(self):
        """Record current cutting force for history"""
        force = self.get_total_cutting_force()
        self.robot.state.get(FORCE_HISTORY).append(force.copy())
        return force

    def get_force_magnitude(self):
        """Get magnitude of total cutting force"""
        force = self.get_total_cutting_force()
        return np.linalg.norm(force)

    def estimate(self):...
