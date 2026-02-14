import numpy as np
from data import bpTrajDataLoader

__all__ = ["GMRReferenceGenerator"]


class GMRReferenceGenerator:
    def __init__(self, gmr_sequence: bpTrajDataLoader):
        self.trajectory = gmr_sequence.pos
        self.velocity_profile = gmr_sequence.vel
        self.force_profile = gmr_sequence.force
        self.stiffness_profile = None # should be from optimizer
        self.current_idx = 0

    def get_reference(self, time_elapsed):
        # Interpolate to get current desired values
        idx = np.searchsorted(self.trajectory[:, 0], time_elapsed)
        alpha = (time_elapsed - self.trajectory[idx-1, 0]) / \
                (self.trajectory[idx, 0] - self.trajectory[idx-1, 0])

        pos_des = self.interpolate(self.trajectory[idx-1:idx+1, 1:4], alpha)
        vel_des = self.interpolate(self.velocity_profile[idx-1:idx+1, 1:4], alpha)
        force_des = self.interpolate(self.force_profile[idx-1:idx+1, 1:4], alpha)

        if self.stiffness_profile is not None:
            stiffness_des = self.interpolate(self.stiffness_profile[idx-1:idx+1, 1:4], alpha)
        else:
            stiffness_des = None

        return pos_des, vel_des, force_des, stiffness_des

    def interpolate(self, values, alpha):
        return values[0] * (1 - alpha) + values[1] * alpha