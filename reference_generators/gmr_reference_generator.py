import sys
from typing import Union

import numpy as np
from data import bpTrajDataLoader, NamedArray

__all__ = ["GMRReferenceGenerator"]

class GMRReferenceGenerator:
    def __init__(self, gmr_sequence: bpTrajDataLoader):
        if isinstance(gmr_sequence, bpTrajDataLoader):
            self.trajectory = gmr_sequence.pos
            self.velocity_profile = gmr_sequence.vel
            self.force_profile = gmr_sequence.force
        else:
            raise TypeError(f"gmr_sequence must be of type bpTrajDataLoader, got {type(gmr_sequence)} instead")

        self.stiffness_profile = None
        self.dt = 0.002 # Default simulation step
        self.time_elapsed = 0.0

    def __iter__(self):
        self.time_elapsed = 0.0
        return self

    def __next__(self):
        # Stop if we exceed the trajectory duration
        if self.time_elapsed > self.trajectory[-1, 0]:
            raise StopIteration

        # Predict using interpolation logic
        refs = self.predict(self.time_elapsed)
        self.time_elapsed += self.dt
        return refs

    def predict(self, time_elapsed):
        # Use a safety clip for the index to prevent out-of-bounds at t=0
        idx = np.searchsorted(self.trajectory[:, 0], time_elapsed)
        idx = max(1, min(idx, len(self.trajectory) - 1))

        alpha = (time_elapsed - self.trajectory[idx-1, 0]) / \
                (self.trajectory[idx, 0] - self.trajectory[idx-1, 0])

        pos_des = self.interpolate(self.trajectory[idx-1:idx+1, 1:4], alpha)
        vel_des = self.interpolate(self.velocity_profile[idx-1:idx+1, 1:4], alpha)
        force_des = self.interpolate(self.force_profile[idx-1:idx+1, 1:4], alpha)

        stiffness_des = None
        if self.stiffness_profile is not None:
            stiffness_des = self.interpolate(self.stiffness_profile[idx-1:idx+1, 1:4], alpha)

        return pos_des, vel_des, force_des, stiffness_des

    def interpolate(self, values, alpha):
        return values[0] * (1 - alpha) + values[1] * alpha
