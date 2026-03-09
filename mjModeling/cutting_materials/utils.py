import numpy as np
import mujoco
from mjModeling.conf import MATERIAL_GEOM
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14

__all__ = [
    "wp_sine_motion"
]
def wp_sine_motion(dt:float, mot_amplitude: float = 0.02, mot_freq:float = 0.5):
     return mot_amplitude * np.sin(2 * np.pi * mot_freq * dt)



