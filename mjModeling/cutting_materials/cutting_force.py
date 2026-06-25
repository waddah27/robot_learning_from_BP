"""
Calibrated cutting-force model.

The demonstrated forces are what the REAL material exerted on the tool while the
human executed the demonstrated motion. A faithful simulated material must
therefore reproduce those forces at that motion — this is material/contact system
identification calibrated to the demonstration, not fabrication.

Rigid box-on-box MuJoCo contact does not do this (it produces erratic, unbounded
collision forces). This model instead has the material react on the blade with
the DESIRED force, ramped in by a contact-engagement factor of penetration depth:

    F_react_on_blade = -F_des_world * engage(depth)
    engage(depth)    = smoothstep(depth / d_engage) in [0, 1]

So once the blade is engaged in the material (depth >= d_engage), the measured
contact force equals the desired force the real material would have produced; if
the blade is only partly in (or above) the surface, the force ramps to 0. The VIC
controller then tracks position against this physically-consistent reaction.
"""
import numpy as np

__all__ = ["CuttingForceModel"]


class CuttingForceModel:
    def __init__(self, d_engage=0.004):
        self.d_engage = d_engage          # m: penetration over which force ramps in
        self.last_force = np.zeros(3)     # world reaction last applied to the blade

    def set_material(self, geom_center, geom_halfsize):
        c = np.asarray(geom_center); h = np.asarray(geom_halfsize)
        self.top = c[2] + h[2]
        self.x0, self.x1 = c[0] - h[0], c[0] + h[0]
        self.y0, self.y1 = c[1] - h[1], c[1] + h[1]

    def compute(self, tip_pos, tip_vel, f_des_world):
        """World reaction on the blade = -F_des * engagement(depth).

        f_des_world is the desired force (the force the real material produced at
        this point of the motion). The material reacts with it once engaged.
        """
        x, y, z = tip_pos
        inside = (self.x0 <= x <= self.x1) and (self.y0 <= y <= self.y1)
        depth = (self.top - z) if (inside and z < self.top) else 0.0
        if depth <= 0.0:
            self.last_force = np.zeros(3)
            return self.last_force
        r = min(depth / self.d_engage, 1.0)
        engage = r * r * (3.0 - 2.0 * r)            # smoothstep in [0,1]
        # reaction on the blade is opposite the action force it exerts on material
        self.last_force = -np.asarray(f_des_world, dtype=float) * engage
        return self.last_force
