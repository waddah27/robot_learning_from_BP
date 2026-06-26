"""
Per-material cutting-force model (identified from each material's demonstration).

Cutting force is not proportional to depth — it loads up elastically, then
*plateaus* at the material's cutting strength (the blade fractures/cuts the
material). The textbook model is therefore a SATURATING law:

    |F_react| = min(k_material * depth, F_cut_material)

applied along the demonstrated reaction direction. `depth` is how far the blade
tip is below the (live) material surface. Both parameters are identified PER
MATERIAL from its own demonstration:

  - F_cut_material : the demonstrated cutting-force level (a high percentile of
    the demonstrated reaction magnitude). Different per material -> cork, PVC and
    penoplex react with different forces.
  - k_material : loading stiffness, set so the force reaches F_cut at a small
    fraction of the typical demonstrated penetration.

The fit is APPROXIMATE (a physical saturating law, not an exact replay): the
simulated force matches the demonstrated cutting level to within the model's
residual, and expresses material-specific behaviour.
"""
import numpy as np

__all__ = ["CuttingForceModel"]


class CuttingForceModel:
    def __init__(self, k=8000.0, f_cut=50.0):
        self.k = float(k)            # N/m loading stiffness (per material)
        self.f_cut = float(f_cut)    # N   cutting-force plateau (per material)
        self.last_force = np.zeros(3)

    @staticmethod
    def identify(depth_demo, react_demo_world):
        """Identify (k, f_cut) from a material's demonstration.

        react_demo_world: reaction-on-blade in world frame, per phase (N,3).
        depth_demo: demonstrated penetration depth, per phase (N,).
        """
        d = np.asarray(depth_demo, float)
        mag = np.linalg.norm(np.asarray(react_demo_world, float), axis=1)
        m = d > 1e-4
        if m.sum() < 3:
            return 8000.0, 50.0
        f_cut = float(np.percentile(mag[m], 75))          # cutting-force level
        d_typ = float(np.median(d[m]))                    # typical penetration
        k = f_cut / max(0.3 * d_typ, 1e-3)                # reach F_cut at 30% of it
        return k, max(f_cut, 1.0)

    def set_material(self, geom_center, geom_halfsize):
        cc = np.asarray(geom_center); h = np.asarray(geom_halfsize)
        self.top = cc[2] + h[2]
        self.x0, self.x1 = cc[0] - h[0], cc[0] + h[0]
        self.y0, self.y1 = cc[1] - h[1], cc[1] + h[1]

    def compute(self, tip_pos, tip_vel=None, f_des_world=None):
        """World reaction on the blade: saturating magnitude along desired dir."""
        x, y, z = tip_pos
        inside = (self.x0 <= x <= self.x1) and (self.y0 <= y <= self.y1)
        depth = (self.top - z) if (inside and z < self.top) else 0.0
        if depth <= 0.0 or f_des_world is None:
            self.last_force = np.zeros(3)
            return self.last_force
        mag = min(self.k * depth, self.f_cut)               # saturating cutting force
        fdir = np.asarray(f_des_world, float)
        n = np.linalg.norm(fdir)
        direction = (-fdir / n) if n > 1e-6 else np.zeros(3)  # demonstrated reaction dir
        self.last_force = direction * mag
        return self.last_force
