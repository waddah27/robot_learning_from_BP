from logger import Logger
from typing import Union

from mjModeling.cutting_materials.utils import wp_sine_motion
from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14
import numpy as np
import mujoco
from cvxopt import matrix, solvers
from mjModeling.controllers.vic_controller import BasicVariableImpedanceControl
from mjModeling.conf import paramVIC, MATERIAL_GEOM
from data import bpTrajDataLoader, NamedArray
from reference_generators import GMRReferenceGenerator

__all__ = ["BpVariableImpedanceControl"]
solvers.options['show_progress'] = False

class BpVariableImpedanceControl(BasicVariableImpedanceControl):
    def __init__(self, robot: iiwa14, use_behaviour_priors: bool = False):
        super().__init__(robot)
        self.use_bp = use_behaviour_priors
        self.dt = self.model.opt.timestep

        # Passivity Tank State
        self.tank_energy = 20.0
        self.tank_initial = self.tank_energy
        self.tank_max = 50.0
        self.tank_min = 0.001
        self.tank_ref = 5.0
        self.passivity_enabled = True
        self.phase_gating_enabled = True
        self.last_power_nominal = 0.0
        self.last_power_safe = 0.0
        self.last_power_limit = np.inf

        # Optional validation hook: inject a known positive joint-power burst
        # before the passivity QP. Disabled in normal experiments.
        self.adversarial_power_enabled = False
        self.adversarial_power_start = 0.35
        self.adversarial_power_end = 0.55
        self.adversarial_power_watts = 0.0
        self.last_adversarial_power = 0.0
        self.wp_mobile = robot.work_piece.is_movable
        self.sim_time = 0.0

        self.mat_geom_id = self.model.geom(MATERIAL_GEOM).id
        try:
            self.mat_joint_id = self.model.joint("material_slide").id
            Logger.info(f"Material joint ID = {self.mat_joint_id} – motion enabled")
        except:
            self.mat_joint_id = None
            Logger.warning("material_slide joint not found – material will be static.")
        if self.mat_joint_id is None:
            Logger.warning("Material is static – no vertical motion.")
        else:
            Logger.info(f"Material joint ID = {self.mat_joint_id} – motion enabled")

        self.cut_dim_x = robot.work_piece.size[0]
        self.cut_dim_y = robot.work_piece.size[1]
        self.cut_dim_z = robot.work_piece.size[2]

        if self.use_bp:
            if isinstance(robot.work_piece.bp_data, NamedArray):
                self.traj_loader = bpTrajDataLoader(robot.work_piece.bp_data)
            elif isinstance(robot.work_piece.bp_data, bpTrajDataLoader):
                self.traj_loader = robot.work_piece.bp_data
            else:
                raise TypeError("robot.work_piece.bp_data must be NamedArray or bpTrajDataLoader")
            # KLUDGE: todo move it to straight_cutting experiment!
            self.bp_generator = GMRReferenceGenerator(self.traj_loader)

            raw_pos = self.traj_loader.pos[:, 0:]

    def _gmr_to_world(self, gmr_point):
        raise NotImplementedError

    def _solve_passivity_qp(self, tau_nominal, qvel, power_limit=None):
        if not self.passivity_enabled:
            self.last_power_nominal = float(qvel @ tau_nominal)
            self.last_power_safe = self.last_power_nominal
            self.last_power_limit = (
                np.inf if power_limit is None else max(0.0, float(power_limit))
            )
            return tau_nominal

        nv = self.model.nv
        if power_limit is None:
            power_limit = (self.tank_energy - self.tank_min) / self.dt
        power_limit = max(0.0, float(power_limit))
        self.last_power_limit = power_limit
        self.last_power_nominal = float(qvel @ tau_nominal)

        if np.linalg.norm(qvel) < 1e-9 or self.last_power_nominal <= power_limit:
            self.last_power_safe = self.last_power_nominal
            return tau_nominal

        P = matrix(np.eye(nv).astype(float))
        q = matrix(-tau_nominal.astype(float))
        G = matrix(qvel.reshape(1, -1).astype(float))
        h = matrix(np.array([power_limit]).astype(float))
        try:
            sol = solvers.qp(P, q, G, h)
            if sol.get("status") == "optimal":
                tau_safe = np.array(sol['x']).flatten()
                self.last_power_safe = float(qvel @ tau_safe)
                return tau_safe
        except Exception as exc:
            Logger.debug(f"Passivity QP failed, using analytic projection: {exc}")

        # Closed-form Euclidean projection onto qvel.T tau <= power_limit.
        tau_safe = tau_nominal - (
            (self.last_power_nominal - power_limit) / (np.dot(qvel, qvel) + 1e-12)
        ) * qvel
        self.last_power_safe = float(qvel @ tau_safe)
        return tau_safe

    def move_to_position(self, use_default=True, target_pos=None, viewer=None):
        return super().move_to_position(target_pos, viewer)
