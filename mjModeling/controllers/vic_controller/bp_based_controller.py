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
        self.tank_max = 50.0
        self.tank_min = 0.001
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

    def _solve_passivity_qp(self, tau_nominal, qvel):
        nv = self.model.nv
        P = matrix(np.eye(nv).astype(float))
        q = matrix(-tau_nominal.astype(float))
        power_limit = (self.tank_energy - self.tank_min) / self.dt
        G = matrix(qvel.reshape(1, -1).astype(float))
        h = matrix(np.array([max(30.0, power_limit)]).astype(float))
        try:
            sol = solvers.qp(P, q, G, h)
            return np.array(sol['x']).flatten()
        except:
            return np.zeros(nv)

    def move_to_position(self, use_default=True, target_pos=None, viewer=None):
        return super().move_to_position(target_pos, viewer)