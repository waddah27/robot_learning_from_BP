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

__all__ = ["VariableImpedanceControl"]
solvers.options['show_progress'] = False

class VariableImpedanceControl(BasicVariableImpedanceControl):
    def __init__(self, robot: iiwa14, use_behaviour_priors: bool = False):
        super().__init__(robot)
        self.use_bp = use_behaviour_priors
        self.dt = self.model.opt.timestep

        # Passivity Tank State
        self.tank_energy = 20.0
        self.tank_max = 50.0
        self.tank_min = 0.001
        self.wp_mobile = robot.work_piece.is_movable

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

        self.cut_width_x = robot.work_piece.size[0]
        self.cut_width_y = robot.work_piece.size[1]
        self.cut_depth_z = robot.work_piece.size[2]

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
            self.gmr_min = np.min(raw_pos, axis=0)
            self.gmr_max = np.max(raw_pos, axis=0)
            self.gmr_range = np.maximum(self.gmr_max - self.gmr_min, 1e-6)

    def _gmr_to_world(self, gmr_point):
        mat_pos = self.data.geom_xpos[self.mat_geom_id].copy()
        surface_z = mat_pos[2] + self.cut_depth_z   # top of moving material

        p_safe = np.zeros(3)
        p_safe[:min(len(gmr_point), 3)] = gmr_point[:min(len(gmr_point), 3)]

        norm = (p_safe - self.gmr_min) / self.gmr_range
        norm = np.clip(norm, 0, 1)

        world_x = mat_pos[0] + (norm[0] - 0.5) * self.cut_width_x
        world_y = mat_pos[1] + (norm[1] - 0.5) * self.cut_width_y
        world_z = surface_z   # desired Z is always at the surface

        return np.array([world_x, world_y, world_z])

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

    def move_to_position(self, use_default=True, target_pos=None, v_raw=None, f_raw=None, viewer=None):
        if use_default:
            return super().move_to_position(target_pos, viewer)

        tcp_id = self.model.site("scalpel_tip").id
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])
        self.error_accumulated = np.zeros(3)

        max_steps = int(self.opt_max_steps/100)   # full horizon per waypoint

        if not hasattr(self, 'sim_time'):
            self.sim_time = 0.0

        for step in range(max_steps):
            # --- Update material position ---
            if self.mat_joint_id is not None and self.wp_mobile:
                wp_hieght = wp_sine_motion(self.sim_time)
                self.data.qpos[self.mat_joint_id] = wp_hieght
                mujoco.mj_forward(self.model, self.data)

            # --- Desired position at current surface ---
            pos_des = self._gmr_to_world(target_pos)

            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[tcp_id].copy()
            surface_z = self.data.geom_xpos[self.mat_geom_id][2] + self.cut_depth_z

            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)

            site_rot = self.data.site_xmat[tcp_id].reshape(3, 3)
            f_raw_array = np.asarray(f_raw).flatten()
            f_ff_world = site_rot @ f_raw_array
            # Scale feedforward: reduce Z component significantly
            f_ff_world[2] *= 0.1   # only 10% in Z
            f_ff_world[:2] *= 0.3  # keep 30% in XY (or as desired)

            v_tip = jac @ self.data.qvel

            error = pos_des - current_pos
            dist = np.linalg.norm(error)

            if step % int(self.opt_max_steps/10) == 0:
                Logger.debug(f"step {step}: current pos tcp = {current_pos} -- pos_des = {pos_des} -- err = {error}")

            # variable gains scheduling
            kp, kd = self.get_variable_gains(error)

            # Optional: boost Z gain if penetrating (as before)
            penetration_depth = max(0.0, surface_z - current_pos[2])
            if penetration_depth > 0.001:
                kp[2] *= 5.0
                kd[2] *= 3.0
                # Re‑clamp to bounds (optional)
                kp = np.clip(kp, paramVIC.VIC_KP_MIN, paramVIC.VIC_KP_MAX)
                kd = np.clip(kd, 0.5*np.sqrt(paramVIC.VIC_KP_MIN), None)

            # Virtual force (now with vector gains)
            f_virtual = kp * error + paramVIC.VIC_KI * self.error_accumulated - kd * v_tip + f_ff_world

            # Integral update (unchanged)
            if penetration_depth < 0.005 and dist < 0.05:
                self.error_accumulated += error * self.dt
                self.error_accumulated = np.clip(self.error_accumulated, -0.05, 0.05)

            # Torque calculation
            jjt = jac @ jac.T
            tau_task = jac.T @ np.linalg.solve(jjt + 1e-4, f_virtual)
            tau_posture_robot = 10.0 * (q_home - self.data.qpos[:self.n_robot]) - 2.0 * self.data.qvel[:self.n_robot]
            tau_posture = np.zeros(self.model.nv)
            tau_posture[:self.n_robot] = tau_posture_robot
            j_inv = jac.T @ np.linalg.solve(jjt + 1e-4, np.eye(3))
            tau_null = (np.eye(self.model.nv) - (j_inv @ jac)) @ tau_posture

            tau_nominal = tau_task + tau_null + self.data.qfrc_bias[:self.model.nv]

            tau_safe = self._solve_passivity_qp(tau_nominal, self.data.qvel)

            power_flow = self.data.qvel.dot(tau_safe)
            self.tank_energy -= power_flow * self.dt
            self.tank_energy += (np.sum(kd * (v_tip**2)) + 1.0) * self.dt
            self.tank_energy = np.clip(self.tank_energy, 0, self.tank_max)

            self.data.ctrl[:self.model.nu] = np.clip(tau_safe[:self.model.nu], -300, 300)
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt
            K = np.array([kp, kp, kp])
            self.record_contact_forces(kp)

            if viewer and step % 4 == 0:
                viewer.sync()

        return True