from logger import Logger
from typing import Union

from mjModeling.kuka_iiwa_14.iiwa14_model import iiwa14
import numpy as np
import mujoco
from cvxopt import matrix, solvers
from mjModeling.controllers.vic_controller import BasicVariableImpedanceControl
from mjModeling.conf import paramVIC
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

        if self.use_bp:
            # Trajectory loader type check
            if isinstance(robot.work_piece.bp_data, NamedArray):
                self.traj_loader = bpTrajDataLoader(robot.work_piece.bp_data)
            elif isinstance(robot.work_piece.bp_data, bpTrajDataLoader):
                self.traj_loader = robot.work_piece.bp_data
            else:
                raise TypeError("robot.work_piece.bp_data sequence must be NamedArray or bpTrajDataLoader")
            self.bp_generator = GMRReferenceGenerator(self.traj_loader)

            # --- ROBUST 3D PADDING LOGIC ---
            # Extract everything after the time column
            raw_pos = self.traj_loader.pos[:, 0:]

            # Create a 3D array regardless of raw_pos shape (e.g., if it's only N x 2)
            pos_3d =raw_pos # np.zeros((raw_pos.shape[0], 3))
            # cols_to_copy = min(raw_pos.shape[1], 3)
            # pos_3d[:, :cols_to_copy] = raw_pos[:, :cols_to_copy]

            self.gmr_min = np.min(pos_3d, axis=0)
            self.gmr_max = np.max(pos_3d, axis=0)
            self.gmr_range = np.maximum(self.gmr_max - self.gmr_min, 1e-6)

    def _gmr_to_world(self, gmr_point):
        """Maps GMR point to World Workspace. Guaranteed 3D math."""
        # Ensure gmr_point is exactly 3D (pad with zeros if traj_loader yields 2D)
        p_safe = np.zeros(3)
        p_safe[:min(len(gmr_point), 3)] = gmr_point[:min(len(gmr_point), 3)]

        # Now both are (3,) - No broadcast error
        norm = (p_safe - self.gmr_min) / self.gmr_range
        norm = np.clip(norm, 0, 1)

        # Workspace: X[0.35,0.65], Y[-0.15,0.15], Z[0.02,0.04]
        world_x = 0.35 + norm[0] * (0.65 - 0.35)
        world_y = 0.0 + norm[1] * (0.15 - (-0.15))
        world_z = self.working_piece.center[2] + norm[2] * (self.working_piece.surface_height - 0.02)

        return np.array([world_x, world_y, world_z])

    def _solve_passivity_qp(self, tau_nominal, qvel):
        nv = self.model.nv
        P = matrix(np.eye(nv).astype(float))
        q = matrix(-tau_nominal.astype(float))
        power_limit = (self.tank_energy - self.tank_min) / self.dt
        G = matrix(qvel.reshape(1, -1).astype(float))
        # 30W floor to ensure penetration through material resistance
        h = matrix(np.array([max(30.0, power_limit)]).astype(float))
        try:
            sol = solvers.qp(P, q, G, h)
            return np.array(sol['x']).flatten()
        except:
            return np.zeros(nv)

    def move_to_position(self, use_default = True, target_pos=None, v_raw=None, f_raw=None, viewer=None):
        if use_default:
            return super().move_to_position(target_pos, viewer)


        tcp_id = self.model.site("scalpel_tip").id
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])
        self.error_accumulated = np.zeros(3)

        max_steps = int(self.opt_max_steps/100)
        # Map raw data to 3D world (p_raw is padded inside helper)
        pos_des = self._gmr_to_world(target_pos)
        Logger.debug(f"move to {pos_des}")

        for step in range(max_steps):


            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.site_xpos[tcp_id].copy()


            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            # v_tip = v_raw #jac @ self.data.qvel
            # --- FIX: actual tip velocity and force transformation ---
            # Get current site orientation (rotation from site frame to world)
            site_rot = self.data.site_xmat[tcp_id].reshape(3, 3)

            # Transform force from site frame to world frame
            # (f_raw is a 3D vector in the sensor/tip frame)
            f_raw_array = np.asarray(f_raw).flatten()  # ensure shape (3,)
            f_ff_world = site_rot @ f_raw_array

            # Compute actual tip velocity (for damping)
            v_tip = jac @ self.data.qvel


            error = pos_des - current_pos
            dist = np.linalg.norm(error)

            if step % int(self.opt_max_steps/10) == 0:
                Logger.debug(f"step {step}: current pos tcp = {current_pos} -- pos_des = {pos_des} -- err = {dist}")

            kp, kd = self.get_variable_gains(dist)

            if dist < paramVIC.VIC_TOL.value:
                return True

            if dist < 0.05:
                self.error_accumulated += error * self.dt

            # calcs of virtual force using world quantities
            f_virtual = (kp * error) + (paramVIC.VIC_KI.value * self.error_accumulated) \
                        - (kd * v_tip) + f_ff_world
            # ---------------------------------------------------------


            # Torque calculation
            jjt = jac @ jac.T
            tau_task = jac.T @ np.linalg.solve(jjt + 1e-4, f_virtual)
            tau_posture = 10.0 * (q_home[:self.model.nv] - self.data.qpos[:self.model.nv]) - 2.0 * self.data.qvel
            j_inv = jac.T @ np.linalg.solve(jjt + 1e-4, np.eye(3))
            tau_null = (np.eye(self.model.nv) - (j_inv @ jac)) @ tau_posture

            tau_nominal = tau_task + tau_null + self.data.qfrc_bias[:self.model.nv]

            # QP Optimization with Energy Tank
            tau_safe = self._solve_passivity_qp(tau_nominal, self.data.qvel)

            # Update Tank
            power_flow = self.data.qvel.dot(tau_safe)
            self.tank_energy -= power_flow * self.dt
            self.tank_energy += (np.sum(kd * (v_tip**2)) + 1.0) * self.dt
            self.tank_energy = np.clip(self.tank_energy, 0, self.tank_max)

            self.data.ctrl[:self.model.nu] = np.clip(tau_safe[:self.model.nu], -300, 300)
            mujoco.mj_step(self.model, self.data)
            self.record_contact_forces()

            if viewer and step % 4 == 0: viewer.sync()

        return True
