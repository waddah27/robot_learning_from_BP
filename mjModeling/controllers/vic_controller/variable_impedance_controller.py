from typing import Optional, Union
from mjModeling.controllers.vic_controller import BasicVariableImpedanceControl
from mjModeling.cutting_materials import Material
import numpy as np
import mujoco
from mjModeling.conf import paramVIC, workingPiece
from mjModeling.controllers.controller_api import Controller
from mjModeling.estimators import ImpedanceEstimator
from mjModeling.mjRobot import Robot
from reference_generators import GMRReferenceGenerator
from system_analysis.validation import PassivityMonitor
from Optimizers import EnergyTankPassivityOptimizer
from data import bpTrajDataLoader, MaterialData, NamedArray

__all__ = ["VariableImpedanceControl"]

class VariableImpedanceControl(BasicVariableImpedanceControl):
    """
    Extension of BasicVariableImpedanceControl that optionally uses GMR.
    move_to_position() behaves differently based on whether GMR data exists.
    """

    def __init__(self, robot: Robot, gmr_sequence: Union[NamedArray, bpTrajDataLoader]=None):
        # Initialize parent first
        super().__init__(robot)

        # GMR-specific initialization
        self.use_gmr = gmr_sequence is not None

        if self.use_gmr:
            # Handle GMR sequence
            if isinstance(gmr_sequence, NamedArray):
                self.traj_loader = bpTrajDataLoader(gmr_sequence)
            elif isinstance(gmr_sequence, bpTrajDataLoader):
                self.traj_loader = gmr_sequence
            else:
                raise TypeError(f"gmr_sequence must be NamedArray or bpTrajDataLoader, got {type(gmr_sequence)}")

            # GMR generator
            self.gmr_generator = GMRReferenceGenerator(self.traj_loader)

            # Passivity optimizer
            self.optimizer = EnergyTankPassivityOptimizer(
                dt=self.model.opt.timestep,
                safe_mode=True
            )

            # Storage for optimal profiles
            self.K_optimal = None
            self.D_optimal = None
            self.T_energy_history = []

            print(f"GMR mode enabled: {self.traj_loader.material_name}")
            print(f"Trajectory length: {len(self.traj_loader)} steps")

    def move_to_position(self, target_pos=None, viewer=None, max_steps=8000):
        """
        OVERRIDDEN: If GMR data exists, follows trajectory to target_pos as endpoint.
        If no GMR data, behaves EXACTLY like parent.
        """
        # CASE 1: NO GMR DATA - use parent behavior exactly
        if not self.use_gmr:
            return super().move_to_position(target_pos, viewer, max_steps)

        # CASE 2: GMR DATA EXISTS - follow trajectory to reach target_pos
        tcp_id = self.model.site("scalpel_tip").id
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])

        self.error_accumulated = np.zeros(3)
        lambda_sq = paramVIC.VIC_LAMBDA_SQ.value

        # Start timing
        self.start_time = self.data.time

        # Pre-compute optimal impedance from trajectory
        self._compute_optimal_impedance_from_trajectory()

        # Use trajectory iterator
        traj_iter = iter(self.traj_loader)
        step = 0

        print(f"\nFollowing GMR trajectory to target: {target_pos}")
        print(f"Material: {self.traj_loader.material_name}")

        try:
            while step < max_steps:
                mujoco.mj_forward(self.model, self.data)

                # Get next trajectory point
                pos_des, vel_des, force_des = next(traj_iter)
                time_elapsed = self.data.time - self.start_time

                # Current state
                current_pos = self.data.site_xpos[tcp_id].copy()
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
                v_tip = jac @ self.data.qvel

                # Calculate errors relative to trajectory
                pos_error = pos_des - current_pos
                vel_error = vel_des - v_tip
                dist_to_traj = np.linalg.norm(pos_error)

                # Check if we've reached final target
                dist_to_target = np.linalg.norm(target_pos - current_pos) if target_pos is not None else 0
                if target_pos is not None and dist_to_target < paramVIC.VIC_TOL.value:
                    print(f"\n✓ Target reached at step {step}")
                    return True

                # Get gains (optimal if available)
                if self.K_optimal is not None:
                    kp_val, kd_val = self._get_impedance_at_time(time_elapsed)
                else:
                    # Fallback to original gain scheduling
                    kp_val, kd_val = self.get_variable_gains(dist_to_traj)

                # Integral term (same logic as parent)
                if dist_to_traj < 0.05:
                    self.error_accumulated += pos_error * self.model.opt.timestep
                ki_val = paramVIC.VIC_KI.value

                # Cutting resistance (using parent method)
                f_res = self.compensate_cutting_resistance(current_pos, v_tip)

                # Force calculation with GMR feedforward
                f_impedance = kp_val * pos_error - kd_val * vel_error
                f_integral = ki_val * self.error_accumulated
                f_virtual = f_impedance + f_integral + force_des + f_res

                # Store tank energy for monitoring
                if hasattr(self.optimizer, 'T_energy'):
                    self.T_energy_history.append(self.optimizer.T_energy)

                # === IDENTICAL MAPPING TO PARENT ===
                jjt = jac @ jac.T
                tau_task = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), f_virtual)

                # Null-space posture (same as parent)
                k_posture, d_posture = 10.0, 2.0
                tau_posture = k_posture * (q_home[:self.model.nv] - self.data.qpos[:self.model.nv]) - d_posture * self.data.qvel

                j_inv = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), np.eye(3))
                null_projection = np.eye(self.model.nv) - (j_inv @ jac)
                tau_null = null_projection @ tau_posture

                tau_total = tau_task + tau_null + self.data.qfrc_bias[:self.model.nv]
                self.data.ctrl[:self.model.nu] = np.clip(tau_total[:self.model.nu], -300, 300)

                mujoco.mj_step(self.model, self.data)

                # Update state (same as parent)
                if self.estimator:
                    self.robot.state["shared_array"][:-1] = self.robot.state["shared_array"][1:]
                    self.robot.state["shared_array"][-1] = self.estimator.get_total_cutting_force()

                # Progress reporting
                if step % 100 == 0:
                    progress = (step + 1) / len(self.traj_loader) * 100
                    print(f"Step {step}: Progress {progress:.1f}% | "
                          f"Dist to target: {dist_to_target:.4f}m | "
                          f"Kp_z: {kp_val[2]:.1f} N/m")

                if viewer and step % 4 == 0:
                    viewer.sync()

                step += 1

        except StopIteration:
            # Trajectory completed, now go directly to target if needed
            print("\nTrajectory completed, moving directly to target...")
            return super().move_to_position(target_pos, viewer, max_steps - step)

        return False

    # ===== PRIVATE HELPER METHODS (NOT PART OF PUBLIC API) =====

    def _compute_optimal_impedance_from_trajectory(self):
        """Compute optimal impedance using loaded trajectory"""
        X_des = self.traj_loader.pos
        V_des = self.traj_loader.vel
        F_des = self.traj_loader.force

        M_cart = self._estimate_cartesian_inertia()

        K_opt, D_opt, info = self.optimizer.optimize_impedance_profile(
            X_des, V_des, F_des, M_cart
        )

        if info and not info.get('passivity_violated', True):
            self.K_optimal = K_opt
            self.D_optimal = D_opt
            self.impedance_times = np.arange(len(K_opt)) * self.model.opt.timestep
            print(f"✓ Optimal impedance computed for {len(K_opt)} points")

    def _get_impedance_at_time(self, t):
        """Get optimal impedance at time t"""
        if self.K_optimal is None:
            return np.ones(3) * 500, np.ones(3) * 20

        idx = min(int(t / self.model.opt.timestep), len(self.K_optimal) - 1)
        return self.K_optimal[idx], self.D_optimal[idx]

    def _estimate_cartesian_inertia(self):
        """Estimate Cartesian inertia matrix"""
        tcp_id = self.model.site("scalpel_tip").id
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)

        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        try:
            M_inv = np.linalg.inv(M[:self.model.nv, :self.model.nv] + 1e-6 * np.eye(self.model.nv))
            Lambda_inv = jac @ M_inv @ jac.T
            return np.linalg.inv(Lambda_inv + 1e-6 * np.eye(3))
        except:
            return np.diag([0.5, 0.5, 0.5])

# # USAGE:
# # Without GMR - pure parent behavior
# controller = VariableImpedanceControl(robot)
# controller.move_to_position(target_pos)  # ← Exactly like BasicVariableImpedanceControl

# # With GMR - follows trajectory to reach target
# cork_data = MaterialData.cork
# controller_gmr = VariableImpedanceControl(robot, gmr_sequence=cork_data)
# controller_gmr.move_to_position(target_pos)  # ← Follows GMR trajectory to target