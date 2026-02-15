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

__all__ = ["VariableImpedanceControl"]

class VariableImpedanceControl(BasicVariableImpedanceControl):
    """
    Extension of BasicVariableImpedanceControl that optionally uses GMR
    for optimal impedance profiles while maintaining passivity.

    When no GMR sequence is provided, behaves EXACTLY like BasicVariableImpedanceControl.
    """

    def __init__(self, robot: Robot, gmr_sequence=None):
        # Initialize parent first - this sets up ALL base behavior
        super().__init__(robot)

        # GMR-specific initialization (ONLY if GMR is provided)
        self.use_gmr = gmr_sequence is not None

        if self.use_gmr:
            # Set up GMR components
            self.gmr_generator = GMRReferenceGenerator(gmr_sequence)
            self.start_time = None

            # Passivity optimizer for GMR mode
            self.optimizer = EnergyTankPassivityOptimizer(
                dt=self.model.opt.timestep,
                safe_mode=True
            )

            # Storage for optimal profiles
            self.K_optimal = None
            self.D_optimal = None
            self.T_energy_history = []

            # GMR-specific gains (separate from base gains)
            self.Kp_nominal = np.ones(3) * paramVIC.VIC_KP_MAX.value
            self.Kd_nominal = np.ones(3) * 0.5 * np.sqrt(paramVIC.VIC_KP_MAX.value)

            # Monitoring (only in GMR mode)
            self.passivity_monitor = PassivityMonitor()

            print("VariableImpedanceControl: GMR mode enabled")
        else:
            # No GMR - nothing extra needed, will use pure parent behavior
            print("VariableImpedanceControl: Standard mode (no GMR)")

    def compute_optimal_impedance(self, X_des, V_des, F_des):
        """Compute optimal impedance profiles from GMR data (GMR mode only)"""
        if not self.use_gmr:
            return False

        # Estimate Cartesian inertia at initial pose
        M_cart = self._estimate_cartesian_inertia()

        # Optimize with passivity guarantees
        K_opt, D_opt, info = self.optimizer.optimize_impedance_profile(
            X_des, V_des, F_des, M_cart
        )

        if info and not info.get('passivity_violated', True):
            self.K_optimal = K_opt
            self.D_optimal = D_opt
            print(f"Optimal impedance computed. Tank energy range: "
                  f"{info['min_tank_energy']:.2f} - {info['max_tank_energy']:.2f} J")

            # Store for real-time use
            self.impedance_times = np.arange(len(K_opt)) * self.model.opt.timestep
            self.current_imp_idx = 0
            return True
        else:
            print("Warning: Could not compute optimal impedance with passivity guarantees")
            return False

    def get_impedance_at_time(self, t):
        """Get GMR-optimized impedance at time t (GMR mode only)"""
        if not self.use_gmr or self.K_optimal is None:
            # Fall back to base gains
            dist = 0.1  # Default distance
            return self.get_variable_gains(dist)

        idx = min(int(t / self.model.opt.timestep), len(self.K_optimal) - 1)
        self.current_imp_idx = idx

        K_opt = self.K_optimal[idx].copy()
        D_opt = self.D_optimal[idx].copy()

        # Clip to safe bounds (using same limits as base controller)
        k_min, k_max = paramVIC.VIC_KP_MIN.value, paramVIC.VIC_KP_MAX.value
        K_safe = np.clip(K_opt, k_min, k_max)

        # Ensure damping is reasonable (using base controller's formula)
        D_safe = np.clip(D_opt, 0.1 * np.sqrt(k_min), 2.0 * np.sqrt(k_max))

        return K_safe, D_safe

    def _estimate_cartesian_inertia(self):
        """Estimate Cartesian inertia matrix at current configuration"""
        tcp_id = self.model.site("scalpel_tip").id
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)

        # Get joint-space inertia matrix
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M, self.data.qM)

        # Convert to Cartesian space: M_cart = (J M^-1 J^T)^-1
        try:
            M_inv = np.linalg.inv(M[:self.model.nv, :self.model.nv])
            Lambda_inv = jac @ M_inv @ jac.T
            M_cart = np.linalg.inv(Lambda_inv + 1e-6 * np.eye(3))
            return M_cart
        except:
            # Return diagonal estimate if inversion fails
            return np.diag([0.5, 0.5, 0.5])  # Rough estimate

    def adapt_impedance_online(self, current_state, desired_state,
                               measured_force, predicted_force):
        """
        Real-time adaptation with passivity preservation (GMR mode only)
        """
        if not self.use_gmr:
            return None, None

        # Only adapt if we have optimal profiles
        if self.K_optimal is None:
            return None, None

        # Check if adaptation is needed
        force_error = np.linalg.norm(measured_force - predicted_force)
        pos_error = np.linalg.norm(current_state['pos'] - desired_state['pos'])

        force_threshold = getattr(paramVIC, 'FORCE_ERROR_THRESHOLD', 3.0)
        pos_threshold = getattr(paramVIC, 'POS_ERROR_THRESHOLD', 0.01)

        if (force_error < force_threshold and pos_error < pos_threshold):
            # Use precomputed optimal impedance
            return self.get_impedance_at_time(self.data.time - self.start_time)

        # Adaptation needed - blend with optimal for stability
        idx = self.current_imp_idx
        K_opt = self.K_optimal[idx]
        D_opt = self.D_optimal[idx]

        # Conservative adaptation - weight heavily toward optimal
        blend = 0.8  # 80% optimal, 20% adapted
        K_adapted = K_opt  # In a full implementation, compute adapted gains
        D_adapted = D_opt

        K_blend = blend * K_opt + (1 - blend) * K_adapted
        D_blend = blend * D_opt + (1 - blend) * D_adapted

        # Clip to safe bounds
        k_min, k_max = paramVIC.VIC_KP_MIN.value, paramVIC.VIC_KP_MAX.value
        K_safe = np.clip(K_blend, k_min, k_max)
        D_safe = np.clip(D_blend, 0.1 * np.sqrt(k_min), 2.0 * np.sqrt(k_max))

        # Monitor passivity
        self.passivity_monitor.update(K_safe, D_safe,
                                     current_state['vel'], measured_force)

        return K_safe, D_safe

    def move_to_position(self, target_pos, viewer=None, max_steps=8000):
        """
        Override move_to_position to optionally use GMR optimization.
        When GMR is not enabled, calls parent method directly.
        """
        # NO GMR: Use pure parent behavior (exactly as original)
        if not self.use_gmr:
            return super().move_to_position(target_pos, viewer, max_steps)

        # GMR ENABLED: Use enhanced version with optimization
        tcp_id = self.model.site("scalpel_tip").id
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])

        self.error_accumulated = np.zeros(3)
        lambda_sq = paramVIC.VIC_LAMBDA_SQ.value

        # Initialize GMR tracking
        self.start_time = self.data.time

        # Pre-compute optimal impedance from GMR data if available
        if hasattr(self.gmr_generator, 'trajectory'):
            X_des = self.gmr_generator.trajectory[:, 1:4]
            V_des = self.gmr_generator.velocity_profile[:, 1:4]
            F_des = self.gmr_generator.force_profile[:, 1:4]
            self.compute_optimal_impedance(X_des, V_des, F_des)

        for step in range(max_steps):
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[tcp_id].copy()
            error = target_pos - current_pos
            dist = np.linalg.norm(error)

            if dist < paramVIC.VIC_TOL.value:
                return True

            # Get GMR reference at current time
            time_elapsed = self.data.time - self.start_time
            pos_des, vel_des, force_des, _ = self.gmr_generator.get_reference(time_elapsed)

            # Compute errors relative to GMR trajectory
            pos_error = pos_des - current_pos
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            v_tip = jac @ self.data.qvel
            vel_error = vel_des - v_tip

            # Get optimal impedance from GMR
            kp_val, kd_val = self.get_impedance_at_time(time_elapsed)

            # Integral term (same as base controller)
            if dist < 0.05:
                self.error_accumulated += pos_error * self.model.opt.timestep
            ki_val = paramVIC.VIC_KI.value

            # Cutting resistance (using base controller's method)
            f_res = self.compensate_cutting_resistance(current_pos, v_tip)

            # Force calculation with GMR feedforward
            f_impedance = kp_val * pos_error - kd_val * vel_error
            f_integral = ki_val * self.error_accumulated
            f_feedforward = force_des

            f_virtual = f_impedance + f_integral + f_feedforward + f_res

            # Rest is identical to base controller's mapping
            jjt = jac @ jac.T
            tau_task = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), f_virtual)

            # Null-space posture (same as base)
            k_posture, d_posture = 10.0, 2.0
            tau_posture = k_posture * (q_home[:self.model.nv] - self.data.qpos[:self.model.nv]) - d_posture * self.data.qvel

            j_inv = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), np.eye(3))
            null_projection = np.eye(self.model.nv) - (j_inv @ jac)
            tau_null = null_projection @ tau_posture

            tau_total = tau_task + tau_null + self.data.qfrc_bias[:self.model.nv]
            self.data.ctrl[:self.model.nu] = np.clip(tau_total[:self.model.nu], -300, 300)

            mujoco.mj_step(self.model, self.data)

            # Update state (same as base)
            if self.estimator:
                self.robot.state["shared_array"][:-1] = self.robot.state["shared_array"][1:]
                self.robot.state["shared_array"][-1] = self.estimator.get_total_cutting_force()

            if viewer and step % 4 == 0:
                viewer.sync()

        return False