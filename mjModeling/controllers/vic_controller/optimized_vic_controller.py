import numpy as np
from mjModeling.conf import paramVIC
from Optimizers import ImpedanceProfileOptimizer, MPCImpedanceOptimizer
from mjModeling.controllers.vic_controller import GMRVariableImpedanceControl
from mjModeling.mjRobot import Robot

__all__ = ["OptimizedVariableImpedanceControl"]


class OptimizedVariableImpedanceControl(GMRVariableImpedanceControl):
    def __init__(self, robot: Robot, gmr_sequence=None):
        super().__init__(robot, gmr_sequence)

        # Impedance optimizer
        self.optimizer = ImpedanceProfileOptimizer(dt=self.model.opt.timestep)

        # Optimal impedance profiles
        self.K_optimal = None
        self.D_optimal = None

        # Real-time MPC optimizer for adaptation
        self.mpc_optimizer = MPCImpedanceOptimizer(
            dt=self.model.opt.timestep,
            horizon=30  # 60ms lookahead at 2ms timestep
        )

        # Adaptation state
        self.adaptation_active = True
        self.force_error_integral = np.zeros(3)
        self.last_contact_state = False

    def precompute_optimal_impedance(self, X_des, V_des, F_des):
        """Offline computation of optimal impedance profiles"""
        # Estimate effective mass at TCP
        J = self.compute_jacobian()
        M_q = self.compute_joint_space_inertia()
        M_x = np.linalg.inv(J @ np.linalg.inv(M_q) @ J.T)

        # Optimize
        self.K_optimal, self.D_optimal, _, _ = \
            self.optimizer.optimize_trajectory(X_des, V_des, F_des, M_x)

        # Store time vector
        self.optimal_times = np.arange(len(X_des)) * self.model.opt.timestep
        self.current_opt_idx = 0

        print(f"Precomputed optimal impedance profiles for {len(X_des)} steps")
        print(f"K range: [{self.K_optimal.min():.1f}, {self.K_optimal.max():.1f}] N/m")
        print(f"D range: [{self.D_optimal.min():.1f}, {self.D_optimal.max():.1f}] Ns/m")

    def get_current_optimal_impedance(self, time_elapsed):
        """Get optimal impedance at current time"""
        if self.K_optimal is None:
            return None, None

        idx = min(int(time_elapsed / self.model.opt.timestep), len(self.K_optimal)-1)
        return self.K_optimal[idx], self.D_optimal[idx]

    def adapt_impedance_online(self, current_state, desired_state,
                              measured_force, predicted_force):
        """
        Real-time impedance adaptation using MPC
        """
        # Only adapt if error is significant
        force_error = np.linalg.norm(measured_force - predicted_force)
        if force_error < 2.0:  # 2N threshold
            return self.K_optimal[self.current_opt_idx], \
                   self.D_optimal[self.current_opt_idx]

        # Estimate current mass matrix
        J = self.compute_jacobian()
        M_q = self.compute_joint_space_inertia()
        M_x = np.linalg.inv(J @ np.linalg.inv(M_q) @ J.T)

        # Get reference window
        horizon = self.mpc_optimizer.horizon
        idx = self.current_opt_idx
        idx_end = min(idx + horizon, len(self.K_optimal)-1)

        X_ref = self.gmr_generator.trajectory[idx:idx_end, 1:4]
        V_ref = self.gmr_generator.velocity_profile[idx:idx_end, 1:4]
        F_ref = self.gmr_generator.force_profile[idx:idx_end, 1:4]

        # Current state
        x0 = np.concatenate([current_state['pos'], current_state['vel']])

        # Solve MPC
        K_adapted, D_adapted = self.mpc_optimizer.solve_mpc(
            x0, X_ref, V_ref, F_ref, M_x
        )

        # Blend with optimal (conservative)
        blend_factor = 0.7
        K_opt = self.K_optimal[idx]
        D_opt = self.D_optimal[idx]

        K_final = blend_factor * K_opt + (1 - blend_factor) * K_adapted
        D_final = blend_factor * D_opt + (1 - blend_factor) * D_adapted

        # Apply safety bounds
        K_final = np.clip(K_final,
                         self.optimizer.K_min,
                         self.optimizer.K_max)
        D_final = np.clip(D_final,
                         self.optimizer.D_min,
                         self.optimizer.D_max)

        return K_final, D_final

    def compute_control_force_with_optimization(self, current_pos, v_tip,
                                               time_elapsed, measured_force):
        """Enhanced force computation with optimized impedance"""
        # Get GMR references
        pos_des, vel_des, force_des, _ = \
            self.gmr_generator.get_reference(time_elapsed)

        # Get optimal or adapted impedance
        if self.adaptation_active and measured_force is not None:
            current_state = {
                'pos': current_pos,
                'vel': v_tip,
                'force': measured_force
            }
            desired_state = {
                'pos': pos_des,
                'vel': vel_des,
                'force': force_des
            }

            K, D = self.adapt_impedance_online(
                current_state, desired_state,
                measured_force, force_des
            )
        else:
            K, D = self.get_current_optimal_impedance(time_elapsed)
            if K is None:
                # Fallback to variable gain scheduling
                dist = np.linalg.norm(pos_des - current_pos)
                K, D = self.get_variable_gains(dist)
                K = np.ones(3) * K
                D = np.ones(3) * D

        # Store for monitoring
        self.robot.state["optimal_K"] = K.copy()
        self.robot.state["optimal_D"] = D.copy()

        # Compute errors
        pos_error = pos_des - current_pos
        vel_error = vel_des - v_tip

        # Impedance control force
        f_impedance = K * pos_error - D * vel_error

        # Integral term (with anti-windup)
        if np.linalg.norm(pos_error) < 0.05:  # Near target
            self.error_accumulated += pos_error * self.model.opt.timestep
            # Anti-windup: limit integral term
            int_max = 0.1  # 10cm maximum integral correction
            self.error_accumulated = np.clip(self.error_accumulated, -int_max, int_max)

        ki_val = paramVIC.VIC_KI.value
        f_integral = ki_val * self.error_accumulated

        # Feedforward from GMR
        f_feedforward = force_des

        # Cutting resistance
        f_resistance = self.compensate_cutting_resistance(current_pos, v_tip)

        # Total force
        f_total = f_impedance + f_integral + f_feedforward + f_resistance

        return f_total, K, D