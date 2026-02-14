from Optimizers import EnergyTankPassivityOptimizer
from mjModeling.controllers.vic_controller.vic_controller_gmr import VariableImpedanceControl


class PassiveVariableImpedanceControl(VariableImpedanceControl):
    """
    VIC with convex optimization and guaranteed passivity
    """
    def __init__(self, robot: Robot, gmr_sequence=None):
        super().__init__(robot, gmr_sequence)

        # Passivity-preserving optimizer
        self.optimizer = EnergyTankPassivityOptimizer(
            dt=self.model.opt.timestep,
            safe_mode=True
        )

        # Optimal profiles storage
        self.K_optimal = None
        self.D_optimal = None
        self.T_energy_history = []

        # Adaptation thresholds
        self.force_error_threshold = 3.0  # N
        self.pos_error_threshold = 0.01   # m

        # Monitoring
        self.passivity_monitor = PassivityMonitor()

    def compute_optimal_impedance(self, X_des, V_des, F_des):
        """Compute optimal impedance profiles from GMR data"""
        # Estimate Cartesian inertia at initial pose
        M_cart = self.estimate_cartesian_inertia()

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
        """Get impedance at time t with safety checks"""
        if self.K_optimal is None or self.D_optimal is None:
            return self.get_variable_gains_simple(t)

        idx = min(int(t / self.model.opt.timestep), len(self.K_optimal) - 1)
        self.current_imp_idx = idx

        # Get optimal values
        K_opt = self.K_optimal[idx].copy()
        D_opt = self.D_optimal[idx].copy()

        # Check passivity condition
        if not self.check_impedance_passivity(K_opt, D_opt):
            print(f"Warning: Impedance at t={t:.3f}s violates passivity, applying safety filter")
            K_opt, D_opt = self.apply_safety_filter(K_opt, D_opt)

        return K_opt, D_opt

    def check_impedance_passivity(self, K, D):
        """Check if impedance parameters satisfy passivity conditions"""
        M_cart = self.estimate_cartesian_inertia()

        # Check each direction
        for i in range(3):
            m_eff = M_cart[i, i] if M_cart.ndim == 2 else M_cart[i]
            # Damping must be sufficient for the given stiffness and inertia
            if D[i] < 0.3 * np.sqrt(m_eff * K[i]):
                return False

        return True

    def apply_safety_filter(self, K, D):
        """Project impedance to safe, passive set"""
        M_cart = self.estimate_cartesian_inertia()

        K_safe = np.clip(K, self.optimizer.K_min, self.optimizer.K_max)

        # Ensure sufficient damping for passivity
        for i in range(3):
            m_eff = M_cart[i, i] if M_cart.ndim == 2 else M_cart[i]
            D_min_passive = 0.5 * np.sqrt(m_eff * K_safe[i])
            D_safe_min = max(self.optimizer.D_min[i], D_min_passive)
            D[i] = max(D[i], D_safe_min)

        D_safe = np.clip(D, self.optimizer.D_min, self.optimizer.D_max)

        return K_safe, D_safe

    def adapt_impedance_online(self, current_state, desired_state,
                              measured_force, predicted_force):
        """
        Real-time adaptation with passivity preservation
        """
        # Check if adaptation is needed
        force_error = np.linalg.norm(measured_force - predicted_force)
        pos_error = np.linalg.norm(current_state['pos'] - desired_state['pos'])

        if (force_error < self.force_error_threshold and
            pos_error < self.pos_error_threshold):
            # Use precomputed optimal impedance
            return self.get_impedance_at_time(self.data.time - self.start_time)

        # Adaptation needed - solve MPC problem
        M_cart = self.estimate_cartesian_inertia()

        # Get reference window
        horizon = 20  # Short horizon for real-time
        idx = self.current_imp_idx
        idx_end = min(idx + horizon, len(self.K_optimal) - 1)

        if idx_end - idx < 5:
            # Not enough horizon, use optimal
            return self.K_optimal[idx], self.D_optimal[idx]

        X_ref = self.gmr_generator.trajectory[idx:idx_end, 1:4]
        V_ref = self.gmr_generator.velocity_profile[idx:idx_end, 1:4]
        F_ref = self.gmr_generator.force_profile[idx:idx_end, 1:4]

        # Solve MPC
        K_adapted, D_adapted, mpc_info = self.optimizer.optimize_impedance_profile(
            X_ref, V_ref, F_ref, M_cart, current_state, horizon=horizon
        )

        if mpc_info['status'] not in ['optimal', 'optimal_inaccurate']:
            print(f"MPC failed: {mpc_info['status']}, using optimal")
            return self.K_optimal[idx], self.D_optimal[idx]

        # Blend with optimal (conservative)
        K_opt = self.K_optimal[idx]
        D_opt = self.D_optimal[idx]

        blend = 0.6  # Weight on optimal (more conservative)
        K_blend = blend * K_opt + (1 - blend) * K_adapted
        D_blend = blend * D_opt + (1 - blend) * D_adapted

        # Apply safety filter
        K_final, D_final = self.apply_safety_filter(K_blend, D_blend)

        # Update tank energy tracker
        self.T_energy_history.append(self.optimizer.T_energy)

        # Monitor passivity
        self.passivity_monitor.update(K_final, D_final,
                                     current_state['vel'], measured_force)

        return K_final, D_final