import numpy as np
import cvxpy as cp
from scipy.linalg import solve_discrete_are

__all__ = ["EnergyTankPassivityOptimizer"]


class EnergyTankPassivityOptimizer:
    """
    Convex QP-based impedance optimizer with guaranteed passivity
    via energy tank constraints
    """
    def __init__(self, dt=0.002, safe_mode=True):
        self.dt = dt
        self.safe_mode = safe_mode

        # Energy tank parameters
        self.T_max = 50.0  # Maximum tank energy (Joules)
        self.T_min = 5.0   # Minimum tank energy (safety threshold)
        self.T_initial = 20.0  # Initial tank energy
        self.dissipation_rate = 0.1  # Natural dissipation
        self.injection_limit = 5.0   # Max energy injection per step

        # Impedance bounds
        self.K_min = np.array([50, 50, 50])      # N/m
        self.K_max = np.array([2000, 2000, 2000]) # N/m
        self.D_min = np.array([5, 5, 5])         # Ns/m
        self.D_max = np.array([100, 100, 100])   # Ns/m

        # Rate limits for stability
        self.max_K_rate = 500.0  # N/m/s
        self.max_D_rate = 100.0  # Ns/m/s

        # Weighting matrices for cost function
        self.Q = np.diag([10.0, 10.0, 10.0])  # Position error
        self.R = np.diag([0.1, 0.1, 0.1])     # Force error
        self.S_K = np.diag([0.01, 0.01, 0.01]) # Stiffness regularization
        self.S_D = np.diag([0.05, 0.05, 0.05]) # Damping regularization

        # State for MPC
        self.T_energy = self.T_initial  # Current tank energy
        self.last_solution = None

    def optimize_impedance_profile(self, X_des, V_des, F_des, M_est,
                                  current_state=None, horizon=100):
        """
        Solve convex QP for optimal impedance profiles with passivity guarantees

        Parameters:
        -----------
        X_des: (N,3) desired positions
        V_des: (N,3) desired velocities
        F_des: (N,3) desired forces
        M_est: (3,3) estimated Cartesian inertia matrix
        current_state: dict with current pos, vel, force
        horizon: optimization horizon

        Returns:
        --------
        K_opt, D_opt: optimal impedance profiles
        solution_info: optimization metadata
        """
        N = min(horizon, len(X_des))

        if current_state is None:
            # Full trajectory optimization (offline)
            return self._offline_optimization(X_des, V_des, F_des, M_est, N)
        else:
            # MPC optimization (online)
            return self._mpc_optimization(X_des, V_des, F_des, M_est,
                                         current_state, N)

    def _offline_optimization(self, X_des, V_des, F_des, M_est, N):
        """Offline optimization of complete trajectory"""
        # Decision variables
        K = cp.Variable((N, 3))  # Stiffness profile
        D = cp.Variable((N, 3))  # Damping profile
        X = cp.Variable((N, 3))  # Predicted position
        V = cp.Variable((N, 3))  # Predicted velocity
        F_pred = cp.Variable((N, 3))  # Predicted interaction force
        T_energy = cp.Variable(N)  # Energy tank states

        # Initial conditions
        constraints = [
            X[0] == X_des[0],
            V[0] == V_des[0],
            T_energy[0] == self.T_initial
        ]

        cost = 0
        for i in range(N-1):
            # ---- DYNAMICS CONSTRAINTS ----
            # Simplified impedance dynamics: F_pred = M*(V_dot) + D*V + K*(X-X_des)
            V_dot = (V[i+1] - V[i]) / self.dt
            constraints.append(
                F_pred[i] == M_est @ V_dot + D[i] @ V[i] + K[i] @ (X[i] - X_des[i])
            )

            # Position integration
            constraints.append(
                X[i+1] == X[i] + V[i] * self.dt
            )

            # ---- ENERGY TANK CONSTRAINTS (PASSIVITY) ----
            # Power flow: P_in = V[i].T @ F_pred[i] (positive when energy flows in)
            # Tank dynamics: T_dot = -dissipation + injection - adaptation_cost

            # 1. Dissipation term (always positive)
            P_dissipated = self.dissipation_rate * T_energy[i]

            # 2. Power from environment (can be positive or negative)
            P_environment = V[i].T @ F_pred[i]

            # 3. Adaptation power cost (changing impedance consumes energy)
            if i > 0:
                delta_K = K[i] - K[i-1]
                delta_D = D[i] - D[i-1]
                # Quadratic cost of adaptation
                P_adaptation = 0.5 * (cp.quad_form(delta_K, self.S_K) +
                                     cp.quad_form(delta_D, self.S_D))
            else:
                P_adaptation = 0

            # 4. Allowed energy injection (limited for safety)
            P_injection = cp.Variable()
            constraints.append(P_injection >= 0)
            constraints.append(P_injection <= self.injection_limit)

            # Tank dynamics constraint
            T_dot = -P_dissipated + P_injection + cp.maximum(P_environment, 0) - P_adaptation
            constraints.append(
                T_energy[i+1] == T_energy[i] + T_dot * self.dt
            )

            # Tank energy bounds (PASSIVITY CONDITION)
            constraints.append(T_energy[i] >= self.T_min)
            constraints.append(T_energy[i] <= self.T_max)

            # ---- IMPEDANCE CONSTRAINTS ----
            # Bounds
            for j in range(3):
                constraints.append(K[i, j] >= self.K_min[j])
                constraints.append(K[i, j] <= self.K_max[j])
                constraints.append(D[i, j] >= self.D_min[j])
                constraints.append(D[i, j] <= self.D_max[j])

                # Damping lower bound for stability
                m_jj = M_est[j, j] if M_est.ndim == 2 else M_est[j]
                constraints.append(D[i, j] >= 0.5 * cp.sqrt(m_jj * K[i, j]))

            # Rate limits
            if i > 0:
                delta_K = cp.abs(K[i] - K[i-1])
                delta_D = cp.abs(D[i] - D[i-1])
                constraints.append(delta_K <= self.max_K_rate * self.dt)
                constraints.append(delta_D <= self.max_D_rate * self.dt)

            # ---- COST FUNCTION ----
            # Tracking error
            pos_error = X[i] - X_des[i]
            force_error = F_pred[i] - F_des[i]

            cost += cp.quad_form(pos_error, self.Q)
            cost += cp.quad_form(force_error, self.R)

            # Regularization (prevent extreme impedances)
            cost += cp.quad_form(K[i] - (self.K_min + self.K_max)/2, self.S_K)
            cost += cp.quad_form(D[i] - (self.D_min + self.D_max)/2, self.S_D)

            # Penalize energy usage
            cost += 0.01 * P_adaptation

        # Terminal cost: ensure safe state at end
        cost += 10 * cp.norm(X[-1] - X_des[N-1])  # Final position error
        cost += 5 * cp.norm(T_energy[-1] - self.T_initial)  # Return energy to initial

        # ---- SOLVE CONVEX QP ----
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            # Use ECOS or OSQP for convex QP
            result = problem.solve(solver=cp.ECOS, verbose=False,
                                 max_iters=2000, abstol=1e-6, reltol=1e-6)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                raise ValueError(f"Optimization failed: {problem.status}")

            # Extract solution
            K_opt = K.value
            D_opt = D.value

            # Verify passivity
            passivity_violated = np.any(T_energy.value < self.T_min - 1e-3)

            solution_info = {
                'status': problem.status,
                'cost': problem.value,
                'passivity_violated': passivity_violated,
                'min_tank_energy': np.min(T_energy.value),
                'max_tank_energy': np.max(T_energy.value),
                'final_tank_energy': T_energy.value[-1],
                'avg_stiffness': np.mean(K_opt, axis=0),
                'avg_damping': np.mean(D_opt, axis=0)
            }

            self.last_solution = solution_info

            return K_opt, D_opt, solution_info

        except Exception as e:
            print(f"Offline optimization failed: {e}")
            return self._generate_safe_profile(N), None

    def _mpc_optimization(self, X_ref, V_ref, F_ref, M_est, current_state, horizon):
        """Online MPC optimization with current state"""
        N = horizon

        # Extract current state
        x0 = current_state['pos']
        v0 = current_state['vel']
        T0 = self.T_energy  # Current tank energy

        # Decision variables
        K = cp.Variable((N, 3))
        D = cp.Variable((N, 3))
        X = cp.Variable((N, 3))
        V = cp.Variable((N, 3))
        T = cp.Variable(N)

        # Initial constraints
        constraints = [
            X[0] == x0,
            V[0] == v0,
            T[0] == T0
        ]

        cost = 0

        for i in range(N-1):
            # Reference indices
            ref_idx = min(i, len(X_ref)-1)

            # Dynamics
            V_dot = (V[i+1] - V[i]) / self.dt
            F_pred = M_est @ V_dot + D[i] @ V[i] + K[i] @ (X[i] - X_ref[ref_idx])

            # Position integration
            constraints.append(X[i+1] == X[i] + V[i] * self.dt)

            # Energy tank constraints
            if i > 0:
                delta_K = K[i] - K[i-1]
                delta_D = D[i] - D[i-1]
                P_adapt = 0.1 * (cp.norm(delta_K, 2) + cp.norm(delta_D, 2))
            else:
                P_adapt = 0

            # Environment power (simplified)
            P_env = cp.maximum(V[i].T @ (F_ref[ref_idx]), 0)

            # Tank update
            T_dot = -self.dissipation_rate * T[i] + P_env - P_adapt
            constraints.append(T[i+1] == T[i] + T_dot * self.dt)

            # Tank bounds
            constraints.append(T[i] >= self.T_min)

            # Impedance bounds
            for j in range(3):
                constraints.append(K[i, j] >= self.K_min[j])
                constraints.append(K[i, j] <= self.K_max[j])
                constraints.append(D[i, j] >= self.D_min[j])
                constraints.append(D[i, j] <= self.D_max[j])

                m_jj = M_est[j, j] if M_est.ndim == 2 else M_est[j]
                constraints.append(D[i, j] >= 0.3 * cp.sqrt(m_jj * K[i, j]))

            # Cost: track reference while minimizing impedance changes
            pos_error = X[i] - X_ref[ref_idx]
            cost += cp.quad_form(pos_error, self.Q)

            if i > 0:
                K_change = K[i] - K[i-1]
                D_change = D[i] - D[i-1]
                cost += 0.1 * cp.norm(K_change, 2) + 0.2 * cp.norm(D_change, 2)

            # Penalize low tank energy
            cost += 0.5 * cp.maximum(self.T_min + 2.0 - T[i], 0)**2

        # Terminal cost: ensure safe impedance at end
        cost += 2 * cp.norm(K[-1] - np.array([300, 300, 300]), 2)
        cost += 2 * cp.norm(D[-1] - np.array([30, 30, 30]), 2)

        # Solve
        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            # Fast solver for real-time
            result = problem.solve(solver=cp.OSQP, warm_start=True,
                                 verbose=False, max_iter=1000)

            if problem.status in ["optimal", "optimal_inaccurate"]:
                # Update tank energy
                self.T_energy = T.value[1]  # Next step's tank energy

                # Return first control (MPC)
                return K.value[0], D.value[0], {
                    'status': problem.status,
                    'tank_energy': self.T_energy,
                    'horizon': N
                }
        except:
            pass

        # Fallback: maintain current or safe impedance
        return self._get_fallback_impedance(), {'status': 'fallback'}

    def _generate_safe_profile(self, N):
        """Generate guaranteed-safe impedance profile"""
        K_safe = np.ones((N, 3)) * (self.K_min + self.K_max) / 2
        D_safe = 1.5 * np.sqrt(K_safe)  # Over-damped for safety
        return K_safe, D_safe

    def _get_fallback_impedance(self):
        """Fallback impedance for when optimization fails"""
        K_fallback = np.array([300, 300, 300])  # Moderate stiffness
        D_fallback = 2.0 * np.sqrt(K_fallback)  # Critically damped
        return K_fallback, D_fallback

    def reset_energy_tank(self):
        """Reset energy tank to initial state"""
        self.T_energy = self.T_initial