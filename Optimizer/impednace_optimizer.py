import cvxpy as cp
import numpy as np
from scipy import signal

__all__ = ["ImpedanceProfileOptimizer"]


class ImpedanceProfileOptimizer:
    def __init__(self, dt=0.002, horizon=1000):
        self.dt = dt
        self.horizon = horizon
        self.setup_constraints()

    def setup_constraints(self):
        """Define physical and stability limits"""
        self.K_min = np.array([50, 50, 50])      # N/m (softer than your current)
        self.K_max = np.array([2000, 2000, 2000]) # N/m
        self.D_min = np.array([5, 5, 5])         # Ns/m
        self.D_max = np.array([100, 100, 100])   # Ns/m
        self.K_rate_max = 1000                   # N/m/s
        self.D_rate_max = 200                    # Ns/m/s

    def optimize_trajectory(self, X_des, V_des, F_des, mass_matrix):
        """
        Solve for optimal K(t), D(t) given desired profiles
        """
        N = len(X_des)

        # Decision variables
        K = cp.Variable((N, 3))  # Stiffness profile [Kx, Ky, Kz]
        D = cp.Variable((N, 3))  # Damping profile
        X = cp.Variable((N, 3))  # Predicted position
        V = cp.Variable((N, 3))  # Predicted velocity
        F = cp.Variable((N, 3))  # Predicted force

        # Initial conditions
        constraints = [
            X[0] == X_des[0],
            V[0] == V_des[0]
        ]

        # Dynamics constraints (simplified for optimization)
        M = np.diag(mass_matrix) if mass_matrix.ndim == 1 else mass_matrix

        cost_terms = []
        for i in range(N-1):
            # Simplified dynamics: M*a + D*v + K*x = F_ext
            a = (V[i+1] - V[i]) / self.dt
            constraints.append(
                M @ a + D[i] @ V[i] + K[i] @ (X[i] - X_des[i]) == F_des[i]
            )

            # Integrate velocity
            constraints.append(
                X[i+1] == X[i] + V[i] * self.dt
            )

            # Rate limits
            if i > 0:
                constraints.append(
                    cp.abs(K[i] - K[i-1]) <= self.K_rate_max * self.dt
                )
                constraints.append(
                    cp.abs(D[i] - D[i-1]) <= self.D_rate_max * self.dt
                )

            # Cost terms
            tracking_error = cp.norm(X[i] - X_des[i], 2)
            force_error = cp.norm(F[i] - F_des[i], 2)
            impedance_cost = cp.norm(K[i], 'fro') + cp.norm(D[i], 'fro')

            cost_terms.append(1.0 * tracking_error +
                             0.5 * force_error +
                             0.1 * impedance_cost)

        # Bound constraints
        for i in range(N):
            constraints.append(K[i] >= self.K_min)
            constraints.append(K[i] <= self.K_max)
            constraints.append(D[i] >= self.D_min)
            constraints.append(D[i] <= self.D_max)

            # Passivity constraint: D[i] >= ε * sqrt(K[i])
            # Enforced element-wise for each direction
            for j in range(3):
                constraints.append(D[i, j] >= 0.5 * cp.sqrt(K[i, j]))

        # Objective
        objective = cp.sum(cost_terms)
        problem = cp.Problem(cp.Minimize(objective), constraints)

        # Solve
        try:
            result = problem.solve(solver=cp.MOSEK, verbose=False)
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Warning: Optimization status: {problem.status}")
                return self.get_fallback_profile(N)

            return K.value, D.value, X.value, V.value

        except Exception as e:
            print(f"Optimization failed: {e}")
            return self.get_fallback_profile(N)

    def get_fallback_profile(self, N):
        """Generate safe impedance profile when optimization fails"""
        K_safe = np.tile((self.K_min + self.K_max) / 2, (N, 1))
        D_safe = 2 * np.sqrt(K_safe)  # Critically damped
        return K_safe, D_safe, None, None