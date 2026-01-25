from typing import Optional
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

class VICController:
    def __init__(self, F_min:Optional[ndarray]=None, F_max:Optional[ndarray]=None):
        self.k_min = np.array([0, 0, 0])
        self.k_max = np.array([5000, 5000, 5000])
        self.K_d = self.k_min
        self.D_min = np.array([0, 0, 0])
        self.D_max = np.array([1, 1, 1])
        self.D_d = np.array([0.7,0.7,0.7])
        self.f_min = -70 if F_min is None else F_min
        self.f_max = 70 if F_max is None else F_max
        self.epsilon = 0.4
        self.scaler = 1
        self.Q = np.eye(3)
        self.R = np.eye(3) * 1e-9
        self.force_error = np.inf
        self.delta_t = 0.1
        self.TCProtMat = np.array([
            [0,0,1],
            [0,1,0],
            [-1,0,0]
        ])
        self.K_d_list = []
        self.E_tot = None

    def objective(self, params, x_tilde, x_tilde_dot, F_d):
        # x_tilde = x_tilde[:,[1,0,2]]
        # x_tilde_dot = x_tilde_dot[:,[1,0,2]]
        k_d = np.diag(params[:3])
        xi_d = np.diag(params[3:6])
        d_d = self.calculate_damping(xi_d, k_d)
        F_ext = np.dot(k_d, x_tilde) + np.dot(d_d, x_tilde_dot)

        # Calculating the weighted squared norms
        norm_F = np.dot((F_ext - F_d).T, np.dot(self.Q, (F_ext - F_d)))
        norm_k = np.dot((np.diag(k_d) - self.k_min).T, np.dot(self.R, (np.diag(k_d) - self.k_min)))

        # Summing the weighted norms
        # error = norm_F + norm_k

        # Constraints violation penalties (very basic approach)
        force_penalty = np.sum(np.maximum(0, F_ext - self.f_max)**2) + np.sum(np.maximum(0, self.f_min - F_ext)**2)
        self.force_error = norm_F + norm_k #np.sum((F_d - F_ext) ** 2)
        return self.force_error + force_penalty

    def calculate_damping(self,xi_d, k_d):
        sqrt_k_d = np.sqrt(k_d)
        # Step 2: Calculate damping coefficients, d_d = 2 * xi * sqrt(k)
        return  2 * np.diag(xi_d) * sqrt_k_d

    def tank_energy_constraint(self, k_d, x_t, x_tilde, x_tilde_dot, D_d):
        x_t_dot = self.tank_dynamics(x_t, x_tilde_dot, k_d, D_d)
        x_t_next = x_t + x_t_dot * self.delta_t
        self.E_tot += self.T(x_t_next)
        print('*********************')
        return self.epsilon - self.T(x_t_next)  # Constraint function must return >= 0

    def tank_dynamics(self, x_tilde_dot, K_v, D_d):
        # Assuming simplified dynamics
        return -K_v * np.sum(x_tilde_dot) - np.sum(D_d * x_tilde_dot)

    def T(self, x_t):
        return 0.5 * np.sum(x_t**2)

    def optimize(self, x_tilde, x_tilde_dot, F_d):
        # self.scaler = 1/x_tilde # stiffness adapter to increase Force sensitivity
        # x_tilde = x_tilde * self.scaler
        x_t = np.array([self.epsilon, self.epsilon, self.epsilon])
        D_d = self.D_min
        initial_guess = [200, 200, 200, 0.7, 0.7, 0.7]  # Initial guess

        bounds = [(k_min, k_max) for k_min, k_max in zip(self.k_min, self.k_max)] + [(d_min, d_max) for d_min, d_max in zip(self.D_min, self.D_max)]  # Bounds for K_d and no bounds for D_d

        constraints = ({
            'type': 'ineq',
            'fun': self.tank_energy_constraint,
            'args': (x_t, x_tilde, x_tilde_dot, D_d)
        })

        result = minimize(
            self.objective,
            initial_guess,
            args=(x_tilde, x_tilde_dot, F_d),
            bounds=bounds,
            constraints=constraints,
            method='L-BFGS-B'
            )
        self.K_d = result.x[:3]
        self.D_d = result.x[3:6]
        return self.K_d, self.D_d

    def calculate_force(self, x_tilde, x_tilde_dot):
        kd = np.diag(self.K_d)
        dd = np.diag(self.D_d)
        return np.dot(kd, x_tilde) + np.dot(dd, x_tilde_dot)