from typing import Optional
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

class VICController:
    def __init__(self, F_min: Optional[ndarray] = None, F_max: Optional[ndarray] = None):
        self.k_min = np.array([100, 100, 100])
        self.k_max = np.array([5000, 5000, 5000])
        self.K_d = self.k_min
        self.D_min = np.array([0, 0, 0])
        self.D_max = np.array([1, 1, 1])
        self.D_d = np.array([0.7, 0.7, 0.7])
        self.f_min = np.array([-70, -70, -70]) if F_min is None else F_min
        self.f_max = np.array([70, 70, 70]) if F_max is None else F_max
        self.Q = np.eye(3) * 6400
        self.R = np.eye(3)
        self.force_error = np.inf
        self.delta_t = 0.1

    def normalize_params(self, params):
        k_d_norm = (params[:3] - self.k_min) / (self.k_max - self.k_min)
        d_d_norm = (params[3:6] - self.D_min) / (self.D_max - self.D_min)
        return np.concatenate((k_d_norm, d_d_norm))

    def reverse_normalize_params(self, params_norm):
        k_d = params_norm[:3] * (self.k_max - self.k_min) + self.k_min
        d_d = params_norm[3:6] * (self.D_max - self.D_min) + self.D_min
        return np.concatenate((k_d, d_d))

    def normalize_forces(self, F):
        return (F - self.f_min) / (self.f_max - self.f_min)

    def reverse_normalize_forces(self, F_norm):
        return F_norm * (self.f_max - self.f_min) + self.f_min

    def objective(self, params_norm, x_tilde, x_tilde_dot, F_d):
        params = self.reverse_normalize_params(params_norm)
        k_d = np.diag(params[:3])
        d_d = np.diag(params[3:6])

        F_ext = np.dot(k_d, x_tilde) + np.dot(d_d, x_tilde_dot)
        F_ext_norm = self.normalize_forces(F_ext)
        F_d_norm = self.normalize_forces(F_d)

        norm_F = np.dot((F_ext_norm - F_d_norm).T, np.dot(self.Q, (F_ext_norm - F_d_norm)))
        norm_k = np.dot((params[:3] - self.k_min).T, np.dot(self.R, (params[:3] - self.k_min)))

        force_penalty = np.sum(np.maximum(0, F_ext_norm - 1)**2) + np.sum(np.maximum(0, -F_ext_norm)**2)
        return norm_F + norm_k + force_penalty

    def optimize(self, x_tilde, x_tilde_dot, F_d):
        initial_guess = self.normalize_params(np.concatenate((self.K_d, self.D_d)))
        bounds_norm = [(0, 1)] * 6

        result = minimize(
            self.objective,
            initial_guess,
            args=(x_tilde, x_tilde_dot, self.normalize_forces(F_d)),
            bounds=bounds_norm,
            method='L-BFGS-B'
        )

        final_params = self.reverse_normalize_params(result.x)
        self.K_d, self.D_d = final_params[:3], final_params[3:6]
        return self.K_d, self.D_d

    def calculate_force(self, x_tilde, x_tilde_dot):
        kd = np.diag(self.K_d)
        dd = np.diag(self.D_d)
        F_ext = np.dot(kd, x_tilde) + np.dot(dd, x_tilde_dot)
        return F_ext

# # Example use
# vic = VICController()
# x_tilde = np.array([1.0, 0.5, -1.0])
# x_tilde_dot = np.array([0.1, -0.1, 0.05])
# F_d = np.array([5.0, 10.0, 0.0])
# K_d, D_d = vic.optimize(x_tilde, x_tilde_dot, F_d)
# print("K_d:", K_d)
# print("D_d:", D_d)
# force = vic.calculate_force(x_tilde, x_tilde_dot)
# print("Calculated Force:", force)
