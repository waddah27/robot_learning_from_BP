from typing import Any
import numpy as np
from scipy.optimize import minimize
from numpy import ndarray

class Stiffness_global_optimizer:
    def __init__(self, n_dims, k_min: ndarray, k_max: ndarray, epsilon: float, delta_t: float) -> None:
        self.n = n_dims
        self.k_min = k_min
        self.k_max = k_max
        self.epsilon = epsilon
        self.delta_t = delta_t
        self.Q = np.ones(3)
        self.R = np.ones(3)
        self.C = np.eye(self.n)
        self.x_t = 0.3  # Initial tank energy state
        self.damping_factor = 0.707
    def set_weights(self, q_w=3200, r_w = 1):
        self.Q *= q_w
        self.R *= r_w

    def objective(self, k_d, F_ext, F_d):
        self.set_weights()
        # Weight the differences by Q and R before taking the norm
        weighted_diff_F = (F_ext - F_d) * np.sqrt(self.Q)  # Element-wise multiplication
        weighted_diff_K = (k_d - self.k_min) * np.sqrt(self.R)  # Element-wise multiplication

        return 0.5 * (np.sum(weighted_diff_F**2) + np.sum(weighted_diff_K**2))
    # Tank energy dynamics functions
    def T(self, x_t):
        return 0.5 * x_t**2

    def tank_energy_constraint(self, k_d, x_t, x_tilde_dot, K_v, D_d):
        x_t_dot = self.tank_dynamics(x_t, x_tilde_dot, K_v, D_d)
        x_t_next = x_t + x_t_dot * self.delta_t
        return self.T(x_t_next) - self.epsilon

    def tank_dynamics(self, x_t, x_tilde_dot, K_v, D_d):
        sigma = 1 if self.T(x_t) > self.epsilon else 0
        w = (-K_v * x_tilde_dot) if self.T(x_t) > self.epsilon else 0
        return (sigma * np.dot(x_tilde_dot.T, np.dot(D_d, x_tilde_dot))) - np.dot(w.T, x_tilde_dot) / x_t

    def compute_D_d(self, k_d_prev):
        return 2 * self.damping_factor * np.sqrt(k_d_prev)

    def run(self, F_ext, F_d):
        # Optimization process for each instance
        k_d_init = self.k_min + (self.k_min + self.k_max)/2 # Initial guess
        D_d = np.diag([0.7, 0.7, 0.7])
        x_t = 1  # Initial state of the tank
        x_tilde_dot = np.array([0.1,0.1,0.1])  # Assume the robot follows the exact trajectory with the exact velocity
        K_v = k_d_init  # Initial variable part of the stiffness

        # Update tank energy constraint with current instance values
        constraints = [{'type': 'ineq', 'fun': self.tank_energy_constraint, 'args': (x_t, x_tilde_dot, K_v, D_d)}]
        bounds = [(self.k_min[j], self.k_max[j]) for j in range(self.n)]

        result = minimize(self.objective, k_d_init, args=(F_ext, F_d), bounds=bounds, constraints=constraints, method='SLSQP')

        if result.success:
            dd_next = self.compute_D_d(result.x)
            D_d = np.diag(dd_next)
        else:
            print(f"Optimization failed for instance : {result.message}")


if __name__=="__main__":
    # System parameters
    k_min = np.array([200, 200, 200])
    k_max = np.array([1000, 1000, 1000])
    Q = np.array([3200, 3200, 3200])
    R = np.array([1, 1, 1])
    epsilon = 0.4
    delta_t = 1  # Assumed time step
    # optimizer = Stiffness_global_optimizer(n_dims=3, k_max=k_max, k_min=k_min, epsilon=0.4, delta_t=0.1)

    x_tilde_demo = np.array([[0.1, 0.1, 0.1] for _ in range(10)])
    F_d_demo = np.array([[0, 0, 40] for _ in range(10)])
    kd_opt_arr = []
    stiffness_optimizer = Stiffness_global_optimizer(n_dims=3, k_max=k_max, k_min=k_min, epsilon=0.4, delta_t=0.1)
    for x_tilde, F_d in zip(x_tilde_demo, F_d_demo):
        kd_opt = stiffness_optimizer.run(x_tilde, F_d)
        if kd_opt is not None:
            kd_opt_arr.append(kd_opt)


    kd_opt_arr = np.array(kd_opt_arr)
