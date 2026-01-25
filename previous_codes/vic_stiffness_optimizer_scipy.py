import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class OptimalStiffness:
    def __init__(self, n_dims, k_min, k_max, Q_factor=3200, R_factor=1, delta_t=0.1, epsilon=0.4):
        self.n = n_dims
        self.k_min = k_min
        self.k_max = k_max
        self.Q = Q_factor * np.eye(n_dims)
        self.R = R_factor * np.eye(n_dims)
        self.delta_t = delta_t
        self.epsilon = epsilon
        # self.x_t = epsilon

    def objective_function(self, k_d, x_tilde, F_d):
        x_tilde_dot = np.array([0.1, 0.1, 0.1])  # Simplified constant velocity error
        D_d = np.array([0.7, 0.7, 0.7])  # Damping
        F_ext = (np.eye(self.n) @ x_tilde_dot + D_d) * x_tilde_dot + np.dot(k_d, x_tilde)
        term1 = 0.5 * np.dot((F_ext - F_d).T, np.dot(self.Q, (F_ext - F_d)))
        term2 = 0.5 * np.dot(k_d.T, np.dot(self.R, k_d))
        return term1 + term2

    def tank_energy_constraint(self, k_d, x_t, x_tilde, x_tilde_dot, D_d):
        x_t_dot = self.tank_dynamics(x_t, x_tilde_dot, k_d, D_d)
        x_t_next = x_t + x_t_dot * self.delta_t
        return self.epsilon - self.T(x_t_next)  # Constraint function must return >= 0

    def tank_dynamics(self, x_t, x_tilde_dot, K_v, D_d):
        # Assuming simplified dynamics
        return -K_v * np.sum(x_tilde_dot) - np.sum(D_d * x_tilde_dot)

    def T(self, x_t):
        return 0.5 * np.sum(x_t**2)

    def run_global_optimization(self, x_tilde, F_d):
        x_t = np.array([self.epsilon, self.epsilon, self.epsilon])  # Initial tank state
        x_tilde_dot = np.array([0.1, 0.1, 0.1])  # Simulated error dynamics
        D_d = np.array([0.7, 0.7, 0.7])  # Damping values
        k_d_init = self.k_min #np.mean([self.k_min, self.k_max], axis=0)  # Initial stiffness values

        constraints = ({
            'type': 'ineq',
            'fun': self.tank_energy_constraint,
            'args': (x_t, x_tilde, x_tilde_dot, D_d)
        })

        result = minimize(
            self.objective_function,
            k_d_init,
            args=(x_tilde, F_d),
            method='SLSQP',
            bounds=[(self.k_min[i], self.k_max[i]) for i in range(self.n)],
            constraints=constraints
        )
        return result.x, result.fun

    def run(self, x_tilde, F_d):
        k_d_opt, opt_value = self.run_global_optimization(x_tilde, F_d)
        print(f"Optimal stiffness: {k_d_opt}, Objective function value: {opt_value}")
        return k_d_opt

if __name__ == "__main__":
    n = 3
    k_min = np.array([10.0, 10.0, 10.0])
    k_max = np.array([200.0, 200.0, 200.0])
    stiffness_optimizer = OptimalStiffness(n_dims=n, k_min=k_min, k_max=k_max)
    # x_tilde = np.array([0.1, 0.1, 0.1])
    # F_d = np.array([0, 0, 40])
    # kd_opt = stiffness_optimizer.run(x_tilde, F_d)

    f_tr = np.linspace(5,60,10)
    x_tr = np.linspace(10,0,10)
    F_d_demo = np.array([[0, 0, 60] for i in f_tr])
    x_tilde_demo = np.array([[0.0,0.0,1] for i in x_tr])
    kd_opt_arr = []
    stiffness_optimizer = OptimalStiffness(n_dims=n, k_min=k_min, k_max=k_max)
    for x_tilde, F_d in zip(x_tilde_demo, F_d_demo):
        kd_opt = stiffness_optimizer.run(x_tilde, F_d)
        if kd_opt is not None:
            kd_opt_arr.append(kd_opt)


    kd_opt_arr = np.array(kd_opt_arr)

    # Plotting results
    for i in range(n):
    # plt.plot(x_tilde_demo[:,-1], label=f"$Z$")
        plt.plot(kd_opt_arr[:,i], label=f"$K_{i}$")
    plt.legend()
    plt.show()


# import numpy as np
# from scipy.optimize import differential_evolution
# import matplotlib.pyplot as plt

# class OptimalStiffness:
#     def __init__(self, n_dims, k_min, k_max, Q_factor=3200, R_factor=1):
#         self.n = n_dims
#         self.k_min = k_min
#         self.k_max = k_max
#         self.Q = Q_factor * np.eye(n_dims)
#         self.R = R_factor * np.eye(n_dims)

#     def objective_function(self, k_d, x_tilde, F_d):
#         x_tilde_dot = np.array([0.1, 0.1, 0.1])  # Simplified constant velocity error
#         D_d = np.array([0.7, 0.7, 0.7])  # Damping
#         F_ext = np.eye(self.n) @ x_tilde_dot + D_d * x_tilde_dot + np.dot(k_d, x_tilde)
#         term1 = 0.5 * np.dot((F_ext - F_d).T, np.dot(self.Q, (F_ext - F_d)))
#         term2 = 0.5 * np.dot((k_d - self.k_min).T, np.dot(self.R, (k_d - self.k_min)))
#         return term1 + term2
#     # Tank energy dynamics functions
#     def T(self, x_t):
#         return 0.5 * x_t**2

#     def tank_energy_constraint(self, k_d, x_t, x_tilde_dot, K_v, D_d):
#         x_t_dot = self.tank_dynamics(x_t, x_tilde_dot, K_v, D_d)
#         x_t_next = x_t + x_t_dot * self.delta_t
#         return self.T(x_t_next) - self.epsilon

#     def tank_dynamics(self, x_t, x_tilde_dot, K_v, D_d):
#         sigma = 1 if self.T(x_t) > self.epsilon else 0
#         w = (-K_v * x_tilde_dot) if self.T(x_t) > self.epsilon else 0
#         return (sigma * np.dot(x_tilde_dot.T, np.dot(D_d, x_tilde_dot))) - np.dot(w.T, x_tilde_dot) / x_t

#     def compute_D_d(self, k_d_prev):
#         return 2 * self.damping_factor * np.sqrt(k_d_prev)

#     def run_global_optimization(self, x_tilde, F_d):
#          # Optimization process for each instance
#         k_d_init = self.k_min + (self.k_min + self.k_max)/2 # Initial guess
#         D_d = np.diag([0.7, 0.7, 0.7])
#         x_t = 1  # Initial state of the tank
#         x_tilde_dot = np.array([0.1,0.1,0.1])  # Assume the robot follows the exact trajectory with the exact velocity
#         K_v = k_d_init  # Initial variable part of the stiffness

#         # Update tank energy constraint with current instance values
#         constraints = [{'type': 'ineq', 'fun': self.tank_energy_constraint, 'args': (x_t, x_tilde_dot, K_v, D_d)}]
#         bounds = [(self.k_min[j], self.k_max[j]) for j in range(self.n)]

#         # result = minimize(self.objective, k_d_init, args=(F_ext, F_d), bounds=bounds, constraints=constraints, method='SLSQP')

#         bounds = [(self.k_min[i], self.k_max[i]) for i in range(len(self.k_min))]
#         result = differential_evolution(
#             self.objective_function,
#             bounds=bounds,
#             constraints=constraints,
#             args=(x_tilde, F_d)
#         )
#         return result.x, result.fun

#     def run(self, x_tilde, F_d):
#         # kd_opt_arr = []
#         # for x_tilde, F_d in zip(x_tildes, F_ds):
#         k_d_opt, opt_value = self.run_global_optimization(x_tilde, F_d)
#         kd_opt_arr.append(k_d_opt)
#         print(f"Optimal stiffness: {k_d_opt}, Objective function value: {opt_value}")
#         return kd_opt_arr

# if __name__ == "__main__":
#     n = 3
#     k_min = np.array([10.0, 10.0, 10.0])
#     k_max = np.array([200000.0, 200000.0, 200000.0])

#     x_tilde_demo = np.array([[0.1, 0.1, 0.1] for _ in range(10)])
#     F_d_demo = np.array([[0, 0, 40] for _ in range(10)])
#     kd_opt_arr = []
#     stiffness_optimizer = OptimalStiffness(n_dims=n, k_min=k_min, k_max=k_max)
#     for x_tilde, F_d in zip(x_tilde_demo, F_d_demo):
#         kd_opt = stiffness_optimizer.run(x_tilde, F_d)
#         if kd_opt is not None:
#             kd_opt_arr.append(kd_opt)


#     kd_opt_arr = np.array(kd_opt_arr)

#     # Plotting results
#     # for i in range(n):
#     # plt.plot(kd_opt_arr)
#     # plt.legend()
#     # plt.show()
