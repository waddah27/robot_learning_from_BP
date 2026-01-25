import numpy as np
from scipy.optimize import minimize
from numpy import ndarray
import matplotlib.pyplot as plt
class VICController:
    def __init__(self):
        self.k_min = np.array([0, 0, 0])
        self.k_max = np.array([5e3, 5e3, 5e3])
        self.D_min = np.array([0.3, 0.3, 0.3])
        self.D_max = np.array([0.8, 0.8, 0.8])
        self.F_min = -60
        self.F_max = 60
        self.epsilon = 0.4
        self.Q = np.eye(3) * 3200
        self.R = np.eye(3)
        self.F_ext = None
    def objective(self, params:list, x_tilde:ndarray, x_tilde_dot:ndarray, F_d:ndarray):
        K_d = np.diag(params[:3])
        D_d = np.diag(params[3:6])
        self.F_ext = np.dot(K_d, (x_tilde)) + np.dot(D_d, (x_tilde_dot))
        # Calculate the squared error
        # error = np.sum(((F_d - self.F_ext)@self.Q) ** 2) + np.sum(((K_d-self.k_min)@self.R)**2)
        error = np.sum((F_d-self.F_ext)**2)
        # Constraint violation penalties (very basic approach)
        force_penalty = np.sum(np.maximum(0, self.F_ext - self.F_max) ** 2 + np.maximum(0, self.F_min - self.F_ext) ** 2)
        return error + force_penalty

    def optimize(self, x_tilde, x_tilde_dot, F_d):
        initial_guess = [self.k_min[0], self.k_min[1], self.k_min[0], self.D_max[0], self.D_max[1], self.D_max[2]]  # Initial guess
        bounds = [(k_min, k_max) for k_min, k_max in zip(self.k_min, self.k_max)] + [(d_min, d_max) for d_min, d_max in zip(self.D_min, self.D_max)]  # Bounds for K_d and no bounds for D_d
        result = minimize(
            self.objective,
            initial_guess,
            args=(x_tilde, x_tilde_dot, F_d),
            bounds=bounds,
            method='L-BFGS-B'
            )
        self.K_d = result.x[:3]
        self.D_d = result.x[3:6]
        return self.K_d, self.D_d

    def calculate_force(self, x_tilde, x_tilde_dot):
        kd= np.diag(self.K_d)
        dd = np.diag(self.D_d)
        return np.dot(kd, x_tilde) + np.dot(dd, x_tilde_dot)

# Example usage
x_d = [1.0, 1.0, 1.0]
dot_x_d = [0.0, 0.0, 0.0]
x = [0.5, 0.5, 0.5]
dot_x = [0.1, 0.1, 0.1]
x_tilde = np.array(x_d) - np.array(x)
x_tilde_dot = np.array(dot_x_d) - np.array(dot_x)
F_d = [70.0, 70.0, 70.0]
x_tilde_demo = np.random.rand(100,3)/100
F_d_demo = np.random.rand(100,3)*50
F_ext_demo = []
F_actual_demo = []
K_d_demo = []
controller = VICController()
for x_tilde, F_d in zip(x_tilde_demo, F_d_demo):
    K_d_opt, D_d_opt = controller.optimize(x_tilde, x_tilde_dot, F_d)
    F_actual_opt = controller.calculate_force(x_tilde, x_tilde_dot)
    F_ext_demo.append(controller.F_ext)
    F_actual_demo.append(F_actual_opt)
    K_d_demo.append(K_d_opt)
print("Optimized K_d:", K_d_opt)
print("Optimized D_d:", D_d_opt)
print("Achieved Force F_actual:", F_actual_opt)

F_ext_seq = np.array(F_ext_demo)
F_actual_seq = np.array(F_actual_demo)
K_d_seq = np.array(K_d_demo)
ax1 = plt.subplot(311)


ax1.plot(F_ext_seq[:,0], 'go--', linewidth=2, markersize=2, label='F_ext(x)')
# ax1.plot(F_actual_seq[:,0], label='F_act(x)',color='b')
ax1.plot(F_d_demo[:,0], label='F_d(x)')
# ax1.plot(K_d_seq[:,0], label='K_d(x)')
plt.legend()
plt.show()
