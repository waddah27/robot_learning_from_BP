from typing import Any, Optional
import cvxpy as cp
from matplotlib import pyplot as plt
import numpy as np
from numpy import ndarray

class optimal_stiffness:
    def __init__(
        self,
        n_dims,
        k_min: ndarray,
        k_max: ndarray,
        epsilon: float,  # Minimum tank energy
        initial_tank_energy: float,  # Initial energy in the tank
        solver:Any = cp.SCS
        ) -> None:
        self.n = n_dims
        self.k_min = k_min
        self.k_max = k_max
        self.epsilon = epsilon
        self.tank_energy = initial_tank_energy
        self.solver = solver
        self.Q = np.eye(self.n)
        self.R = np.eye(self.n)
        self.objective: Any = None
        self.constraints: Any = None
        self.problem: Any = None
        self.dt = 0.1 # Simulation timestep


    def set_weights(self, q_factor: int = 3200, r_factor: int = 1):
        self.Q *= q_factor
        self.R *= r_factor

    def run(self, x: ndarray, F_d: ndarray, verbose = True):
        self.set_weights()
        x_diag = np.diag(x)  # Diagonal matrix of position errors x
        P = x_diag.T @ self.Q @ x_diag + self.R  # Quadratic term matrix
        print(f"P is positive semidefinate: {np.all(np.linalg.eigvals(P) >= 0)}")
        print(f"Condition Number for P: {np.linalg.cond(P)}")
        q = -2 * (F_d.T @ self.Q @ x_diag + self.k_min.T @ self.R)  # Linear term
        k_d = cp.Variable(self.n)  # Decision variable for stiffness

        # Correct energy rate calculation
        # Using element-wise multiplication to calculate the dot product of k_d and x squared
        T_dot = cp.sum(cp.multiply(k_d, x**2))  # Simplified model

        # Objective and constraints setup
        new_tank_energy = self.tank_energy + T_dot * self.dt
        self.objective = cp.Minimize(0.5 * cp.quad_form(k_d, P) + q.T @ k_d)
        self.constraints = [
            k_d >= self.k_min,
            k_d <= self.k_max,
            new_tank_energy >= self.epsilon  # Energy constraint
        ]
        self.problem = cp.Problem(self.objective, self.constraints)
        solution = self.problem.solve(verbose=verbose, solver=self.solver)

        # Update the internal state of tank energy
        self.tank_energy = new_tank_energy.value

        return k_d.value, solution, self.problem.status, self.tank_energy


if __name__ == "__main__":
    n = 3
    x_tilde = np.array([0.0, 0.0, 0.0])
    F_d = np.array([0, 0, 0])
    K_min = np.array([10.0, 10.0, 10.0])
    k_max = np.array([2000.0, 2000.0, 2000.0])
    epsilon = 0.3
    initial_tank_energy = 0.3  # Example initial energy
    trj = np.linspace(0,10,10)
    x_tr = np.linspace(1,0,10)
    F_d_demo = np.array([[0, 0, 40] for i in trj])
    x_tilde_demo = np.array([[0.1,i,i] for i in x_tr])
    kd_opt_arr = []
    stiffness_optimizer = optimal_stiffness(n_dims=n, k_min=K_min, k_max=k_max, epsilon=epsilon, initial_tank_energy=initial_tank_energy)
    i = 0
    for x_tilde, F_d in zip(x_tilde_demo, F_d_demo):
        try:
            k_d_opt, opt_solution, status, tank_energy = stiffness_optimizer.run(x_tilde, F_d, verbose=False)
            kd_opt_arr.append(k_d_opt)
            print(f"Solved  at {i}:Status = {status}, Optimal value = {opt_solution}, updated tank energy = {tank_energy}, F_d = {F_d}, X_tilde = {x_tilde} -- Kd = {k_d_opt}")
        except Exception as E:
            print(f"Failed at {i}: F_d = {F_d}, x_tilde = {x_tilde} -- Reason: {E}")
            kd_opt_arr.append(K_min)
            pass
        # print("Status:", status)
        # print("The optimal value is", opt_solution)
        # print("The optimal k^d is", k_d_opt)
        # print("Updated tank energy:", tank_energy)
        i+=1
    kd_opt_arr = np.array(kd_opt_arr)
    col = kd_opt_arr[:,0]

    for i in range(kd_opt_arr.shape[1]):
        plt.plot(trj, kd_opt_arr[:,i], label=f'$Kd_{i}$')
    plt.legend()
    plt.show()
