from typing import Any, Optional
import cvxpy as cp
import numpy as np
from numpy import ndarray

class optimal_stiffness:
    def __init__(
        self,
        n_dims,
        k_min:ndarray,
        k_max:ndarray,
        ) -> None:
        self.n = n_dims
        self.k_min = k_min
        self.k_max = k_max
        self.Q = np.eye(self.n)
        self.R = np.eye(self.n)
        self.objective:Any = None
        self.constraints:Any = None
        self.problem:Any = None
        # self.set_weights()

    def set_weights(self, q_factor:int=3200, r_factor:int=1):
        self.Q *= q_factor
        self.R *= r_factor

    def run(self, x:ndarray, F_d:ndarray):
        self.set_weights()
        x_diag = np.diag(x)
        P = x_diag.T @ self.Q @ x_diag + self.R
        q = -2 * (F_d.T @ self.Q @ x_diag + self.k_min.T @ self.R)
        k_d = cp.Variable(self.n)
        self.objective = cp.Minimize(0.5 * cp.quad_form(k_d, P)+q.T @ k_d)
        self.constraints = [k_d >= self.k_min, k_d<=self.k_max]
        self.problem = cp.Problem(self.objective, self.constraints)
        solution = self.problem.solve()
        return k_d.value, solution, self.problem.status


if __name__=="__main__":

    # Define the problem data
    n = 3  # Dimension of k^d
    x_tilde = np.array([1.0, 2.0, 3.0])  # Example values for the Cartesian position error
    F_d = np.array([1.5, 1.5, 1.5])  # Desired forces to be applied by robot ee
    K_min = np.array([0.5, 0.5, 0.5])  # Minimum desired stiffness
    k_min = np.array([0.1, 0.1, 0.1])  # Lower bounds on k^d
    k_max = np.array([10.0, 10.0, 10.0])  # Upper bounds on k^d
    q_factor = 1
    r_factor = 0.1
    stiffness_optimizer = optimal_stiffness(n_dims = n, k_min= K_min, k_max= k_max)
    stiffness_optimizer.set_weights(q_factor=q_factor, r_factor=r_factor)
    k_d_opt, opt_solution, status = stiffness_optimizer.run(x_tilde, F_d)
    

    # Output the results
    print("Status:", status)
    print("The optimal value is", opt_solution)
    print("The optimal k^d is", k_d_opt)
