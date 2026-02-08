import numpy as np
import matplotlib.pyplot as plt
from data_analysis_utils import get_lipschitz_criterion, get_norm_bound_threshold, get_smoothness_threshold
from plotters.plotter_traj_3d import PlotterTraj3D
from vic_controller_with_tank_energy_inside import VICController
from time import time
from exp_stability_analysis import QuadraticLyapunov
from plotters import PlotterOfflineSameRange
from data import MaterialData, bpTrajDataLoader
from motion_planners import MotionPlanner
import sys
from bp_basic_experiment import BasicBPexperiment as bp
from approach_analysis import ResAnalyser
# Load the data
bp_data = MaterialData.cork
bp_traj = bpTrajDataLoader(bp_data)
print(bp_traj.shape)

title = f"Visualization of Dynamics during cutting {bp_traj.material_name}"

# Optimizer configs

analyse_results = True
bp.controller = VICController(F_min=bp_traj.F_min, F_max=bp_traj.F_max)
bp.planner = MotionPlanner(bp_traj)
bp.plotter = PlotterOfflineSameRange(title=title)

plot_desired_3d_pos = True
krasovskii_analyasis = False
quadratic_lyapunov = True

if __name__ == "__main__":
    # tcp_rot = transform_coordinates(th_x_deg=90, th_y_deg=90)
    if plot_desired_3d_pos:
        PlotterTraj3D.plot(bp_traj)
    # sys.exit(0)
    kd_opt, dd_opt = bp.run(bp_traj)


    if analyse_results:

        # Generated forces continuity
        F_d_Continuity = ResAnalyser.run(bp_traj)

        # get F_d smoothness threshold: derivatives of F_d are bounded
        F_d_sm = ResAnalyser.get_smoothness(bp_traj)
        sys.exit(0)

        # get the continuity of F_d_dot
        F_d_dot = np.diff(bp_traj.force, axis=0)
        F_d_dot_array = np.array(F_d_dot).T
        print(f"F_d_dot: {F_d_dot}")
        for ax, i in enumerate(range(3)):
            plt.plot(F_d_dot_array[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{F_d} :1^{st} derivative generated force on X,Y and Z$: '+ bp_traj.material_name)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ bp_traj.material_name)
        plt.legend()
        plt.show()

        F_d_dot_continuity = get_lipschitz_criterion(F_d_dot)
        print(f"F_d_dot_continuity: {F_d_dot_continuity}")

        # get the continuity of F_d_ddot: second derivative of F_d is continuous
        F_d_ddot = np.diff(F_d_dot, axis=0)
        F_d_ddot_array = np.array(F_d_ddot).T
        print(f"F_d_ddot: {F_d_ddot}")
        for ax, i in enumerate(range(3)):
            plt.plot(F_d_ddot_array[i], label=f"{axs[ax]}")
        plt.title(r'$\ddot{F_d}:2^{nd} derivative generated force on X,Y and Z$: '+ bp_traj.material_name)
        plt.xlabel('Step')
        plt.ylabel(r'$F_d$: '+ bp_traj.material_name)
        plt.legend()
        plt.show()
        F_ddot_continuity = get_lipschitz_criterion(F_d_ddot)
        sys.exit(0)

        # get the continuity of X_tilde and X_tilde_dot
        X = np.array(bp.x_tilde_list).T
        X_dot = np.array(bp.x_tilde_dot_list).T
        x_tilde_continuity = get_lipschitz_criterion(X.T)
        print(f"x_tilde_continuity: {x_tilde_continuity}")
        x_tilde_dot_continuity = get_lipschitz_criterion(X_dot.T)
        print(f"x_tilde_dot_continuity: {x_tilde_dot_continuity}")
        x_dot_bound = get_norm_bound_threshold(X_dot.T)
        print(f"x_dot_bound: {x_dot_bound}")
        x_tilde_bound = get_norm_bound_threshold(X.T)
        print(f"x_tilde_bound: {x_tilde_bound}")
        for ax, i in enumerate(range(3)):
            plt.plot(X[i], label=f"{axs[ax]}")
        plt.title(r'$\tilde{x}: position$: '+ bp_traj.material_name)
        plt.xlabel('Step')
        plt.ylabel(r'$\tilde{x}$: '+ bp_traj.material_name)
        plt.legend()
        plt.show()

        for ax, i in enumerate(range(3)):
            plt.plot(X_dot[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{\tilde{x}}: velocity$: '+ bp_traj.material_name)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{\tilde{x}}$: '+ bp_traj.material_name)
        plt.legend()
        plt.show()
        # visualise dissipated energy
        plt.plot(bp.controller.E_tot)
        plt.xlabel('Step')
        plt.ylabel(r'Tank storage $T(x_t)$: '+ bp_traj.material_name)
        plt.show()

        # visualise velocity error ZYZ
        for ax, i in enumerate(range(3)):
            plt.plot(bp.x_tilde_dot_list[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{\tilde{x}}: velocity error$: '+ bp_traj.material_name)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{\tilde{x}}$: '+ bp_traj.material_name)
        plt.legend()
        plt.show()

        norm_bounds = []
        for x in bp.controller.Force_error:
            norm_bounds.append(x[-1])
        plt.plot(norm_bounds)
        plt.xlabel('Step')
        plt.ylabel(r'Force error norm: '+ bp_traj.material_name)
        plt.show()

        for i in range(4):
            rand_idx = np.random.randint(0, len(bp.controller.Force_error))
            plt.plot(bp.controller.Force_error[rand_idx], label=f"Step {rand_idx}")
        plt.plot(bp.controller.Force_error[-1], label=f"Step {len(bp.controller.Force_error)}")
        plt.xlabel('optimizer iterations per step')
        plt.ylabel(r'Force error: '+ bp_traj.material_name)
        plt.legend()
        plt.show()


        plt.plot(bp.controller.fun_value)
        plt.xlabel('Step')
        plt.ylabel(r'optimizer fun_value: '+ bp_traj.material_name)
        plt.show()
        convergence_delta = np.max(bp.controller.step_Force_errors)
        print(f"convergence_delta: {convergence_delta}")

    print(f"Done!")

# roubstness test

if krasovskii_analyasis:
    # Define the matrices M, Kd, and Dd
    M = np.diag(np.ones_like(kd_opt))
    Kd = np.diag(kd_opt) * 0
    Dd = 2 * dd_opt[0] * np.sqrt(Kd)

    # Compute the matrix F(x)
    I = np.eye(M.shape[0])
    F = np.block([
        [np.zeros_like(I), I - np.linalg.inv(M) @ Kd],
        [-np.linalg.inv(M) @ Kd + I, -2 * np.linalg.inv(M) @ Dd]
    ])
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(F)

    # Check if all eigenvalues are negative
    is_negative_definite = np.all(eigenvalues < 0)
    print(f"is_negative_definite: {is_negative_definite}")
    print(f"eigenvalues: {eigenvalues}")

if quadratic_lyapunov:
    x1 = np.array(bp.x_tilde_list)
    x2 = np.array(bp.x_tilde_dot_list)
    M = np.eye(x1.shape[1])
    n_samples = x1.shape[0]
    # x = np.concatenate((x1, x2), axis=1)
    F_ext = np.array(bp.F_ext_list)
    Kd = np.array(bp.kd_list)
    Dd = np.array(bp.dd_list)
    ql = QuadraticLyapunov(x1, x2, Kd, Dd, F_ext, M, n_samples,is_exponential=True)
    V, (dot_V, stability_condition) = ql()
    ql.visualize(V, dot_V)
    print("V(x):", V)
    print("dot_V(x):", dot_V)
    print("is_stable:", stability_condition)
