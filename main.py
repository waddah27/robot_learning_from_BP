from enum import Enum
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin
from data_analysis_utils import get_lipschitz_criterion, get_norm_bound_threshold, get_smoothness_threshold
from transformation_utils import transform_coordinates
from vic_controller_with_tank_energy_inside import VICController
from sklearn.metrics import root_mean_squared_error as rmse
from time import time
from exp_stability_analysis import QuadraticLyapunov
from visualization_utils import VisualizerOnline, VisualizerOffline, VisualizerOfflineSameRanges
EXP_DATA_PATH = './experimental_data'
# Load the data
class MaterialMeta(Enum):
    PVC = 'pvc'
    penoplex = 'peno'
    cork =  'crck'

class MaterialData:
    PVC = np.load(f'./predicted_pose_twist_wrench_{MaterialMeta.PVC.value}.npy')
    penoplex = np.load(f'./predicted_pose_twist_wrench_{MaterialMeta.penoplex.value}.npy')
    cork = np.load(f'./predicted_pose_twist_wrench_{MaterialMeta.cork.value}.npy')


data_gmr = MaterialData.cork
MATERIAL_NAME = MaterialMeta.cork.name
title = f"Visualization of Dynamics during cutting {MATERIAL_NAME}"

# Optimizer configs
use_k_min = False
use_k_max = False
use_specified_k = False
k_specified = [500,500,500] # To test using constant stiffness case
print(data_gmr.shape)

# Define indexes for clarity
pos_idx = np.s_[0:3]
vel_idx = np.s_[12:15]
force_idx = np.s_[6:9]
pos = data_gmr[:, pos_idx]
pos = pos[:,[1,0,2]] # Transform ZXY to ZYX
# vel = data_gmr[:, vel_idx]
vel = np.diff(pos, axis=0)
force = data_gmr[:, force_idx]

F_max = np.max(force, axis=0)
F_min = np.min(force, axis=0)
analyse_results = False
class MotionPlanner:
    def __init__(self, pos: np.ndarray, twist: np.ndarray, wrench: np.ndarray) -> None:
        self.P = pos
        self.V = twist
        self.W = wrench

    def go_to_next(self):
        """ Generator that yields position, velocity, and force at each step. """
        for x, v, f in zip(self.P, self.V, self.W):
            yield x, v, f


if __name__ == "__main__":
    # tcp_rot = transform_coordinates(th_x_deg=90, th_y_deg=90)
    x_tilde_list = []
    x_tilde_dot_list = []
    F_ext_list = []
    kd_list = []
    dd_list = []
    convergence_time_per_step = []
    plot_desired_pos = False
    if plot_desired_pos:
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(pos[:,2],pos[:,1], pos[:,0], color='k')
        ax.set_title(label=f"Desired trajextory expressed in robot tcp frame - {MATERIAL_NAME} material")
        ax.set_xlabel(r'Z_{tcp} [m]')
        ax.set_ylabel(r'Y_{tcp}[m]')
        ax.set_zlabel(r'X_{tcp}[m]')
        plt.tight_layout()
        ax.view_init(elev=65, azim=-30)
        plt.show()
    controller = VICController(F_min=F_min, F_max=F_max)
    planner = MotionPlanner(pos=pos, twist=vel, wrench=force)
    # visualizer = Visualizer(xlim=(min(pos[:,-1]), max(pos[:,-1])), ylim=(0,1))

    visualizer = VisualizerOfflineSameRanges(title=title)

    for x, x_dot, F_d in planner.go_to_next():
        x_tilde = np.maximum(np.abs(pos[-1,:] - x), np.array([0.013, 0.013, 0.013])) # accepted error to avoid division by zero
        x_tilde_dot = np.abs(vel[-1,:] - x_dot)
        x_tilde_list.append(x_tilde)
        x_tilde_dot_list.append(x_tilde_dot)
        start_time = time()
        if not use_k_min and not use_k_max and not use_specified_k:
            kd_opt, dd_opt = controller.optimize(x_tilde, x_tilde_dot, F_d)
            kd_list.append(np.diag(kd_opt))
            dd_list.append(2 * np.diag([0.7, 0.7, 0.7]) * np.sqrt(np.diag(kd_opt)))
        else:
            if use_k_min:
                kd_opt = controller.k_min
                dd_opt = np.array([0.7, 0.7, 0.7])
            elif use_k_max:
                kd_opt = controller.k_max
                dd_opt = np.array([0.7, 0.7, 0.7])
            elif use_specified_k:
                kd_opt = k_specified
                dd_opt = np.array([0.7, 0.7, 0.7])
            controller.K_d = kd_opt
            dd_opt = dd_opt


        F_actual = controller.calculate_force(x_tilde, x_tilde_dot)
        F_ext_list.append(F_actual)
        elapsed_time = time() - start_time
        convergence_time_per_step.append(elapsed_time)
        print(f"Elapsed time for this step: {elapsed_time}s")
        print(f"\t k_d = {kd_opt}, dd = {dd_opt}, F_act = {F_actual}")
        print("Next Position:", x_tilde, "Next Force:", F_d)
        # print("Total energy: "+str(controller.E_tot))
        visualizer.collect_data(x=x_tilde,Fd=F_d, Fext=F_actual,k=kd_opt, d=dd_opt)
    visualizer.show()

    # x_tilde_dot_list = np.array(x_tilde_dot_list).T
    print("Completed all steps.")


    if analyse_results:
        # visualize time per step
        plt.plot(convergence_time_per_step)
        plt.xlabel('Step')
        plt.ylabel('Time [s]')
        plt.show()

        axs = {0: 'X', 1: 'Y', 2: 'Z'} # Axis labels
        # get F_d norm bound threshold
        F_d_bound = get_norm_bound_threshold(force)
        print(f"F_d_bound: {F_d_bound}")

        # get F_d continuity threshold (critertion value)
        F_d_continuity = get_lipschitz_criterion(force)
        print(f"F_d_continuity: {F_d_continuity}")
        # visualize F_d continuity
        F_d = np.array(force).T
        for ax, i in enumerate(range(3)):
            plt.plot(F_d[i], label=f"{axs[ax]}")
        plt.title(f"F_d: generated force on X,Y and Z axis - {MATERIAL_NAME} material")
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ MATERIAL_NAME)
        plt.show()

        # get F_d smoothness threshold: derivatives of F_d are bounded
        F_d_sm = get_smoothness_threshold(force)
        print(f"F_d_sm: {F_d_sm}")

        # get the continuity of F_d_dot
        F_d_dot = np.diff(force, axis=0)
        F_d_dot_array = np.array(F_d_dot).T
        print(f"F_d_dot: {F_d_dot}")
        for ax, i in enumerate(range(3)):
            plt.plot(F_d_dot_array[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{F_d} :1^{st} derivative generated force on X,Y and Z$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ MATERIAL_NAME)
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
        plt.title(r'$\ddot{F_d}:2^{nd} derivative generated force on X,Y and Z$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$F_d$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()
        F_ddot_continuity = get_lipschitz_criterion(F_d_ddot)

        # get the continuity of X_tilde and X_tilde_dot
        X = np.array(x_tilde_list).T
        X_dot = np.array(x_tilde_dot_list).T
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
        plt.title(r'$\tilde{x}: position$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\tilde{x}$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()

        for ax, i in enumerate(range(3)):
            plt.plot(X_dot[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{\tilde{x}}: velocity$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{\tilde{x}}$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()
        # visualise dissipated energy
        plt.plot(controller.E_tot)
        plt.xlabel('Step')
        plt.ylabel(r'Tank storage $T(x_t)$: '+ MATERIAL_NAME)
        plt.show()

        # visualise velocity error ZYZ
        for ax, i in enumerate(range(3)):
            plt.plot(x_tilde_dot_list[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{\tilde{x}}: velocity error$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{\tilde{x}}$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()

        norm_bounds = []
        for x in controller.Force_error:
            norm_bounds.append(x[-1])
        plt.plot(norm_bounds)
        plt.xlabel('Step')
        plt.ylabel(r'Force error norm: '+ MATERIAL_NAME)
        plt.show()

        for i in range(4):
            rand_idx = np.random.randint(0, len(controller.Force_error))
            plt.plot(controller.Force_error[rand_idx], label=f"Step {rand_idx}")
        plt.plot(controller.Force_error[-1], label=f"Step {len(controller.Force_error)}")
        plt.xlabel('optimizer iterations per step')
        plt.ylabel(r'Force error: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()


        plt.plot(controller.fun_value)
        plt.xlabel('Step')
        plt.ylabel(r'optimizer fun_value: '+ MATERIAL_NAME)
        plt.show()
        convergence_delta = np.max(controller.step_Force_errors)
        print(f"convergence_delta: {convergence_delta}")

    print(f"Done!")

# roubstness test
krasovskii_analyasis = False
quadratic_lyapunov = True
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
    x1 = np.array(x_tilde_list)
    x2 = np.array(x_tilde_dot_list)
    M = np.eye(x1.shape[1])
    n_samples = x1.shape[0]
    # x = np.concatenate((x1, x2), axis=1)
    F_ext = np.array(F_ext_list)
    Kd = np.array(kd_list)
    Dd = np.array(dd_list)
    ql = QuadraticLyapunov(x1, x2, Kd, Dd, F_ext, M, n_samples,is_exponential=True)
    V, (dot_V, stability_condition) = ql()
    ql.visualize(V, dot_V)
    print("V(x):", V)
    print("dot_V(x):", dot_V)
    print("is_stable:", stability_condition)
