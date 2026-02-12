import numpy as np
from time import time
from data import bpTrajDataLoader
from plotters import PlotterOfflineSameRange
from reference_generators import SimpleGenerator
from vic_controller_with_tank_energy_inside import VICController

__all__ = ["BasicBPexperiment"]


class BasicBPexperiment:

    x_tilde_list: list = []
    x_tilde_dot_list: list = []
    F_ext_list: list = []
    kd_list: list = []
    dd_list: list = []
    convergence_time_per_step: list = []
    use_k_min: bool = False
    use_k_max: bool = False
    use_specified_k: bool = False
    controller: VICController
    planner: SimpleGenerator
    plotter: PlotterOfflineSameRange
    k_specified: list = [500,500,500] # To test using constant stiffness case


    def __init__(self, planner: SimpleGenerator, plotter: PlotterOfflineSameRange):
        self.plotter = plotter
        self.planner = planner

    @classmethod
    def run(cls, bp_traj: bpTrajDataLoader):
        for x, x_dot, F_d in cls.planner.go_to_next():
            x_tilde = np.maximum(np.abs(bp_traj.pos[-1,:] - x), np.array([0.013, 0.013, 0.013])) # accepted error to avoid division by zero
            x_tilde_dot = np.abs(bp_traj.vel[-1,:] - x_dot)
            cls.x_tilde_list.append(x_tilde)
            cls.x_tilde_dot_list.append(x_tilde_dot)
            start_time = time()
            if not cls.use_k_min and not cls.use_k_max and not cls.use_specified_k:
                kd_opt, dd_opt = cls.controller.optimize(x_tilde, x_tilde_dot, F_d)
                cls.kd_list.append(np.diag(kd_opt))
                cls.dd_list.append(2 * np.diag([0.7, 0.7, 0.7]) * np.sqrt(np.diag(kd_opt)))
            else:
                if cls.use_k_min:
                    kd_opt = cls.controller.k_min
                    dd_opt = np.array([0.7, 0.7, 0.7])
                elif cls.use_k_max:
                    kd_opt = cls.controller.k_max
                    dd_opt = np.array([0.7, 0.7, 0.7])
                elif cls.use_specified_k:
                    kd_opt = cls.k_specified
                    dd_opt = np.array([0.7, 0.7, 0.7])
                cls.controller.K_d = kd_opt
                dd_opt = dd_opt

            F_actual = cls.controller.calculate_force(x_tilde, x_tilde_dot)
            cls.F_ext_list.append(F_actual)
            elapsed_time = time() - start_time
            cls.convergence_time_per_step.append(elapsed_time)
            print(f"Elapsed time for this step: {elapsed_time}s")
            print(f"\t k_d = {kd_opt}, dd = {dd_opt}, F_act = {F_actual}")
            print("Next Position:", x_tilde, "Next Force:", F_d)
            # print("Total energy: "+str(controller.E_tot))
            cls.plotter.collect_data(x=x_tilde, Fd=F_d, Fext=F_actual, k=kd_opt, d=dd_opt)
        cls.plotter.show()
        print("Completed all steps.")
        return kd_opt, dd_opt

