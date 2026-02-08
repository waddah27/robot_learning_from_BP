import numpy as np
import matplotlib.pyplot as plt
from bp_basic_experiment import BasicBPexperiment as bp
from data import bpTrajDataLoader
from data_analysis_utils import get_lipschitz_criterion, get_norm_bound_threshold, get_smoothness_threshold

__all__ = ["ResAnalyser"]


class ResAnalyser:
    def __init__(self):
        pass


    @classmethod
    def get_smoothness(cls, bp_traj: bpTrajDataLoader):
        F_d_sm = get_smoothness_threshold(bp_traj.force)
        print(f"F_d_sm: {F_d_sm}")
        return F_d_sm

    @classmethod
    def run(cls, bp_traj: bpTrajDataLoader):
        # visualize time per step
        plt.plot(bp.convergence_time_per_step)
        plt.xlabel('Step')
        plt.ylabel('Time [s]')
        plt.show()

        axs = {0: 'X', 1: 'Y', 2: 'Z'} # Axis labels
        # get F_d norm bound threshold
        F_d_bound = get_norm_bound_threshold(bp_traj.force)
        print(f"F_d_bound: {F_d_bound}")

        # get F_d continuity threshold (critertion value)
        F_d_continuity = get_lipschitz_criterion(bp_traj.force)
        print(f"F_d_continuity: {F_d_continuity}")
        # visualize F_d continuity
        F_d = np.array(bp_traj.force).T
        for ax, i in enumerate(range(3)):
            plt.plot(F_d[i], label=f"{axs[ax]}")
        plt.title(f"F_d: generated force on X,Y and Z axis - {bp_traj.material_name} material")
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ bp_traj.material_name)
        plt.show()
        return F_d_continuity