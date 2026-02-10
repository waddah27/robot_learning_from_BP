import numpy as np
import matplotlib.pyplot as plt
from bp_basic_experiment import BasicBPexperiment as bp
from data import bpTrajDataLoader, NamedArray
from data_analysis_utils import get_lipschitz_criterion, get_norm_bound_threshold, get_smoothness_threshold

__all__ = ["ResAnalyser", "AXES"]

AXES = {0: 'X', 1: 'Y', 2: 'Z'} # Axis labels


class ResAnalyser:

    @classmethod
    def get_smoothness(cls, bp_traj: bpTrajDataLoader):
        F_d_sm = get_smoothness_threshold(bp_traj.force)
        print(f"F_d_sm: {F_d_sm}")
        return F_d_sm

    @classmethod
    def get_boundness(cls, data: np.ndarray):
        data_bound = get_norm_bound_threshold(data)
        return data_bound

    @classmethod
    def get_continuity(cls, material_name:str, seq: NamedArray):
        # visualize time per step
        plt.plot(bp.convergence_time_per_step)
        plt.xlabel('Step')
        plt.ylabel('Time [s]')
        plt.show()


        # get data continuity threshold (critertion value)
        data_continuity = get_lipschitz_criterion(seq.data)
        print(f"F_d_continuity: {data_continuity}")
        # visualize F_d continuity
        data_d = np.array(seq.data).T
        for ax, i in enumerate(range(3)):
            plt.plot(data_d[i], label=f"{AXES[ax]}")
        plt.title(f"{seq.name}: generated on X,Y and Z axis - {material_name} material")
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ material_name)
        plt.show()
        return data_continuity

    @classmethod
    def grad(cls, material_name: str, data: np.ndarray):
        data_dot = np.diff(data, axis=0)
        data_dot_array = np.array(data_dot).T
        for ax, i in enumerate(range(3)):
            plt.plot(data_dot_array[i], label=f"{AXES[ax]}")
        plt.title(r'$\dot{F_d} :1^{st} derivative generated force on X,Y and Z$: '+ material_name)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ material_name)
        plt.legend()
        plt.show()
        return data_dot