import numpy as np
import matplotlib.pyplot as plt
from data import bpTrajDataLoader
from approach_analysis import AXES

__all__ = ["PlotterCartesian"]


class PlotterCartesian:

    @staticmethod
    def plot(data: np.ndarray, material_name):
        for ax, i in enumerate(range(3)):
            plt.plot(data[i], label=f"{AXES[ax]}")
        plt.title(r'$\ddot{F_d}:2^{nd} derivative generated force on X,Y and Z$: '+ material_name)
        plt.xlabel('Step')
        plt.ylabel(r'$F_d$: '+ material_name)
        plt.legend()
        plt.show()
