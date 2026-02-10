import matplotlib.pyplot as plt
from data import bpTrajDataLoader

__all__ = ["PlotterTraj3D"]

class PlotterTraj3D:

    @staticmethod
    def plot(bp_traj: bpTrajDataLoader):
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(bp_traj.pos[:,2], bp_traj.pos[:,1], bp_traj.pos[:,0], color='k')
        ax.set_title(label=f"Desired trajextory expressed in robot tcp frame - {bp_traj.material_name} material")
        ax.set_xlabel(r'Z_{tcp} [m]')
        ax.set_ylabel(r'Y_{tcp}[m]')
        ax.set_zlabel(r'X_{tcp}[m]')
        plt.tight_layout()
        ax.view_init(elev=65, azim=-30)
        plt.show()