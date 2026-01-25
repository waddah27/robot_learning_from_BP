from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from previous_codes.vic_controller_with_tank_energy import VICController
from matplotlib.animation import FuncAnimation
from time import time

data_gmr = np.load('./predicted_pose_twist_wrench_peno.npy')
print(data_gmr.shape)
x_idx, y_idx, z_idx = 1,2,3
x_pred_idx, y_pred_idx, z_pred_idx = 0,1,2
Fx_idx, Fy_idx, Fz_idx = 4,5,6
# Fx_pred_idx, Fy_pred_idx, Fz_pred_idx = 3,4,5
Fx_idx, Fy_idx, Fz_idx =7,8, 9
Tx_idx,Ty_idx,Tz_idx =10,11, 12
vx_idx, vy_idx, vz_idx = 13,14,15
Fx_pred_idx, Fy_pred_idx, Fz_pred_idx =6,7, 8
Tx_pred_idx,Ty_pred_idx,Tz_pred_idx =9,10, 11
vx_pred_idx, vy_pred_idx, vz_pred_idx = 12,13,14

pos = data_gmr[:, x_pred_idx:z_pred_idx+1]
vel = data_gmr[:,vx_pred_idx:vz_pred_idx+1]
force = data_gmr[:,Fx_pred_idx:Fz_pred_idx+1]


class MotionPlanner:
    def __init__(self, pos:ndarray,twist:ndarray, wrench:ndarray) -> None:
        """
        initializing position, twist and wrench learned from gmr model
        """
        self.P = pos
        self.V = twist
        self.W = wrench

    def go_to_next(self):
        """ Generator that yields the position, velocity and force at each step. """
        for x, v, f in zip(self.P,self.V, self.W):
            # Here we simulate a cutting operation with moving along x with velocity v and adjusting force (f)
            # current_position = [x, self.start_pt[1], self.pt_height - self.pt_cut_depth]
            yield x, v, f

class Visualizer:
    def __init__(self, x,y) -> None:
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'bo-', markersize=5)  # blue circles with solid line
        self.ax.set_xlim(min(x), max(x))
        self.ax.set_ylim(min(y) - 5, max(y) + 5)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Force')
        self.xdata:Any = []
        self.ydata:Any = []

    def init_line(self):
        self.line.set_data([], [])
        return self.line,

    # def update(self, frame):
    #     x, y, z = frame
    #     self.xdata.append(x)
    #     self.ydata.append(z)
    #     self.line.set_data(self.xdata, self.ydata)
    #     return self.line,

    def plot(self, x,y, time):
        # ani = FuncAnimation(self.fig, self.update, frames=frames, init_func=self.init_line, blit=True)
            # plt.show()
        self.xdata.append(x)
        self.ydata.append(y)
        self.line.set_data(self.xdata, self.ydata)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line)
        plt.pause(time)




if __name__=="__main__":
    controller = VICController()
    planner = MotionPlanner(pos=pos, twist=vel, wrench=force)
    steps = planner.go_to_next()
    visualizer = Visualizer(x=planner.P[-1], y=planner.W[-1])
    # Set up the plot
    # visualizer.plot(frames=steps)

    try:
        while True:
            start_time = time()
            x_tilde, x_tilde_dot, F_d = next(steps)
            kd_opt, dd_opt = controller.optimize(x_tilde, x_tilde_dot, F_d)
            F_actual = controller.calculate_force(x_tilde, x_tilde_dot)
            print(f"\t k_d = {kd_opt}, dd = {dd_opt}, F_act = {F_actual}")
            print("Next Position:", x_tilde, "Next Force:", F_d)
            now = time() - start_time
            visualizer.plot(x_tilde[0], x_tilde[-1], time=now)
            plt.show()

    except StopIteration:
        print("Completed all steps.")




