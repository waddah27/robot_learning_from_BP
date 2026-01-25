"""
This script plots the experimental data for the penoplex robot with and without
GMR.
"""
import os
import matplotlib.pyplot as plt
import numpy as np


PATH_TO_EXP_DATA = './experimental_data/'
class ExpDataPaths:
    """
    This class contains the paths to the experimental data files.
    """
    cork_robot_only = os.path.join(PATH_TO_EXP_DATA,"Fd_robot_cork_good_res_1.txt")
    cork_robot_gmr = os.path.join(PATH_TO_EXP_DATA, 'Fd_gmr_cork_good_res_1.txt')
    penoplex_robot_noly = os.path.join(PATH_TO_EXP_DATA, 'Fd_robot_peno_good_res_1.txt')
    penoplex_robot_gmr = os.path.join(PATH_TO_EXP_DATA, 'Fd_gmr_peno_good_res_1.txt')


gmr_data_pth = ExpDataPaths.penoplex_robot_gmr
robot_data_pth = ExpDataPaths.penoplex_robot_noly
robot_ar=np.fromfile(robot_data_pth, dtype=float, sep="\n ").reshape((-1, 1, 12))
gmr_ar=np.fromfile(gmr_data_pth, dtype=float, sep="\n ").reshape((-1, 1, 12))
# pos_ar=np.fromfile("pose_log.txt", dtype=float, sep="\n ").reshape((-1, 1, 6))
print(robot_ar.shape)
# print(pos_ar.shape)
F_axes = ['Fx', 'Fy', 'Fz']
K_axes = ['Kdx', 'Kdy', 'Kdz']
D_axes = ['Ddx', 'Ddy', 'Ddz']

F_ext_robot = robot_ar[:,0, 0:3]
Kd_robot = robot_ar[:,0,6:9]
Dd_robot = robot_ar[:,0,9:12]

F_ext_gmr = gmr_ar[:,0, 0:3]
Kd_gmr = gmr_ar[:,0,6:9]
Dd_gmr = gmr_ar[:,0,9:12]

fig, axs = plt.subplots(2,2,figsize=(12,10))
for i, c in zip(range(F_ext_robot.shape[1]), F_axes):
    axs[0,0].plot(F_ext_robot[:,i], label=c)
axs[0,0].set_title('robot only')
axs[0,0].legend()

for i, c in zip(range(Kd_robot.shape[1]), K_axes):
    axs[0,1].plot(Kd_robot[:,i], label=c)
axs[0,1].legend()

for i, c in zip(range(F_ext_gmr.shape[1]), F_axes):
    axs[1,0].plot(F_ext_gmr[:,i], label=c)
axs[1,0].legend()

for i, c in zip(range(Kd_gmr.shape[1]), K_axes):
    axs[1,1].plot(Kd_gmr[:,i], label=c)
axs[1,1].legend()
plt.show()

