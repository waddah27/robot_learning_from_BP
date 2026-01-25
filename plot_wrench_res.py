import matplotlib.pyplot as plt
import numpy as np

rea_ar=np.fromfile("Force_peno_from_human_demo_4.txt", dtype=float, sep="\n ").reshape((-1, 1, 12))
pos_ar=np.fromfile("Pos_peno_from_human_demo_4.txt", dtype=float, sep="\n ").reshape((-1, 1, 6))
print(rea_ar.shape)
print(pos_ar.shape)
F_axes = ['Fx', 'Fy', 'Fz']
T_axes = ['Tx', 'Ty', 'Tz']
K_axes = ['Kdx', 'Kdy', 'Kdz']
D_axes = ['Ddx', 'Ddy', 'Ddz']
Pos_axes = ['X', 'Y', 'X']
orient_axes = ['A', 'B', 'C']
F_ext = rea_ar[:,0, 0:3]
T_ext = rea_ar[:,0, 3:6]
Kd = rea_ar[:,0,6:9]
Dd = rea_ar[:,0,9:12]
Err = rea_ar[:,0,-1]
Pos = pos_ar[:,0,0:3]
orient = pos_ar[:, 0, 3:6]

ax1 = plt.subplot(511)
for i, c in zip(range(F_ext.shape[1]), F_axes):
    plt.plot(F_ext[:,i], label=c)
ax1.legend()
ax2 = plt.subplot(512)
for i, c in zip(range(T_ext.shape[1]), T_axes):
    plt.plot(T_ext[:,i], label=c)
ax2.legend()

ax3 = plt.subplot(513)
for i, c in zip(range(Kd.shape[1]), K_axes):
    plt.plot(Kd[:,i], label=c)
ax3.legend()

ax4 = plt.subplot(514)
for i, c in zip(range(Dd.shape[1]), D_axes):
    plt.plot(Dd[:,i], label=c)
ax4.legend()

# ax5 = plt.subplot(515)
# plt.plot(Err, label='$||F_{d} - F_{ext}||$')
# ax5.legend()


plt.show()

ax1 = plt.subplot(211)
plt.plot(Pos[:10,0],Pos[:10,1])
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend()

ax2 = plt.subplot(212)
for i, c in zip(range(orient.shape[1]), orient_axes):
    plt.plot(orient[:,i], label=c)
ax2.legend()
plt.show()
