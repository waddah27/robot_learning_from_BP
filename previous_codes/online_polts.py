import numpy as np
import matplotlib.pyplot as plt

# plt.axis([0, 10, 0, 1])
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)
y1_l, y2_l, y3_l, t = [], [], [], []
for i in range(10):
    t.append(i)
    y1 = np.random.random()
    y2 = np.random.random()
    y3 = np.random.random()
    y1_l.append(y1)
    y2_l.append(y2)
    y3_l.append(y3)
    ax1.plot(t,y1_l)
    ax2.plot(y2_l)
    ax3.plot(y3_l)
    plt.pause(0.05)
    plt.draw()

plt.show()