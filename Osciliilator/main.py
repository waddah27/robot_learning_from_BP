import sys
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
from multiprocessing import shared_memory
import multiprocessing as mp
import time

from mjModeling.conf import oscillatorConfigs as oscConf
__all__ = ["RealTimeDrawer", "run_drawer"]


class RealTimeDrawer(QtWidgets.QWidget):
    def __init__(self, shm_name, num_signals=oscConf.N_SIGS.value, buffer_size=oscConf.BUFFER_SIZE.value):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(title="Real-Time Oscillator")
        self.layout.addWidget(self.plot_widget)
        
        # Connect to Shared Memory
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.data_buffer = np.frombuffer(self.shm.buf, dtype=np.float64).reshape((buffer_size, num_signals))
        
        # Setup Curves
        self.curves = [self.plot_widget.plot(pen=pg.intColor(i)) for i in range(num_signals)]
        
        # High-speed timer for UI updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16) # ~60 FPS

    def update_plot(self):
        # In a real setup, you'd read a shared 'write_ptr' to sync.
        # For simplicity, we just grab the whole buffer.
        for i, curve in enumerate(self.curves):
            curve.setData(self.data_buffer[:, i])

    def closeEvent(self, event):
        self.shm.close()


def run_drawer(shm_name):
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimeDrawer(shm_name)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    BUF_SIZE = 1000
    NUM_SIGS = 3
    
    # 1. Create Shared Memory (8 bytes * total elements)
    shm = shared_memory.SharedMemory(create=True, size=BUF_SIZE * NUM_SIGS * 8)
    shared_array = np.frombuffer(shm.buf, dtype=np.float64).reshape((BUF_SIZE, NUM_SIGS))
    shared_array[:] = 0 # Initialize

    # 2. Start Drawer Process
    drawer_proc = mp.Process(target=run_drawer, args=(shm.name,))
    drawer_proc.start()

    # 3. YOUR SIMULATION LOOP
    try:
        t = 0
        idx = 0
        while True:
            # Simulate 10 different signals
            for s in range(NUM_SIGS):
                shared_array[idx, s] = np.sin(2 * np.pi * (s+1) * 0.1 * t)
            
            idx = (idx + 1) % BUF_SIZE
            t += 0.01
            time.sleep(0.005) # Simulate 200Hz physics
    except KeyboardInterrupt:
        pass
    finally:
        drawer_proc.terminate()
        shm.close()
        shm.unlink()
