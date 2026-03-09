import sys
import time
import numpy as np
import pyqtgraph as pg
from multiprocessing import shared_memory, Process
from PyQt6 import QtWidgets, QtCore

class RealTimeOscillator(QtWidgets.QWidget):
    def __init__(self, shm_name, signal_names=None, num_signals=3, buffer_size=1000):
        super().__init__()
        self.setWindowTitle("Contact Force Monitor")
        self.layout = QtWidgets.QVBoxLayout(self)

        # Plot widget
        self.plot_widget = pg.PlotWidget(title="Scalpel Contact Forces (Robot – Workpiece)")
        self.plot_widget.setLabel('left', 'Force', units='N')
        self.plot_widget.setLabel('bottom', 'Sample')
        self.plot_widget.addLegend()
        self.layout.addWidget(self.plot_widget)

        # Info panel for current values
        info_panel = QtWidgets.QWidget()
        info_layout = QtWidgets.QGridLayout(info_panel)
        self.value_labels = []
        self.layout.addWidget(info_panel)

        # Shared memory connection
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.data_buffer = np.frombuffer(self.shm.buf, dtype=np.float64).reshape((buffer_size, num_signals))
        self.buffer_size = buffer_size
        self.num_signals = num_signals

        # Signal names
        if signal_names is None:
            signal_names = [f"F{i} (N)" for i in range(num_signals)]
        elif len(signal_names) != num_signals:
            raise ValueError("Number of signal names must match number of signals")
        self.signal_names = signal_names

        # Create curves (fixed colours for X, Y, Z)
        self.curves = []
        colours = [(255,0,0), (0,255,0), (0,0,255)] if num_signals == 3 else None
        for i, name in enumerate(signal_names):
            pen = colours[i] if colours else pg.intColor(i)
            curve = self.plot_widget.plot(pen=pen, name=name)
            self.curves.append(curve)

            # Add a label for current value
            label = QtWidgets.QLabel(f"{name}: 0.000 N")
            label.setStyleSheet("font: bold 10pt;")
            info_layout.addWidget(label, i, 0)
            self.value_labels.append(label)

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)  # ~60 Hz

    def update_plot(self):
        # Plot the whole buffer (you may later optimise with a rolling window)
        for i, curve in enumerate(self.curves):
            curve.setData(self.data_buffer[:, i])

        # Update current value labels (most recent sample)
        latest = self.data_buffer[-1, :]
        for i, label in enumerate(self.value_labels):
            label.setText(f"{self.signal_names[i]}: {latest[i]:.3f} N")

    def closeEvent(self, event):
        self.shm.close()
        event.accept()


def run_drawer(shm_name):
    """Function to start the Qt application in a separate process."""
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimeOscillator(
        shm_name,
        signal_names=["Fx (N)", "Fy (N)", "Fz (N)"],
        num_signals=3,
        buffer_size=1000
    )
    window.show()
    sys.exit(app.exec())   # PyQt6 uses exec() (not exec_())


if __name__ == "__main__":
    BUF_SIZE = 1000
    NUM_SIGS = 3

    # 1. Create Shared Memory (8 bytes * total elements)
    shm = shared_memory.SharedMemory(create=True, size=BUF_SIZE * NUM_SIGS * 8)
    shared_array = np.frombuffer(shm.buf, dtype=np.float64).reshape((BUF_SIZE, NUM_SIGS))
    shared_array[:] = 0  # Initialize

    # 2. Start Drawer Process
    drawer_proc = Process(target=run_drawer, args=(shm.name,))
    drawer_proc.start()

    # 3. YOUR SIMULATION LOOP (here simulated with sine waves)
    try:
        t = 0
        idx = 0
        while True:
            # Simulate 3 contact force signals (replace with real data)
            for s in range(NUM_SIGS):
                shared_array[idx, s] = np.sin(2 * np.pi * (s+1) * 0.1 * t)

            idx = (idx + 1) % BUF_SIZE
            t += 0.01
            time.sleep(0.005)  # Simulate 200Hz physics
    except KeyboardInterrupt:
        pass
    finally:
        drawer_proc.terminate()
        shm.close()
        shm.unlink()