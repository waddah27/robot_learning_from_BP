import sys
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
from shmemory import SharedMemoryBuffer

class RealTimeOscillator(QtWidgets.QWidget):
    def __init__(self, shm_name):
        super().__init__()
        self.setWindowTitle("Contact Force Monitor")
        self.layout = QtWidgets.QVBoxLayout(self)

        # Attach to shared memory buffer (reader side)
        self.buffer = SharedMemoryBuffer(name=shm_name, create=False)
        self.signal_names = self.buffer.get_signal_names()
        self.num_signals = len(self.signal_names)

        # Plot widget
        self.plot_widget = pg.PlotWidget(title="Scalpel Contact Forces")
        self.plot_widget.setLabel('left', 'Force', units='N')
        self.plot_widget.setLabel('bottom', 'Sample')
        self.plot_widget.addLegend()
        self.layout.addWidget(self.plot_widget)

        # Info panel for current values
        info_panel = QtWidgets.QWidget()
        info_layout = QtWidgets.QGridLayout(info_panel)
        self.value_labels = []
        self.layout.addWidget(info_panel)

        # Create curves
        self.curves = []
        colours = [(255,0,0), (0,255,0), (0,0,255)] if self.num_signals == 3 else None
        for i, name in enumerate(self.signal_names):
            pen = colours[i] if colours else pg.intColor(i)
            curve = self.plot_widget.plot(pen=pen, name=name)
            self.curves.append(curve)

            label = QtWidgets.QLabel(f"{name}: 0.000 N")
            label.setStyleSheet("font: bold 10pt;")
            info_layout.addWidget(label, i, 0)
            self.value_labels.append(label)

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(16)

    def update_plot(self):
        data = self.buffer.read_latest()   # now returns chronological order
        for i, curve in enumerate(self.curves):
            curve.setData(data[:, i])
        if data.shape[0] > 0:
            latest = data[-1, :]
            for i, label in enumerate(self.value_labels):
                label.setText(f"{self.signal_names[i]}: {latest[i]:.3f} N")
                label.setText(f"{self.signal_names[i]}: {latest[i]:.3f} N")

    def closeEvent(self, event):
        self.buffer.close()
        event.accept()


def run_drawer(shm_name):
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimeOscillator(shm_name)
    window.show()
    sys.exit(app.exec())