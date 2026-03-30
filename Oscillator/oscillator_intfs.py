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

        # --- Buttons ---
        button_layout = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)

        self.screenshot_button = QtWidgets.QPushButton("Take Screenshot")
        self.screenshot_button.clicked.connect(self.capture_screenshot)
        button_layout.addWidget(self.screenshot_button)
        self.layout.addLayout(button_layout)

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
        self.timer.start(16)          # ~60 fps

        self.paused = False            # pause flag

    def toggle_pause(self):
        """Pause or resume the live updates."""
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def capture_screenshot(self):
        """Save the current plot widget as a PNG image."""
        # Use a timestamp to avoid overwriting
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        filename = f"screenshot_{timestamp}.png"
        # Capture the plot widget
        pixmap = self.plot_widget.grab()
        pixmap.save(filename)
        print(f"Screenshot saved as {filename}")

    def update_plot(self):
        """Update plot and values from shared memory, unless paused."""
        if self.paused:
            return                     # freeze display

        data = self.buffer.read_latest()   # returns chronological order
        for i, curve in enumerate(self.curves):
            curve.setData(data[:, i])
        if data.shape[0] > 0:
            latest = data[-1, :]
            for i, label in enumerate(self.value_labels):
                # Updated label text
                label.setText(f"{self.signal_names[i]}: {latest[i]:.3f} N")

    def closeEvent(self, event):
        self.buffer.close()
        event.accept()


def run_drawer(shm_name):
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimeOscillator(shm_name)
    window.show()
    sys.exit(app.exec())