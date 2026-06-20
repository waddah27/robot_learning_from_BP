"""
Real-time controller monitor (refactored).

Replaces the single-axis "everything in Newtons" oscilloscope with a professional
multi-panel monitor that groups signals by physical quantity, each with its own
correctly-labelled axis:

    Panel 1  Position (m)      desired (dashed) vs actual (solid), per axis X/Y/Z
    Panel 2  Force (N)         desired (dashed) vs actual (solid), per axis
    Panel 3  Stiffness (N/m)   commanded K per axis
    Panel 4  Tracking error (mm)  |desired - actual| position error per axis

Axes are colour-coded (X=red, Y=green, Z=blue) and share a common time axis.
Signal grouping is derived from the buffer's signal names, so it adapts if the
layout changes.
"""
import sys
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtWidgets, QtCore
from shmemory import SharedMemoryBuffer

# axis -> colour (R/G/B for X/Y/Z)
_AXIS_COLOUR = {"x": (200, 40, 40), "y": (40, 160, 60), "z": (40, 90, 210), "?": (120, 120, 120)}


def classify(name):
    """Map a signal name to (group, axis, is_desired)."""
    n = name.replace("(N)", "").strip()
    low = n.lower()
    axis = "x" if "x" in low else ("y" if "y" in low else ("z" if "z" in low else "?"))
    if n.startswith("K"):
        return "stiffness", axis, False
    if "F" in n:
        return "force", axis, n.startswith("Fd")
    return "position", axis, n.endswith("d")


class RealTimeOscillator(QtWidgets.QWidget):
    def __init__(self, shm_name):
        super().__init__()
        self.setWindowTitle("Learnt-Skill Controller Monitor")
        self.resize(1100, 900)

        self.buffer = SharedMemoryBuffer(name=shm_name, create=False)
        self.signal_names = self.buffer.get_signal_names()
        self.num_signals = len(self.signal_names)
        self.meta = [classify(n) for n in self.signal_names]

        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        root = QtWidgets.QVBoxLayout(self)

        # --- controls ---
        bar = QtWidgets.QHBoxLayout()
        self.pause_button = QtWidgets.QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        bar.addWidget(self.pause_button)
        self.shot_button = QtWidgets.QPushButton("Take Screenshot")
        self.shot_button.clicked.connect(self.capture_screenshot)
        bar.addWidget(self.shot_button)
        # per-axis visibility toggles (X/Y/Z) instead of 15 checkboxes
        self.axis_visible = {"x": True, "y": True, "z": True}
        for ax in ("x", "y", "z"):
            cb = QtWidgets.QCheckBox(f"{ax.upper()} axis")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda s, a=ax: self._toggle_axis(a, s))
            c = _AXIS_COLOUR[ax]
            cb.setStyleSheet(f"color: rgb{c}; font: bold 10pt;")
            bar.addWidget(cb)
        bar.addStretch(1)
        root.addLayout(bar)

        # --- 4 stacked, x-linked plots ---
        self.glw = pg.GraphicsLayoutWidget()
        root.addWidget(self.glw)
        specs = [("position", "Position", "m"),
                 ("force", "Contact force", "N"),
                 ("stiffness", "Stiffness", "N/m"),
                 ("error", "Position tracking error", "mm")]
        self.plots, self.curves, self.err_curves = {}, {}, {}
        prev = None
        for r, (key, title, unit) in enumerate(specs):
            p = self.glw.addPlot(row=r, col=0, title=title)
            p.setLabel("left", title, units=unit)
            p.showGrid(x=True, y=True, alpha=0.25)
            p.addLegend(offset=(10, 5), labelTextSize="8pt")
            if prev is not None:
                p.setXLink(prev)
            prev = p
            self.plots[key] = p
        self.plots["error"].setLabel("bottom", "Sample")

        # build curves for the three native groups
        for i, name in enumerate(self.signal_names):
            group, axis, desired = self.meta[i]
            p = self.plots[group]
            col = _AXIS_COLOUR[axis]
            pen = pg.mkPen(color=col, width=1.6,
                           style=QtCore.Qt.PenStyle.DashLine if desired else QtCore.Qt.PenStyle.SolidLine)
            self.curves[i] = p.plot(pen=pen, name=name)

        # error curves: one per axis (computed = desired - actual position)
        self._pos_idx = self._pair_position_indices()
        for axis, (id_des, id_act) in self._pos_idx.items():
            pen = pg.mkPen(color=_AXIS_COLOUR[axis], width=1.8)
            self.err_curves[axis] = self.plots["error"].plot(pen=pen, name=f"|e_{axis}|")

        for p in self.plots.values():
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)

        # --- hover readout: crosshair + value label on each panel ---
        self._units = {"position": "mm", "force": "N", "stiffness": "kN/m", "error": "mm"}
        self._vlines, self._readouts, self._proxies = {}, {}, {}
        for key, p in self.plots.items():
            vline = pg.InfiniteLine(angle=90, movable=False,
                                    pen=pg.mkPen((100, 100, 100), width=0.8,
                                                 style=QtCore.Qt.PenStyle.DashLine))
            p.addItem(vline, ignoreBounds=True)
            self._vlines[key] = vline
            txt = pg.TextItem(anchor=(0, 1), color=(20, 20, 20),
                              fill=pg.mkBrush(255, 255, 255, 200))
            p.addItem(txt, ignoreBounds=True)
            self._readouts[key] = txt
            self._proxies[key] = pg.SignalProxy(
                p.scene().sigMouseMoved, rateLimit=60,
                slot=lambda ev, k=key: self._on_mouse_move(ev, k))

        # display limits so the GUI never chokes regardless of buffer fill
        self._max_read = 8000    # most-recent samples read per frame
        self._max_draw = 4000    # points actually rendered per curve (decimated)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(33)          # ~30 fps
        self.paused = False

    def _pair_position_indices(self):
        """Find (desired_idx, actual_idx) of position signals per axis."""
        out = {}
        for axis in ("x", "y", "z"):
            des = act = None
            for i, (g, a, d) in enumerate(self.meta):
                if g == "position" and a == axis:
                    if d:
                        des = i
                    else:
                        act = i
            if des is not None and act is not None:
                out[axis] = (des, act)
        return out

    def _toggle_axis(self, axis, state):
        visible = (state == QtCore.Qt.CheckState.Checked.value) or bool(state)
        self.axis_visible[axis] = visible
        for i, (g, a, d) in enumerate(self.meta):
            if a == axis:
                self.curves[i].setVisible(visible)
        if axis in self.err_curves:
            self.err_curves[axis].setVisible(visible)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("Resume" if self.paused else "Pause")

    def capture_screenshot(self):
        ts = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        fn = f"monitor_{ts}.png"
        self.glw.grab().save(fn)
        print(f"Screenshot saved as {fn}")

    def update_plot(self):
        if self.paused:
            return
        # Guard the whole update: a single bad frame must NEVER kill the QTimer
        # (an unhandled exception in this slot would stop all future plotting).
        try:
            wcount = self.buffer.get_write_index()        # total samples written
            if wcount <= 0:
                return
            # Cap how many recent samples we READ each frame so the GUI can never
            # choke (e.g. during the long approach phase) — and free of zero-pad.
            n_read = min(wcount, self.buffer.buffer_size, self._max_read)
            data = self.buffer.read_latest(n_points=n_read)
            if data is None or len(data) == 0:
                return
            x = np.arange(wcount - len(data), wcount)     # real sample numbers
            # Decimate for rendering speed (keeps ~_max_draw points on screen).
            if len(x) > self._max_draw:
                s = len(x) // self._max_draw + 1
                data = data[::s]
                x = x[::s]
            for i, curve in self.curves.items():
                curve.setData(x, data[:, i])
            for axis, (id_des, id_act) in self._pos_idx.items():
                e = np.abs(data[:, id_des] - data[:, id_act]) * 1e3
                self.err_curves[axis].setData(x, e)
        except Exception as ex:                            # keep the timer alive
            print(f"[monitor] update skipped this frame: {ex}")

    def _on_mouse_move(self, event, key):
        """Show sample index + value at the cursor on the hovered panel."""
        pos = event[0]
        p = self.plots[key]
        if not p.sceneBoundingRect().contains(pos):
            return
        mp = p.getViewBox().mapSceneToView(pos)
        x, y = mp.x(), mp.y()
        self._vlines[key].setPos(x)
        unit = self._units.get(key, "")
        self._readouts[key].setText(f"sample={x:.0f}\n{y:.2f} {unit}")
        # place the label near the cursor, just inside the view
        self._readouts[key].setPos(x, y)

    def closeEvent(self, event):
        self.buffer.close()
        event.accept()


def run_drawer(shm_name):
    app = QtWidgets.QApplication(sys.argv)
    window = RealTimeOscillator(shm_name)
    window.show()
    sys.exit(app.exec())
