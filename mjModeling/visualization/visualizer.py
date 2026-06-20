import time
import mujoco
import mujoco.viewer as mjViewer
from mjModeling.conf import VIS_SITE_FRAME, VIS_LABEL_NAME, VIS_JOINTS
from mjModeling import Robot


class ViewerControl:
    """Shared run/pause/restart flags toggled from the MuJoCo viewer keyboard.

    Keys (handled via launch_passive key_callback):
        SPACE  -> pause / resume
        R      -> re-run the experiment
        Q      -> quit (also: just close the window)
    The controller's long loops poll `paused`/`quit` so a run can be paused
    mid-cut and the simulation does not run forever.
    """
    def __init__(self):
        self.paused = False
        self.restart = False
        self.quit = False

    def key_callback(self, keycode):
        if keycode == 32:                       # SPACE
            self.paused = not self.paused
            print("[viewer] PAUSED" if self.paused else "[viewer] RESUMED")
        elif keycode in (ord('R'), ord('r')):   # R
            self.restart = True
            print("[viewer] RESTART requested")
        elif keycode in (ord('Q'), ord('q')):   # Q
            self.quit = True
            print("[viewer] QUIT requested")


class Visualize:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.viewer = None
        self.control = ViewerControl()

    def run(self, callback):
        # Simulate and display video.
        # Reset state and time.
        mujoco.mj_resetData(self.robot.model, self.robot.data)
        # expose control flags to the controller (polled inside long loops)
        self.robot.viewer_control = self.control
        # Launch the viewer with keyboard control
        self.viewer = mjViewer.launch_passive(
            self.robot.model, self.robot.data,
            key_callback=self.control.key_callback
            )
        with self.viewer as viewer:
            # --- Enable joint visualization *after* the viewer starts ---
            with viewer.lock():
                # Show virtual joints between componnets
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = VIS_JOINTS
                # Show the Site Label (displays the name "scalpel_tip" in
                # the 3D view)
                if VIS_LABEL_NAME:
                    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
                # Show the Site Frame (displays RGB axes at the TCP)
                if VIS_SITE_FRAME:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                # Ensure the group site belongs to is visible
                # (default is Group 0)
                # This bitmask enables groups 0, 1, and 2
                viewer.opt.sitegroup = 3
                # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            model, data = self.robot.model, self.robot.data
            print("[viewer] Controls:  SPACE = pause/resume,  R = re-run,  Q / close = quit")
            # Run the experiment ONCE, then idle-hold (do NOT loop the cut forever).
            while viewer.is_running() and not self.control.quit:
                self.control.restart = False
                # Run the full experiment once (pause/quit are polled inside the
                # controller's long loops so a run can be paused mid-cut).
                self.robot.experiment_exeucte_wrapper(callback, viewer)
                print("[viewer] Run complete — holding. SPACE=pause, R=re-run, Q/close=quit.")
                # Idle-hold: keep the window responsive and the robot steady,
                # without re-running the experiment, until R / Q / window close.
                while viewer.is_running() and not self.control.restart and not self.control.quit:
                    step_start = time.time()
                    if not self.control.paused:
                        # gravity/Coriolis compensation so the arm holds steady
                        data.ctrl[:model.nu] = data.qfrc_bias[:model.nu]
                        mujoco.mj_step(model, data)
                    viewer.sync()
                    dt_sleep = model.opt.timestep - (time.time() - step_start)
                    if dt_sleep > 0:
                        time.sleep(dt_sleep)
                if self.control.restart:
                    mujoco.mj_resetData(model, data)   # clean state for re-run
        sim_time = self.robot.data.time          # capture before releasing robot
        self.robot.shutdown()
        self.robot = None
        del self.viewer
        print(f"Simulation loop terminated after running for {sim_time:.2f} seconds of simulation time.")
