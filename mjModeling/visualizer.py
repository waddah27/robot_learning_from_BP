from startup import *
from base import Robot
class Visualize:
    def __init__(self, robot: Robot):
        self.model = robot.model
        self.data = robot.data

    def run(self):
        # Simulate and display video.
        mujoco.mj_resetData(self.model, self.data)  # Reset state and time.
            
        # Launch the viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # --- Enable joint visualization *after* the viewer starts ---
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
                # You can add other flags here too, e.g.:
                # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            # Loop as long as the user has not closed the viewer window.
            while viewer.is_running():
                step_start = time.time()
                # Step the simulation forward
                mujoco.mj_step(self.model, self.data)  
                # Sync the viewer display with the current data state and options
                viewer.sync()
                # Rudimentary time keeping to run near real-time
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        print(f"Simulation loop terminated after running for {self.data.time:.2f} seconds of simulation time.") 