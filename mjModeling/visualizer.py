from mjModeling import *
from base import Robot
class Visualize:
    def __init__(self, robot: Robot):
        self.robot = robot

    def run(self):
        # Simulate and display video.
        mujoco.mj_resetData(self.robot.model, self.robot.data)  # Reset state and time.
            
        # Launch the viewer
        with mujoco.viewer.launch_passive(self.robot.model, self.robot.data) as viewer:
            # --- Enable joint visualization *after* the viewer starts ---
            with viewer.lock():
                # Show virtual joints between componnets 
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = VIS_VIRTUAL_JOINTS
                # Show the Site Label (displays the name "scalpel_tip" in the 3D view)
                if VIS_LABEL_NAME:
                    viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
            
                # Show the Site Frame (displays RGB axes at the TCP)
                if VIS_SITE_FRAME:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            
            # 3. Ensure the group site belongs to is visible (default is Group 0)
            # This bitmask enables groups 0, 1, and 2
                viewer.opt.sitegroup = 3
                # You can add other flags here too, e.g.:
                # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            # Loop as long as the user has not closed the viewer window.
            while viewer.is_running():
                step_start = time.time()
                
                self.robot.run_cutting_experiment()
                # Step the simulation forward
                mujoco.mj_step(self.robot.model, self.robot.data)  
                # Sync the viewer display with the current data state and options
                viewer.sync()
                # Rudimentary time keeping to run near real-time
                time_until_next_step = self.robot.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        print(f"Simulation loop terminated after running for {self.robot.data.time:.2f} seconds of simulation time.") 