from mjModeling.mjRobot.base import Robot
from mjModeling import *
import numpy as np

class iiwa14(Robot):
    def __init__(self):
        self._model = None
        self._data = None
        self.state = {}
        self.reset_state()
        
    def reset_state(self):
        """Reset the state dictionary"""
        if not self.state.get(force_history):
            self.state[force_history] = []  # Store cutting forces
        else:
            self.state.get(force_history).clear()
    
    @property
    def model(self):
        return self._model
    
    def create(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found at {xml_path}")

        spec = mujoco.MjSpec.from_file(xml_path)

        # 1. Register Mesh 1
        mesh1 = spec.add_mesh(name="mesh1", file=scalpelHandler1_path)
        mesh1.refquat = [0.707, -0.707, 0, 0]
        mesh1.scale = [0.001, 0.001, 0.001] 

        # 2. Register Mesh 2
        mesh2 = spec.add_mesh(name="mesh2", file=scalpelHandler2_path)
        mesh2.refquat = [0.0, 1.0, 0, 0] # Match the orientation of part 1
        mesh2.scale = [0.001, 0.001, 0.001]
        # 3. Register the scalpel mesh
        mesh3 = spec.add_mesh(name="mesh3", file=scalpel_path)
        #  Rotate 60 degrees To mach scalpel being perpendicular to ee  
        mesh3.refquat = [0, 0.5, 0, 0.866] 
        mesh3.scale = [0.0001, 0.0001, 0.0001]
        # 4. Setup Body Hierarchy
        ee_body = spec.body("link7")
        attach_site = spec.site("attachment_site")
        
        handler = ee_body.add_body(name="3d_printed_handler")
        # Align handler origin to the attachment site
        handler.pos =  attach_site.pos
        handler.quat = attach_site.quat
        handler_depth = -0.09
        # 5. Add Part 1 (First half of handler)
        handler.add_geom(
            name="handler_part1_geom", 
            type=mujoco.mjtGeom.mjGEOM_MESH, 
            meshname="mesh1", 
            rgba=[0.9, 0.9, 0.0, 1], # Orange
            pos = attach_site.pos + [-0.045 ,0.045, handler_depth]
        )

        # 6. Part 2 (Second half of handler)
        # We add this to the SAME body so they are fixed together.
        handler.add_geom(
            name="handler_part2_geom", 
            type=mujoco.mjtGeom.mjGEOM_MESH, 
            meshname="mesh2", 
            rgba=[0.8, 0.8, 0.8, 1], # Grey to distinguish
            pos=attach_site.pos + [-0.0175, -0.002, 0.025] 
        )
        # 7. Scalpel geometry from registered scalpel mesh
        handler.add_geom(
            name="Scalpel_geom", 
            type=mujoco.mjtGeom.mjGEOM_MESH, 
            meshname="mesh3", 
            rgba=[0.8, 0.8, 0.8, 1], # Grey to distinguish
            pos=attach_site.pos
        )
        # 8. TCP defined at the tip of scalpel
        tip_offset = [0, 0, 0.113] 
        tcp_site = handler.add_site(
        name="scalpel_tip",
        pos=attach_site.pos + tip_offset,
        size=[0.002, 0.002, 0.002], # Small visual marker
        rgba=[1, 0, 0, 1], # Red tip
        group=1  # Ensure group 1 is enabled in your viewer
        )
        
        # 9. Cutting material definition - MAKE IT SOFTER for cutting
        material = spec.worldbody.add_body(name="cutting_material")
        material.add_geom(
            name="material_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.3, 0.3, 0.02],  # Adjust dimensions
            pos=[0.5, 0, 0.02],      # Position under scalpel
            rgba=[0.2, 0.8, 0.2, 0.7],
            solref=[0.02, 1],  # Softer contact
            # solimp=[0.9, 0.95, 0.001,2],
            margin=0.001
        )
        
        # 10. Compile model, forward it with data
        self._model = spec.compile()
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        
        print("✓ Model created")
        print(f"Scalpel geom ID: {self._model.geom('Scalpel_geom').id}")
        print(f"Material geom ID: {self._model.geom('material_geom').id}")
        
    @property
    def data(self):
        return self._data
    
    # ========== CONTACT FORCE MEASUREMENT METHODS ==========
    
    def get_scalpel_contact_forces(self):
        """Get all contact forces on the scalpel"""
        scalpel_forces = []
        scalpel_geom_id = self._model.geom("Scalpel_geom").id
        
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            
            # Check if scalpel is involved in this contact
            if contact.geom1 == scalpel_geom_id or contact.geom2 == scalpel_geom_id:
                # Get contact force in global frame
                force = np.zeros(6)
                mujoco.mj_contactForce(self._model, self._data, i, force)
                
                # Extract force vector (first 3 elements are force)
                force_vector = force[:3]
                scalpel_forces.append({
                    'force': force_vector.copy(),
                    'position': contact.pos.copy(),
                    'frame': contact.frame.copy()
                })
                
                # Debug print
                # print(f"Contact {i}: force = {np.linalg.norm(force_vector):.4f} N")
        
        return scalpel_forces
    
    def get_total_cutting_force(self):
        """Get total force vector on scalpel (sum of all contacts)"""
        contacts = self.get_scalpel_contact_forces()
        total_force = np.zeros(3)
        
        for contact in contacts:
            total_force += contact['force']
        
        return total_force
    
    def record_force_step(self):
        """Record current cutting force for history"""
        force = self.get_total_cutting_force()
        self.state.get(force_history).append(force.copy())
        return force
    
    def get_force_magnitude(self):
        """Get magnitude of total cutting force"""
        force = self.get_total_cutting_force()
        return np.linalg.norm(force)
    
    # ========== CUTTING CONTROL METHODS ==========
    
    def perform_cutting_stroke(self, depth=0.03, steps=500):
        """Perform a simple downward cutting motion"""
        print(f"\nStarting cutting stroke: depth={depth}m, steps={steps}")
        
        # Reset force history
        self.reset_state()
        
        # Get initial TCP position
        tcp_id = self._model.site("scalpel_tip").id
        start_pos = self._data.site_xpos[tcp_id].copy()
        print(f"start pos = {start_pos}")
        
        # ==== ADD CONTROL COMMANDS HERE ====
        print(f"Applying control to {self._model.nu} actuators")
        print(f"Control vector shape: {self._data.ctrl.shape}")
        
        # Simple downward motion
        for step in range(steps):
            # ==== THIS IS WHERE CONTROL THE ROBOT ====
            # Try different control indices:
            
            # OPTION 1: Try position control on joint 6 (usually last joint)
            # self._data.ctrl[5] = -0.01  # Negative = move down
            
            # OPTION 2: Try all joints small negative
            # for i in range(min(6, self._model.nu)):
            #     self._data.ctrl[i] = -0.005
            
            # OPTION 3: Direct Cartesian control (if your robot supports it)
            # current_z = self._data.site_xpos[tcp_id][2]
            # target_z = start_pos[2] - depth
            # error = target_z - current_z
            # self._data.ctrl[2] = error * 10.0  # P-control on Z
            
            # OPTION 4: Find which control moves Z - try each
            if step < 50:
                # Test control 0
                self._data.ctrl[0] = -0.01
            elif step < 100:
                # Test control 1  
                self._data.ctrl[1] = -0.01
            elif step < 150:
                # Test control 2
                self._data.ctrl[2] = -0.01
            # Continue for all controls...
            
            # Step simulation
            mujoco.mj_step(self._model, self._data)
            
            # Record forces
            current_force = self.record_force_step()
            force_mag = np.linalg.norm(current_force)
            
            # Get current depth
            current_depth = start_pos[2] - self._data.site_xpos[tcp_id][2]
            
            # Print progress
            if step % 50 == 0:
                print(f"  Step {step:3d}: Depth={current_depth:.4f}m, Force={force_mag:.2f}N")
                print(f"    TCP Z position: {self._data.site_xpos[tcp_id][2]:.4f}")
                print(f"    Control values: {self._data.ctrl[:min(6, len(self._data.ctrl))]}")
            
            # Stop if reached target depth
            if current_depth >= depth:
                print(f"  ✓ Reached target depth at step {step}")
                break
        
        print(f"Cutting completed. Max force: {np.max([np.linalg.norm(f) for f in self.state.get(force_history)]):.2f}N")
        return self.state.get(force_history)   
    # ========== IMPEDANCE ESTIMATION ==========
    
    def estimate_impedance(self, displacement=0.001, steps=100):
        """Simple impedance estimation by applying small displacement"""
        print("\nEstimating impedance parameters...")
        
        tcp_id = self._model.site("scalpel_tip").id
        start_pos = self._data.site_xpos[tcp_id].copy()
        
        # Store initial force
        initial_force = self.get_total_cutting_force()
        
        # Apply small downward displacement
        displacement_forces = []
        for step in range(steps):
            # Apply small control (adjust for your robot)
            # self._data.ctrl[2] = -0.001
            
            mujoco.mj_step(self._model, self._data)
            
            # Measure force
            force = self.get_total_cutting_force()
            displacement_forces.append(force.copy())
            
            # Check displacement
            current_pos = self._data.site_xpos[tcp_id]
            if abs(current_pos[2] - start_pos[2]) >= displacement:
                break
        
        # Calculate average force during displacement
        avg_force = np.mean(displacement_forces, axis=0)
        force_change = avg_force - initial_force
        
        # Simple stiffness estimation: K = ΔF / Δx
        stiffness = np.linalg.norm(force_change) / displacement
        
        # Simple damping estimation (using velocity)
        # Get site velocity from Jacobian
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self._model, self._data, 
                                mujoco.mjtObj.mjOBJ_SITE, tcp_id, vel, 0)
        final_vel = vel[:3]  # Linear velocity part
        if np.linalg.norm(final_vel) > 1e-6:
            damping = np.linalg.norm(force_change) / np.linalg.norm(final_vel)
        else:
            damping = 0.0
        
        print(f"  Stiffness estimate: {stiffness:.2f} N/m")
        print(f"  Damping estimate: {damping:.2f} N·s/m")
        
        return stiffness, damping

    # ========== MAIN EXPERIMENT ==========

    def run_cutting_experiment(self):
        """ TODO refactor this later
        """

        print("\n" + "="*60)
        print("CUTTING EXPERIMENT STARTING")
        print("="*60)
        
        # 1. Move robot to starting position above material
        # (You need to implement this based on your robot)
        print("\n1. Positioning robot...")
        # position_robot_above_material(robot)
        
        # 2. Measure initial impedance (no contact)
        print("\n2. Measuring initial impedance...")
        initial_stiffness, initial_damping = self.estimate_impedance()
        
        # 3. Perform cutting stroke
        print("\n3. Performing cutting stroke...")
        force_data = self.perform_cutting_stroke(depth=0.02, steps=300)
        
        # 4. Measure impedance after cutting
        print("\n4. Measuring impedance after cutting...")
        final_stiffness, final_damping = self.estimate_impedance()
        
        # 5. Analyze results
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS")
        print("="*60)
        
        force_magnitudes = [np.linalg.norm(f) for f in force_data]
        
        print(f"Cutting duration: {len(force_data)} steps")
        print(f"Max cutting force: {np.max(force_magnitudes):.2f} N")
        print(f"Avg cutting force: {np.mean(force_magnitudes):.2f} N")
        print(f"Stiffness change: {final_stiffness - initial_stiffness:.2f} N/m")
        print(f"Damping change: {final_damping - initial_damping:.2f} N·s/m")
        if False:
            # Save data
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(force_magnitudes)
            plt.xlabel('Time Step')
            plt.ylabel('Cutting Force (N)')
            plt.title('Cutting Force Profile')
            plt.grid(True)
            plt.savefig('cutting_force_profile.png')
            plt.show()
        
        print("\n✓ Experiment complete. Data saved.")

# Run the experiment
if __name__ == "__main__":
    pass