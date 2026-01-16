

import mujoco
import numpy as np
from mjModeling import Robot
from mjModeling.conf import (
    FORCE_HISTORY,
    SCALPEL_GEOM
    )
from mjModeling.experiments import Experiment


class ImpedanceEstimator(Experiment):
    def __init__(self, robot: Robot):
        self.robot = robot

    def get_scalpel_contact_forces(self):
        """Get all contact forces on the scalpel"""
        scalpel_forces = []
        scalpel_geom_id = self.robot.model.geom(SCALPEL_GEOM).id
        
        for i in range(self.robot.data.ncon):
            contact = self.robot.data.contact[i]
            
            # Check if scalpel is involved in this contact
            if contact.geom1 == scalpel_geom_id or contact.geom2 == scalpel_geom_id:
                # Get contact force in global frame
                force = np.zeros(6)
                mujoco.mj_contactForce(self.robot.model, self.robot.data, i, force)
                
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
        
        for i,contact in enumerate(contacts):
            total_force += contact['force']
            print(f"contact {i} -- force = {contact['force']}")
        
        return total_force
    
    def record_force_step(self):
        """Record current cutting force for history"""
        force = self.get_total_cutting_force()
        self.robot.state.get(FORCE_HISTORY).append(force.copy())
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
        self.robot.reset_state()
        
        # Get initial TCP position
        tcp_id = self.robot.model.site("scalpel_tip").id
        start_pos = self.robot.data.site_xpos[tcp_id].copy()
        print(f"start pos = {start_pos}")
        
        # ==== ADD CONTROL COMMANDS HERE ====
        print(f"Applying control to {self.robot.model.nu} actuators")
        print(f"Control vector shape: {self.robot.data.ctrl.shape}")
        
        # Simple downward motion
        for step in range(steps):
            # ==== THIS IS WHERE CONTROL THE ROBOT ====
            # Try different control indices:
            
            # OPTION 1: Try position control on joint 6 (usually last joint)
            # self.robot.data.ctrl[5] = -0.01  # Negative = move down
            
            # OPTION 2: Try all joints small negative
            # for i in range(min(6, self.robot.model.nu)):
            #     self.robot.data.ctrl[i] = -0.005
            
            # OPTION 3: Direct Cartesian control (if robot supports it)
            # current_z = self.robot.data.site_xpos[tcp_id][2]
            # target_z = start_pos[2] - depth
            # error = target_z - current_z
            # self.robot.data.ctrl[2] = error * 10.0  # P-control on Z
            
            # OPTION 4: Find which control moves Z - try each
            if step < 50:
                # Test control 0
                self.robot.data.ctrl[0] = -0.01
            elif step < 100:
                # Test control 1  
                self.robot.data.ctrl[1] = -0.01
            elif step < 150:
                # Test control 2
                self.robot.data.ctrl[2] = -0.01
            # Continue for all controls...
            
            # Step simulation
            mujoco.mj_step(self.robot.model, self.robot.data)
            
            # Record forces
            current_force = self.record_force_step()
            force_mag = np.linalg.norm(current_force)
            
            # Get current depth
            current_depth = start_pos[2] - self.robot.data.site_xpos[tcp_id][2]
            
            # Print progress
            if step % 50 == 0:
                print(f"  Step {step:3d}: Depth={current_depth:.4f}m, Force={force_mag:.2f}N")
                print(f"    TCP Z position: {self.robot.data.site_xpos[tcp_id][2]:.4f}")
                print(f"    Control values: {self.robot.data.ctrl[:min(6, len(self.robot.data.ctrl))]}")
            
            # Stop if reached target depth
            if current_depth >= depth:
                print(f"  ✓ Reached target depth at step {step}")
                break
        
        print(f"Cutting completed. Max force: {np.max([np.linalg.norm(f) for f in self.robot.state.get(FORCE_HISTORY)]):.2f}N")
        return self.robot.state.get(FORCE_HISTORY)   
    # ========== IMPEDANCE ESTIMATION ==========
    
    def estimate_impedance(self, displacement=0.00001, steps=100):
        """Simple impedance estimation by applying small displacement"""
        print("\nEstimating impedance parameters...")
        
        tcp_id = self.robot.model.site("scalpel_tip").id
        start_pos = self.robot.data.site_xpos[tcp_id].copy()
        
        # Store initial force
        initial_force = self.get_total_cutting_force()
        
        # Apply small downward displacement
        displacement_forces = []
        for step in range(steps):
            # Apply small control (adjust for robot)
            # self.robot.data.ctrl[2] = -0.001
            
            mujoco.mj_step(self.robot.model, self.robot.data)
            
            # Measure force
            force = self.get_total_cutting_force()
            displacement_forces.append(force.copy())
            
            # Check displacement
            current_pos = self.robot.data.site_xpos[tcp_id]
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
        mujoco.mj_objectVelocity(self.robot.model, self.robot.data, 
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

    def execute(self):
        """ TODO refactor this later
        """

        print("\n" + "="*60)
        print("CUTTING EXPERIMENT STARTING")
        print("="*60)
        
        # 1. Move robot to starting position above material
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