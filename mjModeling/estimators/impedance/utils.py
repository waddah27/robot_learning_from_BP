# ========== CUTTING CONTROL METHODS ==========

from mjModeling.estimators.impedance.impedance_estimation import ImpedanceEstimator

"""
utilities to be later registerd to impedance estimator if they are needed 
"""
def perform_cutting_stroke(self: ImpedanceEstimator, depth=0.03, steps=500):
    """NOT USED YET Perform a simple downward cutting motion"""
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
    """NOT USED YET: Simple impedance estimation by applying small displacement"""
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

def estimate(self):
    """ TODO NOT USED YET refactor this later
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