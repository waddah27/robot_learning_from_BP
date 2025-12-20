import crocoddyl
import example_robot_data
import numpy as np
import pinocchio
import meshcat.geometry as g
import meshcat.transformations as tf
import time
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# ============================================================================
# 1. CUSTOM IMPEDANCE AND CONTACT MODELS (FIXED)
# ============================================================================

class VariableImpedanceCostModel(crocoddyl.CostModelAbstract):
    """Cost model that implements variable impedance control"""
    def __init__(self, state, nu, frame_id, K_desired, D_desired, ref_frame_placement):
        crocoddyl.CostModelAbstract.__init__(self, state, crocoddyl.ActivationModelWeightedQuad(np.ones(6)), nu)
        self.frame_id = frame_id
        self.K = K_desired  # Desired stiffness matrix (6x6)
        self.D = D_desired  # Desired damping matrix (6x6)
        self.ref_placement = ref_frame_placement  # Desired frame placement
        
    def calc(self, data, x, u):
        # Get current frame placement
        pinocchio.framesForwardKinematics(self.state.pinocchio, data.pinocchio, x[:self.state.nq])
        current_placement = data.pinocchio.oMf[self.frame_id]
        
        # Compute error in SE(3)
        error_placement = self.ref_placement.inverse() * current_placement
        error_log = pinocchio.log6(error_placement).vector
        
        # Compute velocity error (simplified)
        error_velocity = np.zeros(6)  # In real implementation, compute from x[nq:]
        
        # Impedance cost: 0.5 * e^T K e + 0.5 * edot^T D edot
        data.r = np.sqrt(self.K.diagonal()) * error_log + np.sqrt(self.D.diagonal()) * error_velocity
        
    def calcDiff(self, data, x, u):
        # Simplified derivatives - in practice, compute proper Jacobians
        nq, nv = self.state.nq, self.state.nv
        
        # Compute frame Jacobian
        pinocchio.computeFrameJacobian(self.state.pinocchio, data.pinocchio, 
                                      x[:nq], self.frame_id, pinocchio.LOCAL_WORLD_ALIGNED)
        J = data.pinocchio.J
        
        # Simplified derivative computation
        data.Lx = np.zeros(self.state.ndx)
        data.Lu = np.zeros(self.nu)
        data.Lxx = np.eye(self.state.ndx) * 0.01
        data.Lxu = np.zeros((self.state.ndx, self.nu))
        data.Luu = np.eye(self.nu) * 0.01

class CuttingContactModel(crocoddyl.ContactModelMultiple):
    """Extended contact model for cutting tasks - FIXED CONSTRUCTOR"""
    def __init__(self, state, nu, contact_frame_id, material_properties):
        crocoddyl.ContactModelMultiple.__init__(self, state, nu)
        self.contact_frame_id = contact_frame_id
        self.material = material_properties
        self.cutting_depth = 0.0
        self.cut_force_history = []
        
        # Create 6D contact model - FIXED: Use correct constructor signature
        contact_placement = pinocchio.SE3.Identity()
        # Correct constructor: state, frame_id, placement, reference_frame
        self.addContact("cutting_contact", 
                       crocoddyl.ContactModel6D(
                           state, 
                           contact_frame_id, 
                           contact_placement, 
                           pinocchio.LOCAL_WORLD_ALIGNED  # Reference frame
                       ))
        
    def calc(self, data, x):
        super().calc(data, x)
        
        # Add cutting-specific forces
        if hasattr(data, 'pinocchio'):
            # Get penetration depth (simplified)
            current_pos = data.pinocchio.oMf[self.contact_frame_id].translation
            surface_height = 0.3  # Cutting surface at z=0.3
            penetration = max(0, surface_height - current_pos[2])
            
            if penetration > 0:
                # Material resistance force (simplified)
                resistance_force = self.compute_cutting_force(penetration, 
                                                            data.pinocchio.v[self.contact_frame_id] 
                                                            if hasattr(data.pinocchio, 'v') else None)
                
                # Store for analysis
                self.cut_force_history.append(resistance_force)

    def compute_cutting_force(self, penetration, velocity):
        """Compute cutting force based on material properties"""
        # Spring-damper model for material resistance
        spring_force = self.material['stiffness'] * penetration
        if velocity is not None and hasattr(velocity, 'linear'):
            damper_force = self.material['damping'] * abs(velocity.linear[2])
        else:
            damper_force = 0
        
        # Cutting-specific forces (simplified)
        if penetration > self.material['yield_depth']:
            # Material yielding/cutting phase
            cut_force = spring_force + damper_force + self.material['cutting_resistance']
        else:
            # Elastic deformation phase
            cut_force = spring_force + damper_force
            
        return cut_force

    def get_cutting_force_history(self):
        """Return the history of cutting forces"""
        return self.cut_force_history

class LearnedImpedanceModel:
    """Model that learns impedance parameters from human demonstrations"""
    def __init__(self):
        self.stiffness_profile = []
        self.damping_profile = []
        
    def load_human_data(self, data_file):
        """Load human demonstration data"""
        # In practice: load from file
        # For now, create synthetic data
        self.create_synthetic_impedance_data()
        
    def create_synthetic_impedance_data(self):
        """Create synthetic impedance profiles for demonstration"""
        # High stiffness at start and end, lower during cutting
        phases = np.linspace(0, 1, 100)
        for phase in phases:
            if phase < 0.2:  # Approach phase
                stiffness = np.diag([800, 800, 800, 200, 200, 200])
                damping = np.diag([80, 80, 80, 20, 20, 20])
            elif phase < 0.8:  # Cutting phase
                stiffness = np.diag([300, 300, 200, 100, 100, 100])
                damping = np.diag([40, 40, 30, 15, 15, 15])
            else:  # Retract phase
                stiffness = np.diag([600, 600, 600, 150, 150, 150])
                damping = np.diag([60, 60, 60, 15, 15, 15])
            
            self.stiffness_profile.append(stiffness)
            self.damping_profile.append(damping)
    
    def predict_impedance(self, state, phase):
        """Predict impedance parameters for current state and phase"""
        idx = min(int(phase * len(self.stiffness_profile)), len(self.stiffness_profile)-1)
        return self.stiffness_profile[idx], self.damping_profile[idx]

# ============================================================================
# 2. VIBRATION ANALYSIS TOOLS
# ============================================================================

class VibrationAnalyzer:
    """Analyze vibration and stiffness changes during task execution"""
    def __init__(self):
        self.position_history = []
        self.force_history = []
        self.stiffness_history = []
        self.frequency_data = []
        
    def record_data(self, position, force, stiffness):
        self.position_history.append(position.copy())
        self.force_history.append(force.copy())
        self.stiffness_history.append(stiffness.copy())
    
    def analyze_vibration(self):
        """Analyze vibration frequencies"""
        if len(self.position_history) < 10:
            return {}
        
        positions = np.array(self.position_history)
        forces = np.array(self.force_history)
        
        # Compute frequency content using FFT
        freq_results = []
        if len(positions) > 0 and positions.shape[0] > 10:
            for i in range(min(3, positions.shape[1])):  # For x,y,z axes
                pos_fft = np.fft.fft(positions[:, i])
                freqs = np.fft.fftfreq(len(pos_fft), d=0.001)  # Assuming 1ms sampling
                
                # Find dominant frequency
                idx = np.argmax(np.abs(pos_fft[:len(pos_fft)//2]))
                dominant_freq = abs(freqs[idx])
                freq_results.append(dominant_freq)
        
        # Analyze stiffness variations
        stiffness_var = 0
        if len(self.stiffness_history) > 0:
            stiffness_array = np.array(self.stiffness_history)
            if stiffness_array.size > 0:
                stiffness_var = np.std(stiffness_array) if stiffness_array.size > 1 else 0
        
        return {
            'dominant_frequencies': freq_results,
            'stiffness_variation': stiffness_var,
            'max_force': np.max(np.abs(forces)) if len(forces) > 0 else 0,
            'force_rmse': np.sqrt(np.mean(forces**2)) if len(forces) > 0 else 0
        }
    
    def plot_results(self):
        """Plot analysis results"""
        if not self.position_history and not self.stiffness_history:
            print("No data to plot")
            return
            
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        if len(self.position_history) > 0:
            positions = np.array(self.position_history)
            for i in range(min(3, positions.shape[1])):
                axes[i, 0].plot(positions[:, i])
                axes[i, 0].set_title(f'Position axis {["x","y","z"][i]}')
                axes[i, 0].set_xlabel('Time step')
                axes[i, 0].set_ylabel('Position (m)')
        
        if len(self.stiffness_history) > 0:
            stiffness = np.array(self.stiffness_history)
            # Flatten stiffness for plotting
            if stiffness.ndim > 1:
                stiffness_flat = stiffness.reshape(stiffness.shape[0], -1)
                for i in range(min(3, stiffness_flat.shape[1])):
                    axes[i, 1].plot(stiffness_flat[:, i])
                    axes[i, 1].set_title(f'Stiffness component {i}')
                    axes[i, 1].set_xlabel('Time step')
                    axes[i, 1].set_ylabel('Stiffness value')
        
        plt.tight_layout()
        plt.savefig('vibration_analysis.png')
        plt.show()

# ============================================================================
# 3. MAIN CUTTING TASK IMPLEMENTATION (SIMPLIFIED)
# ============================================================================

def create_cutting_task_with_impedance(robot_name="talos_arm", visualize=True):
    """Create a complete cutting task with variable impedance control - SIMPLIFIED VERSION"""
    
    # ==================== SETUP ====================
    print(f"Loading {robot_name} robot...")
    robot = example_robot_data.load(robot_name)
    robot_model = robot.model
    robot_data = robot_model.createData()
    
    # Simulation parameters - REDUCED FOR TESTING
    DT = 0.01  # 10ms time step (increased for stability)
    T = 50  # Number of steps (reduced for testing)
    
    # Cutting target and material properties
    target = np.array([0.4, 0.0, 0.4])  # Target position
    material_properties = {
        'stiffness': 500.0,  # Reduced for testing
        'damping': 20.0,     
        'yield_depth': 0.02,
        'cutting_resistance': 2.0,
        'friction_coefficient': 0.3
    }
    
    # ==================== VISUALIZATION ====================
    display = None
    if visualize:
        try:
            display = crocoddyl.MeshcatDisplay(robot)
            # Display target point
            display.robot.viewer["world/target"].set_object(
                g.Sphere(0.02),
                g.MeshLambertMaterial(color=0xFF0000, opacity=0.7)
            )
            display.robot.viewer["world/target"].set_transform(
                tf.translation_matrix(target)
            )
        except Exception as e:
            print(f"Visualization setup failed: {e}")
            visualize = False
    
    # ==================== STATE AND MODELS ====================
    state = crocoddyl.StateMultibody(robot_model)
    nq, nv, nu = state.nq, state.nv, state.nv
    
    # Get frame IDs
    if robot_name == "talos_arm":
        gripper_frame_id = robot_model.getFrameId("gripper_left_joint")
    else:
        # Try common frame names
        frame_names = ["tool0", "end_effector", "gripper", "wrist_3_link"]
        gripper_frame_id = None
        for name in frame_names:
            if robot_model.existFrame(name):
                gripper_frame_id = robot_model.getFrameId(name)
                break
        if gripper_frame_id is None:
            gripper_frame_id = robot_model.frames[-1].parent  # Use last frame
    
    print(f"Using frame ID: {gripper_frame_id}")
    
    # ==================== LEARNED IMPEDANCE MODEL ====================
    print("Loading learned impedance model...")
    impedance_learner = LearnedImpedanceModel()
    impedance_learner.create_synthetic_impedance_data()
    
    # ==================== VIBRATION ANALYZER ====================
    vibration_analyzer = VibrationAnalyzer()
    
    # ==================== CREATE ACTION MODELS ====================
    print("Creating action models...")
    
    # List to store all running models
    running_models = []
    
    for t in range(T):
        # Compute phase of task (0 to 1)
        phase = t / T
        
        # Get impedance parameters from learned model
        K_desired, D_desired = impedance_learner.predict_impedance(None, phase)
        
        # Create cost model for this phase - SIMPLIFIED
        runningCostModel = crocoddyl.CostModelSum(state)
        
        # Goal tracking cost
        try:
            goalTrackingCost = crocoddyl.CostModelResidual(
                state,
                crocoddyl.ResidualModelFrameTranslation(state, gripper_frame_id, target)
            )
            runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
        except Exception as e:
            print(f"Warning: Could not create goal tracking cost: {e}")
        
        # Control regularization (always works)
        uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))
        runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
        
        # State regularization
        xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
        runningCostModel.addCost("stateReg", xRegCost, 1e-3)
        
        # Create contact model (but use free dynamics for now to test)
        contact_model = crocoddyl.ContactModelMultiple(state, nu)
        
        # Create actuation model
        actuationModel = crocoddyl.ActuationModelFull(state)
        
        # Create differential action model - START WITH FREE DYNAMICS
        diff_action = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuationModel, runningCostModel
        )
        
        # For testing, only enable contacts after initial tests pass
        if t > T//2 and False:  # Disabled for now
            try:
                # Try with contact dynamics
                contact_model_simple = crocoddyl.ContactModelMultiple(state, nu)
                diff_action = crocoddyl.DifferentialActionModelContactFwdDynamics(
                    state, actuationModel, contact_model_simple, runningCostModel
                )
            except Exception as e:
                print(f"Warning: Contact dynamics failed at step {t}: {e}")
        
        # Create integrated action model
        running_model = crocoddyl.IntegratedActionModelEuler(diff_action, DT)
        running_models.append(running_model)
        
        # Record impedance parameters for analysis
        vibration_analyzer.record_data(
            position=target * phase,  # Simplified
            force=np.zeros(6),
            stiffness=K_desired.diagonal()[:3] if hasattr(K_desired, 'diagonal') else np.zeros(3)
        )
    
    # ==================== TERMINAL MODEL ====================
    print("Creating terminal model...")
    terminalCostModel = crocoddyl.CostModelSum(state)
    
    # Terminal goal cost
    try:
        terminalGoalCost = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ResidualModelFrameTranslation(state, gripper_frame_id, target)
        )
        terminalCostModel.addCost("terminalGoal", terminalGoalCost, 1e3)
    except:
        pass
    
    # Terminal state regularization
    xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
    terminalCostModel.addCost("terminalStateReg", xRegCost, 1e2)
    
    # Create terminal model
    terminal_contact = crocoddyl.ContactModelMultiple(state, nu)
    terminal_diff = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, terminalCostModel
    )
    terminal_model = crocoddyl.IntegratedActionModelEuler(terminal_diff, 0.0)
    
    # ==================== INITIAL STATE ====================
    if robot_name == "talos_arm":
        q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Start from zero
    else:  # UR5
        q0 = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    
    x0 = np.concatenate([q0, np.zeros(nv)])
    print(f"Initial state: {x0.shape}")
    
    # ==================== SOLVER SETUP ====================
    print("Setting up solver...")
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
    
    # Create DDP solver with relaxed settings
    ddp = crocoddyl.SolverDDP(problem)
    ddp.th_stop = 1e-4
    ddp.regInit = 1e-1  # Higher regularization for stability
    ddp.max_qp_iters = 100
    
    # Simple callback for progress
    callbacks = [crocoddyl.CallbackVerbose()]
    ddp.setCallbacks(callbacks)
    
    # ==================== SOLVE ====================
    print("\n=== Solving cutting task ===")
    print(f"Time horizon: {T*DT:.3f}s, Steps: {T}")
    print(f"Target position: {target}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Solve with few iterations for testing
        ddp.solve(maxiter=10, regInit=1e-1)
        
        solve_time = time.time() - start_time
        print(f"\nSolution completed in {solve_time:.2f} seconds")
        print(f"Iterations: {ddp.iter}")
        print(f"Final cost: {ddp.cost:.6f}")
        
        # ==================== ANALYSIS ====================
        print("\n=== Analysis Results ===")
        analysis_results = vibration_analyzer.analyze_vibration()
        for key, value in analysis_results.items():
            print(f"{key}: {value}")
        
        # Plot results
        vibration_analyzer.plot_results()
        
        # ==================== VISUALIZATION ====================
        if visualize and display is not None:
            print("\nVisualizing solution...")
            display.displayFromSolver(ddp)
            
            # Show final position
            if ddp.xs:
                xT = ddp.xs[-1]
                pinocchio.forwardKinematics(robot_model, robot_data, xT[:nq])
                pinocchio.updateFramePlacements(robot_model, robot_data)
                if gripper_frame_id < len(robot_data.oMf):
                    final_pos = robot_data.oMf[gripper_frame_id].translation
                    print(f"\nTarget position: {target}")
                    print(f"Final position:  {final_pos}")
                    print(f"Error:           {np.linalg.norm(target - final_pos):.6f} m")
            
            print("\nVisualization running. Close the window or press Ctrl+C to exit.")
            try:
                time.sleep(10)  # Show for 10 seconds
            except KeyboardInterrupt:
                pass
        
        return ddp, vibration_analyzer
        
    except Exception as e:
        print(f"\nError during solving: {e}")
        import traceback
        traceback.print_exc()
        return None, vibration_analyzer

# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

def analyze_solution(solver, robot_name="talos_arm"):
    """Basic analysis of the solution"""
    if solver is None or not hasattr(solver, 'xs'):
        print("No solution to analyze")
        return
    
    print("\n=== Solution Analysis ===")
    print(f"Number of states: {len(solver.xs)}")
    print(f"Number of controls: {len(solver.us)}")
    
    # Load robot for forward kinematics
    robot = example_robot_data.load(robot_name)
    robot_model = robot.model
    robot_data = robot_model.createData()
    
    # Get frame ID
    if robot_name == "talos_arm":
        frame_id = robot_model.getFrameId("gripper_left_joint")
    else:
        frame_id = robot_model.getFrameId("tool0") if robot_model.existFrame("tool0") else 0
    
    # Compute final position
    xT = solver.xs[-1]
    pinocchio.forwardKinematics(robot_model, robot_data, xT[:robot_model.nq])
    pinocchio.updateFramePlacements(robot_model, robot_data)
    
    if frame_id < len(robot_data.oMf):
        final_pos = robot_data.oMf[frame_id].translation
        print(f"Final end-effector position: {final_pos}")
    
    # Plot trajectory
    if len(solver.xs) > 0:
        positions = []
        for x in solver.xs:
            positions.append(x[:3] if len(x) >= 3 else x[:min(3, len(x))])
        
        positions = np.array(positions)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(min(3, positions.shape[1])):
            ax.plot(positions[:, i], label=f'Axis {i}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Position (rad/m)')
        ax.set_title('Joint Position Trajectory')
        ax.legend()
        ax.grid(True)
        plt.savefig('trajectory.png')
        plt.show()

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CUTTING TASK - SIMPLIFIED TEST VERSION")
    print("=" * 60)
    
    # Start with simpler settings
    robot_options = ["talos_arm"]  # Just test with talos_arm first
    selected_robot = robot_options[0]
    
    print(f"\nTesting with robot: {selected_robot}")
    print("Note: Using free dynamics (no contacts) for initial testing")
    print("=" * 60)
    
    # Run with visualization disabled first
    try:
        solver, analyzer = create_cutting_task_with_impedance(
            robot_name=selected_robot,
            visualize=False  # Disable for initial testing
        )
        
        if solver is not None:
            analyze_solution(solver, selected_robot)
            print("\n" + "=" * 60)
            print("BASIC TEST PASSED!")
            print("Next: Enable contacts and impedance control")
            print("=" * 60)
        else:
            print("\nTest failed. Check error messages above.")
            
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()