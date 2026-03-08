from mjModeling.controllers.vic_controller import BasicVariableImpedanceControl
from mjModeling.cutting_materials import Material
import numpy as np
import mujoco
from mjModeling.conf import paramVIC, workPiece
from mjModeling.controllers import Controller
from mjModeling.estimators import ImpedanceEstimator
from mjModeling.mjRobot import Robot
from reference_generators import GMRReferenceGenerator

__all__ = ["GMRVariableImpedanceControl"]


class GMRVariableImpedanceControl(BasicVariableImpedanceControl):
    def __init__(self, robot: Robot, gmr_sequence=None):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data
        self.error_accumulated = np.zeros(3)
        self.estimator = ImpedanceEstimator(robot)
        self._working_piece: Material = None

        # GMR Integration
        if gmr_sequence is not None:
            self.gmr_generator = GMRReferenceGenerator(gmr_sequence)
            self.use_gmr_priors = True
            self.start_time = None
        else:
            self.use_gmr_priors = False

        # Adaptive gains storage
        self.Kp_gmr = np.ones(3) * paramVIC.VIC_KP_MAX.value
        self.Kd_gmr = np.ones(3) * 0.5 * np.sqrt(paramVIC.VIC_KP_MAX.value)

    def compute_control_force(self, current_pos, v_tip, error, dist):
        """
        Enhanced control force with GMR behavior priors
        """
        # 1. GET GMR REFERENCES (if available)
        if self.use_gmr_priors:
            time_elapsed = self.data.time - self.start_time
            pos_des, vel_des, force_des, stiffness_des = \
                self.gmr_generator.predict(time_elapsed)

            # Compute errors relative to GMR trajectory
            pos_error = pos_des - current_pos
            vel_error = vel_des - v_tip

            # Use GMR-suggested stiffness if available
            if stiffness_des is not None:
                self.Kp_gmr = stiffness_des
                self.Kd_gmr = 2 * np.sqrt(self.Kp_gmr)  # Critically damped
        else:
            # Fallback to original error calculation
            pos_error = error
            vel_error = -v_tip  # Original used -Kd*v

        # 2. ADAPTIVE GAIN SELECTION
        if self.use_gmr_priors:
            # Blend between GMR stiffness and safety bounds
            kp_val = np.clip(self.Kp_gmr,
                            paramVIC.VIC_KP_MIN.value,
                            paramVIC.VIC_KP_MAX.value)
            kd_val = np.clip(self.Kd_gmr,
                            0.1 * np.sqrt(paramVIC.VIC_KP_MIN.value),
                            2.0 * np.sqrt(paramVIC.VIC_KP_MAX.value))
        else:
            # Original variable gain scheduling
            kp_val, kd_val = self.get_variable_gains(dist)

        # 3. FORCE CALCULATION WITH GMR PRIORS
        # Base impedance term
        f_impedance = kp_val * pos_error - kd_val * vel_error

        # Add integral term (carefully - can cause instability with force priors)
        if dist < 0.05:  # Only near target
            self.error_accumulated += pos_error * self.model.opt.timestep
        ki_val = paramVIC.VIC_KI.value
        f_integral = ki_val * self.error_accumulated

        # Add GMR force prior (feedforward term)
        if self.use_gmr_priors:
            f_feedforward = force_des
        else:
            f_feedforward = np.zeros(3)

        # Cutting resistance compensation
        f_resistance = self.compensate_cutting_resistance(current_pos, v_tip)

        # TOTAL VIRTUAL FORCE
        f_total = f_impedance + f_integral + f_feedforward + f_resistance

        # 4. IMPEDANCE ADAPTATION LOGIC
        # Adjust gains based on contact prediction vs reality
        if self.use_gmr_priors and hasattr(self, 'contact_predictor'):
            self.adapt_impedance_to_contact(current_pos, force_des, f_resistance)

        return f_total, kp_val, kd_val

    def adapt_impedance_to_contact(self, current_pos, force_desired, force_measured):
        """
        Adapt impedance parameters when actual contact differs from GMR prediction
        """
        # Calculate prediction error
        force_error = np.linalg.norm(force_desired - force_measured)

        # Check if we're in contact region (using working piece surface)
        surface_z = self.working_piece.surface_height if self.working_piece else 0
        depth = surface_z - current_pos[2]

        if depth > 0:  # In contact
            # If force error is large, GMR prediction is inaccurate
            if force_error > paramVIC.FORCE_ERROR_THRESHOLD.value:
                # Increase damping for stability
                self.Kd_gmr *= 1.5
                # Reduce stiffness to be safer
                self.Kp_gmr *= 0.7

                # Limit changes
                self.Kd_gmr = np.clip(self.Kd_gmr,
                                    0.5 * np.sqrt(paramVIC.VIC_KP_MIN.value),
                                    2.0 * np.sqrt(paramVIC.VIC_KP_MAX.value))
                self.Kp_gmr = np.clip(self.Kp_gmr,
                                    paramVIC.VIC_KP_MIN.value,
                                    paramVIC.VIC_KP_MAX.value)

                # Log adaptation event
                self.robot.state["adaptation_events"] += 1
        else:  # Free motion
            # Gradually return to GMR-suggested stiffness
            self.Kp_gmr = 0.9 * self.Kp_gmr + 0.1 * self.gmr_generator.Kp_nominal
            self.Kd_gmr = 0.9 * self.Kd_gmr + 0.1 * self.gmr_generator.Kd_nominal

    def move_to_position(self, target_pos, viewer=None, max_steps=8000):
        tcp_id = self.model.site("scalpel_tip").id
        q_home = np.array([0.0, -0.7, 0.0, 1.5, 0.0, 0.7, 3.14159])

        self.error_accumulated = np.zeros(3)
        lambda_sq = paramVIC.VIC_LAMBDA_SQ.value

        # Initialize adaptation tracking
        self.robot.state["adaptation_events"] = 0
        self.robot.state["force_tracking_error"] = []

        for step in range(max_steps):
            mujoco.mj_forward(self.model, self.data)

            current_pos = self.data.site_xpos[tcp_id].copy()
            error = target_pos - current_pos
            dist = np.linalg.norm(error)

            # Check completion (using either target or GMR endpoint)
            if dist < paramVIC.VIC_TOL.value:
                return True

            # Get Jacobian and tip velocity
            jac = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac, None, tcp_id)
            v_tip = jac @ self.data.qvel

            # COMPUTE ENHANCED CONTROL FORCE
            f_virtual, kp_val, kd_val = self.compute_control_force(
                current_pos, v_tip, error, dist
            )

            # Store for visualization/analysis
            self.robot.state["current_Kp"] = kp_val
            self.robot.state["current_Kd"] = kd_val

            # 5. STABLE MAPPING (Damped Least Squares)
            jjt = jac @ jac.T
            tau_task = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), f_virtual)

            # 6. NULL-SPACE POSTURE CONTROL
            k_posture, d_posture = 10.0, 2.0
            tau_posture = k_posture * (q_home[:self.model.nv] - self.data.qpos[:self.model.nv]) \
                        - d_posture * self.data.qvel

            j_inv = jac.T @ np.linalg.solve(jjt + lambda_sq * np.eye(3), np.eye(3))
            null_projection = np.eye(self.model.nv) - (j_inv @ jac)
            tau_null = null_projection @ tau_posture

            # 7. FINAL TORQUE
            tau_total = tau_task + tau_null + self.data.qfrc_bias[:self.model.nv]

            # Apply with limits
            self.data.ctrl[:self.model.nu] = np.clip(tau_total[:self.model.nu], -300, 300)

            # 8. STEP PHYSICS
            mujoco.mj_step(self.model, self.data)

            # Update estimators and state
            if self.estimator:
                cutting_force = self.estimator.get_total_cutting_force()
                self.robot.state["shared_array"][:-1] = self.robot.state["shared_array"][1:]
                self.robot.state["shared_array"][-1] = cutting_force

                # Track force error if using GMR
                if self.use_gmr_priors:
                    time_elapsed = self.data.time - self.start_time
                    _, _, force_des, _ = self.gmr_generator.predict(time_elapsed)
                    force_error = np.linalg.norm(force_des - cutting_force)
                    self.robot.state["force_tracking_error"].append(force_error)

            # Visualization
            if viewer and step % 4 == 0:
                # Optional: Visualize desired vs actual in viewer
                if self.use_gmr_priors and step % 20 == 0:
                    self.visualize_gmr_reference(viewer, time_elapsed)
                viewer.sync()

        return False

    def visualize_gmr_reference(self, viewer, time_elapsed):
        """Optional: Visualize GMR references in MuJoCo viewer"""
        if hasattr(self, 'gmr_generator'):
            pos_des, _, force_des, _ = self.gmr_generator.predict(time_elapsed)

            # Add visualization markers
            mujoco.mjv_initGeom(viewer.user_scn.geoms[0],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=[0.005, 0, 0],
                            pos=pos_des,
                            mat=np.eye(3).flatten(),
                            rgba=[0, 1, 0, 0.5])

            # Visualize desired force as arrow
            force_scale = 0.001
            force_end = pos_des + force_des * force_scale
            # (we'd need to add line geometry for proper arrow visualization)

    @property
    def working_piece(self):
        return self._working_piece

    @working_piece.setter
    def working_piece(self, material: Material):
        self._working_piece = material
