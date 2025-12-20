import crocoddyl
import example_robot_data
import numpy as np
import pinocchio
import meshcat.geometry as g
import sys

robot_name = "talos_arm"
# robot_name = "ur5"
robot = example_robot_data.load(robot_name)
robot_model = robot.model

DT = 1e-3
T = 25
target = np.array([0.4, 0.0, 0.4])

display = crocoddyl.MeshcatDisplay(robot)
display.robot.viewer["world/point"].set_object(g.Sphere(0.05))
display.robot.viewer["world/point"].set_transform(
    np.array(
        [
            [1.0, 0.0, 0.0, target[0]],
            [0.0, 1.0, 0.0, target[1]],
            [0.0, 0.0, 1.0, target[2]],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
)

# Create the cost functions
state = crocoddyl.StateMultibody(robot.model)
# Create cost model per each action model
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

actuationModel = crocoddyl.ActuationModelFull(state)

xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))

if robot_name != "ur5":
    goalTrackingCost = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualModelFrameTranslation(
            state, robot_model.getFrameId("gripper_left_joint"), target
        ),
    )
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e5)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("stateReg", xRegCost, 1e-4)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-7)
terminalCostModel.addCost("stateReg", xRegCost, 1e-4)
terminalCostModel.addCost("ctrlReg", uRegCost, 1e-7)

# Create the actuation model
actuationModel = crocoddyl.ActuationModelFull(state)

# Create the action model
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, runningCostModel
    ),
    DT,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuationModel, terminalCostModel
    )
)
# runningModel.differential.armature = 0.2 * np.ones(state.nv)
# terminalModel.differential.armature = 0.2 * np.ones(state.nv)

# Create the problem
q0 = np.array([2.0, 1.5, -2.0, 0.0, 0.0, 0.0, 0.0])
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
if robot_name == "ur5":
    x0 = x0[:-1]
print(f"{robot_name} : {x0.shape}")
# sys.exit(0)
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
ddp.solve()

# Visualizing the solution in gepetto-viewer
display.displayFromSolver(ddp)

robot_data = robot_model.createData()
xT = ddp.xs[-1]
pinocchio.forwardKinematics(robot_model, robot_data, xT[: state.nq])
pinocchio.updateFramePlacements(robot_model, robot_data)
print(
    "Finally reached = ",
    robot_data.oMf[robot_model.getFrameId("gripper_left_joint")].translation.T,
)

