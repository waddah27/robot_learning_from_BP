from Oscillator import run_drawer
from mjModeling.conf import ROBOT_SCENE
from kuka_iiwa_14.iiwa14_model import iiwa14
from mjModeling.controllers import JacobianIK, VariableImpedanceControl
from mjModeling.experiments.impedance import (
    ImpedanceEstimator)
from mjModeling.experiments.motion import InitPos
from mjModeling.experiments import Experiment
from visualization.visualizer import Visualize
import multiprocessing as mp

# 1 - build experiment env
robot = iiwa14().create(ROBOT_SCENE)
# Experiments
impedanceEstimator: Experiment = ImpedanceEstimator(robot)
init_pos: Experiment = InitPos(robot)

# Controllers
vic = VariableImpedanceControl(robot)
jik = JacobianIK(robot)
print(f"Gravity = {robot.model.opt.gravity}")
# 2 - simulator
visualizer = Visualize(robot)

if __name__ == '__main__':
    #  Start Oscillator Process
    drawer_proc = mp.Process(target=run_drawer, args=(robot.shm.name,))
    drawer_proc.start()
    controllers = {
        "vic": vic,
        "jik": jik
    }

    experiments = {
        "motion": init_pos,
        "impedance": impedanceEstimator
        }

    init_pos.controller = vic
    visualizer.run(callback=lambda x: experiments["motion"].execute(x))
