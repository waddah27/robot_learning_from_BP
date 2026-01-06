from mjModeling.conf import ROBOT_SCENE
from kuka_iiwa_14.iiwa14_model import iiwa14
from mjModeling.experiments.impedance import (
    ImpedanceEstimator)
from mjModeling.experiments.motion import InitPos
from mjModeling.experiments import Experiment
from visualization.visualizer import Visualize

# 1 - build experiment env
robot = iiwa14().create(ROBOT_SCENE)
impedanceEstimator: Experiment = ImpedanceEstimator(robot)
init_pos: Experiment = InitPos(robot)
print(robot.model.opt.gravity)
# 2 - simulator
visualizer = Visualize(robot)

if __name__ == '__main__':
    experiments = {
        "motion": init_pos,
        "impedance": impedanceEstimator
        }
    visualizer.run(callback=lambda: experiments["impedance"].execute())
