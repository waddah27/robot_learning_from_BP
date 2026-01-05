from mjModeling import robot_scene_xml
from kuka_iiwa_14.iiwa14_model import iiwa14
from mjModeling.experiments import (
    ImpedanceEstimator,
    InitPos
    )
from visualization.visualizer import Visualize

# 1 - build experiment env
robot = iiwa14().create(robot_scene_xml)
impedanceEstimator = ImpedanceEstimator(robot)
init_pos = InitPos(robot)
print(robot.model.opt.gravity)
# 2 - simulator
visualizer = Visualize(robot)

if __name__ == '__main__':
    visualizer.run(callback=lambda: init_pos.execute())
