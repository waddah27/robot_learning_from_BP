from mjModeling import robot_scene_xml
from kuka_iiwa_14.iiwa14_model import iiwa14
from visualization.visualizer import Visualize
from kinematics import JacobianIK, quat_to_mat

# 1 - build experiment env
robot = iiwa14().create(robot_scene_xml)
print(robot.model.opt.gravity)
# 2 - simulator
visualizer = Visualize(robot)

if __name__ == '__main__':
    visualizer.run()
