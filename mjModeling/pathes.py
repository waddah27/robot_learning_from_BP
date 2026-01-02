import os

from mjModeling.startup import CONF
# Get the directory where your Python script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full absolute path to the STL
SCALPEL_HANDLER_PART1 = CONF["scalpel_handler"]["part1"]
SCALPEL_HANDLER_PART2 = CONF["scalpel_handler"]["part2"]
SCALPEL_DIRNAME = CONF["scalpel_dir"]
scalpelHandler1_path = os.path.join(base_dir, SCALPEL_HANDLER_PART1)
scalpelHandler2_path = os.path.join(base_dir, SCALPEL_HANDLER_PART2)
scalpel_path = os.path.join(base_dir, SCALPEL_DIRNAME)
robot_env_dir = os.path.join(os.path.dirname(__file__), CONF['robot_name'])
robot_scene_xml = os.path.join(robot_env_dir, 'scene.xml')