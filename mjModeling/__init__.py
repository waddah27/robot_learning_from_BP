# Other imports and helper functions
import os
import time
import mujoco
import itertools
import numpy as np
from PIL import Image
from mujoco import viewer
import json
# Graphics and plotting.
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import distutils.util
import subprocess

# load configs 
root_dir = os.path.dirname(__file__)
CONF = None
with open(os.path.join(root_dir,'configs.json')) as f:
  CONF = json.load(f)
# Construct the full absolute path to the STL
SCALPEL_HANDLER_PART1 = CONF["scalpel_handler"]["part1"]
SCALPEL_HANDLER_PART2 = CONF["scalpel_handler"]["part2"]
SCALPEL_DIRNAME = CONF["scalpel_dir"]
scalpelHandler1_path = os.path.join(root_dir, SCALPEL_HANDLER_PART1)
scalpelHandler2_path = os.path.join(root_dir, SCALPEL_HANDLER_PART2)
scalpel_path = os.path.join(root_dir, SCALPEL_DIRNAME)
robot_env_dir = os.path.join(os.path.dirname(__file__), CONF['robot_name'])
robot_scene_xml = os.path.join(robot_env_dir, 'scene.xml')
