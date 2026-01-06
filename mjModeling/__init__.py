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
import re

# load configs 
root_dir = os.path.dirname(__file__)

# Load configs from json igonring comments
configs = json.loads(re.sub(r'//.*|/\*[\s\S]*?\*/', '', open(os.path.join(root_dir,'configs.json')).read()))

# Construct the full absolute path to the STL
SCALPEL_HANDLER_PART1 = configs["scalpel_handler"]["part1"]
SCALPEL_HANDLER_PART2 = configs["scalpel_handler"]["part2"]
SCALPEL_DIRNAME = configs["scalpel_dir"]
VIS_SITE_FRAME = configs["vis_site_frame"]
VIS_JOINTS = configs["vis_joints"]
VIS_LABEL_NAME = configs["vis_label_name"]
MATERIAL_GEOM = configs["material_geom_name"]
SCALPEL_GEOM = configs["scalpel_geom_name"]

# get state parameters names
force_history = configs["force_history"]
scalpelHandler1_path = os.path.join(root_dir, SCALPEL_HANDLER_PART1)
scalpelHandler2_path = os.path.join(root_dir, SCALPEL_HANDLER_PART2)
scalpel_path = os.path.join(root_dir, SCALPEL_DIRNAME)

robot_xml_dir = os.path.join(os.path.dirname(__file__), configs['robot_name'])
robot_env_dir = os.path.join(robot_xml_dir, 'xml')
robot_scene_xml = os.path.join(robot_env_dir, 'scene.xml')
