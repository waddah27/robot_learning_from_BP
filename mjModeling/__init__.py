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
SCALPEL_HANDLER_PART1 = configs["scalpel_handler_dir"]["part1"]
SCALPEL_HANDLER_PART2 = configs["scalpel_handler_dir"]["part2"]
SCALPEL_DIRNAME = configs["scalpel_dir"]
SCALPEL_HANDLER_1_PATH = os.path.join(root_dir, SCALPEL_HANDLER_PART1)
SCALPEL_HANDLER_2_PATH = os.path.join(root_dir, SCALPEL_HANDLER_PART2)
SCALPEL_PATH = os.path.join(root_dir, SCALPEL_DIRNAME)
# flags for mujoco viewer
VIS_SITE_FRAME = configs["vis_site_frame"]
VIS_JOINTS = configs["vis_joints"]
VIS_LABEL_NAME = configs["vis_label_name"]
# Basic robot and scene xml files
ROBOT_XML = os.path.join(os.path.dirname(__file__), configs['robot_name'])
ROBOT_ENV = os.path.join(ROBOT_XML, 'xml')
ROBOT_SCENE = os.path.join(ROBOT_ENV, 'scene.xml')

# added geometries parameters
MATERIAL_GEOM = configs["material_geom_name"]
SCALPEL_GEOM = configs["scalpel_geom_name"]
# get robot state dict keys
FORCE_HISTORY = configs["force_history"]