import os
import re
import json
from enum import Enum, IntEnum
import numpy as np
# load configs
file_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(file_dir)

# Load configs from json igonring comments
configs = json.loads(re.sub(r'//.*|/\*[\s\S]*?\*/', '', open(os.path.join(file_dir,'configs.json')).read()))


# IK parameters
class paramIK(Enum):
    IK_MAX_STEPS = configs["IK_params"]["ik_max_steps"]
    IK_KP = configs["IK_params"]["kp"]
    IK_KD = configs["IK_params"]["kd"]
    IK_TOL = configs["IK_params"]["tolerance"]


class paramVIC(Enum):
    VIC_MAX_STEPS = configs["vic_params"]["vic_max_steps"]
    VIC_TOL = configs["vic_params"]["tolerance"]
    VIC_KP_MIN = configs["vic_params"]["kp"]["min"]
    VIC_KP_MAX = configs["vic_params"]["kp"]["max"]
    VIC_M = configs["vic_params"]["m"]
    VIC_KI = configs["vic_params"]["ki"]
    VIC_LAMBDA_SQ = configs["vic_params"]["lambda_sq"]

# GMR Integration Parameters
class GMRParams:
    FORCE_ERROR_THRESHOLD = 5.0  # N, threshold for triggering adaptation
    ADAPTATION_RATE = 0.1  # How quickly to adapt gains
    BLENDING_FACTOR = 0.7  # Weight of GMR vs reactive control (0=GMR only, 1=reactive only)
    MIN_STIFFNESS_CONTACT = 100.0  # N/m, minimum in contact
    MAX_STIFFNESS_FREE = 2000.0  # N/m, maximum in free motion

# Add to existing paramVIC class
paramVIC.GMR = GMRParams

class workPiece:
    MATERIAL_RESISTANCE = configs["material_params"]["material_resistance"]
    MATERIAL_NAME = configs["material_params"]["material_name"]
    MATERIAL_IS_SOLID = True
    POS = np.array([0.5, 0.0, 0.02])
    SIZE = np.array([0.3, 0.3, 0.02])
    SURFACE: float = SIZE[2] + POS[2]
    MOVABLE: bool = True


class Signal(IntEnum):
    FX = 0
    FY = 1
    FZ = 2

class oscillatorConfigs:
    BUFFER_SIZE = 1000
    N_SIGS = len(Signal)

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
ROBOT_DIR = os.path.join(root_dir, configs['robot_name'])
ROBOT_XML_DIR = os.path.join(ROBOT_DIR, 'xml')
ROBOT_SCENE = os.path.join(ROBOT_XML_DIR, 'scene.xml')

# attached assets parameters
MATERIAL_GEOM = configs["material_params"]["material_geom_name"]
SCALPEL_GEOM = configs["scalpel_geom_name"]
# get robot state dict keys
FORCE_HISTORY = configs["force_history"]


