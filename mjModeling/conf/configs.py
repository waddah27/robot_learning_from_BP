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


class paramVIC:
    VIC_MAX_STEPS = 5000
    VIC_TOL = 0.004
    VIC_KP_MIN = 400.0
    VIC_KP_MAX = 1500.0
    VIC_M = 1 # mass
    VIC_KI = 200
    VIC_LAMBDA_SQ = 1e-4
    ADAPTIVE = True
    CONTINUOUS_TRAJ = True
    PHASE_SPEED = 1.0

class ImpedanceOptimizer:
    qp = "qp"
    lsq = "lsq"
    gd = "gd"

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
    MATERIAL_RESISTANCE = 27 #N (OBSOLETE)
    MATERIAL_NAME ="cutting_material"
    MATERIAL_IS_SOLID = True
    POS = np.array([0.5, 0.0, 0.02])
    SIZE = np.array([0.3, 0.3, 0.02])
    SURFACE: float = SIZE[2] + POS[2]
    PENETRATION_RATE = .8 # how much the scalpel could penatrate the material from surface to center
    MOBILE: bool = False


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
VIS_SITE_FRAME = False
VIS_JOINTS = False
VIS_LABEL_NAME = True
# Basic robot and scene xml files
ROBOT_DIR = os.path.join(root_dir, configs['robot_name'])
ROBOT_XML_DIR = os.path.join(ROBOT_DIR, 'xml')
ROBOT_SCENE = os.path.join(ROBOT_XML_DIR, 'scene.xml')

# attached assets parameters
MATERIAL_GEOM = "material_geom"
SCALPEL_GEOM = "scalpel_geom"
# get robot state dict keys
FORCE_HISTORY = "force_history"


