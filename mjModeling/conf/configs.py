import os
import re
import json
from enum import Enum, IntEnum
import numpy as np

file_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(file_dir)


# IK parameters
class paramIK(Enum):
    IK_MAX_STEPS = 500
    IK_KP = 50
    IK_KD = 0.1
    IK_TOL = 0.04

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

class paramVIC:
    VIC_MAX_STEPS = 50000
    VIC_TOL = 0.001
    VIC_KP_MIN = 100.0
    VIC_KP_MAX = 2000.0
    VIC_M = 1 # mass
    VIC_KI = 200
    VIC_LAMBDA_SQ = 1e-4
    ADAPTIVE = True
    CONTINUOUS_TRAJ = True
    PHASE_SPEED = 1.0
    # Disable using variable gains for other motion phases than cutting
    DISABLE_PTP_VIC = True
    USE_LEARNT_GAINS = True
    TRACK_WP_MOTION = True
    # GMR params to vic
    GMR = GMRParams

class workPiece:
    MATERIAL_RESISTANCE = 27 #N (OBSOLETE)
    MATERIAL_NAME ="cutting_material"
    MATERIAL_IS_SOLID = True
    HIGHT = 0.02 #TODO different hight material behaves strange at first WHY?
    POS = np.array([0.5, 0.0, HIGHT])
    SIZE = np.array([0.3, 0.3, 0.02])
    SURFACE: float = SIZE[2] + POS[2]
    MOBILE: bool = True
    MOT_AMPL = 0.05 # amplitude of motion sine/cos wave


class Signal(IntEnum):
    FX = 0
    FY = 1
    FZ = 2

class oscillatorConfigs:
    BUFFER_SIZE = 1000
    N_SIGS = len(Signal)

# Construct the full absolute path to the STL
SCALPEL_HANDLER_PART1 = "scalpel_model/scalpel/scalpelHandler1.STL"
SCALPEL_HANDLER_PART2 = "scalpel_model/scalpel/scalpelHandler2.STL"
SCALPEL_DIRNAME = "scalpel_model/scalpel/Scalpel.stl"
SCALPEL_HANDLER_1_PATH = os.path.join(root_dir, SCALPEL_HANDLER_PART1)
SCALPEL_HANDLER_2_PATH = os.path.join(root_dir, SCALPEL_HANDLER_PART2)
SCALPEL_PATH = os.path.join(root_dir, SCALPEL_DIRNAME)
# flags for mujoco viewer
VIS_SITE_FRAME = False
VIS_JOINTS = False
VIS_LABEL_NAME = True
# Basic robot and scene xml files
ROBOT_NAME = "kuka_iiwa_14"
ROBOT_DIR = os.path.join(root_dir, ROBOT_NAME)
ROBOT_XML_DIR = os.path.join(ROBOT_DIR, 'xml')
ROBOT_SCENE = os.path.join(ROBOT_XML_DIR, 'scene.xml')

# attached assets parameters
MATERIAL_GEOM = "material_geom"
SCALPEL_GEOM = "scalpel_geom"
# get robot state dict keys
FORCE_HISTORY = "force_history"
TCP_POS = "tcp_pos"

