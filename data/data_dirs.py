import os
import glob
from enum import Enum
__all__ = [
    "BP_DIR", 
    "CLEAN_TRAIN_DATA", 
    "EXPERIMENTAL_DATA",
    "MaterialMeta"
    ]
FILE_DIR = os.path.dirname(__file__)
BP_DIR = os.path.join(FILE_DIR, 'bp_store')
mat_dirs = sorted(glob.glob(os.path.join(BP_DIR, '*.npy')))
BP_CORK_DATA_DIR = mat_dirs[0]
BP_PENO_DATA_DIR = mat_dirs[1]
BP_PVC_DATA_DIR = mat_dirs[2]
CLEAN_DATA_DIR = os.path.join(FILE_DIR, 'CleanData')
CLEAN_DATA_GROUPED = os.path.join(CLEAN_DATA_DIR, 'grouped')
CLEAN_DATA_UNGROUPED = os.path.join(CLEAN_DATA_DIR, 'ungrouped')
CLEAN_TRAIN_DATA = os.path.join(CLEAN_DATA_GROUPED, 'ready')
EXPERIMENTAL_DATA = os.path.join(FILE_DIR, 'experimental_data')


class MaterialMeta(Enum):
    PVC = BP_PVC_DATA_DIR
    penoplex = BP_PENO_DATA_DIR
    cork = BP_CORK_DATA_DIR