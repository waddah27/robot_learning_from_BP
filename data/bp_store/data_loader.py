import os
import glob
from enum import Enum
__all__ = ["MaterialMeta"]

_file_dir = os.path.dirname(__file__)
_mat_dirs = sorted(glob.glob(os.path.join(_file_dir, "*.npy")))
_bp_corck_dir = _mat_dirs[0]
_bp_peno_dir = _mat_dirs[1]
_bp_pvc_dir = _mat_dirs[2]


class MaterialMeta(Enum):
    PVC = _bp_pvc_dir
    penoplex = _bp_peno_dir
    cork = _bp_corck_dir