import os
import glob
from enum import Enum
import numpy as np

__all__ = ["MaterialMeta", "bpTrajDataLoader"]

_file_dir = os.path.dirname(__file__)
_mat_dirs = sorted(glob.glob(os.path.join(_file_dir, "*.npy")))
_bp_corck_dir = _mat_dirs[0]
_bp_peno_dir = _mat_dirs[1]
_bp_pvc_dir = _mat_dirs[2]


class MaterialMeta(Enum):
    PVC = _bp_pvc_dir
    penoplex = _bp_peno_dir
    cork = _bp_corck_dir


class bpTrajDataLoader:
    def __init__(self, data_gmr: np.ndarray):
        _pos_idxs = np.s_[0:3]
        _vel_idxs = np.s_[12:15]
        _force_idxs = np.s_[6:9]
        _pos = data_gmr[: , _pos_idxs]
        # Transform ZXY to ZYX
        self.pos = _pos[:, [1,0,2]]
        self.vel = data_gmr[:, _vel_idxs]
        self.force = data_gmr[:, _force_idxs]
        self.F_max = np.max(self.force, axis=0)
        self.F_min = np.min(self.force, axis=0)
        self.step = 0
        self.max_steps = len(self.pos)
        self._shape = data_gmr.shape
        self.state = None
    
    def __len__(self):
        return self.max_steps
    
    def __iter__(self):
        self.step = 0
        return self
    
    def __next__(self):
        if self.step >= self.max_steps: raise StopIteration
        self.state = {
            "pos": self.pos[self.step],
            "vel": self.vel[self.step],
            "force": self.force[self.step],
            "index": self.step,
            "progress": (self.step +1) / self.max_steps
            
        }
        self.step += 1
        return self.state.get("pos"), self.state.get("vel"), self.state.get("force")
    
    @property
    def shape(self):
        return self._shape
    
    def get_window(self, start, end):
        return {
            "pos": self.pos[start:end],
            "vel": self.vel[start:end],
            "force": self.force[start:end]
        }


class MaterialData:
    PVC = np.load(MaterialMeta.PVC.value)
    penoplex = np.load(MaterialMeta.penoplex.value)
    cork = np.load(MaterialMeta.cork.value)
