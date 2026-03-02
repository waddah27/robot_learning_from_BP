import os
import glob
from enum import Enum
import numpy as np
import inspect

__all__ = ["MaterialData", "bpTrajDataLoader", "NamedArray"]

_file_dir = os.path.dirname(__file__)
_mat_dirs = sorted(glob.glob(os.path.join(_file_dir, "*.npy")))
_bp_corck_dir = _mat_dirs[0]
_bp_peno_dir = _mat_dirs[1]
_bp_pvc_dir = _mat_dirs[2]


class NamedArray:
    def __init__(self, data_or_path):
        self._name = None
        self._data = data_or_path if isinstance(data_or_path, np.ndarray) else None
        self.path = data_or_path if isinstance(data_or_path, str) else None

    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        return self.data

    @property
    def data(self):
        if self._data is None and self.path:
            self._data = np.load(self.path)
        return self._data

    @property
    def name(self):
        if self._name is not None:
            return self._name
        # Look at the caller of the method accessing this property
        frame = inspect.currentframe().f_back.f_back
        # 1. Search Locals (Functions/Methods)
        if self._helper_lookup_name(frame.f_locals):
            return self._name
        # 2. Search Globals (Module level)
        if self._helper_lookup_name(frame.f_globals):
            return self._name
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        if self._helper_lookup_name(frame.f_locals):
            return self._name
        if self._helper_lookup_name(frame.f_globals):
            return self._name
        return None

    def _helper_lookup_name(self, dic: dict):
        for name, val in dic.items():
            if val is self:
                self._name = name
                return self._name
        return None


class MaterialMeta(Enum):
    PVC = _bp_pvc_dir
    penoplex = _bp_peno_dir
    cork = _bp_corck_dir


class MaterialData:
    PVC = NamedArray(MaterialMeta.PVC.value)
    penoplex = NamedArray(MaterialMeta.penoplex.value)
    cork = NamedArray(MaterialMeta.cork.value)


class bpTrajDataLoader:
    def __init__(self, data_gmr: NamedArray):
        _pos_idxs = np.s_[0:3]
        _vel_idxs = np.s_[12:15]
        _force_idxs = np.s_[6:9]
        _pos = data_gmr.data[:, _pos_idxs]
        # Transform ZXY to ZYX
        self.pos = _pos#[:, [1, 0, 2]]
        self.vel = data_gmr.data[:, _vel_idxs]
        self.force = data_gmr.data[:, _force_idxs]
        self.F_max = np.max(self.force, axis=0)
        self.F_min = np.min(self.force, axis=0)
        self.step = 0
        self.max_steps = len(self.pos)
        self._shape = data_gmr.data.shape
        self._name = data_gmr.name
        self.state = None

    def __len__(self):
        return self.max_steps

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step >= self.max_steps:
            raise StopIteration
        self.state = {
            "pos": self.pos[self.step],
            "vel": self.vel[self.step],
            "force": self.force[self.step],
            "index": self.step,
            "progress": (self.step + 1) / self.max_steps
        }
        self.step += 1
        return self.state.get("pos"), self.state.get("vel"), self.state.get("force")

    @property
    def shape(self):
        return self._shape

    @property
    def material_name(self):
        return self._name

    def get_window(self, start, end):
        return {
            "pos": self.pos[start:end],
            "vel": self.vel[start:end],
            "force": self.force[start:end]
        }


if __name__ == '__main__':
    A = MaterialData.cork
    print(A.data.shape)
    print(A.name)
