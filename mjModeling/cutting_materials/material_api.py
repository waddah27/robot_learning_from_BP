from data import MaterialData
from mjModeling.conf import workPiece
import numpy as np
__all__ = ["Material"]
class Material:
    def __init__(self):
        super().__init__()
        self._cut_resistance = None
        self._surface_height = None
        self._size = None
        self._center = None
        self._name = None
        self._is_solid = None
        self._is_movable = None
        self._bp_data = None

    @classmethod
    def from_work_piece(cls):
        if not workPiece:
            raise ValueError(f"class {workPiece.__name__} was not found")
        # obj = cls.__new__(cls)
        obj = cls()
        obj.cut_resistance = workPiece.MATERIAL_RESISTANCE
        obj.surface_height = workPiece.SURFACE
        obj.size = workPiece.SIZE
        obj.center = workPiece.POS
        obj.name = workPiece.MATERIAL_NAME
        obj.is_solid = workPiece.MATERIAL_IS_SOLID
        obj.is_movable = workPiece.MOVABLE
        return obj

    @property
    def cut_resistance(self):
        return self._cut_resistance

    @cut_resistance.setter
    def cut_resistance(self, value):
        self._cut_resistance = value

    @property
    def surface_height(self):
        return self._surface_height

    @surface_height.setter
    def surface_height(self, value):
        self._surface_height = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, val:np.array):
        if not isinstance(val, np.ndarray):
            raise TypeError(f"size must be np.ndarray of shape (3,), got {type(val)}")
        if val.shape != (3,):
            raise ValueError(f"material size is expected of shape (3,), got {val.shape}")
        self._size = val

    @property
    def center(self):
        return self._center

    @center.setter
    def center(self, val:np.array):
        if not isinstance(val, np.ndarray):
            raise TypeError(f"material center must be np.ndarray of shape (3,), got {type(val)}")
        if val.shape != (3,):
            raise ValueError(f"material size is expected of shape (3,), got {val.shape}")
        self._center = val

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, val: str):
        if isinstance(val, str):
            self._name = val
        else:
            raise ValueError(f"expected a string, got {type(val)}")

    @property
    def is_solid(self):
        return self._is_solid

    @is_solid.setter
    def is_solid(self, val: bool):
        if isinstance(val, bool):
            self._is_solid = val
        else:
            raise ValueError(f"is_solid is expected to be a bool, got {type(val)}")

    @property
    def is_movable(self):
        return self._is_movable

    @is_movable.setter
    def is_movable(self, val: bool):
        if isinstance(val, bool):
            self._is_movable = val
        else:
            raise ValueError(f"is_solid is expected to be a bool, got {type(val)}")

    @property
    def bp_data(self):
        return self._bp_data

    @bp_data.setter
    def bp_data(self, val: str):
        if not isinstance(val, str):
            raise ValueError(f"bp_data is expecting str name got {type(val)}")
        match val:
            case MaterialData.cork.name:
                self._bp_data = MaterialData.cork
            case MaterialData.penoplex.name:
                self._bp_data = MaterialData.penoplex
            case MaterialData.PVC.name:
                self._bp_data = MaterialData.PVC
            case _:
                raise NotImplementedError(f"material name {val} was not found!")

