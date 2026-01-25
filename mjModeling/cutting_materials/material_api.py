from abc import ABC, abstractmethod
__all__ = ["Material"]
class Material(ABC):
    def __init__(self):
        super().__init__()
        self._cut_resistance = None
        self._surface_hight = None

    @property
    def cut_resistance(self):
        return self._cut_resistance
    @cut_resistance.setter
    def cut_resistance(self, value):
        self._cut_resistance = value

    @property
    def surface_hight(self):
        return self._surface_hight

    @surface_hight.setter
    def surface_hight(self, value):
        self._surface_hight = value