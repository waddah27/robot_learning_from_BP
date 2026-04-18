from abc import ABC, abstractmethod

from mjModeling.conf.configs import paramVIC

__all__ = ["Controller"]


class Controller(ABC):
    opt_max_steps = paramVIC.VIC_MAX_STEPS
    def __init__(self):
        super().__init__()

    @abstractmethod
    def move_to_position(self, *args, **kwargs): ...
