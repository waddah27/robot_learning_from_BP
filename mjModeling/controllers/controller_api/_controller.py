from abc import ABC, abstractmethod

__all__ = ["Controller"]


class Controller(ABC):
    opt_max_steps = 8000
    def __init__(self):
        super().__init__()

    @abstractmethod
    def move_to_position(self, *args, **kwargs): ...
