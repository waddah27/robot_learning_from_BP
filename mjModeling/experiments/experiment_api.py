from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self):
        super().__init__()
        self._controller = None

    @abstractmethod
    def execute(self, *args, **kwargs): ...
    
    @property
    def controller(self): ...
    
    @controller.setter
    def controller(self, *args, **kwargs): ...
