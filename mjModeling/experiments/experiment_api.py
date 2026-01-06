from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def execute(self): ...
