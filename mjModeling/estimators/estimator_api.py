from abc import ABC, abstractmethod

from mjModeling.controllers.controller_api import Controller
from mjModeling.mjRobot import Robot


class Estimator(ABC):
    def __init__(self):
        super().__init__()
        self._robot = None
        self._controller = None

    @abstractmethod
    def execute(self, *args, **kwargs): ...

    @property
    def robot(self):
        return self._robot

    @robot.setter
    def robot(self, robot: Robot):
        self._robot = robot

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, controller: Controller):
        self._controller = controller