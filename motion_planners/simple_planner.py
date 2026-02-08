from data import bpTrajDataLoader
__all__ = ["MotionPlanner"]


class MotionPlanner:
    def __init__(self, data: bpTrajDataLoader) -> None:
        self.data = data

    def go_to_next(self):
        """ Generator that yields position, velocity, and force at each step.
        """
        for d in self.data:
            yield d

