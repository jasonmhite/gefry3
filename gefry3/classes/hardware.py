import numpy as np
from gefry3.classes import *
from gefry3.classes.meta import Dictable

__all__ = ["Source", "Detector"]

class Source(Dictable):
    def __init__(self, R, I0):
        self.R = R
        self.I0 = I0

    def _as_dict(self):
        return {"R": self.R, "I0": self.I0}

    @classmethod
    def _from_dict(cls, data):
        return cls(data["R"], data["I0"])

class Detector(Dictable):
    def __init__(self, R, epsilon, area, dwell):
        self.R = R
        self.epsilon = epsilon
        self.area = area
        self.dwell = dwell

    def compute_response(self, I):
        return I * self.area * self.dwell * self.epsilon

    def _as_dict(self):
        return {
            "R": self.R,
            "epsilon": self.epsilon,
            "area": self.area,
            "dwell": self.dwell,
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(
            data["R"],
            data["epsilon"],
            data["area"],
            data["dwell"],
        )
