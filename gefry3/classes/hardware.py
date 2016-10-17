import numpy as np
from gefry3.classes import *

__all__ = ["Source", "Detector"]

class Source(object):
    def __init__(self, R, I0):
        self.R = R
        self.I0 = I0

class Detector(object):
    def __init__(self, R, epsilon, area, dwell):
        self.R = R
        self.epsilon = epsilon
        self.area = area
        self.dwell = dwell

    def compute_response(self, I):
        return I * self.area * self.dwell * self.epsilon
