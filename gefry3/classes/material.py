# from collections import namedtuple
import numpy as np

__all__ = ["Material"]

# Material = namedtuple("Material", ["name", "number_dens", "sigma_t", "Sigma_T"])

class Material(object):
    def __init__(self, Sigma_T):
        self.Sigma_T = Sigma_T

    def __init__(self, number_dens, sigma_t):
        self.Sigma_T = number_dens * sigma_t

    def alpha(self, dx):
        return np.exp(-self.Sigma_T * dx)
