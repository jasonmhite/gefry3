# from collections import namedtuple
import numpy as np

from gefry3.classes.meta import Dictable

__all__ = ["Material"]

# Material = namedtuple("Material", ["name", "number_dens", "sigma_t", "Sigma_T"])

class Material(Dictable):
    # def __init__(self, Sigma_T):
        # self.number_dens = None
        # self.sigma_t = None
        # self.Sigma_T = Sigma_T

    def __init__(self, number_dens, sigma_t):
        self.number_dens = np.float64(number_dens)
        self.sigma_t = np.float64(sigma_t)
        self.Sigma_T = number_dens * sigma_t

    def _as_dict(self):
        return {"number_dens": self.number_dens, "sigma_t": self.sigma_t}

    @classmethod
    def _from_dict(cls, data):
        return cls(data["number_dens"], data["sigma_t"])

    def __eq__(self, other):
        return self.number_dens == other.number_dens and self.sigma_t == other.sigma_t
