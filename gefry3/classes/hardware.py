import numpy as np
import pyst
from gefry3.classes import *
from gefry3.classes.meta import Dictable

from shapely.geometry import MultiPoint, LineString
from functools import partial


__all__ = ["Source", "Detector", "OrientedPrismDetector"]

class Source(Dictable):
    def __init__(self, R, I0):
        self.R = np.array(R)
        self.I0 = np.float64(I0)

    def _as_dict(self):
        return {"R": self.R, "I0": self.I0}

    @classmethod
    def _from_dict(cls, data):
        return cls(data["R"], data["I0"])

class Detector(Dictable):
    def __init__(self, R, epsilon, area, dwell):
        self.R = np.array(R)
        self.epsilon = np.float64(epsilon)
        self.area = np.float64(area)
        self.dwell = np.float64(dwell)

    def compute_response(self, I, r):
        r = np.array(r)
        I = np.float64(I)

        dr = np.linalg.norm(self.R - r)
        beta = 4 * np.pi * (np.linalg.norm(dr) ** 2)

        return I * self.area * self.dwell * self.epsilon / beta

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

def vec_2d_to_3d(x, val=0.0):
    return np.hstack((x, np.atleast_1d(val)))

class OrientedPrismDetector(Detector):
    # derives form detector but it's gonna override everything

    def __init__(self, R, L, theta, epsilon, dwell):
        """
        R is the coordinate of the center of mass
        L is [l, w, h] dimensions
        Theta is rotation in radians ccw from the x-axis.

        Detector will be placed s.t. the COM is coplanar to the source.
        """

        self.R = np.array(R)
        self._R3 = vec_2d_to_3d(self.R)
        self.dims = np.array(L)
        self.theta = np.float64(theta)
        self.dwell = np.float64(dwell)
        self.epsilon = np.float64(epsilon)
        self.d = np.linalg.norm(self.R)

        l, w, h = self.dims

        self.vertices = np.array([
            [0, 0, 0],
            [l, 0, 0],
            [l, w, 0],
            [0, w, 0],

            [0, 0, h],
            [l, 0, h],
            [l, w, h],
            [0, w, h],
        ])

        # Center because I gave the coordinates as corner-relative
        self.vertices -= self.dims / 2.

        self._rotation_matrix = pyst.RotationMatrix \
            . rot(self.theta, direction="z")

        self.vertices = self._rotation_matrix(self.vertices)
        self.vertices += self._R3

        corner = self.vertices[0]
        self.center = self.vertices[0] + self._rotation_matrix(self.dims) / 2.
        
        # Vectors defining the prism
        self.r_x, self.r_y, self.r_z = self._rotation_matrix(np.diag([l, w, h]))

        # Assuming they are coplanar we can skip the top and bottom to save computation
        self.facets = [
            pyst.RectangularFacet(corner, self.r_x, self.r_z, sense=1.0, name="Front"),
            pyst.RectangularFacet(corner + self.r_y, self.r_x, self.r_z, sense=-1.0, name="Back"),
            # pyst.RectangularFacet(corner + self.r_z, self.r_x, self.r_y, sense=1.0, name="Top"),
            # pyst.RectangularFacet(corner, self.r_x, self.r_y, sense=-1.0, name="Bottom"),
            pyst.RectangularFacet(corner + self.r_x, self.r_y, self.r_z, sense=1.0, name="Right"),
            pyst.RectangularFacet(corner, self.r_y, self.r_z, sense=-1.0, name="Left"),
        ]
