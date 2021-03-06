import numpy as np
from gefry3.classes import *
from gefry3.classes.meta import Dictable

try:
    import pyst
    PYST_AVAIL = True
except ImportError:
    PYST_AVAIL = False

from shapely.geometry import MultiPoint, LineString, Polygon
from functools import partial
from collections import defaultdict

__all__ = ["Source", "Detector", "detectorRegistry"]

if PYST_AVAIL:
    __all__.append("OrientedPrismDetector")

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
        self.R = np.array(R, dtype=np.float64)
        self.epsilon = np.float64(epsilon)
        self.area = np.float64(area)
        self.dwell = np.float64(dwell)

    def compute_response(self, I, r):
        r = np.array(r, dtype=np.float64)
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

if PYST_AVAIL:
    class OrientedPrismDetector(Detector):
        # derives from detector but it's gonna override everything

        def __init__(self, R, L, theta, epsilon, dwell):
            """
            R is the coordinate of the center of mass
            L is [l, w, h] dimensions
            Theta is rotation in radians ccw from the x-axis.

            Detector will be placed s.t. the COM is coplanar to the source.
            """

            self.R = np.array(R, dtype=np.float64)
            self._R3 = vec_2d_to_3d(self.R)
            self.dims = np.array(L, dtype=np.float64)
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

            # Hrm... maybe use center = r, they should be the same...
            self.corner = self.vertices[0]
            self.center = self.vertices[0] + self._rotation_matrix(self.dims) / 2.

            # Vectors defining the prism
            self.r_x, self.r_y, self.r_z = self._rotation_matrix(np.diag([l, w, h]))

            # Assuming they are coplanar we can skip the top and bottom to save computation
            self.facets = [
                pyst.RectangularFacet(
                    self.corner,
                    self.r_x,
                    self.r_z,
                    sense=1.0,
                    name="Front",
                ),
                pyst.RectangularFacet(
                    self.corner + self.r_y,
                    self.r_x,
                    self.r_z,
                    sense=-1.0,
                    name="Back",
                ),
                # pyst.RectangularFacet(corner + self.r_z, self.r_x, self.r_y, sense=1.0, name="Top"),
                # pyst.RectangularFacet(corner, self.r_x, self.r_y, sense=-1.0, name="Bottom"),
                pyst.RectangularFacet(
                    self.corner + self.r_x,
                    self.r_y,
                    self.r_z,
                    sense=1.0,
                    name="Right",
                ),
                pyst.RectangularFacet(
                    self.corner,
                    self.r_y,
                    self.r_z,
                    sense=-1.0,
                    name="Left",
                ),
            ]

        def omega(self, r):
            r = np.array(r, dtype=np.float64)
            r3 = vec_2d_to_3d(r)

            return sum([facet(r3) for facet in self.facets if facet.is_facing(r3)])

        def compute_response(self, I, r):
            r = np.array(r, dtype=np.float64)
            I = np.float64(I)

            dr = np.linalg.norm(self.R - r)
            beta = self.omega(r) / (4. * np.pi)

            return I * beta * self.dwell * self.epsilon

        def _as_dict(self):
            return {
                "R": self.R,
                "lwh": self.dims.tolist(),
                "theta": self.theta,
                "epsilon": self.epsilon,
                "dwell": self.dwell,
            }

        @classmethod
        def _from_dict(cls, data):
            return cls(
                data["R"],
                data["lwh"],
                data["theta"],
                data["epsilon"],
                data["dwell"],
            )


    class OrientedPrismDetectorIntrinsic(OrientedPrismDetector):

        def __init__(self, R, L, theta, sigma_det, dwell):
            # sigma_det is the detection macro cross section, in m^-1
            super().__init__(R, L, theta, 1.0, dwell)

            self.sigma_det = sigma_det
            self.profile = Polygon([
                self.vertices[0, :2],
                self.vertices[1, :2],
                self.vertices[2, :2],
                self.vertices[3, :2]
            ])

            # Extend an intersecting ray by this much to make sure
            # it always travels the full extent of the detector
            self._length_extension = self.dims.max()

        def compute_intrinsic(self, R):
            # source to detector angle
            dR = self.center[:2] - R
            theta = np.arctan2(dR[1], dR[0])

            # extend the ray a bit to make sure it crosses the full
            # length of the detector
            extension = self._length_extension * np.array([
                np.cos(theta),
                np.sin(theta),
            ])

            L = LineString([R, self.center[:2] + extension])
            dL = self.profile.intersection(L)

            return 1.0 - np.exp(-self.sigma_det * dL.length)

        def compute_response(self, I, r):
            r = np.array(r, dtype=np.float64)
            I = np.float64(I)

            dr = np.linalg.norm(self.R - r)
            beta = self.omega(r) / (4. * np.pi)

            return I * beta * self.dwell * self.compute_intrinsic(r)

        def _as_dict(self):
            return {
                "R": self.R,
                "lwh": self.dims.tolist(),
                "theta": self.theta,
                "sigma_det": self.sigma_det,
                "dwell": self.dwell,
            }

        @classmethod
        def _from_dict(cls, data):
            return cls(
                data["R"],
                data["lwh"],
                data["theta"],
                data["sigma_det"],
                data["dwell"],
            ) 

detectorRegistry = {
    "Point": Detector,
}

if PYST_AVAIL:
    detectorRegistry["Oriented_Prism"] = OrientedPrismDetector
    detectorRegistry["Oriented_Prism_Intrinsic"] = OrientedPrismDetectorIntrinsic
