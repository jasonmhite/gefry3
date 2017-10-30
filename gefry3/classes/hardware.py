import numpy as np
from gefry3.classes import *
from gefry3.classes.meta import Dictable

from shapely.geometry import MultiPoint, LineString
from functools import partial

__all__ = ["Source", "Detector", "OrientedPrismDetector"]

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

    def compute_response(self, I, r):
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

def rotationMatrix(theta):
    # Rotation matrix about z axis ccw from x axis

    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1],
    ])

def planarProjection(u, n):
    # Projection of u onto plane with normal n
    dn = np.linalg.norm(n, 2)

    return u - (u.dot(n) / (dn ** 2)) * n

# def sphereProjection(c, d, u):
    # # Project u onto the sphere of radius d centered at u.
    # # https://stackoverflow.com/questions/9604132/how-to-project-a-point-on-to-a-sphere
    # # for the lazy.
    # # c -> source location [3 vector]
    # # d -> source-detector distance
    # # u -> detector absolute position
    # return c + r * (u - c) / np.linalg.norm(u - c)

def sphereProjection(d, u):
    # https://stackoverflow.com/questions/9604132/how-to-project-a-point-on-to-a-sphere
    # for the lazy.
    # d -> source-detector distance
    # u -> detector relative position
    return d * u / np.linalg.norm(u) 

def vectorProjection(u, v):
    return u.dot(v) / np.linalg.norm(v)

class OrientedPrismDetector(Detector):
    # derives form detector but it's gonna override everything

    def __init__(self, R, L, theta, epsilon, dwell):
        """
        R is the coordinate of the center of mass
        L is [l, w, h] dimensions
        Theta is rotation in radians ccw from the x-axis

        Detector will be placed s.t. the COM is coplanar to the source.
        """

        self.R = np.array(R)
        self.dims = np.array(L)
        self.theta = theta
        self.dwell = dwell
        self.epsilon = epsilon
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

        # # Apply the rotation matrix to all vertices, see
        # # http://ajcr.net/Basic-guide-to-einsum/ because einsum is 
        # # pure sorcery.
        self.vertices = np.einsum(
            "ij,kj->ki",
            rotationMatrix(self.theta),
            self.vertices,
        )

        # Translate points to their absolute position
        self.vertices += np.append(self.R, 0)

    def _proj(self, r, sphere=False):
        # center points relative to source
        r_0 = self.R - r
        d_0 = np.linalg.norm(r_0)

        relativePoints = self.vertices - np.append(r, 0)

        # project points onto sphere
        sphereProjectedPoints = np.array([sphereProjection(d_0, x) for x in relativePoints])

        if sphere:
            return sphereProjectedPoints

        # now project them onto a plane to approximate the subtended area

        basis = np.array([
            [0, 0, 1],
            np.cross([0, 0, 1], np.append(r_0, 0))
        ])
        basis[1] /= np.linalg.norm(basis[1])

        verticesPlanar = np.einsum("ij,kj->ik", sphereProjectedPoints, basis)

        return verticesPlanar

        # return MultiPoint(verticesPlanar).convex_hull

    def _getViewFrom(self, r, outline=False):
        v = self._proj(r, sphere=False)

        if outline:
            return LineString([
                v[0],
                v[1],
                v[2],
                v[3],
                v[0],
                v[4],
                v[5],
                v[6],
                v[7],
                v[4],
                v[5],
                v[1],
                v[2],
                v[6],
                v[7],
                v[3],
            ])

        else:
            return MultiPoint(self._proj(r)).convex_hull

    def area(self, r):
        # r is absolute position of source
        return self._getViewFrom(r, outline=False).area
