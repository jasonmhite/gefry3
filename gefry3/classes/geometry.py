import shapely.geometry as G
import shapely.ops as O
import numpy as np

from gefry3.classes.meta import Dictable

__all__ = ["Solid", "Domain"]

class Solid(Dictable):
    def __init__(self, vertices):
        self.vertices = vertices

        self.geom = G.Polygon(vertices)

    def find_path_length(self, L): # Length of path along a -> b intersecting the solid
        # L = g.LineString([a, b])

        intersect = L.intersection(self.geom)

        if intersect.is_empty:
            return 0.0
        else:
            return intersect.length

    def _as_dict(self):
        return {"vertices": self.vertices}

    @classmethod
    def _from_dict(cls, data):
        return cls(data["vertices"])

class Domain(Dictable):
    def __init__(self, bbox, solids):
        self.solids = solids
        self.bbox_verts = bbox
        self.bbox = G.Polygon(bbox)

        self.all = O.cascaded_union([S.geom for S in self.solids])

        # Check bounding box is a bounding box
        assert(self.all.difference(self.bbox).is_empty)

        self.empty = self.bbox.difference(self.all)

    def construct_path(self, a, b):
        L = G.LineString([a, b])

        Li = L.intersection(self.empty)
        return np.array([Li.length] + [S.find_path_length(L) for S in self.solids])

    def _as_dict(self):
        solids = [i._as_dict() for i in self.solids]

        return {"bbox": self.bbox_verts, "solids": solids}

    @classmethod
    def _from_dict(cls, data):
        solids = [Solid._from_dict(i) for i in data["solids"]]

        return cls(data["bbox"], solids)
