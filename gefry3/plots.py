import numpy as np

import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Polygon
# from mpl_toolkits.mplot3d import Axes3D

__all__ = ["render_problem"]

def render_problem(p, ax):
    solids = p.domain.solids
    source = p.source
    detectors = p.detectors

    # Generate the patches for the buildings
    building_patches = [
        Polygon(s.vertices)
        for s in solids
    ]

    for patch in building_patches:
        ax.add_patch(patch)
