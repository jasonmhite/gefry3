import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

__all__ = ["render_patches", "set_bounds", "plot_response", "plot_hist_results"]

def build_patches(p, **patchargs):
    solids = p.domain.solids
    source = p.source
    detectors = p.detectors

    # Generate the patches for the buildings
    building_patches = [
        Polygon(s.vertices, **patchargs)
        for s in solids
    ] 

    return building_patches

def render_patches(p, ax=None):
    if ax is None:
        ax = plt.gca() 
    
    building_patches = build_patches(p)

    for patch in building_patches:
        ax.add_patch(patch)

def set_bounds(p, ax=None):
    if ax is None:
        ax = plt.gca()

    xl, yl, xh, yh = p.domain.bbox.bounds

    ax.set_xlim([xl, xh])
    ax.set_ylim([yl, yh])

# TODO: detectors and sources

def plot_response(p, response, bar_width=1, source_loc=None, fig=None):
    if fig is None:
        fig = plt.gcf()
        
    if source_loc is None:
        source_loc = p.source.R
        
    detector_locs = np.array([i.R for i in p.detectors])
    
    ax = fig.add_subplot(111, projection='3d')
    
    patches = build_patches(p, alpha=0.3)
    
    for patch in patches:
        ax.add_patch(patch)
        art3d.patch_2d_to_3d(patch)    
        
    ax.bar3d(detector_locs[:, 0], detector_locs[:, 1], np.zeros_like(response), bar_width, bar_width, response, color="#00ceaa")
    ax.scatter(*detector_locs.T)
    
    ax.scatter([source_loc[0]], [source_loc[1]], marker='*', color="red")
    
    set_bounds(p, ax=ax)
        
    ax.set_zlim([0, 1.1 * max(response)])

def plot_hist_results(data, source_loc, gridsize, bins):
    # gridsize = 20 and bins = 31 work well

    g = sb.jointplot(
        data[:, 0],
        data[:, 1],
        kind="hex",
        stat_func=None,
        marginal_kws={"bins": bins},
        joint_kws={"gridsize": gridsize},
        space=0,
    )

    g.set_axis_labels("x (m)", "y (m)")

    g.ax_joint.axvline([source_loc[0]], color="red", linestyle="--", alpha=0.5)
    g.ax_joint.axhline([source_loc[1]], color="red", linestyle="--", alpha=0.5)

    g.ax_marg_x.axvline([source_loc[0]], color="red", linestyle="--", alpha=0.5)
    g.ax_marg_y.axhline([source_loc[1]], color="red", linestyle="--", alpha=0.5)

    def set_ax_lim(xmin, ymin, xmax, ymax):
        g.ax_joint.set_xlim(xmin, xmax)
        g.ax_joint.set_ylim(ymin, ymax)
        g.ax_marg_x.set_xlim(xmin, xmax)
        g.ax_marg_y.set_ylim(ymin, ymax) 

    def set_ax_lim_to_problem(p):
        set_ax_lim(*p.domain.bbox.bounds)

    def set_ax_lim_to_stddev(dx, dy):
        c_x, c_y = np.mean(data, axis=0) 
        s_x, s_y = np.std(data, axis=0)

        xmin, ymin, xmax, ymax = c_x - dx * s_x, c_y - dy * s_y, c_x + dx * s_x, c_y + dx * s_x

        set_ax_lim(xmin, ymin, xmax, ymax)

        return xmin, ymin, xmax, ymax

    def set_ax_lim_about_mean(xl, yl, xr, yr):
        c_x, c_y = np.mean(data, axis=0)

        xmin, ymin, xmax, ymax = c_x - xl, c_y - yl, c_x + xr, c_y + yr 

        set_ax_lim(xmin, ymin, xmax, ymax)

        return xmin, ymin, xmax, ymax

    g.set_ax_lim = set_ax_lim
    g.set_ax_lim_to_problem = set_ax_lim_to_problem
    g.set_ax_lim_to_stddev = set_ax_lim_to_stddev
    g.set_ax_lim_about_mean = set_ax_lim_about_mean

    return g
