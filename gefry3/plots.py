import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

# NOTE: These are some tools I've made to make specific plots that I use
# regularly. The options and variations are probably specific to the
# exact sorts of plots I need, so maybe you should consider these more
# as examples than something to use directly.

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

def render_patches(p, ax=None, patchargs={}):
    if ax is None:
        ax = plt.gca() 
    
    building_patches = build_patches(p, **patchargs)

    for patch in building_patches:
        ax.add_patch(patch)

def set_bounds(p, ax=None):
    if ax is None:
        ax = plt.gca()

    xl, yl, xh, yh = p.domain.bbox.bounds

    ax.set_xlim([xl, xh])
    ax.set_ylim([yl, yh])

# TODO: detectors and sources



class GefryJointHistogram(sb.axisgrid.JointGrid):
    def __init__(self, other, data):
        if isinstance(other, sb.axisgrid.JointGrid):
            self.__dict__ = other.__dict__.copy()

        self.data = data

    def set_ax_lim(self, xmin, ymin, xmax, ymax):
        self.ax_joint.set_xlim(xmin, xmax)
        self.ax_joint.set_ylim(ymin, ymax)
        self.ax_marg_x.set_xlim(xmin, xmax)
        self.ax_marg_y.set_ylim(ymin, ymax)  

    def set_ax_lim_to_problem(self, p):
        self.set_ax_lim(*p.domain.bbox.bounds)

    def set_ax_lim_to_stddev(self, dx, dy):
        c_x, c_y = np.mean(self.data, axis=0) 
        s_x, s_y = np.std(self.data, axis=0)

        xmin, ymin, xmax, ymax = c_x - dx * s_x, c_y - dy * s_y, c_x + dx * s_x, c_y + dx * s_x

        self.set_ax_lim(xmin, ymin, xmax, ymax)

        return xmin, ymin, xmax, ymax

    def set_ax_lim_about_mean(self, xl, yl, xr, yr):
        c_x, c_y = np.mean(self.data, axis=0)

        xmin, ymin, xmax, ymax = c_x - xl, c_y - yl, c_x + xr, c_y + yr 

        self.set_ax_lim(xmin, ymin, xmax, ymax)

        return xmin, ymin, xmax, ymax 

def plot_response(
    p,
    response,
    bar_width=1,
    source_loc=None,
    fig=None,
    offset=(0.0, 0.0),
    color="#00ceaa",
    draw_extras=True,
    ax=None,
    label=None,
    set_zlim=True,
):
    if fig is None:
        fig = plt.gcf()
        
    if source_loc is None:
        source_loc = p.source.R

    dx, dy = offset
        
    detector_locs = np.array([i.R for i in p.detectors])
    
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
        
    ax.bar3d(detector_locs[:, 0] + dx, detector_locs[:, 1] + dy, np.zeros_like(response), bar_width, bar_width, response, color=color, edgecolor=color)
    if draw_extras:
        ax.scatter([source_loc[0]], [source_loc[1]], marker='*', color="red", label="Source")
        ax.scatter(*detector_locs.T, label="Detector")
        patches = build_patches(p, alpha=0.3)
        
        for patch in patches:
            ax.add_patch(patch)
            art3d.patch_2d_to_3d(patch)        

        set_bounds(p, ax=ax)
        
    if set_zlim:
        ax.set_zlim([0, 1.1 * max(response)])

    return (fig, ax)

def lazy_mode(data, bins=100):
    w, edges = np.histogram(data, bins=bins)
    
    wmax_ind = w.argmax()
    mode = 0.5 * (edges[wmax_ind + 1] + edges[wmax_ind])

    return mode 

def plot_hist_results(data, source_loc, gridsize, bins, draw_loc=True, draw_mean=False, use_plus=True, draw_mode=True, mode_bins=100):
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

    if draw_loc:
        if use_plus:
            g.ax_joint.scatter([source_loc[0]], [source_loc[1]], marker="+", color="red", zorder=10, s=500, linewidths=1.5, label="Source Location")
        else:
            g.ax_joint.axvline([source_loc[0]], color="red", linestyle="--", alpha=0.5)
            g.ax_joint.axhline([source_loc[1]], color="red", linestyle="--", alpha=0.5)

        g.ax_marg_x.axvline([source_loc[0]], color="red", linestyle="--", alpha=0.5)
        g.ax_marg_y.axhline([source_loc[1]], color="red", linestyle="--", alpha=0.5)

    if draw_mean:
        m = np.mean(data, axis=0)

        if draw_mode:
            c = "blue"
        else:
            c = "black"
        if use_plus:
            g.ax_joint.scatter([m[0]], [m[1]], marker="+", color=c, zorder=10, s=500, linewidths=1.5, label="Posterior Mean")
        else:
            g.ax_joint.axvline([m[0]], color=c, linestyle="--", alpha=0.5)
            g.ax_joint.axhline([m[1]], color=c, linestyle="--", alpha=0.5) 

        g.ax_marg_x.axvline([m[0]], color=c, linestyle="--", alpha=0.5)
        g.ax_marg_y.axhline([m[1]], color=c, linestyle="--", alpha=0.5) 

    if draw_mode:
        m = (lazy_mode(data[:, 0], bins=mode_bins), lazy_mode(data[:, 1], bins=mode_bins))

        if use_plus:
            g.ax_joint.scatter([m[0]], [m[1]], marker="+", color="black", zorder=10, s=500, linewidths=1.5, label="Posterior Mode")
        else:
            g.ax_joint.axvline([m[0]], color="black", linestyle="--", alpha=0.5)
            g.ax_joint.axhline([m[1]], color="black", linestyle="--", alpha=0.5) 

        g.ax_marg_x.axvline([m[0]], color="black", linestyle="--", alpha=0.5)
        g.ax_marg_y.axhline([m[1]], color="black", linestyle="--", alpha=0.5)  

    g = GefryJointHistogram(g, data)

    return g
