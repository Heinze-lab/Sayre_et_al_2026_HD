import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm

from .model import Model

def add_pva_subplot(fig: plt.Figure, pos = 111) -> plt.Axes:
    ax = fig.add_subplot(pos, projection="polar")
    ax.set_aspect("equal")
    ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False), labels=["L5/R5", "L6/R4", "L7/R3", "L8/R2", "L1/R1", "L2/R8", "L3/R7", "L4/R6"])
    return ax

def plot_trajectories(ax, time, trajectories, cmap="plasma"):
    norm = plt.Normalize(0, np.sqrt(np.max(time)))
    segments = []
    for trajectory in trajectories:
        segments.append([
            (np.angle(trajectory[i]), np.abs(trajectory[i])), (np.angle(trajectory[i+1]), np.abs(trajectory[i+1]))
        ])
    ax.add_collection(LineCollection(segments, color=cmap(norm(np.sqrt(t))), zorder=i))

def plot_weights(weights, labels = None, ax = None, figsize=None, **kwargs):
    with InAxes(ax, figsize=figsize) as ax:
        if labels is not None:
            ax.set_xticks(np.arange(0, len(labels)), labels, fontsize=2, rotation=90)
            ax.set_yticks(np.arange(0, len(labels)), labels, fontsize=2)
        norm = TwoSlopeNorm(vcenter=0)
        c = ax.imshow(weights, interpolation="nearest", rasterized=True, **{ "cmap": "bwr_r", "norm": norm, **kwargs})
        plt.colorbar(c)
        return c

def plot_grouped_weights(weights, labels = None, ax = None, **kwargs):
    with InAxes(ax) as ax:
        if labels is not None:
            ax.set_xticks(np.arange(0, len(labels)), labels, fontsize=6, rotation=90)
            ax.set_yticks(np.arange(0, len(labels)), labels, fontsize=6)
        norm = TwoSlopeNorm(vcenter=0)
        c = ax.imshow(weights, interpolation="nearest", rasterized=True, **{ "cmap": "bwr_r", "norm": norm, **kwargs})
        cbar = plt.colorbar(c)
        return c, cbar

def plot_activity(model: Model, states: np.array, ax = None):
    with InAxes(ax) as ax:
        ax.imshow(states, vmin=0, vmax=1, interpolation="nearest")
        label_yaxis(model, ax)

def plot_activity_transpose(model: Model, states: np.array, ax = None):
    with InAxes(ax) as ax:
        ax.imshow(states.T, vmin=0, vmax=1)
        label_xaxis(model, ax)

def label_xaxis(model: Model, ax = None):
    if ax is None:
        ax = plt.gca()
    
    ax.set_xticks(np.arange(len(model.cells)), model.cell_names(), fontsize=5, rotation=90)

def label_yaxis(model: Model, ax = None):
    if ax is None:
        ax = plt.gca()
    
    ax.set_yticks(np.arange(len(model.cells)), model.cell_names(), fontsize=5)

class InAxes:
    def __init__(self, ax, figsize=None):
        if ax == None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = plt.subplot()
        else:
            self.fig = None
            self.ax = ax

    def __enter__(self):
        return self.ax

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.fig is not None:
            #self.ax.legend()
            plt.show()