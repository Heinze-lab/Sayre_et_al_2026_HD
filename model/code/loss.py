import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import math
import pickle
from itertools import product
from tqdm.notebook import tqdm
import sys
import os

from cx.model import *

CCW_ROTATION = 1.0
CW_ROTATION = -1.0

NUM_EQUILIBRIA = 0
MEAN_SPECTRAL_GAP = 1
BIAS = 2
AMP_MIN = 3
AMP_MAX = 4
AMP_VAR = 5
INTEGRATION_RANGE = 6
SPEED_MAX = 7
ANGULAR_SPEED_MAX = 8
CCW_SPEED_MIN = 9
CW_SPEED_MIN = 10
CCW_SPEED_VAR = 11
CW_SPEED_VAR = 12
MIN_ROTATION_AMP = 13
SPEED_VAR = 14

def compute_loss(model, data):
    filename = f"../results/data/tuning/{model.name}_loss.npy"
    if os.path.isfile(filename):
        return np.load(filename)

    parameters = list(product(data["Es"], data["Is"], data["biases"], data["compass_strengths"]))
    loss = np.nan * np.zeros((SPEED_VAR + 1, len(data["Es"]), len(data["Is"]), len(data["biases"]), len(data["compass_strengths"])))

    for i, (params, (equilibria, ring, ring_speed)) in enumerate(zip(tqdm(parameters), data["grid"])):
        index = np.unravel_index(i, loss[0,...].shape)
        
        if len(ring) == 0 or ring.shape[1] == 0:
            continue

        m = model.tune(params)
        n = ring.shape[1]

        if n < 190:
            continue

        pva = m.decode(equilibria)
        cutoff = 0.5 * np.max(np.abs(pva))
        ring_mask = np.abs(pva) > cutoff

        # Compute mean spectral gap of the equilibria on the ring
        mean_gap = 0
        for u in equilibria[:,ring_mask].T:
            eigvals = np.linalg.eigvals(m.jacobian(u, m.compass_current(0.0, 0.0)))
            real_indices = np.isreal(eigvals)
            eigvals = np.real(eigvals[real_indices])
            indices = np.argsort(eigvals)
            gap = eigvals[indices[-1]] - eigvals[indices[-2]]
            mean_gap += gap
        mean_gap /= np.sum(ring_mask)

        loss[MEAN_SPECTRAL_GAP,*index] = mean_gap

        loss[NUM_EQUILIBRIA,*index] = np.sum(ring_mask)
        pva = m.decode(ring)
        loss[BIAS,*index] = np.abs(np.sum(pva) / n)
        loss[AMP_MIN,*index] = np.min(np.abs(pva))
        loss[AMP_MAX,*index] = np.max(np.abs(pva))
        loss[AMP_VAR,*index] = np.var(np.abs(pva))
        loss[SPEED_MAX,*index] = np.max(ring_speed)
        loss[SPEED_VAR,*index] = np.var(ring_speed)

        dadt = m.dangledt(ring, m.dudt(ring, m.compass_current(np.zeros(ring.shape[1]), 0.0)))
        loss[ANGULAR_SPEED_MAX,*index] = np.max(np.abs(dadt))

        ccw_rotation = CCW_ROTATION
        cycle, is_cycle = trace_cycle(m, equilibria, rotation=ccw_rotation, dt=0.1, max_steps=10000)
        if is_cycle:
            #cycle_speed = np.array([np.linalg.norm(m.dudt(state, m.compass_current(np.zeros(state.shape[0]), 0.0))) for state in cycle.T])
            dadt = m.dangledt(cycle, m.dudt(cycle, m.compass_current(np.zeros(cycle.shape[1]), 0.0) + m.rotation_current(ccw_rotation * np.ones(cycle.shape[1]))))
            loss[CCW_SPEED_VAR,*index] = np.var(dadt)
            loss[CCW_SPEED_MIN,*index] = np.min(dadt)
            loss[MIN_ROTATION_AMP,*index] = np.min(np.abs(m.decode(cycle)))

        cw_rotation = CW_ROTATION
        cycle, is_cycle = trace_cycle(m, equilibria, rotation=cw_rotation, dt=0.1, max_steps=10000)
        if is_cycle:
            #cycle_speed = np.array([np.linalg.norm(m.dudt(state, m.compass_current(np.zeros(state.shape[0]), 0.0))) for state in cycle.T])
            dadt = m.dangledt(cycle, m.dudt(cycle, m.compass_current(np.zeros(cycle.shape[1]), 0.0) + m.rotation_current(cw_rotation * np.ones(cycle.shape[1]))))
            loss[CW_SPEED_VAR,*index] = np.var(dadt)
            loss[CW_SPEED_MIN,*index] = np.min(-dadt)
            loss[MIN_ROTATION_AMP,*index] = np.minimum(loss[MIN_ROTATION_AMP,*index], np.min(np.abs(m.decode(cycle))))

    np.save(filename, loss)
    return loss


def plot_loss(model, Es, Is, slice, best):
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$E$")
        ax.set_ylabel("$I$")

    X, Y = np.meshgrid(Es, Is)

    axes[NUM_EQUILIBRIA].set_title("equilibria count")
    c = axes[NUM_EQUILIBRIA].pcolormesh(X, Y, slice[NUM_EQUILIBRIA,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[MEAN_SPECTRAL_GAP].set_title("mean spectral gap")
    c = axes[MEAN_SPECTRAL_GAP].pcolormesh(X, Y, slice[MEAN_SPECTRAL_GAP,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[BIAS].set_title("ring bias")
    c = axes[BIAS].pcolormesh(X, Y, slice[BIAS,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[AMP_MIN].set_title("amplitude min")
    c = axes[AMP_MIN].pcolormesh(X, Y, slice[AMP_MIN,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[AMP_MAX].set_title("amplitude max")
    c = axes[AMP_MAX].pcolormesh(X, Y, slice[AMP_MAX,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[AMP_VAR].set_title("amplitude variance")
    c = axes[AMP_VAR].pcolormesh(X, Y, slice[AMP_VAR,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[INTEGRATION_RANGE].set_title("angular integration range")
    c = axes[INTEGRATION_RANGE].pcolormesh(X, Y, slice[INTEGRATION_RANGE,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[CCW_SPEED_VAR].set_title("counter-clockwise da/dt norm variance")
    c = axes[CCW_SPEED_VAR].pcolormesh(X, Y, slice[CCW_SPEED_VAR,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[CW_SPEED_VAR].set_title("clockwise da/dt norm variance")
    c = axes[CW_SPEED_VAR].pcolormesh(X, Y, slice[CW_SPEED_VAR,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[CCW_SPEED_MIN].set_title("counter-clockwise da/dt norm min")
    c = axes[CCW_SPEED_MIN].pcolormesh(X, Y, slice[CCW_SPEED_MIN,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[CW_SPEED_MIN].set_title("clockwise da/dt norm min")
    c = axes[CW_SPEED_MIN].pcolormesh(X, Y, slice[CW_SPEED_MIN,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[MIN_ROTATION_AMP].set_title("min rotation bump amplitude")
    c = axes[MIN_ROTATION_AMP].pcolormesh(X, Y, slice[MIN_ROTATION_AMP,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[SPEED_MAX].set_title("du/dt norm max")
    c = axes[SPEED_MAX].pcolormesh(X, Y, slice[SPEED_MAX,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[ANGULAR_SPEED_MAX].set_title("da/dt norm max")
    c = axes[ANGULAR_SPEED_MAX].pcolormesh(X, Y, slice[ANGULAR_SPEED_MAX,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    axes[SPEED_VAR].set_title("du/dt norm variance")
    c = axes[SPEED_VAR].pcolormesh(X, Y, slice[SPEED_VAR,...].T, rasterized=True, shading="nearest")
    plt.colorbar(c)

    for ax in axes:
        ax.scatter(best[0], best[1], marker="+", c="red")

    plt.tight_layout()
    plt.savefig(f"../results/{model.name}-loss.pdf")
    #plt.show()
