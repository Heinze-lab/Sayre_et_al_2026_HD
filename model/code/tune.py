import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from itertools import product
import multiprocessing
import pickle
import os

from cx.model import *
from loss import *

# Tuning configuration:
RESOLUTION = 20


class Task:
    def __init__(self, model):
        self.model = model

    def __call__(self, params):
        m = self.model.tune(params)
        equilibria = find_equilibria(m)
        ring, ring_speeds = trace_ring(m, equilibria)
        return equilibria, ring, ring_speeds

def run_search(model, Es, Is, biases, compass_strengths):
    parameters = list(product(Es, Is, biases, compass_strengths))

    filename = f"../results/data/tuning/{model.name}_E={Es[0]}..{Es[-1]}_I={Is[0]}..{Is[-1]}_bias={biases[0]}..{biases[-1]}_compass={compass_strengths[0]}..{compass_strengths[-1]}.pkl"
    if os.path.isfile(filename):
        return pickle.load(open(filename, "rb")), parameters

    with multiprocessing.Pool(os.cpu_count()) as pool:
        iterator = pool.imap(Task(model), parameters)
        equilibria_grid = list(tqdm(iterator, total=len(parameters)))

    result = {
        "Es": Es,
        "Is": Is,
        "biases": biases,
        "compass_strengths": compass_strengths,
        "grid": equilibria_grid,
    }

    with open(filename, "wb") as f:
        pickle.dump(result, f)

    return result, parameters


def choose_params(parameters, loss):
    mask = loss[AMP_MIN,...] > 0.0
    mask &= (loss[BIAS,...] < 0.1)

    # There should be a continuous loop when rotating:
    mask &= (loss[CCW_SPEED_MIN,...] > 0.0)
    mask &= (loss[CW_SPEED_MIN,...] > 0.0)

    filtered_loss = np.nan * np.zeros_like(loss)
    filtered_loss[:,mask] = loss[:,mask]

    # Optimize for smallest max drift speed:
    best = np.unravel_index(np.nanargmin(filtered_loss[ANGULAR_SPEED_MAX,...]), loss.shape[1:])
    slice = filtered_loss[:,:,:,*best[-2:]]

    index = np.ravel_multi_index(best, loss[0,...].shape)
    best_parameters = parameters[index]

    return best_parameters, slice


if __name__ == "__main__":
    fly_model = Model("../data/connectome/fly-cells-grouped.csv", "../results/data/connectome/fly-mean-conn.npy", "fly-mean-conn")
    bee_model = Model("../data/connectome/bee-cells-grouped.csv", "../results/data/connectome/bee-mean-conn.npy", "bee-mean-conn")

    print("Running fly grid search...")
    Es = np.logspace(-1, 0, RESOLUTION)
    Is = np.logspace(0, 1, RESOLUTION)
    data, params = run_search(fly_model, Es, Is, np.array([-0.125]), np.array([1.0]))
    print("Computing loss scores...")
    loss = compute_loss(fly_model, data)
    best, slice = choose_params(params, loss)
    plot_loss(fly_model, Es, Is, slice, best)
    print("Fly parameters:", best)
    np.save("../results/data/tuning/fly-params.npy", best)

    print("Running bee grid search...")
    Es = np.logspace(-0.75, -0.50, RESOLUTION)
    data, params = run_search(bee_model, Es, Is, np.array([-0.125]), np.array([1.0]))
    print("Computing loss scores...")
    loss = compute_loss(bee_model, data)
    best, slice = choose_params(params, loss)
    plot_loss(bee_model, Es, Is, slice, best)
    print("Bee parameters:", best)
    np.save("../results/data/tuning/bee-params.npy", best)
