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

from cx.model import *

model = Model("data/fly-cells.csv", "data/fly-truth.npy", "fly-truth")

if __name__ == "__main__":
    from threadpoolctl import threadpool_info
    print(threadpool_info())

    # (np.float64(0.2104904144512021), np.float64(4.912190125853851), np.float64(-0.25), np.float64(1.0))
    Es, Is, biases, compass_strengths = np.logspace(-0.9, -0.5, 50), np.logspace(0.5, 0.9, 50), np.array([-0.25]), np.array([1.0])
    parameters = list(product(Es, Is, biases, compass_strengths))

    def compute(params):
        m = model.tune(params)
        equilibria = find_equilibria(m)
        ring, ring_speeds = trace_ring(m, equilibria)
        return equilibria, ring, ring_speeds

    with multiprocessing.Pool(8) as pool:
        iterator = pool.imap(compute, parameters)
        equilibria_grid = list(tqdm(iterator, total=len(parameters)))

    with open(f"data/tuning/{model.name}_E={Es[0]}..{Es[-1]}_I={Is[0]}..{Is[-1]}_bias={biases[0]}..{biases[-1]}_compass={compass_strengths[0]}..{compass_strengths[-1]}.pkl", "wb") as f:
        pickle.dump({
            "Es": Es,
            "Is": Is,
            "biases": biases,
            "compass_strengths": compass_strengths,
            "grid": equilibria_grid,
        }, f)
