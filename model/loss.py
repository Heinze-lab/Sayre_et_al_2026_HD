import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import math
import pickle
from itertools import product
from tqdm.notebook import tqdm
import sys

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

if __name__ == "__main__":
    model_name = sys.argv[1]
    model = Model(f"data/{model_name[:3]}-cells.csv", f"data/{model_name}.npy", model_name)

    #with open(f"data/tuning/{model_name}_E=0.06309573444801933..6.309573444801933_I=0.1..10.0_bias=-0.25..-0.25_compass=1.0..1.0.pkl", "rb") as f:
    with open(f"data/tuning/{model_name}_E=0.1..10.0_I=0.1..10.0_bias=-0.5..-0.5_compass=1.0..1.0.pkl", "rb") as f:
        data = pickle.load(f)
        
    parameters = list(product(data["Es"], data["Is"], data["biases"], data["compass_strengths"]))

    loss = np.nan * np.zeros((SPEED_VAR + 1, len(data["Es"]), len(data["Is"]), len(data["biases"]), len(data["compass_strengths"])))

    for i, (params, (equilibria, ring, ring_speed)) in enumerate(zip(tqdm(parameters), data["grid"])):
        index = np.unravel_index(i, loss[0,...].shape)
        
        if len(ring) == 0 or ring.shape[1] == 0:
            continue

        #params = [*params]
        #params[2] = 0.22222222
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

        loss[NUM_EQUILIBRIA,*index] = np.sum(ring_mask) #equilibria.shape[1]
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

        cycles =  0
        for s in np.linspace(0.2, 2.0, 10.0):
            _, is_cycle = trace_cycle(m, equilibria, rotation=s * cw_rotation, dt=0.1, max_steps=10000)
            if is_cycle:
                cycles += 1
        loss[INTEGRATION_RANGE,*index] = cycles


    np.save(f"data/tuning/{model_name}_loss.npy", loss)
