from turtle import speed

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.colors import Normalize, TwoSlopeNorm, CenteredNorm, LinearSegmentedColormap, SymLogNorm
import os.path
import pickle
from typing import Callable, Any

from cx.model import Model, simulate, find_equilibria, relax_to_stable, trace_ring, simulate_tracking, trace_cycle
from cx.plotting import *
from cx.math import wrap_nans


def load_or_create(model: Model, experiment_name: str, generator: Callable[[Model], Any], **kwargs) -> Any:
    filename = f"../results/data/experiments/{model.name}-{experiment_name}.pkl"
    try:
        with open(filename, "rb") as f:
            print(f"loading {experiment_name} data for {model.name} from {filename}...")
            return pickle.load(f)
    except FileNotFoundError:
        print(f"generating {experiment_name} data for {model.name}...")
        data = generator(model, **kwargs)
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        return data

def generate_trajectories(model: Model, bump_shaped = True):
    trajectories = []
    compass = model.compass_current(0.0, 0.0)
    for _ in tqdm(range(40*40)):
        if bump_shaped:
            x, y = np.random.uniform(-1, 1, 2)
            r = np.abs(x + 1j*y)
            if r > 1:
                continue
            theta = np.angle(x + 1j*y)
            initial = r * model.ideal_bump(theta).reshape(-1)
        else:
            #initial = np.random.random(len(model.cells))
            initial = np.random.beta(1, 2, len(model.cells))
        trajectories.append(simulate(model, initial, dt=0.1, external=compass, steps=300))
    return trajectories

def plot_trajectories(model: Model):
    filename = f"../results/experiments/{model.name}-trajectories.pdf"
    if not os.path.isfile(filename):
        trajectories_bump = load_or_create(model, "trajectories", lambda m: generate_trajectories(m, bump_shaped=True))
        trajectories_random = load_or_create(model, "trajectories-random", lambda m: generate_trajectories(m, bump_shaped=False))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'})

        pva_trajectories = [model.decode(trajectory) for trajectory in trajectories_bump]
        time = np.arange(0, trajectories_bump[0].shape[1])
        norm = plt.Normalize(0, np.sqrt(np.max(time)))
        cmap = plt.get_cmap("plasma")
        for i, t in enumerate(tqdm(time[:-1])):
            segments = [[(np.angle(trajectory[i]), np.abs(trajectory[i])), (np.angle(trajectory[i+1]), np.abs(trajectory[i+1]))] for trajectory in pva_trajectories]
            ax1.add_collection(LineCollection(segments, color=cmap(norm(t**(1/2))), zorder=i, rasterized=True))

        ax1.set_title(f"{model.name} trajectories (bump-shaped)")
        ax1.set_aspect("equal")
        ax1.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False), labels=["L5/R5", "L6/R4", "L7/R3", "L8/R2", "L1/R1", "L2/R8", "L3/R7", "L4/R6"])
        ax1.set_yticks([])

        pva_trajectories = [model.decode(trajectory) for trajectory in trajectories_random]
        time = np.arange(0, trajectories_bump[0].shape[1])
        norm = plt.Normalize(0, np.sqrt(np.max(time)))
        cmap = plt.get_cmap("plasma")
        for i, t in enumerate(tqdm(time[:-1])):
            segments = [[(np.angle(trajectory[i]), np.abs(trajectory[i])), (np.angle(trajectory[i+1]), np.abs(trajectory[i+1]))] for trajectory in pva_trajectories]
            ax2.add_collection(LineCollection(segments, color=cmap(norm(t**(1/2))), zorder=i, rasterized=True))

        ax2.set_title(f"{model.name} trajectories (random)")
        ax2.set_aspect("equal")
        ax2.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False), labels=["L5/R5", "L6/R4", "L7/R3", "L8/R2", "L1/R1", "L2/R8", "L3/R7", "L4/R6"])
        ax2.set_yticks([])

        plt.tight_layout()
        plt.savefig(filename)

def plot_equilibria(model: Model) -> np.array:
    filename = f"../results/experiments/{model.name}-equilibria.pdf"
    if os.path.isfile(filename):
        return

    equilibria = find_equilibria(model, n_theta=50, n_r=20, threshold=0.01, dt=0.1)

    fig = plt.figure(figsize=(15, 5*3))
    ax1 = plt.subplot(321, projection="polar")
    ax2 = plt.subplot(322)


    for s in equilibria.T.tolist():
        eigvals, eig = np.linalg.eig(model.jacobian(s, model.compass_current(0, 0)))

        if np.all(np.real(eigvals) < 0):
            stable = True
        else:
            stable = False

        real_indices = np.isreal(eigvals)
        eigvals = np.real(eigvals[real_indices])
        eig = np.real(eig[:,real_indices])

        ring_pva = model.decode(np.array(s))

        # least stable direction
        i = np.argmax(eigvals)
        slow_direction = ring_pva - model.decode(s + eig[:,i])
        slow_direction /= np.abs(slow_direction)
        #line = np.array([pva + 0.5*slow_direction, pva - 0.5*slow_direction])
        #ax.plot(np.angle(line), np.abs(line), color="black", alpha=np.clip(0, 1, np.abs(eigvals[i])))

        if stable:
            ax1.scatter(np.angle(ring_pva), np.abs(ring_pva), color="black", s=80, zorder=10)
        else:
            ax1.scatter(np.angle(ring_pva), np.abs(ring_pva), color="black", s=80, facecolors="white", zorder=10)

    nth = 5

    ring, _ = trace_ring(model, equilibria, start_offset=0.01, trace_step=0.05)
    ring_pva = model.decode(ring)
    ring_dudt = model.dudt(ring, model.compass_current(np.zeros(ring.shape[1]), 0.0))
    ring_dudt_pva = model.decode(ring + ring_dudt * 0.01) - ring_pva
    ring_dudt_pva /= np.abs(ring_dudt_pva)
    dadt = model.dangledt(ring, model.dudt(ring, model.compass_current(np.ones(ring.shape[1]), 0)))

    norm = CenteredNorm() #TwoSlopeNorm(vcenter=0, vmin=np.min(dadt), vmax=np.max(dadt))

    #c = ax1.scatter(np.angle(ring_pva), np.abs(ring_pva), marker=".", c=dadt, cmap="PRGn", norm=TwoSlopeNorm(0))
    c = ax1.quiver(np.angle(ring_pva[::nth]), np.abs(ring_pva[::nth]), np.real(ring_dudt_pva[::nth]), np.imag(ring_dudt_pva[::nth]), dadt[::nth], scale=30, cmap="PRGn", norm=norm) #dadt[::nth])
    plt.colorbar(c, label="angular velocity (rad/time step)", pad=0.1, shrink=0.5)
    
    ax1.set_ylim(0, np.max(np.abs(ring_pva)) * 1.25)
    ax1.set_aspect("equal")
    ax1.set_title(f"{model.name} equilibria and ring manifold")
    ax2.set_title(f"{model.name} ring states")

    #ring, ring_speeds = trace_ring(model, equilibria, start_offset=0.01, trace_step=0.1)
    #pva = model.decode(ring)

    indices = np.argsort(np.angle(ring_pva))
    ring = ring[:,indices]
    ring_pva = ring_pva[indices]

    x = np.angle(ring_pva)
    y = -np.arange(len(model.cells))
    X, Y = np.meshgrid(x, y)
    ax2.pcolormesh(X, Y, ring, rasterized=True, shading="nearest", cmap="BuPu")
    ax2.set_yticks(y, labels=model.cell_names(), fontsize=6)
    ax2.set_xlabel("bump location (rad)")

    ax1.set_yticks([])
    ax1.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False), labels=["L5/R5", "L6/R4", "L7/R3", "L8/R2", "L1/R1", "L2/R8", "L3/R7", "L4/R6"])
    ax1.set_ylabel("drift", labelpad=50)
    ax1.tick_params(axis='both', which='major', pad=5)

    norm = CenteredNorm() #TwoSlopeNorm(vcenter=0, vmin=np.min(dadt), vmax=np.max(dadt))

    # CCW
    ax3 = plt.subplot(323, projection="polar")
    ax4 = plt.subplot(324)

    cycle, is_cycle = trace_cycle(model, equilibria, rotation=1.0, dt=0.1, max_steps=10000)
    cycle_pva = model.decode(cycle)
    cycle_dudt = model.dudt(cycle, model.compass_current(np.zeros(cycle.shape[1]), 0.0) + model.rotation_current(1.0*np.ones(cycle.shape[1])))
    cycle_dudt_pva = model.decode(cycle + cycle_dudt * 0.01) - cycle_pva
    cycle_dudt_pva /= np.abs(cycle_dudt_pva)
    dadt = model.dangledt(cycle, cycle_dudt)
    #norm.autoscale(dadt)
    #c = ax1.scatter(np.angle(pva), np.abs(pva), marker=".", c=dadt, cmap="PRGn", norm=TwoSlopeNorm(0))
    c = ax3.quiver(np.angle(cycle_pva[::nth]), np.abs(cycle_pva[::nth]), np.real(cycle_dudt_pva[::nth]), np.imag(cycle_dudt_pva[::nth]), dadt[::nth], scale=30, cmap="PRGn", norm=norm) #dadt[::nth])
    ax3.set_yticks([])
    ax3.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False), labels=["L5/R5", "L6/R4", "L7/R3", "L8/R2", "L1/R1", "L2/R8", "L3/R7", "L4/R6"])
    plt.colorbar(c, label="angular velocity (rad/time step)", pad=0.1, shrink=0.5)
    ax3.set_ylim(0, np.max(np.abs(ring_pva)) * 1.25)
    ax3.set_ylabel("counter-clockwise rotation", labelpad=50)
    ax3.tick_params(axis='both', which='major', pad=5)

    indices = np.argsort(np.angle(cycle_pva))
    cycle = cycle[:,indices]
    cycle_pva = cycle_pva[indices]

    x = np.angle(cycle_pva)
    y = -np.arange(len(model.cells))
    X, Y = np.meshgrid(x, y)
    ax4.pcolormesh(X, Y, cycle, rasterized=True, shading="nearest", cmap="BuPu")
    ax4.set_yticks(y, labels=model.cell_names(), fontsize=6)
    ax4.set_xlabel("bump location (rad)")


    # CW
    ax5 = plt.subplot(325, projection="polar")
    ax6 = plt.subplot(326)

    cycle, is_cycle = trace_cycle(model, equilibria, rotation=-1.0, dt=0.1, max_steps=10000)
    cycle_pva = model.decode(cycle)
    cycle_dudt = model.dudt(cycle, model.compass_current(np.zeros(cycle.shape[1]), 0.0) + model.rotation_current(-1.0*np.ones(cycle.shape[1])))
    cycle_dudt_pva = model.decode(cycle + cycle_dudt * 0.01) - cycle_pva
    cycle_dudt_pva /= np.abs(cycle_dudt_pva)
    dadt = model.dangledt(cycle, cycle_dudt)
    #norm.autoscale(dadt)
    c = ax5.quiver(np.angle(cycle_pva[::nth]), np.abs(cycle_pva[::nth]), np.real(cycle_dudt_pva[::nth]), np.imag(cycle_dudt_pva[::nth]), dadt[::nth], scale=30, cmap="PRGn", norm=norm) #dadt[::nth])
    ax5.set_yticks([])
    ax5.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False), labels=["L5/R5", "L6/R4", "L7/R3", "L8/R2", "L1/R1", "L2/R8", "L3/R7", "L4/R6"])
    plt.colorbar(c, label="angular velocity (rad/time step)", pad=0.1, shrink=0.5)
    ax5.set_ylim(0, np.max(np.abs(ring_pva)) * 1.25)
    ax5.set_ylabel("clockwise rotation", labelpad=50)
    ax5.tick_params(axis='both', which='major', pad=5)

    indices = np.argsort(np.angle(cycle_pva))
    cycle = cycle[:,indices]
    cycle_pva = cycle_pva[indices]

    x = np.angle(cycle_pva)
    y = -np.arange(len(model.cells))
    X, Y = np.meshgrid(x, y)
    ax6.pcolormesh(X, Y, cycle, rasterized=True, shading="nearest", cmap="BuPu")
    ax6.set_yticks(y, labels=model.cell_names(), fontsize=6)
    ax6.set_xlabel("bump location (rad)")


    plt.tight_layout()
    plt.savefig(filename)


def choose_initial_state(model: Model) -> np.array:
    equilibria = find_equilibria(model, n_theta=50, n_r=20, threshold=0.01, dt=0.1)

    pva_all = model.decode(equilibria)
    cutoff = 0.5 * np.max(np.abs(pva_all))

    for s in equilibria.T.tolist():
        eigvals, _ = np.linalg.eig(model.jacobian(s, model.compass_current(0, 0)))
        pva = model.decode(np.array(s))
        stable = False
        if np.all(np.real(eigvals) < 0):
            stable = True

        if np.abs(pva) > cutoff and stable:
            return np.array(s)


ROTATION_SPEEDS = np.radians(np.linspace(-20, 20, 50)) # degrees per time unit
DT = 0.01
NOISE = 0.05

def generate_rotation_data(model: Model) -> np.array:
    ROTATIONS = 5

    without_pen = []

    initial_state = choose_initial_state(model)
    for rotation_speed in tqdm(ROTATION_SPEEDS):
        duration = ROTATIONS * 2 * np.pi / np.abs(rotation_speed)
        steps = int(duration / DT)

        time = np.arange(0, steps) * DT
        states = np.zeros((len(model.cells), steps))
        states[:,0] = initial_state
        thetas = np.arange(0, steps) * DT * rotation_speed + np.angle(model.decode(initial_state))

        for i in range(1, steps):
            compass_current = model.compass_current(thetas[i], disinhibition_strength=1.0)
            u = states[:,i-1]
            dudt = model.dudt(u, compass_current)
            states[:,i] = np.clip(u + model.step(dudt, 0.05, DT), 0, 1)

        without_pen.append((rotation_speed, time, thetas, states))

    with_pen = []

    initial_state = choose_initial_state(model)
    for rotation_speed in tqdm(ROTATION_SPEEDS):
        duration = ROTATIONS * 2 * np.pi / np.abs(rotation_speed)
        steps = int(duration / DT)

        time = np.arange(0, steps) * DT
        states = np.zeros((len(model.cells), steps))
        states[:,0] = initial_state
        thetas = np.arange(0, steps) * DT * rotation_speed + np.angle(model.decode(initial_state))

        for i in range(1, steps):
            compass_current = model.compass_current(thetas[i], disinhibition_strength=1.0)
            rotation_current = model.rotation_current(model.rotation_to_strength(rotation_speed))
            u = states[:,i-1]
            dudt = model.dudt(u, compass_current + rotation_current)
            states[:,i] = np.clip(u + model.step(dudt, 0.05, DT), 0, 1)

        with_pen.append((rotation_speed, time, thetas, states))

    without_compass = []

    initial_state = choose_initial_state(model)
    for rotation_speed in tqdm(ROTATION_SPEEDS):
        duration = ROTATIONS * 2 * np.pi / np.abs(rotation_speed)
        steps = int(duration / DT)

        time = np.arange(0, steps) * DT
        states = np.zeros((len(model.cells), steps))
        states[:,0] = initial_state
        thetas = np.arange(0, steps) * DT * rotation_speed + np.angle(model.decode(initial_state))

        for i in range(1, steps):
            compass_current = model.compass_current(thetas[i], disinhibition_strength=0.0)
            rotation_current = model.rotation_current(model.rotation_to_strength(rotation_speed))
            u = states[:,i-1]
            dudt = model.dudt(u, compass_current + rotation_current)
            states[:,i] = np.clip(u + model.step(dudt, 0.05, DT), 0, 1)

        without_compass.append((rotation_speed, time, thetas, states))

    return (with_pen, without_pen, without_compass)

def generate_pen_rotation(model: Model) -> np.array:
    strengths = np.linspace(-2, 2, 51)
    #initial_state = choose_initial_state(model)
    #trajectories = [simulate(model, initial_state, model.compass_current(0.0, 0.0) + model.rotation_current(strength), noise=0.01) for strength in tqdm(strengths)]
    equilibria = find_equilibria(model, n_theta=50, n_r=20, threshold=0.01, dt=0.1)
    return strengths, [trace_cycle(model, equilibria, rotation=s, dt=0.1, max_steps=10000) for s in strengths]

def result_filename(experiment_name: str, model: Model = None):
    if model is not None:
        filename = f"../results/experiments/{model.name}-{experiment_name}.pdf"
    else:
        filename = f"../results/experiments/{experiment_name}.pdf"
    if os.path.isfile(filename):
        return None
    else:
        return filename

def plot_rotation(model: Model):
    NAME = "rotation"
    if (filename := result_filename(NAME, model)):
        pen_trials, _, _ = load_or_create(model, NAME, generate_rotation_data)

        fig, axes = plt.subplots(len(pen_trials), 1, figsize=(6, 2 * len(pen_trials)))
        for ax, (rotation_speed, time, thetas, states) in zip(axes, pen_trials):
            ax.plot(*wrap_nans(time, np.angle(model.decode(states))))
            ax.plot(*wrap_nans(time, thetas))
            ax.set_ylabel("bump angle (rad)")
            ax.set_xlabel("rotation (rad)")
            ax.set_title(f"$\\omega = {rotation_speed:.2f}~\\text{{rad/time}}$")
            ax.margins(0)
        plt.tight_layout()
        plt.suptitle(f"{model.name} rotation")
        plt.savefig(filename)

def plot_gain(model: Model):
    if (filename := result_filename("gain", model)):
        pen_trials, nopen_trials, nocompass_trials = load_or_create(model, "rotation", generate_rotation_data)

        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(ROTATION_SPEEDS, ROTATION_SPEEDS, color="black", linestyle="--")

        mean_bump_rotation_speeds = []
        for (rotation_speed, time, thetas, states) in pen_trials:
            headings = np.angle(model.decode(states))
            mean_bump_rotation_speed = np.mean(np.diff(np.unwrap(headings)) / DT)
            mean_bump_rotation_speeds.append(mean_bump_rotation_speed)
        ax.plot(ROTATION_SPEEDS, mean_bump_rotation_speeds, label="both")

        mean_bump_rotation_speeds = []
        for (rotation_speed, time, thetas, states) in nopen_trials:
            headings = np.angle(model.decode(states))
            mean_bump_rotation_speed = np.mean(np.diff(np.unwrap(headings)) / DT)
            mean_bump_rotation_speeds.append(mean_bump_rotation_speed)
        ax.plot(ROTATION_SPEEDS, mean_bump_rotation_speeds, "--", label="compass only")

        mean_bump_rotation_speeds = []
        for (rotation_speed, time, thetas, states) in nocompass_trials:
            headings = np.angle(model.decode(states))
            mean_bump_rotation_speed = np.mean(np.diff(np.unwrap(headings)) / DT)
            mean_bump_rotation_speeds.append(mean_bump_rotation_speed)
        ax.plot(ROTATION_SPEEDS, mean_bump_rotation_speeds, "--", label="PEN only")

        ax.set_xlabel("angular velocity (rad/time)")
        ax.set_ylabel("mean bump angular velocity (rad/time)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)

def plot_pen_rotation(model: Model):
    NAME = "pen-rotation"
    if (filename := result_filename(NAME, model)):
        strengths, trajectories = load_or_create(model, NAME, generate_pen_rotation)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        speed = lambda theta: (theta[-1] - theta[0]) / (len(theta) * DT)
        ax1.scatter(strengths, [speed(np.unwrap(np.angle(model.decode(cycle)))) for cycle, _ in trajectories], c=["blue" if is_cycle else "red" for _, is_cycle in trajectories])
        for strength, (cycle, _) in list(zip(strengths, trajectories))[::5]:
            time = np.linspace(0, DT*cycle.shape[1], cycle.shape[1])
            ax2.plot(*wrap_nans(time, np.unwrap(np.angle(model.decode(cycle)) - np.angle(model.decode(cycle[:,0])))), label=f"{strength:.02f}")

        ax1.set_xlabel("inhibition strength")
        ax1.set_ylabel("mean bump angular velocity (rad/time)")

        ax2.set_xlabel("time")
        ax2.set_ylabel("relative bump location (rad)")

        plt.tight_layout()
        plt.legend()
        plt.savefig(filename)

def generate_drift_data(model: Model, rotation=0, noise=0):
    equilibria = find_equilibria(model, n_theta=50, n_r=20, threshold=0.01, dt=0.1)
    ring, _ = trace_ring(model, equilibria, start_offset=0.01, trace_step=0.5, verbose=False)
    return ring, [simulate(model, initial_state, model.compass_current(0.0, 0.0) + model.rotation_current(rotation), dt=DT, steps=10000, noise=noise) for initial_state in tqdm(ring.T)]

def plot_drift(model: Model, rotation=0, noise=0):
    NAME = f"drift-{rotation}-{noise}"
    if (filename := result_filename(NAME, model)):
        _, trajectories = load_or_create(model, NAME, generate_drift_data, rotation=rotation, noise=noise)

        plt.figure()
        for trajectory in trajectories:
            time = np.arange(0, trajectory.shape[1]) * DT
            plt.plot(*wrap_nans(time, np.angle(model.decode(trajectory))), color=model.color or "black", alpha=0.5)
        plt.title(f"{model.name} drift")
        plt.xlabel("time")
        plt.ylabel("angle (rad)")
        plt.margins(0)
        plt.savefig(filename)

def generate_tracking_data(model: Model, input: str, cell_mask=None, darkness_onset=None):
    time, compass = np.load(f"data/{input}.npy")
    states = simulate_tracking(model, compass, dt=DT, noise=0.01, darkness_onset=darkness_onset)
    return time, compass, states

def generate_jumping_tracking_data(model: Model):
    time, compass = np.load("data/compass-jumps.npy")
    states = simulate_tracking(model, compass, dt=DT, noise=0.01)
    return time, compass, states

def plot_ablations(model: Model):
    NAME = "ablations"
    if (filename := result_filename(NAME, model)):
        print("plotting ablations")

        cell_types = ["Delta7", "PEG", "PEN_a", "PEN_b"]
        #fig, axes = #plt.subplots(2**len(cell_types), 2, figsize=(8, 2**len(cell_types) * 2))
        fig = plt.figure(figsize=(12, 2**len(cell_types) * 2))

        speed_axes = []

        norm = CenteredNorm(halfrange=0.0001)
        speed_lim = 0
        max_dadt = 0
        for mask in range(2**len(cell_types)):
            cell_mask = np.zeros(len(model.cells), dtype=bool)
            ablation_name = []
            for i, cell_type in enumerate(cell_types):
                if mask & (1 << i):
                    ablation_name.append(cell_type)
                    cell_mask[model.cell_indices(cell_type)] = True
            ablation_name = "+".join(ablation_name) if len(ablation_name) > 0 else "none"
            print(ablation_name)

            m = model.ablate(cell_mask)

            ax1 = plt.subplot(2**len(cell_types), 3, 3*mask+1, projection="polar")
            ax2 = plt.subplot(2**len(cell_types), 3, 3*mask+2)
            ax3 = plt.subplot(2**len(cell_types), 3, 3*mask+3)

            speed_axes.append(ax2)

            strengths, cycles = load_or_create(m, f"ablation-pen-rotation-{ablation_name}", generate_pen_rotation)
            speed = lambda theta: (theta[-1] - theta[0]) / (len(theta) * DT) if len(theta) > 0 else 0
            speeds = [speed(np.unwrap(np.angle(model.decode(cycle)))) for cycle, _ in cycles]
            speed_lim = max(speed_lim, np.max(np.abs(speeds)))
            ax2.scatter(strengths, speeds, c=["blue" if is_cycle else "red" for _, is_cycle in cycles])

            # TODO: clean up this mess
            print("finding equilibria...")
            equilibria = find_equilibria(m, n_theta=50, n_r=20, threshold=0.01, dt=0.1)

            for s in equilibria.T:
                eigvals, eig = np.linalg.eig(m.jacobian(s, m.compass_current(0, 0)))

                if np.all(np.real(eigvals) < 0):
                    stable = True
                else:
                    stable = False

                real_indices = np.isreal(eigvals)
                eigvals = np.real(eigvals[real_indices])
                eig = np.real(eig[:,real_indices])

                ring_pva = m.decode(s)

                # least stable direction
                i = np.argmax(eigvals)
                slow_direction = ring_pva - m.decode(s + eig[:,i])
                slow_direction /= np.abs(slow_direction)
                #line = np.array([pva + 0.5*slow_direction, pva - 0.5*slow_direction])
                #ax.plot(np.angle(line), np.abs(line), color="black", alpha=np.clip(0, 1, np.abs(eigvals[i])))

                if stable:
                    ax1.scatter(np.angle(ring_pva), np.abs(ring_pva), color="black", s=40, zorder=10)
                else:
                    ax1.scatter(np.angle(ring_pva), np.abs(ring_pva), color="black", s=40, facecolors="white", zorder=10)

            nth = 5

            print("tracing ring...")
            ring, _ = trace_ring(m, equilibria, start_offset=0.01, trace_step=0.1, cutoff=0.0)

            if len(ring) != 0 and ring.shape[1] != 0:
                ring_pva = m.decode(ring)
                ring_dudt = m.dudt(ring, m.compass_current(np.zeros(ring.shape[1]), 0.0))
                ring_dudt_pva = m.decode(ring + ring_dudt * 0.01) - ring_pva
                ring_dudt_pva /= np.abs(ring_dudt_pva)
                dadt = m.dangledt(ring, m.dudt(ring, m.compass_current(np.ones(ring.shape[1]), 0)))

                if len(dadt) > 0:
                    max_dadt = max(max_dadt, np.max(np.abs(dadt[np.isfinite(dadt)])))

                c = ax1.quiver(np.angle(ring_pva[::nth]), np.abs(ring_pva[::nth]), np.real(ring_dudt_pva[::nth]), np.imag(ring_dudt_pva[::nth]), dadt[::nth], scale=20, width=0.01, cmap="PRGn", norm=norm) #dadt[::nth])
                ax1.set_ylim(0, np.max(np.abs(ring_pva)) * 1.25)

            plt.colorbar(c, pad=0.1, shrink=1.0, ax=ax1)
                
            ax1.set_yticks([])
            ax1.set_aspect("equal")

            time, compass, states = load_or_create(m, f"{NAME}-{ablation_name}", generate_tracking_data, input="compass")
            X, Y = np.meshgrid(time, np.linspace(-np.pi, np.pi, len(m.cell_indices("EPG"))//2))
            u = states[m.cell_indices("EPG"),:]
            ax3.pcolormesh(X, Y, u[:u.shape[0]//2,:] + u[u.shape[0]//2:,:], rasterized=True, shading="nearest", cmap="BuPu")

            ax3.plot(*wrap_nans(time, np.angle(m.decode(states))), color="red", label="bump")
            ax3.plot(*wrap_nans(time, compass), "--", label="heading", color="lightgray")
            
            if mask == 0:
                ax3.legend()

            if mask == 2**len(cell_types)-1:
                ax2.set_xlabel("inhibition strength")
                ax3.set_xlabel("time")

            ax1.set_ylabel(ablation_name, labelpad=50)

        norm.halfrange = max_dadt
        for ax in speed_axes:
            ax.set_ylim(-speed_lim * 1.05, speed_lim * 1.05)

        fig.suptitle(f"{model.name} ablations", y=0.99)
        plt.tight_layout()
        plt.savefig(filename)


def compare_tracking(models: list[Model], input, darkness_onset=None):
    NAME = f"tracking_{'+'.join([m.name for m in models])}_{input}"

    if (filename := result_filename(NAME)):
        fig, (*axes, ax1) = plt.subplots(len(models)+1, 1, figsize=(8, 2*(len(models)+1)))

        for ax, model in zip(axes, models):
            time, compass, states = load_or_create(model, NAME, generate_tracking_data, input=input, darkness_onset=darkness_onset)
            ax1.plot(*wrap_nans(time, np.angle(model.decode(states))), label=model.name, color=model.color)
            X, Y = np.meshgrid(time, np.arange(len(model.cell_indices("EPG"))))
            ax.pcolormesh(X, Y, states[model.cell_indices("EPG"),:], rasterized=True, shading="nearest", cmap="BuPu")
            ax.yaxis.set_inverted(True)
            if darkness_onset is not None:
                ax.axvline(time[-1] * darkness_onset, linestyle="dotted", color="gray", label="darkness onset")
            ax.set_yticks(range(len(model.cell_indices("EPG"))), model.cell_names("EPG"), fontsize=6)
            ax.set_aspect("auto")
            ax.set_ylabel(model.name)

        if darkness_onset is not None:
            ax1.axvline(time[-1] * darkness_onset, linestyle="dotted", color="gray", label="darkness onset")
        ax1.plot(*wrap_nans(time, compass), "--", label="heading", color="black")
        ax1.legend()
        ax1.yaxis.set_inverted(True)
        ax1.margins(0)
        ax1.set_ylabel("angle (rad)")
        ax1.set_xlabel("time")
        plt.tight_layout()
        plt.savefig(filename)

def compare_jump_tracking(models: list[Model]):
    if (filename := result_filename("tracking-jumps")):
        fig, (*axes, ax1) = plt.subplots(len(models)+1, 1, figsize=(8, 2*(len(models)+1)))

        for ax, model in zip(axes, models):
            time, compass, states = load_or_create(model, "tracking-jumps", generate_jumping_tracking_data)
            ax1.plot(*wrap_nans(time, np.angle(model.decode(states))), label=model.name, color=model.color)
            X, Y = np.meshgrid(time, np.arange(len(model.cell_indices("EPG"))))
            ax.pcolormesh(X, Y, states[model.cell_indices("EPG"),:], rasterized=True, shading="nearest", cmap="BuPu")
            ax.yaxis.set_inverted(True)
            ax.set_yticks(range(len(model.cell_indices("EPG"))), model.cell_names("EPG"), fontsize=6)
            ax.set_aspect("auto")
            ax.set_ylabel(model.name)

        ax1.plot(*wrap_nans(time, compass), "--", label="heading", color="black")
        ax1.legend()
        ax1.yaxis.set_inverted(True)
        ax1.margins(0)
        ax1.set_ylabel("rad")
        ax1.set_xlabel("time")
        plt.tight_layout()
        plt.savefig(filename)

def analyze(model: Model):
    print("analyzing", model.name)
    plot_trajectories(model)
    plot_equilibria(model)
    plot_ablations(model)
    #plot_rotation(model)
    plot_drift(model, noise = 0.01)
    #plot_drift(model, 1.0, noise = 0.01)
    #plot_drift(model, -1.0, noise = 0.1)
    #plot_drift(model, 1.0, noise=0.05)
    #plot_drift(model, -1.0, noise=0.05)
    ##plot_drift(model, 1.0, noise=0.01)
    ##plot_drift(model, -1.0, noise=0.01)
    plot_gain(model)
    plot_pen_rotation(model)


def comparison_figures(models):
    compare_tracking(models, "compass", darkness_onset=0.5)
    compare_tracking(models, "compass-ramping")
    compare_jump_tracking(models)

    NAME = f"matrices_{'+'.join([m.name for m in models])}_{input}"
    if (filename := result_filename(NAME)):
        fig, axes = plt.subplots(1, len(models) + 1, figsize=(7*(len(models) + 1), 6))
        for ax, model in zip(axes[:-1], models):
            ax.set_title(f"{model.name} weights")
            _, cbar = plot_grouped_weights(model.weights.T, labels=model.cell_names(), ax=ax)
            cbar.set_ticks([model.weights.min(), 0, model.weights.max()])

        diff = np.abs(models[1].weights.T) - np.abs(models[0].weights.T)
        dmin = diff.min()
        dmax = diff.max()
        axes[-1].set_title("difference")
        # assume there are two models
        norm = SymLogNorm(1.0, vmin=min(dmin, -dmax), vmax=max(dmax, -dmin))
        cmap = LinearSegmentedColormap.from_list("fly_bee", [models[0].color, "white", models[1].color], N=256)
        plot_grouped_weights(diff, labels=models[0].cell_names(), ax=axes[-1], cmap=cmap, norm=norm)

        plt.tight_layout()
        plt.savefig(filename)

    if (filename := result_filename("rotation")):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(ROTATION_SPEEDS, ROTATION_SPEEDS, color="black", linestyle="--")

        for model in models:
            trials_pen, trials_nopen, trials_nocompass = load_or_create(model, "rotation", generate_rotation_data)

            def plot(ax, model, trials, **kwargs):
                mean_bump_rotation_speeds = []
                for (rotation_speed, time, thetas, states) in trials:
                    headings = np.angle(model.decode(states))
                    mean_bump_rotation_speed = np.mean(np.diff(np.unwrap(headings)) / DT)
                    mean_bump_rotation_speeds.append(mean_bump_rotation_speed)
                ax.plot(ROTATION_SPEEDS, mean_bump_rotation_speeds, **kwargs)

            plot(ax, model, trials_pen, label=model.name[:3])
            #plot(ax, model, trials_nopen, label=f"{model.name[:3]} (compass only)", linestyle="--")
            #plot(ax, model, trials_nocompass, label=f"{model.name[:3]} (PEN only)", linestyle="--")

        ax.legend()
        ax.set_xlabel("angular velocity (rad/time)")
        ax.set_ylabel("bump angular velocity (rad/time)")
        plt.tight_layout()
        plt.savefig(filename)


if __name__ == "__main__":
    fly_params = np.load("../results/data/tuning/fly-params.npy")
    bee_params = np.load("../results/data/tuning/bee-params.npy")

    # Optimized for low drift:
    fly_mean_conn = Model("../data/connectome/fly-cells.csv", f"../data/connectome/fly-mean-conn.npy", "fly-extrapolated", "darkviolet").tune(fly_params)
    fly_truth = Model("../data/connectome/fly-cells.csv", f"../data/connectome/fly-truth-simplified.npy", "fly-truth", "orange").tune(fly_params)
    bee_mean_conn = Model("../data/connectome/bee-cells.csv", f"../data/connectome/bee-mean-conn.npy", "bee-extrapolated", "green").tune(bee_params)

    # Map highest rotation speed (1.5) due to PEN inhibition to a typical 'high' speed in compass rotation input (about 0.1 rad/time step).
    fly_mean_conn.rotation_inhibition_factor = 1.5 / 0.1
    fly_truth.rotation_inhibition_factor = 1.5 / 0.1
    bee_mean_conn.rotation_inhibition_factor = 1.5 / 0.1

    analyze(fly_mean_conn)
    analyze(bee_mean_conn)
    analyze(fly_truth)
    comparison_figures([fly_mean_conn, bee_mean_conn])
    comparison_figures([fly_mean_conn, fly_truth])
