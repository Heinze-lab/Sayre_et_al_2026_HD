import numpy as np
import polars as pl
import scipy.optimize
import scipy.stats
import copy
import re
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt


def columns(subtype: str):
    return re.findall(r"[LR]\d+", subtype)

def column_index(column: str):
    side = column[0]
    index = int(column[1:]) - 1
    if side == "R":
        index = 8 - index
    return index

COMPASS_COLUMNS = 8
COLUMN_PREFERENCE_ANGLES = np.linspace(-np.pi, np.pi, 8, endpoint=False)
def column_angle(column: str):
    index = column_index(column)
    return COLUMN_PREFERENCE_ANGLES[index % len(COLUMN_PREFERENCE_ANGLES)]


class Model:
    def __init__(self, path_cells, path_weights, name: str, color=None):
        self.color = color
        self.name = name
        cells = pl.read_csv(path_cells)
        self.cells = cells.select(pl.row_index(), "type", "subtype").to_numpy()
        self.cell_counts = cells["count"].to_numpy()
        self.cell_count = self.cell_counts.sum()

        self.weights = np.load(path_weights).T.astype(np.float32)
        if len(self.weights.shape) == 3:
            self.weights = np.sum(self.weights, axis=-1)

        # Normalization by maximum positive / negative weight
        min_neg = np.min(self.weights)
        max_pos = np.max(self.weights)
        norm_neg = np.mean(np.abs(self.weights[self.weights < 0.05 * min_neg]))
        norm_pos = np.mean(np.abs(self.weights[self.weights > 0.05 * max_pos]))
        self.weights[self.weights < 0] /= norm_neg
        self.weights[self.weights > 0] /= norm_pos
        print(f"exc.norm: {norm_pos}, inh.norm: {norm_neg}")

        self.compass_angles = np.linspace(-np.pi, np.pi, COMPASS_COLUMNS, endpoint=False)
        self.weights_compass = np.zeros((self.weights.shape[0], COMPASS_COLUMNS))

        self.columns = [columns(st) for st in cells["subtype"]]
        for epg_index in self.cell_indices("EPG"):
            col_index = column_index(self.columns[epg_index][0])
            self.weights_compass[epg_index, col_index % COMPASS_COLUMNS] = -1.0

        # Assign anatomical preference angle by projections in the PB
        self.preference_angles = np.array(
            [
                scipy.stats.circmean([
                    column_angle(col) for col in columns(st)
                ]) + (np.pi if t == "Delta7" else 0) for t, st in cells.select("type", "subtype").rows()
            ]
        )

        self.weights_angular = np.zeros((self.weights.shape[0], 2))
        for index in self.cell_indices("PEN_a"):
            col = self.columns[index][0]
            if col[0] == "R":
                self.weights_angular[index, 1] = -1
            else:
                self.weights_angular[index, 0] = -1
        for index in self.cell_indices("PEN_b"):
            col = self.columns[index][0]
            if col[0] == "R":
                self.weights_angular[index, 1] = -1
            else:
                self.weights_angular[index, 0] = -1

        self.preference_vectors = np.array([np.exp(1j*theta) for theta in self.preference_angles]).T
        self.decoder = self.preference_vectors

        self.gain = 5 # * np.ones(self.weights.shape[0])
        self.bias = -0.25 # * np.ones(self.weights.shape[0])
        self.compass_strength = 1.0

        self.rotation_inhibition_factor = 1.0

    def tune(self, parameters) -> "Model":
        copy = self.copy()
        E, I, bias, cs = parameters
        copy.weights[self.weights > 0] *= E
        copy.weights[self.weights < 0] *= I
        copy.bias = bias
        copy.compass_strength = cs
        return copy

    def ablate(self, mask: np.array) -> "Model":
        copy = self.copy()
        bias = self.bias * np.ones(len(self.cells))
        bias[mask] = -1000.0
        copy.bias = bias
        return copy

    def cell_indices(self, type):
        return self.cells[self.cells[:,1] == type,0].astype(np.uint16) #.filter(type = type)["index"]

    def total_activity(self, state):
        return state
        #return (state.T * self.cell_counts).T

    def decode(self, state, indices = None, normalize = True):
        # Each state variable corresponds to state of an average cell of that type,
        # so for the population code readout, we convert the state vector to total activity per column.
        state = self.total_activity(state)

        decoder = self.preference_vectors
        if indices is not None:
            decoder = decoder[indices]
            state = state[indices]

        z = decoder @ state
        #norm = np.sqrt(self.cell_count)

        norm = 1.0
        if normalize:
            arg = np.angle(z)
            ideal = self.total_activity(self.ideal_bump(arg))
            if indices is not None:
                ideal = ideal[indices]
            norm = np.abs(decoder @ ideal)

        return z / norm

    def ideal_bump(self, theta):
        a, b = np.meshgrid(self.preference_angles, theta)
        return (0.5 + 0.5 * np.cos((a - b).T))

    def encode(self, pva):
        #return np.real(np.linalg.pinv(self.preference_vectors.reshape((1, len(self.preference_vectors)))) @ pva)
        return np.abs(np.linalg.pinv(self.preference_vectors.reshape(1, -1)) @ pva)

        #print(self.preference_vectors.reshape((152, 1)).shape)
        #print(pva.shape)
        #return 0.5 + 0.5 * np.real(self.preference_vectors.reshape((len(self.preference_vectors), 1)) @ pva)
        #z = pva
        #return np.abs(z) * (0.5 * np.cos(self.preference_angles - np.angle(z)) + 0.5)
        #return np.abs(z) * (0.5 * np.cos(self.preference_angles - np.angle(z)) + 0.5)
        #return [np.abs(z) * (0.5 * np.cos(self.preference_angles - np.angle(z)) + 0.5) for z in pva if np.abs(z) <= 2]

    def copy(self):
        return copy.deepcopy(self)

    def compass_current(self, theta, disinhibition_strength = 1.0):
        if len(np.shape(theta)) > 0:
            theta, compass_angles = np.meshgrid(theta, self.compass_angles)
        else:
            theta, compass_angles = theta, self.compass_angles

        disinhibition = (0.5 + 0.5 * np.cos(compass_angles - theta))
        compass = np.ones_like(compass_angles) - (disinhibition_strength * disinhibition)
        return self.compass_strength * self.weights_compass @ compass

    def rotation_current(self, strength):
        return self.weights_angular @ [np.maximum(0, strength), np.maximum(0, -strength)]

    def activation(self, current):
        return sigmoid(self.gain * (current.T + self.bias)).T
        #return np.clip(self.gain * (current + self.bias), 0, 1)

    def cell_names(self, kind = None):
        if kind is not None:
            idx = self.cell_indices(kind)
            return self.cells[idx,1] + "_" + self.cells[idx,2]
        else:
            return self.cells[:,1] + "_" + self.cells[:,2]

    def dudt(self, state, external):
        # TODO: tau?
        return self.activation(self.weights @ state + external) - state

    def ddecodingdu(self):
        return self.preference_vectors @ np.diag(self.cell_counts)

    def dangledt(self, u, dudt):
        return np.imag(self.ddecodingdu() @ dudt / self.decode(u, normalize=False))

    def step(self, dudt, noise = 0.0, dt = 0.1):
        return dudt * dt + noise * np.random.normal(0, 1, np.shape(dudt)) * np.sqrt(dt)

    def rotation_to_strength(self, rotation):
        return rotation * self.rotation_inhibition_factor

    def jacobian(self, state, external):
        x = self.weights @ state + external
        g = self.gain * (x + self.bias)
        # a = s(g(x))
        # a'(x) = s'(g(x))g'(x)
        # dadx = dsdg(g(x))dgdx(x)
        dsdg = np.diag(sigmoid(g) * (1 - sigmoid(g)))
        #dsdg = np.where((g > 0) & (g < 1), 1, 0)
        dgdx = self.gain
        dadx = dsdg * dgdx
        return dadx @ self.weights - np.eye(len(state))

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

def relax_to_stable(model: Model,
                 initial: np.array,
                 compass: np.array,
                 *,
                 dt = 0.1,
                 dudt_threshold = 0.01,
                 max_steps = 10_000,
                 verbose = False) -> np.array:
    # TODO: do something smarter

    states = np.zeros((len(model.cells), max_steps))
    states[:,0] = initial
    for i in range(1, max_steps):
        u = states[:,i-1]
        dudt = model.dudt(u, compass)

        if np.linalg.norm(dudt) < dudt_threshold:
            if verbose: print(f"reached dudt threshold ({np.linalg.norm(dudt)} < {dudt_threshold}) after {i} iterations")
            return states[:,:i], True

        states[:,i] = np.clip(u + dudt * dt, 0, 1)

    if verbose: print("did not converge")
    return states, False

def simulate(model: Model,
                 initial: np.array,
                 external: np.array,
                 *,
                 dt = 0.1,
                 steps = 1_000,
                 noise = 0.0) -> np.array:
    # TODO: do something smarter

    states = np.zeros((len(model.cells), steps))
    dW = np.random.normal(0.0, 1.0, states.shape)
    states[:,0] = initial
    for i in range(1, steps):
        u = states[:,i-1]
        dudt = model.dudt(u, external)

        states[:,i] = np.clip(u + dudt * dt + noise * dW[:,i] * np.sqrt(dt), 0, 1)

    return states

def simulate_tracking(model: Model,
                 heading: np.array,
                 *,
                 dt = 0.1,
                 darkness_onset = None,
                 noise = 0.0) -> np.array:
    steps = len(heading)
    states = np.zeros((len(model.cells), steps))
    dW = np.random.normal(0.0, 1.0, states.shape)
    dtheta = np.diff(heading, prepend=heading[0])
    step_indices = np.arange(1, steps)
    for i in step_indices:
        theta = heading[i]
        #print(dtheta[i] / dt)
        external = model.compass_current(theta, 1.0 if darkness_onset is None else float(i < darkness_onset * len(step_indices))) + model.rotation_current(model.rotation_to_strength(dtheta[i] / dt))
        u = states[:,i-1]
        dudt = model.dudt(u, external)
        states[:,i] = np.clip(u + dudt * dt + noise * dW[:,i] * np.sqrt(dt), 0, 1)

    return states

# Plotting stuff
def label_xaxis(model: Model, ax = None):
    if ax is None:
        ax = plt.gca()
    
    ax.set_xticks(np.arange(len(model.cells)), model.cell_names(), fontsize=5, rotation=90)

def label_yaxis(model: Model, ax = None):
    if ax is None:
        ax = plt.gca()
    
    ax.set_yticks(np.arange(len(model.cells)), model.cell_names(), fontsize=5)
    

# Experiments
# def tune_model(model: Model, E, I) -> Model:
#     m = model.copy()
#     m.weights[m.weights > 0] *= E
#     m.weights[m.weights < 0] *= I
#     return m

def find_stable_pvas(model: Model, thetas: np.array, **kwargs):
    traj_forced, traj_autonomous, convergence = stabilization_trajectories(model, thetas, **kwargs)

    pva_forced = np.array([model.decode(traj[:,-1]) for traj in traj_forced])
    pva_autonomous = np.array([model.decode(traj[:,-1]) for traj in traj_autonomous])

    return pva_forced, pva_autonomous, convergence

def find_stable_states(model: Model, thetas: np.array, **kwargs):
    stable_states = []

    for theta in thetas:
        stable, _ = find_stable_state(model, theta, **kwargs)
        if stable is None:
            continue

        stable_states.append(stable)

    if len(stable_states) == 0:
        return np.zeros((model.weights.shape[0], 0))

    stable_states, _ = unique_points_by_radius(stable_states, kwargs.get("threshold", 0.01))

    return np.array(stable_states).T


def find_stable_state(model: Model, theta: float, compass_strength: float = 1.0, disinhibition_strength: float = 1.0, dt = 0.1, threshold = 0.01, max_steps = 1_000) -> np.array:
    # Find stable states by applying a compass input until the system converges,
    # then remove the compass input and let it converge again, and finally refine
    # the solution using root finding.

    compass = compass_strength * model.compass_current(theta, disinhibition_strength)
    forced_states, converged = relax_to_stable(
        model,
        np.ones(len(model.cells)),
        compass,
        dt = dt,
        dudt_threshold = threshold,
        max_steps = max_steps,
    )

    #if not converged:
    #    print(f"warning: theta = {theta} did not converge (forcing)")

    compass = compass_strength * model.compass_current(theta, disinhibition_strength = 0.0)
    autonomous_states, converged = relax_to_stable(
        model,
        forced_states[:,-1],
        compass,
        dt = dt,
        dudt_threshold = threshold,
        max_steps = max_steps,
    )

    #if not converged:
    #    print(f"warning: theta = {theta} did not converge (relaxing)")

    result = scipy.optimize.root(
        lambda x: model.dudt(x, compass),
        jac = lambda x: model.jacobian(x, compass),
        x0 = autonomous_states[:,-1],
    )

    if not result.success:
        print(f"failure: theta = {theta} did not converge (root-finding): {result.message}")
        return None, None

    # Check if this is actually a stable state

    return result.x, forced_states[:,-1]


def stabilization_trajectories(model: Model, thetas: np.array, *, dt = 0.1, threshold = 0.01, compass_strength=1.0):
    traj_forced = []
    traj_autonomous = []
    convergence = np.full(len(thetas), True)
    for i, theta in enumerate(thetas):
        compass_current = compass_strength * model.compass_current(theta)

        # Relax system until a stable state is reached
        forced_states, converged = relax_to_stable(
            model,
            np.ones(len(model.cells)),
            compass_current,
            dt = dt,
            dudt_threshold = threshold,
            max_steps = 10_000,
        )

        if not converged:
            print(f"theta = {theta} did not converge (forced)")
            convergence[i] = False
            continue

        # Then remove input and relax again
        compass_current = compass_strength * model.compass_current(theta, disinhibition_strength=0.0)
        autonomous_states, converged = relax_to_stable(
            model,
            forced_states[:,-1],
            compass_current,
            dt = dt,
            dudt_threshold = threshold,
            max_steps = 10_000,
            #verbose=True,
        )

        if not converged:
            print(f"theta = {theta} did not converge (autonomous)")
            convergence[i] = False
            continue

        traj_forced.append(forced_states)
        traj_autonomous.append(autonomous_states)

    return traj_forced, traj_autonomous, convergence

# Math
def unique_points_by_radius(points, tol):
    points = np.asarray(points)
    tree = cKDTree(points)

    keep = np.ones(len(points), dtype=bool)

    for i, p in enumerate(points):
        if not keep[i]:
            continue

        # All points within tol of point i
        neighbors = tree.query_ball_point(p, r=tol)

        # Mark later neighbors as duplicates
        neighbors = [j for j in neighbors if j > i]
        keep[neighbors] = False

    return points[keep], keep


def cluster_points_by_radius(points, tol):
    points = np.asarray(points)
    tree = cKDTree(points)

    pairs = tree.query_pairs(r=tol)
    if not pairs:
        labels = np.arange(len(points))
        return points, labels

    rows, cols = zip(*pairs)
    rows = np.array(rows)
    cols = np.array(cols)

    data = np.ones(len(rows) * 2, dtype=bool)
    graph = coo_matrix(
        (data, (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(len(points), len(points)),
    )

    n_components, labels = connected_components(graph, directed=False)

    # Representative: first point in each component
    reps = np.array([points[np.flatnonzero(labels == k)[0]]
                     for k in range(n_components)])

    return reps, labels


# Find equilibria
def find_equilibria(model: Model, n_theta = 36, n_r = 10, threshold=0.1, max_steps=100, dt=0.1, verbose=False):
    thetas = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    amplitudes = np.linspace(0, 1, n_r)

    epg_indices = model.cell_indices("EPG")

    states = []
    compass = model.compass_current(0, 0)
    for theta in thetas:
        for amplitude in amplitudes:
            epg_state = amplitude * (0.5 + 0.5 * np.cos((model.preference_angles[epg_indices] - theta - np.pi)))

            u = np.zeros(model.weights.shape[0])
            for i in range(max_steps):
                u[epg_indices] = epg_state
                dudt = model.dudt(u, compass)
                dudt[epg_indices] = 0
                u += dt * dudt
                if np.linalg.norm(dudt) < threshold:
                    if verbose:
                        print("convergence: threshold")
                    break
            
            if verbose and i < max_steps:
                print("convergence: max steps")

            # now do root finding
            result = scipy.optimize.root(
                lambda x: model.dudt(x, compass),
                jac = lambda x: model.jacobian(x, compass),
                x0 = u,
            )

            if not result.success:
                if verbose:
                    print("convergence: found no root")
                continue

            states.append(result.x)

    if len(states) == 0:
        return np.zeros((len(model.cells), 0))
        
    return unique_points_by_radius(states, 0.01)[0].T


def trace_ring(model: Model, equilibria, start_offset=0.1, trace_step=0.1, threshold=0.1, cutoff = 0.5, verbose = False, dt = 0.1) -> np.array:
    if equilibria.shape[1] == 0:
        return np.zeros((len(model.cells), 0)), np.zeros(0)

    pva_all = model.decode(equilibria, normalize=False)
    cutoff = cutoff * np.max(np.abs(pva_all))

    stable_states = []
    unstable_states = []

    for s in equilibria.T.tolist():
        eigvals, eig = np.linalg.eig(model.jacobian(s, model.compass_current(0, 0)))

        if np.all(np.real(eigvals) < 0):
            stable = True
        else:
            stable = False

        real_indices = np.isreal(eigvals)
        eigvals = np.real(eigvals[real_indices])
        eig = np.real(eig[:,real_indices])

        # exactly one unstable direction
        if np.sum(eigvals >= 0) > 1:
            continue

        pva = model.decode(np.array(s), normalize=False)

        # least stable direction
        i = np.argmax(eigvals)

        if np.abs(pva) > cutoff:
            if stable:
                stable_states.append((s, eig[:,i]))
            else:
                unstable_states.append((s, eig[:,i]))
        
    compass = model.compass_current(0.0, 0.0)

    def follow_ring(initial, max_steps = 10000, dt = dt):
        states = np.zeros((len(model.cells), max_steps))
        states[:,0] = initial
        path_length = 0
        u = states[:,0]
        j = 1
        for i in range(1, max_steps):
            for s, _ in stable_states:
                if np.linalg.norm(u - s) < threshold:
                    if verbose: print(f"reached stable equilibrium ({np.linalg.norm(u - s)} < {threshold}) after {i} iterations")
                    return states[:,:j], True

            dudt = model.dudt(u, compass)
            step = dudt * dt
            u = np.clip(u + step, 0, 1)
            path_length += np.linalg.norm(step)
            if path_length > trace_step:
                path_length = 0
                states[:,j] = u.copy()
                j += 1
        
        if verbose: print(f"failed to reach stable equilibrium |u - s| < {threshold}) after {i} iterations")
        return states, False

    ring = np.zeros((len(model.cells), 0))

    # Trace out the ring manifold:
    for (s, d) in unstable_states:
        states, _ = follow_ring(s + start_offset * d)
        ring = np.hstack([ring, states])

        states, _ = follow_ring(s - start_offset * d)
        ring = np.hstack([ring, states])

    if ring.shape[1] == 0:
        return np.array([]), np.array([])

    ring_speeds = np.array([np.linalg.norm(model.dudt(state, compass)) for state in ring.T])
    return ring, ring_speeds

def resample_trajectory(states, step_size: float) -> np.array:
    trajectory = [states[0]]
    path_length = 0
    for state in states[1:]:
        v = (state - trajectory[-1])
        path_length += np.linalg.norm(v)
        while path_length >= step_size:
            interpolated_state = trajectory[-1] + v * (step_size / path_length)
            trajectory.append(interpolated_state)
            path_length -= step_size

    return np.array(trajectory).T

#def resample_trajectory(states, step_size: float) -> np.array:
#    trajectory = [states[0]]
#    path_length = 0
#    for state in states[1:]:
#        path_length += np.linalg.norm(state - trajectory[-1])
#        if path_length >= step_size:
#            trajectory.append(state)
#            path_length = 0
#
#    return np.array(trajectory).T

def trace_cycle(model: Model, equilibria, rotation=1.0, max_steps=10000, dt=0.1, trace_step=0.1) -> np.array:
    initial_state = None

    # Find an unstable equilibrium to start at.
    for state in equilibria.T:
        eigvals, eig = np.linalg.eig(model.jacobian(state, model.compass_current(0, 0)))

        if np.all(np.real(eigvals) < 0):
            continue

        real_indices = np.isreal(eigvals)
        eigvals = np.real(eigvals[real_indices])
        eig = np.real(eig[:,real_indices])

        pva = model.decode(np.array(state), normalize=False)

        # least stable direction
        i = np.argmax(eigvals)

        if np.abs(pva) > 0.5 * np.max(np.abs(model.decode(equilibria, normalize=False))):
            initial_state = state
            break

    if initial_state is None:
        return np.zeros((len(model.cells), 0)), False

    compass = model.compass_current(0.0, 0.0)
    rotation = model.rotation_current(rotation)

    visited = []
    tail = [initial_state]
    current_state = initial_state

    # Now trace out the cycle manifold by letting the system evolve under a constant rotation input
    # until we reach a state that is close to a previously recorded state (i.e. we have completed a cycle).
    THRESHOLD = 0.1
    for i in range(max_steps): #tqdm(range(max_steps), desc="tracing cycle"):
        dudt = model.dudt(current_state, compass + rotation)
        if np.linalg.norm(dudt) < 0.01:
            return np.array(visited + tail).T, False
        step = dudt * dt
        current_state = np.clip(current_state + step, 0, 1)

        # Go through points in our "tail" and move those that are outside the threshold into the visited list,
        # so that we can check for convergence against them.
        new_tail = []
        for state in tail:
            if np.linalg.norm(state - current_state) > THRESHOLD:
                visited.append(state)
            else:
                new_tail.append(state)
        tail = new_tail

        # Now check if the next state is close to any of the previously visited states. If it is, we have completed a cycle and can stop.
        if len(visited) > 1:
            tree = cKDTree(visited)
            closest_distance, closest_index = tree.query(current_state)
            if closest_distance < THRESHOLD:
                return resample_trajectory(visited[closest_index:] + tail, trace_step), True

            #for j in range(len(visited)):
            #    if np.linalg.norm(visited[j] - current_state) < THRESHOLD:
            #        return np.array(visited[j:] + tail).T, True

        tail.append(current_state)


    return resample_trajectory(visited + tail, trace_step), False
