import numpy as np

def wrap(values, vmin=-np.pi, vmax=np.pi):
    return (values - vmin) % (vmax - vmin) + vmin

def wrap_nans(x, y, vmin=-np.pi, vmax=np.pi):
    period = vmax - vmin
    wrapped = wrap(y, vmin, vmax)
    mask = np.abs(np.diff(wrapped, prepend=wrapped[0])) > period/2
    x_ = np.insert(x, mask, x[mask])
    y_ = np.insert(wrapped, mask, np.nan)
    return x_, y_

def bump_dft(state, indices, angles):
    #preference_vectors = np.array([[np.cos(theta), np.sin(theta)] for theta in PREFERENCE_ANGLES[indices]]).T
    preference_vectors = np.array([np.exp(1j*theta) for theta in angles[indices]]).T
    return preference_vectors @ state[indices] / np.sqrt(len(indices))

def angular_distance(a, b):
    return (a - b + np.pi) % (2*np.pi) - np.pi
