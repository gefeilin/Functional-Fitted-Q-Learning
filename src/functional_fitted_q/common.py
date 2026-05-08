import hashlib
import random

import numpy as np
from filelock import FileLock
from scipy.interpolate import UnivariateSpline
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def safe_append(path, text):
    # Multiple batch jobs append into the same result files, so writes must be serialized.
    lock_path = path + ".lock"
    with FileLock(lock_path):
        with open(path, "a") as f:
            f.write(text)


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def state_dependent_gp(state):
    kernel = RBF(length_scale=(state[0] ** 2 + state[1] ** 2 + state[2] ** 2) / 3)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, optimizer=None)
    return gp


def generate_action(state, x_index=np.linspace(0, 1, 100)):
    gp = state_dependent_gp(state)
    x_grid = np.atleast_2d(x_index).T
    y_samples = gp.sample_y(x_grid, n_samples=1, random_state=random.seed())
    y_samples = np.clip(y_samples, -2, 2)
    return y_samples.reshape(-1)


def calculate_reward(row):
    theta = row["theta"]
    theta_dot = row["thetadot"]
    actions = row["FunctionalData"]
    sum_torque_squared = np.sum(np.square(actions))
    reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * (1 / len(actions)) * sum_torque_squared)
    reward = (reward - (-(np.pi**2 + 0.1 * (8**2) + 0.001 * (2**2)))) / (
        np.pi**2 + 0.1 * (8**2) + 0.001 * (2**2)
    )
    return reward


def calculate_reward_c(row):
    theta = row["theta"]
    theta_dot = row["thetadot"]
    action = row["Continuous"]
    reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * action**2)
    reward = (reward - (-(np.pi**2 + 0.1 * (8**2) + 0.001 * (2**2)))) / (
        np.pi**2 + 0.1 * (8**2) + 0.001 * (2**2)
    )
    return reward


def random_policy(x_length: int = 100):
    """Generate one smooth behavior-policy action trajectory.

    The paper simulations use ``x_length=100``. Exposing the grid length keeps
    toy examples and tests small without changing the default paper behavior.
    """
    time_points = np.linspace(0, 1, x_length, endpoint=False)
    dt = 1 / x_length
    tau = np.random.uniform(-2, 2)
    sigma = 1
    control_actions = []
    for _ in time_points:
        tau += sigma * np.sqrt(dt) * np.random.normal()
        tau = np.clip(tau, -2, 2)
        control_actions.append(tau)
    spline = UnivariateSpline(time_points, control_actions, s=0.5)
    smoothed_actions = spline(time_points)
    return smoothed_actions


def generate_unique_seed(random_seed, size, iteration, gpu_id):
    unique_string = f"{random_seed}_{size}_{iteration}_{gpu_id}"
    hash_object = hashlib.sha256(unique_string.encode())
    hash_hex = hash_object.hexdigest()
    unique_seed = int(hash_hex, 16) % (2**32)
    return unique_seed


def compute_bspline_penalty_R(B: np.ndarray, x: np.ndarray):
    """Compute the normalized roughness penalty matrix for a spline basis."""
    _, n_grid = B.shape
    dB = np.gradient(B, x, axis=1)
    d2B = np.gradient(dB, x, axis=1)

    dx = np.diff(x)
    weights = np.zeros(n_grid)
    weights[1:-1] = (dx[:-1] + dx[1:]) / 2
    weights[0] = dx[0] / 2
    weights[-1] = dx[-1] / 2
    W = np.diag(weights)

    penalty = d2B @ W @ d2B.T
    penalty = penalty / np.max(np.abs(penalty))
    return penalty


__all__ = [
    "DEFAULT_X",
    "DEFAULT_Y",
    "calculate_reward",
    "calculate_reward_c",
    "compute_bspline_penalty_R",
    "generate_action",
    "generate_unique_seed",
    "random_policy",
    "safe_append",
    "state_dependent_gp",
]
