import hashlib
import numpy as np
from filelock import FileLock
from scipy.interpolate import UnivariateSpline


DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def safe_append(path, text):
    lock_path = path + ".lock"
    with FileLock(lock_path):
        with open(path, "a") as file_obj:
            file_obj.write(text)


def generate_unique_seed(random_seed, size, iteration, gpu_id):
    unique_string = f"{random_seed}_{size}_{iteration}_{gpu_id}"
    hash_object = hashlib.sha256(unique_string.encode())
    hash_hex = hash_object.hexdigest()
    return int(hash_hex, 16) % (2**32)


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


def random_policy():
    dt = 0.01
    total_time = 1
    time_points = np.arange(0, total_time, dt)
    tau = np.random.uniform(-2, 2)
    sigma = 1
    control_actions = []
    for _ in time_points:
        tau += sigma * np.sqrt(dt) * np.random.normal()
        tau = np.clip(tau, -2, 2)
        control_actions.append(tau)
    spline = UnivariateSpline(time_points, control_actions, s=0.5)
    return spline(time_points)


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
