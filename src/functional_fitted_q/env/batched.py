from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

from .dynamics import PendulumConfig, wrap_angle


def _rhs_batch(
    theta: np.ndarray,
    omega: np.ndarray,
    torque: np.ndarray,
    config: PendulumConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrapped = wrap_angle(theta)
    omega_for_cost = np.clip(omega, -config.omega_limit, config.omega_limit)
    running_cost = wrapped**2 + 0.1 * omega_for_cost**2 + 0.001 * torque**2
    dtheta = config.duration * omega
    domega = config.duration * (
        3.0 * config.g / (2.0 * config.l) * np.sin(theta)
        + 3.0 * torque / (config.m * config.l**2)
    )
    return dtheta, domega, running_cost


def deterministic_transition_batch(
    states: np.ndarray,
    torque_values: np.ndarray,
    config: PendulumConfig,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    torque_values = np.asarray(torque_values, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (n,3)")
    if torque_values.shape != (len(states), 2 * config.ode_substeps + 1):
        raise ValueError("torque_values has an incompatible batch/quadrature shape")
    if (
        not np.isfinite(torque_values).all()
        or np.max(np.abs(torque_values)) > config.torque_limit + 1e-10
    ):
        raise ValueError("functional actions violate finite torque bounds")

    theta = np.arctan2(states[:, 1], states[:, 0]).astype(np.float64, copy=True)
    omega = states[:, 2].astype(np.float64, copy=True)
    accumulated_cost = np.zeros(len(states), dtype=np.float64)
    h = 1.0 / config.ode_substeps
    for substep in range(config.ode_substeps):
        torque_left = torque_values[:, 2 * substep]
        torque_middle = torque_values[:, 2 * substep + 1]
        torque_right = torque_values[:, 2 * substep + 2]
        k1 = _rhs_batch(theta, omega, torque_left, config)
        k2 = _rhs_batch(
            theta + 0.5 * h * k1[0],
            omega + 0.5 * h * k1[1],
            torque_middle,
            config,
        )
        k3 = _rhs_batch(
            theta + 0.5 * h * k2[0],
            omega + 0.5 * h * k2[1],
            torque_middle,
            config,
        )
        k4 = _rhs_batch(
            theta + h * k3[0],
            omega + h * k3[1],
            torque_right,
            config,
        )
        theta += h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        omega += h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        accumulated_cost += h * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        omega = np.clip(omega, -config.omega_limit, config.omega_limit)
    deterministic_states = np.column_stack([np.cos(theta), np.sin(theta), omega])
    rewards = np.clip(1.0 - accumulated_cost / config.max_cost, 0.0, 1.0)
    return deterministic_states, rewards


def apply_transition_noise_batch(
    deterministic_states: np.ndarray,
    theta_normals: np.ndarray,
    omega_uniforms: np.ndarray,
    config: PendulumConfig,
) -> np.ndarray:
    deterministic_states = np.asarray(deterministic_states, dtype=np.float64)
    theta = np.arctan2(deterministic_states[:, 1], deterministic_states[:, 0])
    if config.sigma_theta:
        theta = wrap_angle(theta + config.sigma_theta * np.asarray(theta_normals))
    omega = deterministic_states[:, 2]
    if config.sigma_omega:
        lower = (-config.omega_limit - omega) / config.sigma_omega
        upper = (config.omega_limit - omega) / config.sigma_omega
        lower_probability = ndtr(lower)
        probability = lower_probability + np.asarray(omega_uniforms) * (
            ndtr(upper) - lower_probability
        )
        probability = np.clip(
            probability, np.finfo(float).tiny, 1.0 - np.finfo(float).eps
        )
        omega = omega + config.sigma_omega * ndtri(probability)
    return np.column_stack(
        [
            np.cos(theta),
            np.sin(theta),
            np.clip(omega, -config.omega_limit, config.omega_limit),
        ]
    )
