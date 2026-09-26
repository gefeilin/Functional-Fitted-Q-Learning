"""Scalar Pendulum dynamics for one function-valued MDP action.

The within-action coordinate u ranges over [0, 1]; physical time advances by
config.duration. Reward integrates normalized-time cost before endpoint noise
is applied. env/batched.py implements the same calculations in batches.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import truncnorm

from .functional_action import FunctionalAction


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    value = (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi
    return float(value) if np.ndim(theta) == 0 else value


@dataclass(frozen=True)
class PendulumConfig:
    g: float = 10.0
    m: float = 1.0
    l: float = 1.0
    torque_limit: float = 2.0
    omega_limit: float = 8.0
    duration: float = 0.5
    ode_substeps: int = 128
    sigma_theta: float = 0.03
    sigma_omega: float = 0.05
    gamma: float = 0.95

    def __post_init__(self) -> None:
        if self.ode_substeps <= 0 or self.duration <= 0:
            raise ValueError("duration and ode_substeps must be positive")
        if self.sigma_theta < 0 or self.sigma_omega < 0:
            raise ValueError("transition noise scales must be nonnegative")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie in (0,1)")

    @property
    def max_cost(self) -> float:
        return math.pi**2 + 0.1 * self.omega_limit**2 + 0.001 * self.torque_limit**2


class FunctionalPendulumEnv:
    """Continuing Pendulum where one MDP action is a continuous torque curve."""

    def __init__(self, config: PendulumConfig | None = None, seed: int = 0):
        self.config = config or PendulumConfig()
        self.rng = np.random.default_rng(seed)
        self.theta = 0.0
        self.omega = 0.0

    @property
    def state(self) -> np.ndarray:
        return np.array(
            [math.cos(self.theta), math.sin(self.theta), self.omega], dtype=np.float64
        )

    def reset(
        self, *, theta: float | None = None, omega: float | None = None
    ) -> np.ndarray:
        self.theta = float(
            self.rng.uniform(-math.pi, math.pi) if theta is None else wrap_angle(theta)
        )
        self.omega = float(
            self.rng.uniform(-1.0, 1.0)
            if omega is None
            else np.clip(omega, -self.config.omega_limit, self.config.omega_limit)
        )
        return self.state

    def _rhs(self, u: float, y: np.ndarray, action: FunctionalAction) -> np.ndarray:
        theta, omega, _ = y
        torque = float(action(float(np.clip(u, 0.0, 1.0))))
        if not math.isfinite(torque) or abs(torque) > self.config.torque_limit + 1e-10:
            raise ValueError("action violates torque bounds")
        wrapped = float(wrap_angle(theta))
        omega_for_cost = float(
            np.clip(omega, -self.config.omega_limit, self.config.omega_limit)
        )
        running_cost = wrapped**2 + 0.1 * omega_for_cost**2 + 0.001 * torque**2
        # Chain rule from physical time to u. The cost is already integrated
        # over u, so it does not receive a second duration factor.
        dtheta = self.config.duration * omega
        domega = self.config.duration * (
            3.0 * self.config.g / (2.0 * self.config.l) * math.sin(theta)
            + 3.0 * torque / (self.config.m * self.config.l**2)
        )
        return np.array([dtheta, domega, running_cost], dtype=np.float64)

    def deterministic_transition(
        self, theta: float, omega: float, action: FunctionalAction
    ) -> tuple[float, float, float]:
        h = 1.0 / self.config.ode_substeps
        y = np.array([float(theta), float(omega), 0.0], dtype=np.float64)
        for j in range(self.config.ode_substeps):
            u = j * h
            k1 = self._rhs(u, y, action)
            k2 = self._rhs(u + 0.5 * h, y + 0.5 * h * k1, action)
            k3 = self._rhs(u + 0.5 * h, y + 0.5 * h * k2, action)
            k4 = self._rhs(u + h, y + h * k3, action)
            y = y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            y[1] = np.clip(y[1], -self.config.omega_limit, self.config.omega_limit)
        endpoint_theta = float(wrap_angle(y[0]))
        endpoint_omega = float(
            np.clip(y[1], -self.config.omega_limit, self.config.omega_limit)
        )
        reward = float(np.clip(1.0 - y[2] / self.config.max_cost, 0.0, 1.0))
        return endpoint_theta, endpoint_omega, reward

    def _noisy_endpoint(self, theta: float, omega: float) -> tuple[float, float]:
        cfg = self.config
        noisy_theta = float(
            wrap_angle(
                theta
                + (self.rng.normal(0.0, cfg.sigma_theta) if cfg.sigma_theta else 0.0)
            )
        )
        if cfg.sigma_omega == 0:
            noisy_omega = omega
        else:
            lower = (-cfg.omega_limit - omega) / cfg.sigma_omega
            upper = (cfg.omega_limit - omega) / cfg.sigma_omega
            noisy_omega = float(
                truncnorm.rvs(
                    lower,
                    upper,
                    loc=omega,
                    scale=cfg.sigma_omega,
                    random_state=self.rng,
                )
            )
        return noisy_theta, noisy_omega

    def step(self, action: FunctionalAction) -> tuple[np.ndarray, float, dict]:
        theta_det, omega_det, reward = self.deterministic_transition(
            self.theta, self.omega, action
        )
        self.theta, self.omega = self._noisy_endpoint(theta_det, omega_det)
        return (
            self.state,
            reward,
            {
                "theta_deterministic": theta_det,
                "omega_deterministic": omega_det,
                "theta": self.theta,
                "omega": self.omega,
                "raw_cost": (1.0 - reward) * self.config.max_cost,
            },
        )
