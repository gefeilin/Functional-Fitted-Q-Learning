from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def state_features(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (3,):
        raise ValueError("state must be (cos(theta), sin(theta), omega)")
    cos_theta, sin_theta, omega = state
    w = omega / 8.0
    return np.array([1.0, sin_theta, cos_theta, w, w * cos_theta], dtype=np.float64)


def state_features_batch(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("states must have shape (n,3)")
    cos_theta, sin_theta, omega = values.T
    w = omega / 8.0
    return np.column_stack(
        [np.ones(len(values)), sin_theta, cos_theta, w, w * cos_theta]
    )


@dataclass(frozen=True)
class AnalyticReferencePolicy:
    """Calibration-frozen smooth benchmark; beta has shape (3,5)."""

    beta: np.ndarray

    def __post_init__(self) -> None:
        beta = np.asarray(self.beta, dtype=np.float64)
        if beta.shape != (3, 5) or not np.isfinite(beta).all():
            raise ValueError("reference beta must be a finite (3,5) array")
        object.__setattr__(self, "beta", beta)

    def latent(self, state: np.ndarray, u: float | np.ndarray) -> float | np.ndarray:
        b = self.beta @ state_features(state)
        x = np.asarray(u, dtype=np.float64)
        value = b[0] + b[1] * np.cos(np.pi * x) + b[2] * np.sin(2.0 * np.pi * x)
        return float(value) if x.ndim == 0 else value

    def action(self, state: np.ndarray):
        return lambda u: 2.0 * np.tanh(self.latent(state, u))

    def values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        coefficients = state_features_batch(states) @ self.beta.T
        x = np.asarray(u, dtype=np.float64)
        latent = (
            coefficients[:, [0]]
            + coefficients[:, [1]] * np.cos(np.pi * x)[None, :]
            + coefficients[:, [2]] * np.sin(2.0 * np.pi * x)[None, :]
        )
        return 2.0 * np.tanh(latent)
