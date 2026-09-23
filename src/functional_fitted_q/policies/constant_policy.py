from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .bspline_policy import policy_state_features, policy_state_features_batch


@dataclass
class ConstantActionPolicy:
    """State-dependent continuous scalar action embedded as a constant function."""

    parameters: np.ndarray

    def __post_init__(self) -> None:
        self.parameters = np.asarray(self.parameters, dtype=np.float64)
        if self.parameters.shape != (7,) or not np.isfinite(self.parameters).all():
            raise ValueError("constant policy parameters must have shape (7,)")

    def latent_scalar(self, state: np.ndarray) -> float:
        return float(policy_state_features(state) @ self.parameters)

    def scalar(self, state: np.ndarray) -> float:
        return float(np.clip(self.latent_scalar(state), -2.0, 2.0))

    def latent_values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        features = np.stack(
            [policy_state_features(state) for state in np.asarray(states)]
        )
        scalars = features @ self.parameters
        return np.broadcast_to(scalars[:, None], (len(states), len(u))).copy()

    def values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.clip(self.latent_values_batch(states, u), -2.0, 2.0)

    def action(self, state: np.ndarray):
        value = self.scalar(state)
        return lambda u: np.zeros_like(np.asarray(u), dtype=np.float64) + value


@dataclass
class BoundedConstantActionPolicy:
    """State-dependent continuous constant function with a bounded scalar."""

    parameters: np.ndarray

    def __post_init__(self) -> None:
        self.parameters = np.asarray(self.parameters, dtype=np.float64)
        if self.parameters.shape != (7,) or not np.isfinite(self.parameters).all():
            raise ValueError("constant policy parameters must have shape (7,)")

    def scalar(self, state: np.ndarray) -> float:
        return float(2.0 * np.tanh(policy_state_features(state) @ self.parameters))

    def scalars_batch(self, states: np.ndarray) -> np.ndarray:
        return 2.0 * np.tanh(policy_state_features_batch(states) @ self.parameters)

    def values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        scalars = self.scalars_batch(states)
        return np.broadcast_to(scalars[:, None], (len(scalars), len(u))).copy()

    def latent_values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.values_batch(states, u)

    def action(self, state: np.ndarray):
        value = self.scalar(state)
        return lambda u: np.zeros_like(np.asarray(u), dtype=np.float64) + value
