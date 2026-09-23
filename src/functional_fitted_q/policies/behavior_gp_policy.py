from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..env.functional_action import GridFunctionalAction
from .reference_policy import AnalyticReferencePolicy


def matern52_covariance(grid: np.ndarray, lengthscale: float) -> np.ndarray:
    distance = np.abs(grid[:, None] - grid[None, :])
    z = np.sqrt(5.0) * distance / lengthscale
    covariance = (1.0 + z + z**2 / 3.0) * np.exp(-z)
    covariance.flat[:: len(grid) + 1] += 1e-10
    return covariance


@dataclass
class BehaviorGPPolicy:
    reference: AnalyticReferencePolicy
    subject_rng: np.random.Generator
    step_rng: np.random.Generator
    resolution: int = 128
    reference_latent_scale: float = 0.70
    subject_lengthscale: float = 0.50
    subject_amplitude: float = 0.15
    step_lengthscale: float = 0.25
    step_amplitude: float = 0.35

    def __post_init__(self) -> None:
        if self.resolution < 8:
            raise ValueError("GP resolution is too small")
        self.grid = np.linspace(0.0, 1.0, self.resolution)
        self._subject_chol = np.linalg.cholesky(
            matern52_covariance(self.grid, self.subject_lengthscale)
        )
        self._step_chol = np.linalg.cholesky(
            matern52_covariance(self.grid, self.step_lengthscale)
        )
        self._subject_effects: dict[int, np.ndarray] = {}
        self._draw_counter = 0

    def subject_effect(self, subject_id: int) -> np.ndarray:
        if subject_id not in self._subject_effects:
            self._subject_effects[subject_id] = self.subject_amplitude * (
                self._subject_chol @ self.subject_rng.standard_normal(self.resolution)
            )
        return self._subject_effects[subject_id]

    def sample(self, state: np.ndarray, subject_id: int) -> GridFunctionalAction:
        reference_values = np.asarray(
            self.reference.action(state)(self.grid), dtype=np.float64
        )
        safe = np.clip(reference_values / 2.0, -1.0 + 1e-10, 1.0 - 1e-10)
        latent_reference = np.arctanh(safe)
        step_effect = self.step_amplitude * (
            self._step_chol @ self.step_rng.standard_normal(self.resolution)
        )
        latent = (
            self.reference_latent_scale * latent_reference
            + self.subject_effect(subject_id)
            + step_effect
        )
        self._draw_counter += 1
        return GridFunctionalAction(self.grid, 2.0 * np.tanh(latent))

    def draw_subject_effects(self, n_subjects: int) -> np.ndarray:
        draws = self.subject_rng.standard_normal((n_subjects, self.resolution))
        return self.subject_amplitude * (draws @ self._subject_chol.T)

    def draw_step_effects(self, n_subjects: int, t_per_subject: int) -> np.ndarray:
        draws = self.step_rng.standard_normal(
            (n_subjects, t_per_subject, self.resolution)
        )
        return self.step_amplitude * (draws @ self._step_chol.T)

    def sample_values_batch(
        self,
        states: np.ndarray,
        subject_effects: np.ndarray,
        step_effects: np.ndarray,
    ) -> np.ndarray:
        reference_values = self.reference.values_batch(states, self.grid)
        safe = np.clip(reference_values / 2.0, -1.0 + 1e-10, 1.0 - 1e-10)
        latent = (
            self.reference_latent_scale * np.arctanh(safe)
            + subject_effects
            + step_effects
        )
        self._draw_counter += len(states)
        return 2.0 * np.tanh(latent)

    @property
    def draw_counter(self) -> int:
        return self._draw_counter
