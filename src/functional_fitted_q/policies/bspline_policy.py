"""State-dependent spline policies, with paper dimension p_u stored as k.

The seven features z(s) map a (7, p_u) parameter matrix C to spline
coefficients. Paper fits use BoundedCoefficientBSplinePolicy: coefficient
bounds preserve smooth splines without clipping the resulting curve.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline


def policy_state_features(state: np.ndarray) -> np.ndarray:
    """Return z(s), using omega/8 to scale the velocity feature."""
    cos_theta, sin_theta, omega = np.asarray(state, dtype=np.float64)
    w = omega / 8.0
    return np.array(
        [1.0, sin_theta, cos_theta, w, w * sin_theta, w * cos_theta, w**2],
        dtype=np.float64,
    )


def policy_state_features_batch(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 3:
        raise ValueError("states must have shape (n,3)")
    cos_theta, sin_theta, omega = states.T
    w = omega / 8.0
    return np.column_stack(
        [
            np.ones(len(states), dtype=np.float64),
            sin_theta,
            cos_theta,
            w,
            w * sin_theta,
            w * cos_theta,
            w**2,
        ]
    )


def open_uniform_knots(k: int, degree: int = 3) -> np.ndarray:
    if k < degree + 1:
        raise ValueError("K must be at least degree+1")
    n_internal = k - degree - 1
    internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1]
    return np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])


@dataclass
class BSplinePolicy:
    """Pointwise-clipped variant; paper fits use the bounded-coefficient class."""

    parameter_matrix: np.ndarray
    degree: int = 3

    def __post_init__(self) -> None:
        matrix = np.asarray(self.parameter_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != 7 or not np.isfinite(matrix).all():
            raise ValueError("parameter_matrix must be finite with shape (7,K)")
        self.parameter_matrix = matrix
        self.k = matrix.shape[1]
        self.knots = open_uniform_knots(self.k, self.degree)
        self._basis = [
            BSpline(self.knots, np.eye(self.k)[j], self.degree, extrapolate=False)
            for j in range(self.k)
        ]

    def coefficients(self, state: np.ndarray) -> np.ndarray:
        """Latent, unconstrained B-spline coefficients C^T z(s)."""
        return self.parameter_matrix.T @ policy_state_features(state)

    def basis_matrix(self, u: np.ndarray, derivative: int = 0) -> np.ndarray:
        x = np.asarray(u, dtype=np.float64)
        return np.column_stack([basis(x, nu=derivative) for basis in self._basis])

    def latent_values(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.basis_matrix(u) @ self.coefficients(state)

    def values(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.clip(self.latent_values(state, u), -2.0, 2.0)

    def latent_values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        features = policy_state_features_batch(states)
        coefficients = features @ self.parameter_matrix
        return coefficients @ self.basis_matrix(u).T

    def values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.clip(self.latent_values_batch(states, u), -2.0, 2.0)

    def action(self, state: np.ndarray):
        return lambda u: (
            self.values(state, np.atleast_1d(u))
            if np.ndim(u)
            else float(self.values(state, np.array([u]))[0])
        )

    def roughness(self, states: np.ndarray, quadrature_size: int = 257) -> float:
        """Mean latent-preclip second-derivative roughness."""
        grid = np.linspace(0.0, 1.0, quadrature_size)
        second_basis = self.basis_matrix(grid, derivative=2)
        values = []
        for state in np.asarray(states):
            second = second_basis @ self.coefficients(state)
            values.append(np.trapz(second**2, grid))
        return float(np.mean(values))


@dataclass
class BoundedCoefficientBSplinePolicy:
    """Implement pi_C(s)(u) = B_pu(u)^T [2*tanh(C^T z(s))].

    Cubic B-splines are nonnegative and sum to one on [0, 1], so bounding
    each coefficient in [-2, 2] bounds the entire torque curve there.
    """

    parameter_matrix: np.ndarray
    degree: int = 3

    def __post_init__(self) -> None:
        matrix = np.asarray(self.parameter_matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != 7 or not np.isfinite(matrix).all():
            raise ValueError("parameter_matrix must be finite with shape (7,K)")
        self.parameter_matrix = matrix
        self.k = matrix.shape[1]
        self.knots = open_uniform_knots(self.k, self.degree)
        self._basis = [
            BSpline(self.knots, np.eye(self.k)[j], self.degree, extrapolate=False)
            for j in range(self.k)
        ]

    def basis_matrix(self, u: np.ndarray, derivative: int = 0) -> np.ndarray:
        x = np.asarray(u, dtype=np.float64)
        return np.column_stack([basis(x, nu=derivative) for basis in self._basis])

    def coefficients(self, state: np.ndarray) -> np.ndarray:
        return 2.0 * np.tanh(self.parameter_matrix.T @ policy_state_features(state))

    def coefficients_batch(self, states: np.ndarray) -> np.ndarray:
        """Return rows c_C(s)^T, with shape (number_of_states, p_u)."""
        return 2.0 * np.tanh(
            policy_state_features_batch(states) @ self.parameter_matrix
        )

    def values(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.basis_matrix(u) @ self.coefficients(state)

    def values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.coefficients_batch(states) @ self.basis_matrix(u).T

    def latent_values(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.values(state, u)

    def latent_values_batch(self, states: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.values_batch(states, u)

    def action(self, state: np.ndarray):
        return lambda u: (
            self.values(state, np.atleast_1d(u))
            if np.ndim(u)
            else float(self.values(state, np.array([u]))[0])
        )

    def roughness(self, states: np.ndarray, quadrature_size: int = 257) -> float:
        """Grid approximation for inspection; training uses exact W_pu instead."""
        grid = np.linspace(0.0, 1.0, quadrature_size)
        second_basis = self.basis_matrix(grid, derivative=2)
        second = self.coefficients_batch(states) @ second_basis.T
        return float(np.mean(np.trapz(second**2, grid, axis=1)))
