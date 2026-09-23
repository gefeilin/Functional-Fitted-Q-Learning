"""Exact B-spline curvature and CMA-ES policy optimization.

The paper estimator uses the uncentered, state-averaged integrated squared
second derivative defined in the manuscript.  The dimensionless optimizer
rescales that same quantity by a deterministic sieve-dependent constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
import torch

from .cmaes_policy_optimization import (
    CMAESPolicyOptimizationConfig,
    DerivativeFreeCMAES,
)
from .theory_total_curvature import (
    PENALTY_CONVENTION as THEORY_TOTAL_CURVATURE_CONVENTION,
)
from ..policies.bspline_policy import open_uniform_knots, policy_state_features_batch


def coefficient_penalty_components(coefficients: np.ndarray, matrix: np.ndarray):
    """Batched coefficient arrays (..., states, K); return both distinct moments."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    matrix = np.asarray(matrix, dtype=np.float64)
    if coefficients.ndim < 2 or coefficients.shape[-2] < 2:
        raise ValueError("centered GitHub penalty requires at least two states")
    if matrix.shape != (coefficients.shape[-1], coefficients.shape[-1]):
        raise ValueError("coefficient/matrix dimension mismatch")
    if not np.isfinite(coefficients).all() or not np.isfinite(matrix).all():
        raise ValueError("penalty inputs must be finite")
    centered = coefficients - coefficients.mean(axis=-2, keepdims=True)
    n = coefficients.shape[-2]
    covariance_energy = np.einsum("...nk,kl,...nl->...", centered, matrix, centered) / (
        n - 1
    )
    mean_energy = (
        np.einsum("...nk,kl,...nl->...", coefficients, matrix, coefficients) / n
    )
    # Tiny negative values can arise from cancellation in the PSD quadratic form.
    for value in (covariance_energy, mean_energy):
        if np.any(value < -1e-10):
            raise ValueError(
                "negative second-derivative energy beyond roundoff tolerance"
            )
    return {
        "centered_covariance_penalty": np.maximum(covariance_energy, 0.0),
        "uncentered_mean_roughness": np.maximum(mean_energy, 0.0),
    }


def exact_bspline_curvature_matrix(k: int, degree: int = 3) -> np.ndarray:
    """Return integral B''(u)B''(u)^T du without K-dependent normalization.

    For cubic splines each product is piecewise quadratic, so two-node Gaussian
    quadrature on every nonempty knot span is exact up to floating-point error.
    """
    if k < degree + 1 or degree < 2:
        raise ValueError("curvature requires at least a quadratic B-spline basis")
    knots = open_uniform_knots(k, degree)
    basis = [BSpline(knots, np.eye(k)[j], degree, extrapolate=False) for j in range(k)]
    gaussian = 1.0 / np.sqrt(3.0)
    matrix = np.zeros((k, k), dtype=np.float64)
    for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:]):
        if not right > left:
            continue
        midpoint = (left + right) / 2.0
        halfwidth = (right - left) / 2.0
        points = midpoint + halfwidth * np.array([-gaussian, gaussian])
        second = np.column_stack([item(points, nu=2) for item in basis])
        matrix += halfwidth * (second.T @ second)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.isfinite(matrix).all() or eigenvalues[0] < -1e-9 or eigenvalues[-1] <= 0:
        raise ValueError("invalid exact B-spline curvature matrix")
    return matrix


@dataclass(frozen=True)
class TheoryTotalCurvatureCMAESConfig(CMAESPolicyOptimizationConfig):
    penalty_convention: str = THEORY_TOTAL_CURVATURE_CONVENTION

    def __post_init__(self):
        super().__post_init__()
        if self.penalty_convention != THEORY_TOTAL_CURVATURE_CONVENTION:
            raise ValueError("the theory-aligned total-curvature identity changed")
        if not np.isfinite(self.lambda_roughness):
            raise ValueError("lambda must be finite")


class TheoryTotalCurvatureCMAES(DerivativeFreeCMAES):
    """CMA-ES with the manuscript's uncentered raw state-averaged curvature."""

    algorithm_schema = "adafnn_theory_total_curvature_cmaes_v1"

    def __init__(self, config, quadrature_grid, device):
        if not isinstance(config, TheoryTotalCurvatureCMAESConfig):
            raise TypeError("a theory-total-curvature config is required")
        super().__init__(config, quadrature_grid, device)

    def _spline_operators(self, k, degree):
        basis, _ = super()._spline_operators(k, degree)
        return basis, exact_bspline_curvature_matrix(k, degree)

    def _full_data_total_curvature(self, candidates, roughness_operator):
        features = getattr(self, "_full_penalty_features", None)
        if features is None:
            raise RuntimeError("full next-state penalty features are not initialized")
        candidate_tensor = torch.as_tensor(
            candidates, dtype=torch.float64, device=self.device
        )
        matrix = torch.as_tensor(
            roughness_operator, dtype=torch.float64, device=self.device
        )
        output = []
        with torch.no_grad():
            for candidate_start in range(
                0, len(candidates), self.config.candidate_batch_size
            ):
                block = candidate_tensor[
                    candidate_start : candidate_start + self.config.candidate_batch_size
                ]
                energy_sum = torch.zeros(
                    len(block), dtype=torch.float64, device=self.device
                )
                for state_start in range(0, len(features), 8192):
                    state_block = features[state_start : state_start + 8192]
                    coefficients = 2.0 * torch.tanh(
                        torch.einsum("nf,cfk->cnk", state_block, block)
                    )
                    energy_sum += torch.einsum(
                        "cnk,kl,cnl->c", coefficients, matrix, coefficients
                    )
                output.append((energy_sum / len(features)).cpu().numpy())
        return np.concatenate(output)

    def _functional_objective(
        self, candidates, states, critic, basis_values, roughness_operator
    ):
        temporary = not hasattr(self, "_full_penalty_features")
        if temporary:
            self._full_penalty_features = torch.as_tensor(
                policy_state_features_batch(states),
                dtype=torch.float64,
                device=self.device,
            )
        try:
            q_mean = DerivativeFreeCMAES._functional_objective(
                self,
                candidates,
                states,
                critic,
                basis_values,
                np.zeros_like(roughness_operator),
            )
            total_curvature = self._full_data_total_curvature(
                candidates, roughness_operator
            )
            return q_mean - self.config.lambda_roughness * total_curvature
        finally:
            if temporary:
                del self._full_penalty_features

    def optimize(self, critic, states, initial, rng, **kwargs):
        states = np.asarray(states, dtype=np.float64)
        self._full_penalty_features = torch.as_tensor(
            policy_state_features_batch(states), dtype=torch.float64, device=self.device
        )
        try:
            policy, trace, state = super().optimize(
                critic, states, initial, rng, **kwargs
            )
            objective_states = states[state.objective_state_indices]
            metrics = self.penalty_metrics(policy, states)
            basis, matrix = self._spline_operators(policy.k, policy.degree)
            q_mean = DerivativeFreeCMAES._functional_objective(
                self,
                policy.parameter_matrix[None],
                objective_states,
                critic,
                basis,
                np.zeros_like(matrix),
            )[0]
            metrics.update(
                q_mean=float(q_mean),
                penalized_objective=float(state.best_objective),
                objective_state_count=len(objective_states),
                objective_state_indices=state.objective_state_indices.tolist(),
                penalty_state_count=len(states),
                penalty_state_scope="all_dataset_next_states",
                arithmetic_reconstruction_error=float(
                    q_mean - metrics["weighted_policy_penalty"] - state.best_objective
                ),
            )
            self.last_policy_objective_metrics = metrics
            return policy, trace, state
        finally:
            del self._full_penalty_features

    def penalty_metrics(self, policy, states):
        _, matrix = self._spline_operators(policy.k, policy.degree)
        features = policy_state_features_batch(states)
        coefficients = 2.0 * np.tanh(features @ policy.parameter_matrix)
        parts = coefficient_penalty_components(coefficients, matrix)
        n = len(coefficients)
        total = float(parts["uncentered_mean_roughness"])
        centered = float(parts["centered_covariance_penalty"])
        mean_curve = max(0.0, total - (n - 1.0) * centered / n)
        scale = float(np.max(np.abs(matrix)))
        return {
            "penalty_convention": THEORY_TOTAL_CURVATURE_CONVENTION,
            "lambda_roughness": self.config.lambda_roughness,
            "roughness_matrix_raw_maxabs": scale,
            "raw_total_curvature_penalty": total,
            "raw_centered_covariance_curvature": centered,
            "raw_mean_curve_curvature": mean_curve,
            "normalized_total_curvature_diagnostic": total / scale,
            "normalized_centered_covariance_diagnostic": centered / scale,
            # Compatibility fields remain explicitly diagnostic for common readers.
            "normalized_centered_covariance_penalty": centered / scale,
            "normalized_uncentered_mean_roughness": total / scale,
            "raw_fd2_uncentered_mean_roughness": total,
            "weighted_policy_penalty": float(self.config.lambda_roughness * total),
        }
