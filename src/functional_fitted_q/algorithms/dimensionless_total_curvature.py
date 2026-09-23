"""Dimensionless, theory-aligned total-curvature policy improvement.

The manuscript estimand remains the raw, uncentered state-average of integrated
second-derivative energy.  Only deterministic, outcome-independent units are
introduced so that critic value and curvature have comparable numerical scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from .cmaes_policy_optimization import DerivativeFreeCMAES
from .curvature_cmaes import (
    TheoryTotalCurvatureCMAES,
    TheoryTotalCurvatureCMAESConfig,
)
from ..policies.bspline_policy import BoundedCoefficientBSplinePolicy
from ..policies.constant_policy import BoundedConstantActionPolicy


PENALTY_CONVENTION = (
    "theory_uncentered_total_curvature_dimensionless_fixed_sieve_bound_v1"
)
Q_SCALE_RULE = "one_over_one_minus_gamma"
OMEGA_SCALE_RULE = "four_K_times_lambda_max_exact_bspline_curvature"


@dataclass(frozen=True)
class DimensionlessTheoryTotalCurvatureCMAESConfig(TheoryTotalCurvatureCMAESConfig):
    lambda_dimensionless: float = 0.1
    q_scale: float = 20.0
    penalty_convention: str = PENALTY_CONVENTION

    def __post_init__(self) -> None:
        # Bypass the raw-convention subclass check but retain all base checks.
        super(TheoryTotalCurvatureCMAESConfig, self).__post_init__()
        if self.lambda_roughness != 0.0:
            raise ValueError("dimensionless optimizer requires raw lambda_roughness=0")
        if self.penalty_convention != PENALTY_CONVENTION:
            raise ValueError("dimensionless penalty identity changed")
        if not np.isfinite(self.lambda_dimensionless) or self.lambda_dimensionless < 0:
            raise ValueError("dimensionless lambda must be finite and nonnegative")
        if not np.isfinite(self.q_scale) or self.q_scale <= 0:
            raise ValueError("q_scale must be finite and positive")


class DimensionlessTheoryTotalCurvatureCMAES(TheoryTotalCurvatureCMAES):
    """Full CMA-ES with fixed units and a nested zero-curvature validity floor."""

    algorithm_schema = "adafnn_dimensionless_total_curvature_cmaes_v1"

    def __init__(self, config, quadrature_grid, device):
        if not isinstance(config, DimensionlessTheoryTotalCurvatureCMAESConfig):
            raise TypeError("dimensionless total-curvature config required")
        # TheoryTotalCurvatureCMAES checks the raw convention, so initialize the
        # common CMA-ES base directly while retaining its exact-curvature methods.
        DerivativeFreeCMAES.__init__(self, config, quadrature_grid, device)

    def _spline_operators(self, k, degree):
        basis, matrix = TheoryTotalCurvatureCMAES._spline_operators(self, k, degree)
        values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
        tolerance = max(1.0, float(values[-1])) * 1.0e-12
        if float(values[0]) < -tolerance:
            raise RuntimeError("exact curvature matrix is not PSD within roundoff")
        # Exact R_K is PSD with a two-dimensional linear-function nullspace.
        # Project only negative floating-point eigenvalues to zero.
        matrix = (vectors * np.maximum(values, 0.0)) @ vectors.T
        return basis, (matrix + matrix.T) / 2.0

    @staticmethod
    def _omega_scale_from_matrix(matrix: np.ndarray) -> float:
        matrix = np.asarray(matrix, dtype=np.float64)
        largest = float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)[-1])
        scale = 4.0 * matrix.shape[0] * largest
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("invalid deterministic curvature scale")
        return scale

    def _objective_components(
        self, candidates, states, critic, basis_values, roughness_operator
    ):
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
        total_curvature = np.maximum(total_curvature, 0.0)
        omega_scale = self._omega_scale_from_matrix(roughness_operator)
        objective = (
            q_mean / self.config.q_scale
            - self.config.lambda_dimensionless * total_curvature / omega_scale
        )
        return q_mean, total_curvature, objective, omega_scale

    def _functional_objective(
        self, candidates, states, critic, basis_values, roughness_operator
    ):
        temporary = not hasattr(self, "_full_penalty_features")
        if temporary:
            from ..policies.bspline_policy import policy_state_features_batch

            self._full_penalty_features = torch.as_tensor(
                policy_state_features_batch(states),
                dtype=torch.float64,
                device=self.device,
            )
        try:
            return self._objective_components(
                candidates, states, critic, basis_values, roughness_operator
            )[2]
        finally:
            if temporary:
                del self._full_penalty_features

    @staticmethod
    def _constant_seed(initial: BoundedCoefficientBSplinePolicy) -> int:
        digest = hashlib.sha256(
            np.ascontiguousarray(initial.parameter_matrix).view(np.uint8)
        ).digest()
        return int.from_bytes(digest[:8], "little")

    def optimize(self, critic, states, initial, rng, **kwargs):
        """Optimize the dimensionless objective with deterministic safety floors."""
        from ..policies.bspline_policy import policy_state_features_batch

        states = np.asarray(states, dtype=np.float64)
        self._full_penalty_features = torch.as_tensor(
            policy_state_features_batch(states),
            dtype=torch.float64,
            device=self.device,
        )
        try:
            # Call the common optimizer directly: dynamic dispatch still uses the
            # dimensionless objective above, while avoiding the raw-class metrics.
            full_policy, trace, state = DerivativeFreeCMAES.optimize(
                self, critic, states, initial, rng, **kwargs
            )
            objective_states = states[state.objective_state_indices]
            constant_initial = BoundedConstantActionPolicy(
                initial.parameter_matrix.mean(axis=1)
            )
            constant_policy, _, _ = DerivativeFreeCMAES.optimize_constant(
                self,
                critic,
                objective_states,
                constant_initial,
                np.random.default_rng(self._constant_seed(initial)),
                checkpoint_callback=None,
                checkpoint_every_generations=5,
                checkpoint_every_seconds=30.0,
            )
            k = initial.k
            candidates = {
                "full_cmaes": full_policy.parameter_matrix,
                "constant_in_u_cmaes": np.repeat(
                    constant_policy.parameters[:, None], k, axis=1
                ),
                "zero_action": np.zeros_like(initial.parameter_matrix),
            }
            basis, matrix = self._spline_operators(k, initial.degree)
            candidate_names = list(candidates)
            candidate_stack = np.stack([candidates[name] for name in candidate_names])
            q_mean, curvature, objectives, omega_scale = self._objective_components(
                candidate_stack, objective_states, critic, basis, matrix
            )
            best_index = int(np.argmax(objectives))
            selected_name = candidate_names[best_index]
            selected_parameters = candidate_stack[best_index]
            tolerance = 1.0e-10
            if float(objectives[best_index]) + tolerance < max(
                float(objectives[1]), float(objectives[2])
            ):
                raise RuntimeError("optimizer validity floor failed")
            state.best_parameters = selected_parameters.copy()
            state.best_objective = float(objectives[best_index])
            floor_record = {
                "restart": -1,
                "generation": -1,
                "objective_evaluations": int(state.evaluations),
                "best_objective": float(objectives[best_index]),
                "generation_mean_objective": float(np.mean(objectives)),
                "generation_sd_objective": float(np.std(objectives, ddof=1)),
                "sigma": 0.0,
                "covariance_condition": 1.0,
                "termination_reason": "nested_zero_curvature_floor",
                "optimizer_floor_selected": selected_name,
            }
            trace.append(floor_record)
            state.trace = list(trace)
            policy = BoundedCoefficientBSplinePolicy(
                selected_parameters, initial.degree
            )
            metrics = self.penalty_metrics(policy, states)
            metrics.update(
                q_mean=float(q_mean[best_index]),
                q_normalized=float(q_mean[best_index] / self.config.q_scale),
                dimensionless_objective=float(objectives[best_index]),
                penalized_objective=float(objectives[best_index]),
                objective_state_count=len(objective_states),
                objective_state_indices=state.objective_state_indices.tolist(),
                penalty_state_count=len(states),
                penalty_state_scope="all_dataset_next_states",
                optimizer_floor_selected=selected_name,
                optimizer_floor_candidate_objectives={
                    name: float(value)
                    for name, value in zip(candidate_names, objectives)
                },
                optimizer_floor_candidate_q_means={
                    name: float(value) for name, value in zip(candidate_names, q_mean)
                },
                optimizer_floor_candidate_raw_curvatures={
                    name: float(value)
                    for name, value in zip(candidate_names, curvature)
                },
                optimizer_floor_tolerance=tolerance,
                omega_scale_recomputed=float(omega_scale),
                arithmetic_reconstruction_error=float(
                    q_mean[best_index] / self.config.q_scale
                    - self.config.lambda_dimensionless
                    * curvature[best_index]
                    / omega_scale
                    - objectives[best_index]
                ),
            )
            self.last_policy_objective_metrics = metrics
            return policy, trace, state
        finally:
            del self._full_penalty_features

    def penalty_metrics(self, policy, states):
        """Return raw and scaled curvature for the selected functional policy."""
        _, matrix = self._spline_operators(policy.k, policy.degree)
        from ..policies.bspline_policy import policy_state_features_batch

        coefficients = 2.0 * np.tanh(
            policy_state_features_batch(states) @ policy.parameter_matrix
        )
        n = len(coefficients)
        total = max(
            0.0,
            float(np.einsum("nk,kl,nl->", coefficients, matrix, coefficients) / n),
        )
        centered_coefficients = coefficients - coefficients.mean(axis=0, keepdims=True)
        centered = max(
            0.0,
            float(
                np.einsum(
                    "nk,kl,nl->",
                    centered_coefficients,
                    matrix,
                    centered_coefficients,
                )
                / (n - 1)
            ),
        )
        mean_curve = max(0.0, total - (n - 1.0) * centered / n)
        matrix_scale = float(np.max(np.abs(matrix)))
        raw = {
            "penalty_convention": PENALTY_CONVENTION,
            "raw_total_curvature_penalty": total,
            "raw_centered_covariance_curvature": centered,
            "raw_mean_curve_curvature": mean_curve,
            "roughness_matrix_raw_maxabs": matrix_scale,
            "normalized_total_curvature_diagnostic": total / matrix_scale,
            "normalized_centered_covariance_diagnostic": centered / matrix_scale,
            "normalized_centered_covariance_penalty": centered / matrix_scale,
            "normalized_uncentered_mean_roughness": total / matrix_scale,
            "raw_fd2_uncentered_mean_roughness": total,
        }
        omega_scale = self._omega_scale_from_matrix(matrix)
        raw_total = total
        dimensionless_curvature = raw_total / omega_scale
        dimensionless_weighted = (
            self.config.lambda_dimensionless * dimensionless_curvature
        )
        equivalent_raw_lambda = (
            self.config.lambda_dimensionless * self.config.q_scale / omega_scale
        )
        raw.update(
            penalty_convention=PENALTY_CONVENTION,
            lambda_roughness=equivalent_raw_lambda,
            lambda_dimensionless=self.config.lambda_dimensionless,
            q_scale=self.config.q_scale,
            q_scale_rule=Q_SCALE_RULE,
            omega_scale=omega_scale,
            omega_scale_rule=OMEGA_SCALE_RULE,
            dimensionless_total_curvature=dimensionless_curvature,
            dimensionless_weighted_policy_penalty=dimensionless_weighted,
            equivalent_raw_lambda=equivalent_raw_lambda,
            weighted_policy_penalty=equivalent_raw_lambda * raw_total,
        )
        return raw


__all__ = [
    "DimensionlessTheoryTotalCurvatureCMAES",
    "DimensionlessTheoryTotalCurvatureCMAESConfig",
    "OMEGA_SCALE_RULE",
    "PENALTY_CONVENTION",
    "Q_SCALE_RULE",
]
