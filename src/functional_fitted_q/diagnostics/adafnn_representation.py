"""Finite-design feature geometry and projections of observed critic differences.

The uncentered empirical second moment and its ridge inverse are diagnostics.
They do not replace the full-domain projection U_v or spectral weight W_v in
the coverage theorem. Projection residuals are retained because final-layer
features need not represent a changing, clipped nonlinear critic exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from ..critics.adafnn_functional_critic import trapezoid_weights


@dataclass(frozen=True)
class AdaFNNRepresentationGeometry:
    second_moment: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    ridge_scales: dict[str, float]
    inverse_by_eta: dict[str, np.ndarray]
    feature_dimension: int


def resample_action_values(
    values: np.ndarray, source_grid: np.ndarray, target_grid: np.ndarray
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_grid = np.asarray(source_grid, dtype=np.float64)
    target_grid = np.asarray(target_grid, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(source_grid):
        raise ValueError("action values and source grid are inconsistent")
    if np.array_equal(source_grid, target_grid):
        return values.astype(np.float32, copy=False)
    interpolated = CubicSpline(source_grid, values, axis=1, bc_type="natural")(
        target_grid
    )
    return np.clip(interpolated, -2.0, 2.0).astype(np.float32)


@torch.no_grad()
def penultimate_features(
    critic,
    states: np.ndarray,
    action_values: np.ndarray,
    action_grid: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 8192,
    append_bias: bool = True,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    target_grid = critic.quadrature_grid.detach().cpu().numpy().astype(np.float64)
    actions = resample_action_values(action_values, action_grid, target_grid)
    outputs: list[np.ndarray] = []
    feature_network = critic.outer_network[:-1]
    for start in range(0, len(states), batch_size):
        stop = min(len(states), start + batch_size)
        state_tensor = torch.as_tensor(states[start:stop], device=device)
        action_tensor = torch.as_tensor(actions[start:stop], device=device)
        scores = critic.action_scores(action_tensor)
        hidden = feature_network(torch.cat([state_tensor, scores], dim=-1))
        if append_bias:
            hidden = torch.cat(
                [
                    torch.ones(
                        (len(hidden), 1), dtype=hidden.dtype, device=hidden.device
                    ),
                    hidden,
                ],
                dim=1,
            )
        outputs.append(hidden.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0)


@torch.no_grad()
def critic_values(
    critic,
    states: np.ndarray,
    action_values: np.ndarray,
    action_grid: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float32)
    target_grid = critic.quadrature_grid.detach().cpu().numpy().astype(np.float64)
    actions = resample_action_values(action_values, action_grid, target_grid)
    outputs: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        stop = min(len(states), start + batch_size)
        outputs.append(
            critic(
                torch.as_tensor(states[start:stop], device=device),
                torch.as_tensor(actions[start:stop], device=device),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    return np.concatenate(outputs)


def build_representation_geometry(
    design_features: np.ndarray,
    etas: tuple[float, ...] = (1.0e-4, 1.0e-3, 1.0e-2),
) -> AdaFNNRepresentationGeometry:
    features = np.asarray(design_features, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("nonempty two-dimensional design features are required")
    second_moment = features.T @ features / len(features)
    second_moment = (second_moment + second_moment.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    dimension = features.shape[1]
    average_eigenvalue = float(np.trace(second_moment) / dimension)
    if not np.isfinite(average_eigenvalue) or average_eigenvalue <= 0.0:
        raise RuntimeError("AdaFNN representation second moment has zero trace")
    ridge_scales: dict[str, float] = {}
    inverse_by_eta: dict[str, np.ndarray] = {}
    identity = np.eye(dimension, dtype=np.float64)
    for eta in etas:
        if eta <= 0:
            raise ValueError("positive representation ridge multipliers are required")
        label = f"{float(eta):.0e}"
        ridge = float(eta) * average_eigenvalue
        ridge_scales[label] = ridge
        inverse_by_eta[label] = np.linalg.solve(
            second_moment + ridge * identity, identity
        )
    return AdaFNNRepresentationGeometry(
        second_moment=second_moment,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        ridge_scales=ridge_scales,
        inverse_by_eta=inverse_by_eta,
        feature_dimension=dimension,
    )


def representation_query_metrics(
    geometry: AdaFNNRepresentationGeometry,
    query_features: np.ndarray,
    *,
    range_tolerances: tuple[float, ...] = (1.0e-8, 1.0e-10, 1.0e-12),
) -> dict[str, np.ndarray]:
    features = np.asarray(query_features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != geometry.feature_dimension:
        raise ValueError("query features do not match the representation geometry")
    result: dict[str, np.ndarray] = {}
    for label, inverse in geometry.inverse_by_eta.items():
        result[f"representation_leverage_eta_{label}"] = np.einsum(
            "qi,ij,qj->q", features, inverse, features, optimize=True
        )
    maximum = float(np.max(geometry.eigenvalues))
    norm = np.maximum(np.linalg.norm(features, axis=1), np.finfo(np.float64).eps)
    for tolerance in range_tolerances:
        label = f"{float(tolerance):.0e}"
        keep = geometry.eigenvalues > float(tolerance) * maximum
        if np.any(keep):
            basis = geometry.eigenvectors[:, keep]
            projected = (features @ basis) @ basis.T
        else:
            projected = np.zeros_like(features)
        result[f"range_residual_tau_{label}"] = (
            np.linalg.norm(features - projected, axis=1) / norm
        )
    return result


def raw_action_locality_metrics(
    query_actions: np.ndarray,
    training_actions: np.ndarray,
    neighbor_indices: np.ndarray,
    action_grid: np.ndarray,
    *,
    radii: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
) -> dict[str, np.ndarray]:
    query_actions = np.asarray(query_actions, dtype=np.float64)
    training_actions = np.asarray(training_actions, dtype=np.float64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    if query_actions.ndim != 2 or training_actions.ndim != 2:
        raise ValueError("functional actions must be matrices")
    if query_actions.shape[1] != training_actions.shape[1]:
        raise ValueError("query and training action grids differ")
    if neighbor_indices.ndim != 2 or len(neighbor_indices) != len(query_actions):
        raise ValueError("state-neighbor indices do not match query rows")
    weights = trapezoid_weights(np.asarray(action_grid, dtype=np.float64))
    delta = training_actions[neighbor_indices] - query_actions[:, None, :]
    squared = np.einsum("qkg,g,qkg->qk", delta, weights, delta, optimize=True)
    distances = np.sqrt(np.maximum(squared, 0.0))
    result: dict[str, np.ndarray] = {
        "d_min": np.min(distances, axis=1),
        "d_median_32": np.median(distances, axis=1),
    }
    for radius in radii:
        label = f"r{int(round(100 * float(radius))):03d}"
        counts = np.sum(distances <= float(radius), axis=1).astype(np.int64)
        result[f"neighbor_count_{label}"] = counts
        result[f"neighbor_proportion_{label}"] = counts / distances.shape[1]
    return result


def ridge_project_secant(
    geometry: AdaFNNRepresentationGeometry,
    design_features: np.ndarray,
    design_secant: np.ndarray,
    eta_label: str = "1e-03",
) -> dict[str, np.ndarray | float]:
    features = np.asarray(design_features, dtype=np.float64)
    secant = np.asarray(design_secant, dtype=np.float64)
    if features.ndim != 2 or secant.shape != (len(features),):
        raise ValueError("design feature/secant shapes are inconsistent")
    inverse = geometry.inverse_by_eta[eta_label]
    beta = inverse @ (features.T @ secant / len(features))
    fitted = features @ beta
    residual = secant - fitted
    projected_energy = float(np.mean(fitted**2))
    ridge = geometry.ridge_scales[eta_label]
    denominator = projected_energy + ridge * float(beta @ beta)
    return {
        "beta": beta,
        "fitted": fitted,
        "residual": residual,
        "actual_design_energy": float(np.mean(secant**2)),
        "projected_design_energy": projected_energy,
        "behavior_projection_rmse": float(np.sqrt(np.mean(residual**2))),
        "ridge_beta_norm_sq": float(beta @ beta),
        "projected_regularized_denominator": float(denominator),
    }


def extended_ratio(numerator: float, denominator: float, *, exact_zero: bool) -> float:
    """Divide without epsilon: positive/zero is infinity; flagged zero/zero is 0.

    The zero convention is numerical bookkeeping, not evidence of coverage.
    Selected paper fits all have positive actual logged-design denominators.
    """
    numerator = float(numerator)
    denominator = float(denominator)
    if denominator < 0.0 or not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if denominator == 0.0:
        if numerator == 0.0 and exact_zero:
            return 0.0
        if numerator > 0.0:
            return float("inf")
        return float("nan")
    return numerator / denominator


__all__ = [
    "AdaFNNRepresentationGeometry",
    "build_representation_geometry",
    "critic_values",
    "extended_ratio",
    "penultimate_features",
    "raw_action_locality_metrics",
    "representation_query_metrics",
    "resample_action_values",
    "ridge_project_secant",
]
