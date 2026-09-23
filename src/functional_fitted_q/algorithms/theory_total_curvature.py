"""Theory-aligned uncentered total curvature for functional policies.

For a spline action a_s(u)=B(u)^T c(s), the implemented quantity is

    mean_s c(s)^T [integral B''(u) B''(u)^T du] c(s).

The state coefficients are not centered and the curvature matrix is not
max-absolute normalized.  For cubic splines, two-node Gauss--Legendre
quadrature is applied on every nonempty knot span.  Because the product of two
second derivatives is piecewise quadratic, this computes the defining
integral exactly up to floating-point error.
"""

from __future__ import annotations


import numpy as np
import torch


PENALTY_CONVENTION = (
    "theory_uncentered_total_curvature_raw_l2_exact_bspline_d2_gauss_legendre_v1"
)


def validate_action_grid(grid: np.ndarray) -> None:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 3 or not np.isfinite(grid).all():
        raise ValueError("curvature grid must contain at least three finite points")
    spacing = np.diff(grid)
    if np.any(spacing <= 0):
        raise ValueError("curvature grid must be strictly increasing")


def total_curvature_matrix(policy, grid: np.ndarray) -> tuple[np.ndarray, dict]:
    grid = np.asarray(grid, dtype=np.float64)
    validate_action_grid(grid)
    if getattr(policy, "degree", None) != 3:
        raise ValueError("the exact two-node curvature rule is frozen to cubic splines")
    knots = np.asarray(policy.knots, dtype=np.float64)
    spans = [
        (float(left), float(right))
        for left, right in zip(np.unique(knots)[:-1], np.unique(knots)[1:])
        if right > left
    ]
    if not spans or spans[0][0] != 0.0 or spans[-1][1] != 1.0:
        raise ValueError("B-spline knot spans must partition [0,1]")
    gaussian = 1.0 / np.sqrt(3.0)
    matrix = np.zeros((policy.k, policy.k), dtype=np.float64)
    for left, right in spans:
        midpoint = (left + right) / 2.0
        halfwidth = (right - left) / 2.0
        points = midpoint + halfwidth * np.asarray([-gaussian, gaussian])
        second_basis = np.asarray(
            policy.basis_matrix(points, derivative=2), dtype=np.float64
        )
        if second_basis.shape != (2, policy.k) or not np.isfinite(second_basis).all():
            raise ValueError("analytic second-derivative basis matrix is invalid")
        matrix += halfwidth * (second_basis.T @ second_basis)
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = 1.0e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if float(eigenvalues.min()) < -tolerance:
        raise RuntimeError("curvature matrix is not numerically positive semidefinite")
    matrix = np.ascontiguousarray(matrix, dtype="<f8")
    receipt = {
        "schema_version": 1,
        "penalty_convention": PENALTY_CONVENTION,
        "derivative_order": 2,
        "spline_degree": 3,
        "state_coefficient_centering": False,
        "state_average": "all_training_next_states",
        "functional_norm": "squared_L2_0_1",
        "basis_derivative": "analytic_scipy_bspline_derivative_2",
        "quadrature": "two_node_gauss_legendre_per_nonempty_knot_span",
        "quadrature_exactness": "exact_for_piecewise_quadratic_Bspline_second_derivative_products",
        "quadrature_points": int(2 * len(spans)),
        "declared_action_grid_points": int(len(grid)),
        "matrix_normalized": False,
        "matrix_shape": list(matrix.shape),
        "matrix": matrix.tolist(),
        "matrix_max_abs": float(np.max(np.abs(matrix))),
        "matrix_trace": float(np.trace(matrix)),
        "matrix_min_eigenvalue": float(eigenvalues.min()),
        "matrix_max_eigenvalue": float(eigenvalues.max()),
        "action_grid": np.asarray(grid).tolist(),
    }
    return matrix, receipt


def torch_uncentered_total_curvature(
    coefficients: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Average c(s)^T R c(s) over the state axis (second-to-last)."""
    if coefficients.ndim < 2 or coefficients.shape[-2] < 1:
        raise ValueError("curvature coefficients need a nonempty state axis")
    if matrix.shape != (coefficients.shape[-1], coefficients.shape[-1]):
        raise ValueError("coefficient/curvature-matrix dimension mismatch")
    if not torch.isfinite(coefficients).all() or not torch.isfinite(matrix).all():
        raise ValueError("curvature inputs must be finite")
    values = (
        (torch.matmul(coefficients, matrix) * coefficients).sum(dim=-1).mean(dim=-1)
    )
    tolerance = 1.0e-10 * torch.clamp(values.abs().max(), min=1.0)
    if torch.any(values < -tolerance):
        raise RuntimeError("negative total curvature beyond numerical tolerance")
    return values.clamp_min(0.0)
