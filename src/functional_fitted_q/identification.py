"""Shared representation/locality/secant calculations for the oracle design."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .diagnostics.adafnn_representation import (
    build_representation_geometry,
    extended_ratio,
    raw_action_locality_metrics,
    representation_query_metrics,
    ridge_project_secant,
)


def blocked_state_neighbors(
    query_states: np.ndarray,
    training_states: np.ndarray,
    lengthscales: tuple[float, float, float],
    count: int = 32,
    *,
    block_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query_states, dtype=np.float64)
    training = np.asarray(training_states, dtype=np.float64)
    scale = np.asarray(lengthscales, dtype=np.float64)
    if (
        query.ndim != 2
        or query.shape[1] != 3
        or training.ndim != 2
        or training.shape[1] != 3
    ):
        raise ValueError("state matrices must have three columns")
    if count < 1 or count > len(training):
        raise ValueError("invalid state-neighbor count")
    indices = np.empty((len(query), count), dtype=np.int64)
    squared_out = np.empty((len(query), count), dtype=np.float64)
    for start in range(0, len(query), block_size):
        stop = min(len(query), start + block_size)
        squared = np.sum(
            ((query[start:stop, None, :] - training[None, :, :]) / scale) ** 2,
            axis=2,
        )
        selected = np.argsort(squared, axis=1, kind="stable")[:, :count]
        indices[start:stop] = selected
        squared_out[start:stop] = np.take_along_axis(squared, selected, axis=1)
    return indices, squared_out


def representation_identification(
    *,
    metadata: dict,
    splits: np.ndarray,
    subject_ids: np.ndarray,
    time_indices: np.ndarray,
    query_states: np.ndarray,
    training_actions: np.ndarray,
    action_grid: np.ndarray,
    neighbor_indices: np.ndarray,
    design_features: np.ndarray,
    design_secant: np.ndarray,
    actions_by_class: dict[str, np.ndarray],
    features_by_class: dict[str, np.ndarray],
    actual_secants_by_class: dict[str, np.ndarray],
    representation_ridge_etas: tuple[float, ...] = (1.0e-4, 1.0e-3, 1.0e-2),
    range_tolerances: tuple[float, ...] = (1.0e-8, 1.0e-10, 1.0e-12),
    radii: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Compute locality, leverage, and adjacent-critic secant summaries.

    All query classes share the same held-out states and logged-design
    neighbors.  The function returns query-level records, split-level secant
    energies, representation spectra, and a compact geometry receipt.
    """
    classes = tuple(actions_by_class)
    if classes != tuple(features_by_class) or classes != tuple(actual_secants_by_class):
        raise ValueError("query-class action/feature/secant mappings disagree")
    geometry = build_representation_geometry(
        design_features, etas=representation_ridge_etas
    )
    projection = ridge_project_secant(geometry, design_features, design_secant, "1e-03")
    exact_zero = bool(np.all(np.asarray(design_secant) == 0.0))
    frames = []
    for query_class in classes:
        actions = actions_by_class[query_class]
        features = features_by_class[query_class]
        secant = np.asarray(actual_secants_by_class[query_class], dtype=np.float64)
        projected = features @ projection["beta"]
        locality = raw_action_locality_metrics(
            actions,
            training_actions,
            neighbor_indices,
            action_grid,
            radii=radii,
        )
        representation = representation_query_metrics(
            geometry, features, range_tolerances=range_tolerances
        )
        frames.append(
            pd.DataFrame(
                {
                    **metadata,
                    "query_class": query_class,
                    "split": splits,
                    "subject_id": subject_ids,
                    "time_index": time_indices,
                    "state_cos": query_states[:, 0],
                    "state_sin": query_states[:, 1],
                    "state_omega": query_states[:, 2],
                    "h_actual_m20_minus_m19": secant,
                    "h_projected_m20_representation": projected,
                    "h_projection_residual": secant - projected,
                    **locality,
                    **representation,
                }
            )
        )
    query = pd.concat(frames, ignore_index=True)
    primary_leverage = "representation_leverage_eta_1e-03"
    query["relative_representation_leverage"] = np.nan
    query["locality_leverage_discordance"] = False
    discordance = []
    for split in ("A", "B"):
        behavior = query[
            (query["split"] == split) & (query["query_class"] == "behavior")
        ]
        learned_mask = (query["split"] == split) & (query["query_class"] == "learned")
        behavior_leverage = float(behavior[primary_leverage].median())
        behavior_distance = float(behavior["d_min"].quantile(0.75))
        split_mask = query["split"] == split
        query.loc[split_mask, "relative_representation_leverage"] = (
            query.loc[split_mask, primary_leverage] / behavior_leverage
        )
        marked = (
            learned_mask
            & (query["d_min"] > behavior_distance)
            & (query[primary_leverage] <= behavior_leverage)
        )
        query.loc[marked, "locality_leverage_discordance"] = True
        discordance.append(
            {
                "split": split,
                "behavior_q75_d_min": behavior_distance,
                "behavior_median_representation_leverage": behavior_leverage,
                "learned_query_count": int(learned_mask.sum()),
                "discordant_learned_query_count": int(marked.sum()),
                "discordance_proportion": float(marked.sum() / learned_mask.sum()),
            }
        )

    rows = []
    actual_denominator = float(projection["actual_design_energy"])
    projected_denominator = float(projection["projected_regularized_denominator"])
    for split in ("A", "B"):
        mask = splits == split
        for query_class in classes:
            actual = np.asarray(actual_secants_by_class[query_class])[mask]
            features = features_by_class[query_class][mask]
            projected = features @ projection["beta"]
            residual = actual - projected
            g_actual = float(np.mean(actual**2))
            g_projected = float(np.mean(projected**2))
            leverage = float(
                query[
                    (query["split"] == split) & (query["query_class"] == query_class)
                ][primary_leverage].mean()
            )
            ratio = extended_ratio(
                g_projected, projected_denominator, exact_zero=exact_zero
            )
            tolerance = 1.0e-8 * max(1.0, abs(leverage))
            rows.append(
                {
                    **metadata,
                    "split": split,
                    "query_class": query_class,
                    "query_count": int(mask.sum()),
                    "D_actual_sq": actual_denominator,
                    "G_actual_sq": g_actual,
                    "actual_graph_design_ratio": extended_ratio(
                        g_actual, actual_denominator, exact_zero=exact_zero
                    ),
                    "D_projected_sq": float(projection["projected_design_energy"]),
                    "ridge_beta_norm_sq": float(projection["ridge_beta_norm_sq"]),
                    "projected_regularized_denominator": projected_denominator,
                    "G_projected_sq": g_projected,
                    "projected_graph_design_ratio": ratio,
                    "mean_representation_leverage": leverage,
                    "projected_inequality_slack": leverage - ratio,
                    "projected_inequality_tolerance": tolerance,
                    "projected_inequality_pass": bool(
                        np.isfinite(ratio) and ratio <= leverage + tolerance
                    ),
                    "behavior_projection_rmse": float(
                        projection["behavior_projection_rmse"]
                    ),
                    "graph_projection_rmse": float(np.sqrt(np.mean(residual**2))),
                    "exact_zero_secant": exact_zero,
                }
            )
    secant = pd.DataFrame(rows)
    spectrum = pd.DataFrame(
        {
            **{key: value for key, value in metadata.items()},
            "eigenvalue_index": np.arange(len(geometry.eigenvalues)),
            "eigenvalue_ascending": geometry.eigenvalues,
        }
    )
    geometry_metadata = {
        "feature_dimension": geometry.feature_dimension,
        "uncentered_second_moment": True,
        "ridge_scales": geometry.ridge_scales,
        "range_relative_tolerances": list(range_tolerances),
        "minimum_eigenvalue": float(np.min(geometry.eigenvalues)),
        "maximum_eigenvalue": float(np.max(geometry.eigenvalues)),
        "trace": float(np.sum(geometry.eigenvalues)),
        "discordance": discordance,
    }
    return query, secant, spectrum, geometry_metadata


__all__ = ["blocked_state_neighbors", "representation_identification"]
