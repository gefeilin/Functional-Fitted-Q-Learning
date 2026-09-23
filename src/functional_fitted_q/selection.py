"""Choose policy regularization using independent simulator tuning returns."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def select_lambda(
    candidates: Iterable[dict],
    *,
    expected_lambdas: Iterable[float],
    tuning_seed: int,
    episodes: int = 1000,
) -> dict:
    """Select the largest tuning mean, breaking exact ties toward smaller lambda.

    Each candidate must supply ``fit_id``, ``lambda_dimensionless``,
    ``normalized_returns``, ``evaluation_seed``, and ``stream``.  This function
    is intentionally unable to read reporting-stream outcomes.
    """
    rows = list(candidates)
    expected = tuple(sorted(float(value) for value in expected_lambdas))
    if expected != (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0):
        raise ValueError("candidate coefficients do not match the paper grid")
    if len(rows) != len(expected):
        raise ValueError("one tuning result per candidate lambda is required")

    normalized = []
    seen = set()
    for row in rows:
        policy_lambda = float(row["lambda_dimensionless"])
        if policy_lambda in seen or policy_lambda not in expected:
            raise ValueError("duplicate or unexpected candidate lambda")
        seen.add(policy_lambda)
        if row.get("stream") != "tuning":
            raise ValueError("only the tuning stream may select lambda")
        if int(row["evaluation_seed"]) != int(tuning_seed):
            raise ValueError("candidate used a different tuning seed")
        returns = np.asarray(row["normalized_returns"], dtype=np.float64)
        if returns.shape != (episodes,) or not np.isfinite(returns).all():
            raise ValueError("invalid tuning return vector")
        normalized.append(
            {
                "fit_id": str(row["fit_id"]),
                "lambda_dimensionless": policy_lambda,
                "mean_normalized_return": float(returns.mean()),
                "standard_error": float(returns.std(ddof=1) / np.sqrt(episodes)),
            }
        )
    normalized.sort(key=lambda row: row["lambda_dimensionless"])
    # max keeps the first tie, so sorting favors the smaller coefficient.
    selected = max(normalized, key=lambda row: row["mean_normalized_return"])
    ranked = sorted(
        normalized,
        key=lambda row: (-row["mean_normalized_return"], row["lambda_dimensionless"]),
    )
    margin = (
        selected["mean_normalized_return"] - ranked[1]["mean_normalized_return"]
        if len(ranked) > 1
        else float("nan")
    )
    result = {
        "schema_version": 1,
        "selection_rule": "maximize_tuning_mean_normalized_discounted_return",
        "selection_label": "simulation_oracle_monte_carlo_selection",
        "exact_tie_break": "smaller_positive_lambda",
        "evaluation_seed": int(tuning_seed),
        "episodes": int(episodes),
        "candidate_results": normalized,
        "selected_fit_id": selected["fit_id"],
        "selected_lambda_dimensionless": selected["lambda_dimensionless"],
        "selection_margin_over_runner_up": float(margin),
        "selected_at_lower_boundary": selected["lambda_dimensionless"] == expected[0],
        "selected_at_upper_boundary": selected["lambda_dimensionless"] == expected[-1],
    }
    return result


def load_tuning_result(path: Path) -> dict:
    """Load a bounded NPZ tuning artifact without permitting pickle payloads."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as loaded:
        required = {
            "fit_id",
            "lambda_dimensionless",
            "evaluation_seed",
            "stream",
            "normalized_returns",
        }
        if not required.issubset(loaded.files) or not set(loaded.files).issubset(
            required | {"raw_returns"}
        ):
            raise ValueError(f"unexpected tuning artifact fields: {source}")
        return {
            "fit_id": str(loaded["fit_id"].item()),
            "lambda_dimensionless": float(loaded["lambda_dimensionless"].item()),
            "evaluation_seed": int(loaded["evaluation_seed"].item()),
            "stream": str(loaded["stream"].item()),
            "normalized_returns": np.asarray(
                loaded["normalized_returns"], dtype=np.float64
            ),
        }


__all__ = ["load_tuning_result", "select_lambda"]
