"""Fit-level statistics, with the RNG algorithms used for the paper intervals."""

import random
import numpy as np
import pandas as pd

N_GRID = [2000, 4000, 8000, 16000, 32000]
LAMBDAS = [0.0001, 0.001, 0.01, 0.1, 1.0]


def bootstrap_interval(values, *, summary="mean", seed=20260916):
    values = np.asarray(values, dtype=float)
    if values.shape != (20,) or not np.isfinite(values).all():
        raise ValueError("Exactly 20 finite fit-level values are required")
    indices = np.random.RandomState(seed).randint(0, 20, size=(10000, 20))
    fn = np.mean if summary == "mean" else np.median
    lo, hi = np.percentile(fn(values[indices], axis=1), [2.5, 97.5])
    return float(fn(values)), float(lo), float(hi)


def grid_check(frame, field, levels):
    if len(frame) != 20 * len(levels) or frame.duplicated([field, "master_seed"]).any():
        raise ValueError("Incomplete or duplicated experiment grid")
    for value in levels:
        if sorted(frame.loc[frame[field].eq(value), "master_seed"].tolist()) != list(
            range(20)
        ):
            raise ValueError("Incomplete seed grid")


def proxies(frame, method, *, fixed=False):
    """Select the released fit-level identification summaries.

    Held-out queries are summarized within each fitted model before release.
    This keeps the public artifact compact and prevents query rows from being
    mistaken for independent training replicates.
    """
    keys = ["fit_id", "master_seed", "n_transitions", "lambda_dimensionless"]
    flag = "fixed_n_all_lambda_view" if fixed else "selected_sample_size_view"
    selected = frame[frame.approximator.eq(method) & frame[flag]].copy()
    if len(selected) != 100 or selected.duplicated(keys).any():
        raise ValueError("Fit-level identification grid mismatch")
    selected = selected.rename(
        columns={
            "d_min_median_behavior": "d_min_behavior",
            "d_min_median_learned": "d_min_learned",
            "relative_representation_leverage_median_behavior":
                "relative_representation_leverage_behavior",
            "relative_representation_leverage_median_learned":
                "relative_representation_leverage_learned",
        }
    )
    columns = keys + [
        "d_min_behavior",
        "d_min_learned",
        "relative_representation_leverage_behavior",
        "relative_representation_leverage_learned",
        "D_actual_sq",
        "G_actual_sq",
        "actual_graph_design_ratio",
    ]
    if not np.isfinite(selected[columns].select_dtypes(include=[np.number])).all().all():
        raise ValueError("Non-finite released identification summary")
    return selected[columns]


def paired_summary(pair):
    # Resample a whole seed jointly across the nested sample sizes.
    rng = random.Random(20260919)
    idx = np.array([[rng.randrange(20) for _ in range(20)] for _ in range(10000)])
    rows = []
    for n in N_GRID:
        values = (
            pair.loc[pair.n_transitions.eq(n)]
            .sort_values("master_seed")
            .difference.to_numpy()
        )
        lo, hi = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
        rows.append(
            dict(n_transitions=n, center=values.mean(), ci_low=lo, ci_high=hi, count=20)
        )
    return pd.DataFrame(rows)
