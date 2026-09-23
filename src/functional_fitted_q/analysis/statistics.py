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


def proxies(query, secant, method, *, fixed=False):
    keys = ["fit_id", "master_seed", "n_transitions", "lambda_dimensionless"]
    flag = "fixed_n_all_lambda_view" if fixed else "selected_sample_size_view"
    q = query[
        query.approximator.eq(method)
        & query[flag]
        & query.split.eq("B")
        & query.query_class.isin(["learned", "behavior"])
    ]
    s = secant[
        secant.approximator.eq(method)
        & secant[flag]
        & secant.split.eq("B")
        & secant.query_class.eq("learned")
    ]
    counts = q.groupby(keys + ["query_class"]).size()
    if len(counts) != 200 or not counts.eq(256).all() or len(s) != 100:
        raise ValueError("Held-out query grid mismatch")
    med = (
        q.groupby(keys + ["query_class"])[["d_min", "relative_representation_leverage"]]
        .median()
        .reset_index()
    )
    wide = med.pivot_table(
        index=keys,
        columns="query_class",
        values=["d_min", "relative_representation_leverage"],
    ).reset_index()
    wide.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in wide.columns
    ]
    return wide.merge(
        s[keys + ["actual_graph_design_ratio"]], on=keys, validate="one_to_one"
    )


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
