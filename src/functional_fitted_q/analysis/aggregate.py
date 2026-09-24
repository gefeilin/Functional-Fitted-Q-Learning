"""Convert a complete new run grid into the released compact table schema."""

import json
import numpy as np
import pandas as pd
from functional_fitted_q.design import build_manifest, constant_fit_id


QUERY_CLASSES = ("learned", "previous", "behavior", "reference")
QUERY_METRICS = (
    "d_min",
    "d_median_32",
    "relative_representation_leverage",
    "representation_leverage_eta_1e-03",
    "range_residual_tau_1e-08",
    "h_actual_m20_minus_m19",
)


def _summarize_queries(frame, task, *, selected, fixed):
    """Reduce q=n query rows to four fit-level action-class records."""
    n = int(task["n_transitions"])
    if len(frame) != 4 * n or set(frame.query_class) != set(QUERY_CLASSES):
        raise ValueError(f"Incomplete identification queries: {task['fit_id']}")
    if set(frame.split) != {"evaluation"}:
        raise ValueError(f"Unexpected identification split: {task['fit_id']}")
    rows = []
    for query_class in QUERY_CLASSES:
        subset = frame[frame.query_class.eq(query_class)]
        if len(subset) != n:
            raise ValueError(f"Identification does not satisfy q=n: {task['fit_id']}")
        row = {
            "fit_id": task["fit_id"],
            "approximator": task["approximator"],
            "master_seed": int(task["master_seed"]),
            "n_transitions": n,
            "lambda_dimensionless": float(task["lambda_dimensionless"]),
            "query_class": query_class,
            "query_count": n,
            "fixed_n_all_lambda_view": fixed,
            "selected_sample_size_view": selected,
        }
        for metric in QUERY_METRICS:
            values = subset[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {metric}: {task['fit_id']}")
            row[f"{metric}_median"] = float(np.median(values))
            if metric.startswith("h_"):
                row[f"{metric}_mean_sq"] = float(np.mean(values**2))
        rows.append(row)
    return rows


def _build_proxy(query, secant):
    """Join fit-level locality/representation summaries to learned-action energies."""
    keys = [
        "fit_id",
        "approximator",
        "master_seed",
        "n_transitions",
        "lambda_dimensionless",
    ]
    metrics = [
        "d_min_median",
        "d_median_32_median",
        "relative_representation_leverage_median",
        "representation_leverage_eta_1e-03_median",
        "range_residual_tau_1e-08_median",
        "h_actual_m20_minus_m19_mean_sq",
    ]
    wide = query.pivot(index=keys, columns="query_class", values=metrics).reset_index()
    wide.columns = [
        "_".join(str(value) for value in item if value != "").rstrip("_")
        if isinstance(item, tuple)
        else item
        for item in wide.columns
    ]
    energy_columns = keys + [
        "D_actual_sq",
        "G_actual_sq",
        "actual_graph_design_ratio",
        "D_projected_sq",
        "ridge_beta_norm_sq",
        "projected_regularized_denominator",
        "G_projected_sq",
        "projected_graph_design_ratio",
        "mean_representation_leverage",
        "projected_inequality_slack",
        "behavior_projection_rmse",
        "graph_projection_rmse",
        "exact_zero_secant",
        "fixed_n_all_lambda_view",
        "selected_sample_size_view",
    ]
    learned = secant[secant.query_class.eq("learned")]
    result = wide.merge(learned[energy_columns], on=keys, validate="one_to_one")
    return result.sort_values(
        ["approximator", "n_transitions", "lambda_dimensionless", "master_seed"]
    ).reset_index(drop=True)


def aggregate(root, runs, output):
    """Collect a complete run grid into the released result-table schema."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root)
    choices = []
    selected = set()
    for cell in manifest["selection_cells"]:
        document = json.loads(
            (runs / "selection" / cell["selection_id"] / "selection.json").read_text()
        )
        selected.add(document["selected_fit_id"])
        choices.append(document)
    pd.DataFrame(choices).to_csv(output / "selected_lambdas.csv", index=False)

    tuning = []
    reporting = []
    query_summaries = []
    secants = []
    neighbors = []
    constants = []
    for task in manifest["candidate_fits"]:
        fit = task["fit_id"]
        candidate = (
            runs
            / "candidates"
            / ("krr" if task["approximator"] == "nystrom_krr" else "adafnn")
            / fit
        )
        with np.load(candidate / "tuning_mc.npz", allow_pickle=False) as archive:
            tuning.append(
                dict(
                    task,
                    tuning_J_normalized_mean=float(
                        archive["normalized_returns"].mean()
                    ),
                    evaluation_seed=int(archive["evaluation_seed"]),
                )
            )
        fixed = int(task["n_transitions"]) == 8000
        chosen = fit in selected
        if chosen or fixed:
            location = runs / "evaluate" / fit
            document = json.loads((location / "reporting_summary.json").read_text())
            reporting.append(
                dict(
                    document,
                    n_subjects=task["n_subjects"],
                    t_per_subject=20,
                    selected_sample_size_view=chosen,
                    fixed_n_all_lambda_view=fixed,
                )
            )
            identify = runs / "identify" / fit
            query = pd.read_parquet(identify / "query_metrics.parquet")
            query_summaries.extend(
                _summarize_queries(query, task, selected=chosen, fixed=fixed)
            )
            secant = pd.read_parquet(identify / "secant_metrics.parquet")
            if set(secant.query_class) != set(QUERY_CLASSES) or not secant.query_count.eq(
                int(task["n_transitions"])
            ).all():
                raise ValueError(f"Incomplete secant grid: {fit}")
            secant["selected_sample_size_view"] = chosen
            secant["fixed_n_all_lambda_view"] = fixed
            secants.append(secant)
            sensitivity = pd.read_parquet(identify / "neighbor_sensitivity.parquet")
            if (
                len(sensitivity) != 8
                or set(sensitivity.query_class) != {"learned", "behavior"}
                or set(sensitivity.state_neighbor_count) != {16, 32, 64, 128}
                or not sensitivity.query_count.eq(int(task["n_transitions"])).all()
            ):
                raise ValueError(f"Incomplete neighbor sensitivity: {fit}")
            if chosen and task["approximator"] == "adafnn":
                neighbors.append(sensitivity)
        if task["approximator"] == "adafnn" and task["lambda_dimensionless"] == 0.0001:
            constant_id = constant_fit_id(task)
            document = json.loads(
                (runs / "constant" / constant_id / "evaluation_summary.json").read_text()
            )
            constants.append(
                dict(
                    document,
                    method="constant_action_fqi",
                    approximator="adafnn",
                    data_pool_id=task["data_pool_id"],
                    n_subjects=task["n_subjects"],
                    t_per_subject=20,
                    checkpoint_iteration=20,
                    gamma=0.95,
                    episodes=1000,
                    horizon=100,
                    evaluation_seed_namespace="oracle_lambda_reporting_v1",
                    scientific_complete=True,
                )
            )

    proxy = _build_proxy(pd.DataFrame(query_summaries), pd.concat(secants, ignore_index=True))
    neighbor = pd.concat(neighbors, ignore_index=True)
    if len(proxy) != 360 or len(neighbor) != 800:
        raise ValueError("Compact identification result grid is incomplete")
    primary = neighbor[neighbor.state_neighbor_count.eq(32)].pivot(
        index="fit_id", columns="query_class", values="d_min_median"
    )
    check = proxy[
        proxy.approximator.eq("adafnn") & proxy.selected_sample_size_view
    ].set_index("fit_id")
    np.testing.assert_allclose(primary.loc[check.index, "learned"], check.d_min_median_learned)
    np.testing.assert_allclose(primary.loc[check.index, "behavior"], check.d_min_median_behavior)

    pd.DataFrame(tuning).to_csv(output / "tuning_candidate_values.csv", index=False)
    pd.DataFrame(reporting).to_csv(output / "reporting_values.csv", index=False)
    pd.DataFrame(constants).to_csv(output / "constant_returns.csv", index=False)
    proxy.to_csv(output / "identification_proxy_per_fit.csv", index=False)
    neighbor.to_csv(output / "neighbor_sensitivity_per_fit.csv", index=False)
    (output / "identification_design.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "analysis_design_id": "functional_pendulum_oracle_identification_matched_n_v2",
                "checkpoint_pair": [19, 20],
                "heldout_split": "evaluation",
                "heldout_subjects_rule": "n_subjects",
                "heldout_decisions": 20,
                "query_count_per_action_class": "q = n = N x 20",
                "critic_refit": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "candidates": len(tuning),
                "reporting": len(reporting),
                "constant": len(constants),
                "identification_fits": len(proxy),
                "neighbor_sensitivity_rows": len(neighbor),
            }
        )
    )
