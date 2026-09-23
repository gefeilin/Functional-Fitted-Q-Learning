"""Convert a complete new run grid into the same schema as released results."""

import json
import numpy as np
import pandas as pd
from functional_fitted_q.design import build_manifest, constant_fit_id


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
    queries = []
    secants = []
    spectra = []
    constants = []
    for t in manifest["candidate_fits"]:
        fit = t["fit_id"]
        candidate = (
            runs
            / "candidates"
            / ("krr" if t["approximator"] == "nystrom_krr" else "adafnn")
            / fit
        )
        with np.load(candidate / "tuning_mc.npz", allow_pickle=False) as z:
            tuning.append(
                dict(
                    t,
                    tuning_J_normalized_mean=float(z["normalized_returns"].mean()),
                    evaluation_seed=int(z["evaluation_seed"]),
                )
            )
        if fit in selected or t["n_transitions"] == 8000:
            location = runs / "evaluate" / fit
            document = json.loads((location / "reporting_summary.json").read_text())
            reporting.append(
                dict(
                    document,
                    n_subjects=t["n_subjects"],
                    t_per_subject=20,
                    selected_sample_size_view=fit in selected,
                    fixed_n_all_lambda_view=t["n_transitions"] == 8000,
                )
            )
            for name, frames in [
                ("query_metrics", queries),
                ("secant_metrics", secants),
                ("representation_spectrum", spectra),
            ]:
                frame = pd.read_parquet(runs / "identify" / fit / (name + ".parquet"))
                frame["selected_sample_size_view"] = fit in selected
                frame["fixed_n_all_lambda_view"] = t["n_transitions"] == 8000
                frames.append(frame)
        if t["approximator"] == "adafnn" and t["lambda_dimensionless"] == 0.0001:
            fit = constant_fit_id(t)
            document = json.loads(
                (runs / "constant" / fit / "evaluation_summary.json").read_text()
            )
            constants.append(
                dict(
                    document,
                    method="constant_action_fqi",
                    approximator="adafnn",
                    data_pool_id=t["data_pool_id"],
                    n_subjects=t["n_subjects"],
                    t_per_subject=20,
                    checkpoint_iteration=20,
                    gamma=0.95,
                    episodes=1000,
                    horizon=100,
                    evaluation_seed_namespace="oracle_lambda_reporting_v1",
                    scientific_complete=True,
                )
            )
    pd.DataFrame(tuning).to_csv(output / "tuning_candidate_values.csv", index=False)
    pd.DataFrame(reporting).to_csv(output / "reporting_values.csv", index=False)
    pd.DataFrame(constants).to_csv(output / "constant_returns.csv", index=False)
    pd.concat(queries, ignore_index=True).to_parquet(
        output / "identification_query_metrics.parquet", index=False
    )
    pd.concat(secants, ignore_index=True).to_parquet(
        output / "identification_secant_metrics.parquet", index=False
    )
    pd.concat(spectra, ignore_index=True).to_parquet(
        output / "identification_spectrum.parquet", index=False
    )
    print(
        json.dumps(
            dict(
                candidates=len(tuning),
                reporting=len(reporting),
                constant=len(constants),
            )
        )
    )
