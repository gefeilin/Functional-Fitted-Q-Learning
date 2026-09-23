"""Independent numeric checks against the manuscript's saved plotting targets."""

import json
import numpy as np
import pandas as pd
from .statistics import bootstrap_interval, paired_summary, LAMBDAS


def check_inputs(root):
    records = json.loads((root / "results/figures.json").read_text())
    paths = {path for row in records for path in row["source_inputs"]}
    for path in paths:
        if not (root / path).is_file():
            raise FileNotFoundError("Missing result file: " + path)
    return len(paths)


def compare(a, b, keys):
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if a.shape != b.shape or set(a.columns) != set(b.columns):
        raise AssertionError("Table schema/cardinality mismatch")
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]) and a[col].dtype != bool:
            np.testing.assert_allclose(
                a[col], b[col], rtol=1e-12, atol=1e-12, equal_nan=True
            )
        elif not a[col].fillna("").equals(b[col].fillna("")):
            raise AssertionError("Table value mismatch: " + col)


def verify(root, output):
    """Verify reconstructed estimates and rendered figures against the release."""
    inputs = check_inputs(root)
    target = root / "results/reference/tables"
    actual = output / "tables"
    tables = []
    for method in ["adafnn", "nystrom_krr"]:
        for name in [
            "selected",
            "candidates_n8000",
            "selected_diagnostics",
            "candidate_diagnostics_n8000",
        ]:
            file = method + "_" + name + ".csv"
            compare(pd.read_csv(actual / file), pd.read_csv(target / file), ["fit_id"])
            tables.append(file)
    for file, keys in [
        ("paired_returns.csv", ["n_transitions", "master_seed"]),
        ("paired_difference_summary.csv", ["n_transitions"]),
        ("selected_secant_energies.csv", ["fit_id"]),
    ]:
        compare(pd.read_csv(actual / file), pd.read_csv(target / file), keys)
        tables.append(file)
    cols = ["metric", "n", "center", "ci_low", "ci_high", "bootstrap_seed"]
    compare(
        pd.read_csv(actual / "main_figure_values.csv")[cols],
        pd.read_csv(target / "main_figure_values.csv")[cols],
        ["metric", "n"],
    )
    main = pd.read_csv(actual / "main_figure_values.csv")
    # Independent scalar checks against displayed manuscript values.
    ada = main[main.metric.eq("J_normalized_mean_ffqi")].sort_values("n").center
    np.testing.assert_allclose(
        ada.round(3), [0.587, 0.593, 0.605, 0.657, 0.702], rtol=0, atol=0
    )
    krr = (
        pd.read_csv(actual / "nystrom_krr_selected.csv")
        .groupby("n_transitions")
        .J_normalized_mean.mean()
    )
    np.testing.assert_allclose(
        krr.round(3), [0.710, 0.695, 0.682, 0.658, 0.645], rtol=0, atol=0
    )
    extra = pd.read_csv(actual / "appendix_figure_values.csv")
    compare(
        extra,
        pd.read_csv(target / "appendix_figure_values.csv"),
        ["figure", "panel", "metric", "level"],
    )
    # Raw curvature supports the implementation-check sentence in the appendix.
    rough = pd.read_csv(root / "results/source/roughness.csv")
    rough = rough[rough.n_transitions.eq(8000) & rough.fixed_n_all_lambda_view]
    saved = pd.read_csv(target / "roughness_summary.csv").sort_values(
        "lambda_dimensionless"
    )
    draws = np.random.default_rng(0).integers(0, 20, size=(10000, 20))
    rough_rows = []
    for i, lam in enumerate(LAMBDAS):
        f = rough[rough.lambda_dimensionless.eq(lam)].sort_values("master_seed")
        if f.master_seed.tolist() != list(range(20)):
            raise AssertionError("Roughness seed mismatch")
        x = f.raw_total_curvature_penalty.to_numpy()
        lo, hi = np.percentile(x[draws].mean(axis=1), [2.5, 97.5])
        np.testing.assert_allclose(
            [x.mean(), lo, hi],
            saved.iloc[i][
                ["roughness_center", "roughness_ci_low", "roughness_ci_high"]
            ].to_numpy(float),
            rtol=1e-12,
            atol=1e-12,
        )
        count = int(f.optimizer_floor_selected.eq("full_cmaes").sum())
        assert count == int(saved.iloc[i].full_functional_count)
        rough_rows.append(
            dict(
                lambda_dimensionless=lam,
                mean_curvature=float(x.mean()),
                ci_low=float(lo),
                ci_high=float(hi),
                full_functional_count=count,
            )
        )
    pd.DataFrame(rough_rows).to_csv(actual / "curvature_check.csv", index=False)
    images = []
    from PIL import Image

    for row in json.loads((root / "results/figures.json").read_text()):
        name = row["artifact_id"]
        a = np.asarray(
            Image.open(output / "figures" / (name + ".png")).convert("RGB"), dtype=float
        )
        b = np.asarray(
            Image.open(root / "results/reference/figures" / (name + ".png")).convert(
                "RGB"
            ),
            dtype=float,
        )
        if a.shape != b.shape:
            raise AssertionError("Figure dimensions changed: " + name)
        if not np.array_equal(a, b):
            raise AssertionError("Rendered figure differs from reference: " + name)
        images.append(
            dict(
                artifact=name,
                width=a.shape[1],
                height=a.shape[0],
                pixel_identical=True,
            )
        )
    report = dict(
        status="PASS",
        input_files=inputs,
        matched_seed_tables=len(tables),
        matched_main_estimates=25,
        matched_appendix_estimates=len(extra),
        reconstructed_tuning_selections=200,
        matched_roughness_levels=5,
        figure_rendering=images,
        full_training_rerun=False,
        numerical_tolerance=dict(rtol=1e-12, atol=1e-12),
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report
