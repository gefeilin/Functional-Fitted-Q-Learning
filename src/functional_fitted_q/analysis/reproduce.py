"""Rebuild paper statistics from the released per-fit and per-query results."""

import json
import numpy as np
import pandas as pd
from .statistics import (
    N_GRID,
    LAMBDAS,
    bootstrap_interval,
    grid_check,
    proxies,
    paired_summary,
)


def tables(source, output):
    """Recompute selection, diagnostics, and plotted intervals from raw tables."""
    report = pd.read_csv(source / "reporting_values.csv")
    tuning = pd.read_csv(source / "tuning_candidate_values.csv")
    constant = pd.read_csv(source / "constant_returns.csv")
    columns = [
        "fit_id",
        "master_seed",
        "n_transitions",
        "lambda_dimensionless",
        "approximator",
        "split",
        "query_class",
        "selected_sample_size_view",
        "fixed_n_all_lambda_view",
        "d_min",
        "relative_representation_leverage",
    ]
    query = pd.read_parquet(
        source / "identification_query_metrics.parquet", columns=columns
    )
    secant = pd.read_parquet(source / "identification_secant_metrics.parquet")
    winners = (
        tuning.sort_values(
            ["tuning_J_normalized_mean", "lambda_dimensionless"],
            ascending=[False, True],
        )
        .groupby(["approximator", "n_transitions", "master_seed"])
        .head(1)
    )
    if (
        len(tuning) != 1000
        or len(winners) != 200
        or set(winners.fit_id)
        != set(report.loc[report.selected_sample_size_view, "fit_id"])
    ):
        raise ValueError("Tuning-only selection does not reproduce reported winners")
    result = {}
    for method in ["adafnn", "nystrom_krr"]:
        selected = report[
            report.approximator.eq(method) & report.selected_sample_size_view
        ].copy()
        fixed = report[
            report.approximator.eq(method) & report.fixed_n_all_lambda_view
        ].copy()
        grid_check(selected, "n_transitions", N_GRID)
        grid_check(fixed, "lambda_dimensionless", LAMBDAS)
        result[method] = dict(
            selected=selected,
            candidates_n8000=fixed,
            selected_diagnostics=proxies(query, secant, method),
            candidate_diagnostics_n8000=proxies(query, secant, method, fixed=True),
        )
        for name, frame in result[method].items():
            frame.to_csv(output / (method + "_" + name + ".csv"), index=False)
    grid_check(constant, "n_transitions", N_GRID)
    pair = result["adafnn"]["selected"].merge(
        constant,
        on=["n_transitions", "master_seed"],
        suffixes=("_ffqi", "_constant"),
        validate="one_to_one",
    )
    if (
        len(pair) != 100
        or not pair.evaluation_seed_ffqi.eq(pair.evaluation_seed_constant).all()
    ):
        raise ValueError("Matched reporting streams changed")
    pair["difference"] = pair.J_normalized_mean_ffqi - pair.J_normalized_mean_constant
    pair.to_csv(output / "paired_returns.csv", index=False)
    paired = paired_summary(pair)
    paired.to_csv(output / "paired_difference_summary.csv", index=False)
    energies = secant[
        secant.selected_sample_size_view
        & secant.split.eq("B")
        & secant.query_class.eq("learned")
    ].copy()
    if len(energies) != 200 or not energies.D_actual_sq.gt(0).all():
        raise ValueError("Invalid empirical energy denominator")
    np.testing.assert_allclose(
        energies.actual_graph_design_ratio,
        energies.G_actual_sq / energies.D_actual_sq,
        rtol=1e-12,
        atol=1e-12,
    )
    energies.to_csv(output / "selected_secant_energies.csv", index=False)
    values = []
    specs = [
        (
            "J_normalized_mean_ffqi",
            result["adafnn"]["selected"],
            "J_normalized_mean",
            "mean",
            0,
            "Functional-action policy",
        ),
        (
            "J_normalized_mean_constant",
            constant,
            "J_normalized_mean",
            "mean",
            100,
            "Constant-action policy",
        ),
        (
            "d_min_learned",
            result["adafnn"]["selected_diagnostics"],
            "d_min_learned",
            "median",
            0,
            "Learned policy",
        ),
        (
            "d_min_behavior",
            result["adafnn"]["selected_diagnostics"],
            "d_min_behavior",
            "median",
            100,
            "Behavior",
        ),
        (
            "actual_graph_design_ratio",
            result["adafnn"]["selected_diagnostics"],
            "actual_graph_design_ratio",
            "median",
            0,
            "Critic ratio",
        ),
    ]
    for metric, frame, column, stat, shift, label in specs:
        for i, n in enumerate(N_GRID):
            seed = 20260916 + shift + i
            c, lo, hi = bootstrap_interval(
                frame.loc[frame.n_transitions.eq(n)].sort_values("master_seed")[column],
                summary=stat,
                seed=seed,
            )
            values.append(
                dict(
                    metric=metric,
                    n=n,
                    label=label,
                    center=c,
                    ci_low=lo,
                    ci_high=hi,
                    bootstrap_seed=seed,
                )
            )
    main = pd.DataFrame(values)
    main.to_csv(output / "main_figure_values.csv", index=False)
    return result, main, paired, energies


def render_appendix(frames, paired, energies, output):
    """Render the four appendix figures and return their plotted values."""
    from . import plotting as p
    from matplotlib.ticker import NullLocator

    p.OUT = output
    p.SUMMARY.clear()
    p.FIGURES.clear()
    p.plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "figure.constrained_layout.h_pad": 0.025,
            "figure.constrained_layout.w_pad": 0.03,
        }
    )
    fig, ax = p.canvas(height=2.0)
    p.panel(
        ax,
        frames["nystrom_krr"]["selected"],
        "n_transitions",
        "J_normalized_mean",
        statistic="mean",
        color=p.BLUE,
        annotate=True,
        fmt=".3f",
        figure="krr_value",
    )
    p.axis(ax, "n_transitions")
    ax.set_ylabel("Normalized return\n$(1-\\gamma)J_{100}$")
    p.save(fig, "krr_value", 1)
    p.diagnostics(
        frames["nystrom_krr"]["selected_diagnostics"],
        "krr_identification",
        "n_transitions",
        "krr_sample",
    )
    fig, ax = p.canvas(height=2.0)
    ax.errorbar(
        N_GRID,
        paired.center,
        yerr=[paired.center - paired.ci_low, paired.ci_high - paired.center],
        marker="o",
        color=p.BLUE,
        capsize=2,
    )
    ax.axhline(0, color="gray", linestyle=":")
    p.axis(ax, "n_transitions")
    ax.set_ylabel("Paired normalized-\nreturn difference")
    p.save(fig, "paired_value_difference", 1)
    fig, axes = p.canvas(2, 2.15)
    for ax, method, title in zip(
        axes, ["adafnn", "nystrom_krr"], ["(a) AdaFNN", "(b) Nyström KRR"]
    ):
        f = energies[energies.approximator.eq(method)]
        for metric, color, label, marker, dashed in [
            ("G_actual_sq", p.BLUE, "Policy actions", "o", False),
            ("D_actual_sq", p.GRAY, "Logged design", "s", True),
        ]:
            p.panel(
                ax,
                f,
                "n_transitions",
                metric,
                color=color,
                label=label,
                marker=marker,
                dashed=dashed,
                figure="critic_update_energies",
                panel_id=method,
            )
        p.axis(ax, "n_transitions")
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        ax.set_ylabel("Mean squared\ncritic difference")
        ax.legend(frameon=False, fontsize=8)
    p.save(fig, "critic_update_energies", 2)
    return pd.DataFrame(p.SUMMARY)


def reproduce(root, source, output):
    """Reconstruct every released table and figure from saved seed-level data."""
    from . import plotting
    from .verify import check_inputs

    check_inputs(root)
    tab = output / "tables"
    fig = output / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    frames, main, paired, energies = tables(source, tab)
    plotting.render_main(main, fig)
    extra = render_appendix(frames, paired, energies, fig)
    extra.to_csv(tab / "appendix_figure_values.csv", index=False)
    print(
        json.dumps(
            dict(
                figures=6,
                selected_functional_fits=200,
                constant_fits=100,
                tuning_candidates=1000,
                output=str(output),
            )
        )
    )
