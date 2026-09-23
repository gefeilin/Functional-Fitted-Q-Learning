"""Paper plotting primitives; analytical inputs supplied by reproduce.py."""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator, MaxNLocator
from .statistics import bootstrap_interval

NG = [2000, 4000, 8000, 16000, 32000]
LG = [0.0001, 0.001, 0.01, 0.1, 1.0]
BLUE, GRAY, ORANGE = "#2c6ca2", "#63727e", "#bf6530"
OUT = None
SUMMARY = []
FIGURES = []


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def panel(
    ax,
    frame,
    field,
    metric,
    *,
    statistic="median",
    color=BLUE,
    label=None,
    shift=0,
    marker="o",
    dashed=False,
    fmt=".2f",
    annotate=False,
    offset=6,
    scale=1.0,
    figure="",
    panel_id=""
):
    """Draw one bootstrap-summary curve and record each displayed estimate."""
    levels = NG if field == "n_transitions" else LG
    points, lows, highs = [], [], []
    for i, level in enumerate(levels):
        rows = frame.loc[frame[field] == level].sort_values("master_seed")
        check(rows.master_seed.astype(int).tolist() == list(range(20)), "seed grid")
        c, lo, hi = bootstrap_interval(
            rows[metric].to_numpy(float), summary=statistic, seed=20260916 + shift + i
        )
        SUMMARY.append(
            dict(
                figure=figure,
                panel=panel_id,
                metric=metric,
                field=field,
                level=level,
                statistic=statistic,
                center=c,
                ci_low=lo,
                ci_high=hi,
                count=20,
                bootstrap_seed=20260916 + shift + i,
            )
        )
        points.append(c / scale)
        lows.append(lo / scale)
        highs.append(hi / scale)
    points, lows, highs = map(np.asarray, (points, lows, highs))
    ax.errorbar(
        levels,
        points,
        yerr=[points - lows, highs - points],
        color=color,
        marker=marker,
        linestyle="--" if dashed else "-",
        markersize=3.8,
        linewidth=1.25,
        elinewidth=0.8,
        capsize=2,
        label=label,
        zorder=3,
    )
    if annotate:
        for x, y, lo, hi in zip(levels, points, lows, highs):
            # Labels sit beyond CI endpoints, never cover an error-bar stem.
            ax.annotate(
                format(y, fmt),
                (x, hi if offset >= 0 else lo),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset >= 0 else "top",
                fontsize=8,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
            )
    # Explicitly include every interval endpoint; allow annotation breathing room.
    ax.update_datalim(np.column_stack((levels, lows)))
    ax.update_datalim(np.column_stack((levels, highs)))
    ax.autoscale_view()
    ax.margins(y=0.20)


def axis(ax, field):
    sample = field == "n_transitions"
    levels = NG if sample else LG
    labels = (
        ["2k", "4k", "8k", "16k", "32k"]
        if sample
        else [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"]
    )
    ax.set_xscale("log", base=2 if sample else 10)
    ax.xaxis.set_major_locator(FixedLocator(levels))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(levels[0] / (1.3 if sample else 2), levels[-1] * (1.3 if sample else 2))
    ax.set_xlabel(
        "Offline transitions, $n$"
        if sample
        else r"Curvature coefficient, $\lambda_{\Omega,n}/s_\Omega$"
    )
    ax.grid(axis="y", color="#e3e7eb", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.6, pad=2)


def canvas(count=1, height=1.85):
    return plt.subplots(1, count, figsize=(5.5, height), constrained_layout=True)


def save(fig, stem, expected_axes):
    check(len(fig.axes) == expected_axes, "Unexpected axes: " + stem)
    fig.savefig(
        OUT / (stem + ".pdf"),
        metadata={
            "CreationDate": datetime(2000, 1, 1),
            "Creator": "FFQI paper reproduction",
        },
    )
    fig.savefig(OUT / (stem + ".png"), dpi=320)
    FIGURES.append(dict(stem=stem, axes=expected_axes, width_inches=5.5))
    plt.close(fig)


def diagnostics(frame, stem, field, kind="all"):
    """Render locality and identification panels for one approximator."""
    count = 2 if kind == "main" else 3
    fig, axes = canvas(count, 1.9 if count == 2 else 2.1)
    panel(
        axes[0],
        frame,
        field,
        "d_min_learned",
        label="Learned policy" if count == 2 else "Learned",
        figure=stem,
        panel_id="a",
    )
    panel(
        axes[0],
        frame,
        field,
        "d_min_behavior",
        color=GRAY,
        label="Behavior",
        shift=100,
        marker="s",
        dashed=True,
        figure=stem,
        panel_id="a",
    )
    axes[0].set_title(
        "(a) Local action distance" if count == 2 else "(a) Local distance", loc="left"
    )
    axes[0].set_ylabel("Nearest-action\n$L^2$ distance")
    axes[0].legend(
        frameon=False, fontsize=8, handlelength=1.5, labelspacing=0.15, loc="upper left"
    )
    axes[0].set_ylim(bottom=0)
    specs = [
        (
            axes[-1],
            "actual_graph_design_ratio",
            ORANGE,
            "(b) Adjacent-critic energy" if count == 2 else "(c) Energy ratio",
            "Graph/design\nenergy ratio",
        )
    ]
    if count == 3:
        specs.insert(
            0,
            (
                axes[1],
                "relative_representation_leverage_learned",
                BLUE,
                "(b) Leverage",
                "Relative feature\nleverage",
            ),
        )
    for ax, metric, color, title, ylabel in specs:
        panel(ax, frame, field, metric, color=color, figure=stem, panel_id=title[1])
        ax.axhline(1, color="#737b81", linestyle=":", linewidth=0.9, zorder=1)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        if "leverage" in metric or (kind == "krr_sample" and "ratio" in metric):
            ax.set_yscale("log")
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes:
        axis(ax, field)
    if count == 3:
        # Compact, unambiguous labels keep all three panels readable at 5.5 in.
        axes[0].set_ylabel("Action distance")
        axes[1].set_ylabel("Relative leverage")
        axes[2].set_ylabel("Energy ratio")
        if field == "lambda_dimensionless":
            # Keep the legend above every distance CI, including the broad
            # fixed-coefficient intervals, without clipping observations.
            axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.3)
            for ax in axes:
                ax.set_xlabel(r"Curvature, $\lambda_{\Omega,n}/s_\Omega$")
    save(fig, stem, count)


def main_line(
    ax,
    values,
    metric,
    label,
    color,
    *,
    lower=False,
    right=False,
    marker="o",
    dashed=False,
    fmt=".2f"
):
    rows = values[values.metric.eq(metric)].sort_values("n")
    assert rows.n.tolist() == NG
    c, lo, hi = rows[["center", "ci_low", "ci_high"]].to_numpy().T
    ax.errorbar(
        NG,
        c,
        yerr=[c - lo, hi - c],
        color=color,
        label=label,
        marker=marker,
        linestyle="--" if dashed else "-",
        linewidth=1.2,
        elinewidth=0.8,
        capsize=2,
        markersize=3.5,
    )
    for x, y, l, h in zip(NG, c, lo, hi):
        ax.annotate(
            format(y, fmt),
            (x, l if lower else h),
            xytext=(4, 1) if right else ((0, -4) if lower else (0, 4)),
            textcoords="offset points",
            fontsize=7.3,
            color=color,
            ha="left" if right else "center",
            va="top" if lower else "bottom",
            annotation_clip=False,
        )


def main_style(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(NG))
    ax.xaxis.set_major_formatter(FixedFormatter(["2k", "4k", "8k", "16k", "32k"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(1450, 47500)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e3e7eb", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, pad=2)


def main_save(fig, name):
    fig.savefig(
        OUT / (name + ".pdf"),
        metadata={
            "CreationDate": datetime(2000, 1, 1),
            "Creator": "Functional Fitted Q-Iteration",
        },
    )
    fig.savefig(OUT / (name + ".png"), dpi=320)
    plt.close(fig)


def render_main(values, output):
    """Render both main-text figures from recorded plotted values."""
    global OUT
    OUT = output
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )

    fig, ax = plt.subplots(figsize=(4.05, 2.3))
    fig.subplots_adjust(left=0.135, right=0.975, bottom=0.19, top=0.965)
    main_line(
        ax,
        values,
        "J_normalized_mean_ffqi",
        "Functional-action policy",
        BLUE,
        fmt=".3f",
    )
    main_line(
        ax,
        values,
        "J_normalized_mean_constant",
        "Constant-action policy",
        GRAY,
        lower=True,
        marker="s",
        dashed=True,
        fmt=".3f",
    )
    main_style(ax)
    ax.set_ylim(0.485, 0.77)
    ax.set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    ax.set_ylabel("Normalized return $(1-\\gamma)J_{100}$", labelpad=4)
    ax.set_xlabel("Offline transitions, $n$", labelpad=3)
    ax.legend(frameon=False, loc="upper left", labelspacing=0.3)
    main_save(fig, "main_value")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.775, wspace=0.40)
    for ax, title in zip(
        axes, ["(a) Local action distance", "(b) Adjacent-critic energy"]
    ):
        main_style(ax)
        ax.set_title(title, loc="left", pad=24)
    main_line(axes[0], values, "d_min_learned", "Learned policy", BLUE)
    main_line(
        axes[0],
        values,
        "d_min_behavior",
        "Behavior",
        GRAY,
        right=True,
        marker="s",
        dashed=True,
    )
    axes[0].set_ylim(0, 3.3)
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_ylabel("Nearest-action $L^2$ distance", labelpad=4)
    axes[0].legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0, 1.015),
        ncol=2,
        borderaxespad=0,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1,
        fontsize=7,
    )
    main_line(axes[1], values, "actual_graph_design_ratio", None, ORANGE)
    axes[1].set_ylim(0.79, 1.55)
    axes[1].set_yticks([0.8, 1.0, 1.2, 1.4])
    axes[1].set_ylabel("Graph/design energy ratio", labelpad=4)
    axes[1].axhline(1, color="#737b81", linestyle=":", linewidth=0.8)
    fig.supxlabel("Offline transitions, $n$", fontsize=7.5, y=0.04)
    main_save(fig, "main_identification")
