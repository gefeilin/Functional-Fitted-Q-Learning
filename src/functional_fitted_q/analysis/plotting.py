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
BLUE, GRAY, ORANGE = "#0072B2", "#6F7880", "#2F3E46"
PURPLE, GOLD = "#8E5EA2", "#E69F00"
LEVERAGE_BLUE = "#2C6CA2"
KRR_ENERGY_ORANGE = "#C45A24"
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
        markersize=3.2,
        linewidth=1.15,
        elinewidth=0.75,
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
    if annotate:
        ax.margins(y=0.20)
    return points, lows, highs


def axis(ax, field):
    sample = field == "n_transitions"
    levels = NG if sample else LG
    labels = (
        ["2k", "4k", "8k", "16k", "32k"]
        if sample
        else [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"]
    )
    ax.set_xscale("log", base=2 if sample else 10)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(FixedLocator(levels))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    if sample:
        ax.set_xlim(1450, 47500)
    else:
        ax.set_xlim(levels[0] / 2, levels[-1] * 2)
    ax.set_xlabel(
        "Offline transitions, $n$"
        if sample
        else r"Curvature coefficient, $\lambda_{\Omega,n}/s_\Omega$"
    )
    ax.grid(axis="y", color="#e3e7eb", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.6, pad=2)


def canvas(count=1, height=1.85):
    return plt.subplots(1, count, figsize=(5.5, height), constrained_layout=True)


def save(fig, stem, expected_axes):
    check(len(fig.axes) == expected_axes, "Unexpected axes: " + stem)
    fig.savefig(
        OUT / (stem + ".pdf"),
        bbox_inches="tight",
        metadata={
            "CreationDate": datetime(2000, 1, 1),
            "Creator": "FFQI paper reproduction",
        },
    )
    fig.savefig(OUT / (stem + ".png"), dpi=320, bbox_inches="tight")
    FIGURES.append(
        dict(stem=stem, axes=expected_axes, width_inches=fig.get_size_inches()[0])
    )
    plt.close(fig)


def diagnostics(frame, stem, field, kind="all"):
    """Render locality and identification panels for one approximator."""
    count = 2 if kind == "main" else 3
    is_krr_sample = kind == "krr_sample"
    learned_color = LEVERAGE_BLUE if is_krr_sample else PURPLE
    behavior_color = GRAY if is_krr_sample else GOLD
    fig, axes = canvas(count, 1.9 if count == 2 else 2.1)
    panel(
        axes[0],
        frame,
        field,
        "d_min_learned",
        color=learned_color,
        label="Learned policy" if count == 2 else "Learned",
        figure=stem,
        panel_id="a",
    )
    panel(
        axes[0],
        frame,
        field,
        "d_min_behavior",
        color=behavior_color,
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
        frameon=False,
        fontsize=6.1,
        handlelength=1.3,
    )
    axes[0].set_ylim(bottom=0)
    specs = [
        (
            axes[-1],
            "actual_graph_design_ratio",
            KRR_ENERGY_ORANGE if is_krr_sample else ORANGE,
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
                LEVERAGE_BLUE,
                "(b) Leverage",
                "Relative feature\nleverage",
            ),
        )
    for ax, metric, color, title, ylabel in specs:
        panel(ax, frame, field, metric, color=color, figure=stem, panel_id=title[1])
        ax.axhline(1, color="#737b81", linestyle=":", linewidth=0.8)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        if "leverage" in metric or (is_krr_sample and "ratio" in metric):
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes:
        axis(ax, field)
    if count == 3:
        # Compact, unambiguous labels keep all three panels readable at 5.5 in.
        axes[0].set_ylabel("Action distance")
        axes[1].set_ylabel("Relative leverage")
        axes[2].set_ylabel("Energy ratio" if is_krr_sample else "Graph/design energy ratio")
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
    fmt=".2f",
    annotate=True,
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
        linewidth=1.15,
        elinewidth=0.75,
        capsize=2,
        markersize=3.2,
    )
    if annotate:
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
        bbox_inches="tight",
        metadata={
            "CreationDate": datetime(2000, 1, 1),
            "Creator": "Functional Fitted Q-Iteration",
        },
    )
    fig.savefig(OUT / (name + ".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def render_main(values, output):
    """Render the paper's three-panel main experiment figure."""
    global OUT
    OUT = output
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.9))
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.13, top=0.94, wspace=0.42)
    for ax, title in zip(
        axes, ["(a) Policy value", "(b) Action distance", "(c) Critic energy"]
    ):
        main_style(ax)
        ax.set_box_aspect(1)
        ax.set_title(title, loc="left", pad=6)
    main_line(
        axes[0],
        values,
        "J_normalized_mean_ffqi",
        "Functional action",
        BLUE,
        fmt=".3f",
        annotate=False,
    )
    main_line(
        axes[0],
        values,
        "J_normalized_mean_constant",
        "Constant action",
        GRAY,
        lower=True,
        marker="s",
        dashed=True,
        fmt=".3f",
        annotate=False,
    )
    axes[0].set_ylim(0.485, 0.77)
    axes[0].set_yticks([0.50, 0.55, 0.60, 0.65, 0.70, 0.75])
    axes[0].set_ylabel("Normalized policy value")
    axes[0].legend(frameon=False, loc="upper left", borderaxespad=0.35,
                   handlelength=1.45, handletextpad=0.4, labelspacing=0.25)
    main_line(
        axes[1], values, "d_min_learned", "Learned action", PURPLE,
        annotate=False,
    )
    main_line(
        axes[1],
        values,
        "d_min_behavior",
        "Behavior",
        GOLD,
        right=True,
        marker="s",
        dashed=True,
        annotate=False,
    )
    # Leave headroom above the widest bootstrap interval so the legend does
    # not cover the learned-action curve.
    axes[1].set_ylim(0, 3.3)
    axes[1].set_ylabel("Nearest-action $L^2$ distance")
    axes[1].legend(frameon=False, loc="upper left", borderaxespad=0.35,
                   handlelength=1.3, handletextpad=0.35, labelspacing=0.25)
    main_line(
        axes[2], values, "actual_graph_design_ratio", None, ORANGE,
        annotate=False,
    )
    axes[2].set_ylim(bottom=0)
    axes[2].set_ylabel("Graph/design energy ratio")
    axes[2].axhline(1, color="#737b81", linestyle=":", linewidth=0.8)
    main_save(fig, "main_figure")
