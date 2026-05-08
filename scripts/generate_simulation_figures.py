"""Summarize simulation score files and create simple comparison figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from functional_fitted_q.visualization import collect_score_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "figures")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = collect_score_results(args.results_root)
    if scores.empty:
        raise FileNotFoundError(f"No score files found under {args.results_root}")

    summary = (
        scores.groupby(["method", "gamma", "horizon", "size"], as_index=False)["score"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    scores.to_csv(args.output_dir / "simulation_scores_long.csv", index=False)
    summary.to_csv(args.output_dir / "simulation_scores_summary.csv", index=False)

    plot_df = scores[scores["gamma"] == args.gamma].copy()
    if plot_df.empty:
        raise ValueError(f"No scores found for gamma={args.gamma}")

    methods = ["Functional FQI", "DDPG", "TD3", "SAC", "CQL", "BCQ"]
    horizons = sorted(plot_df["horizon"].unique())
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.5 * len(horizons), 4), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for axis, horizon in zip(axes, horizons):
        horizon_df = plot_df[plot_df["horizon"] == horizon]
        grouped = [horizon_df[horizon_df["method"] == method]["score"].to_numpy() for method in methods]
        axis.boxplot(grouped, labels=methods, showfliers=False)
        axis.set_title(f"Horizon {horizon}")
        axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel("Monte Carlo policy value")
    fig.suptitle(f"Policy-learning scores, gamma={args.gamma}")
    fig.tight_layout()
    fig.savefig(args.output_dir / f"policy_learning_g{args.gamma}.png", dpi=200)


if __name__ == "__main__":
    main()
