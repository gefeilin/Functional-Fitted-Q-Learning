"""Create a compact FQE summary table from released result summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results" / "fqe")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "tables")
    args = parser.parse_args()

    rmse_path = args.results_dir / "fqe_rmse_by_size_setting_recomputed_ground_truth.csv"
    truth_path = args.results_dir / "fqe_ground_truth_final_summary.csv"
    if not rmse_path.exists() or not truth_path.exists():
        raise FileNotFoundError("Expected released FQE summary CSV files are missing.")

    rmse = pd.read_csv(rmse_path)
    truth = pd.read_csv(truth_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rmse.to_csv(args.output_dir / "fqe_rmse_summary.csv", index=False)
    truth.to_csv(args.output_dir / "fqe_ground_truth_summary.csv", index=False)

    table = rmse.pivot_table(index=["gamma", "h"], columns="size", values="rmse")
    table.to_csv(args.output_dir / "fqe_rmse_pivot.csv")
    (args.output_dir / "fqe_rmse_pivot.md").write_text(table.to_markdown(floatfmt=".4f") + "\n")


if __name__ == "__main__":
    main()
