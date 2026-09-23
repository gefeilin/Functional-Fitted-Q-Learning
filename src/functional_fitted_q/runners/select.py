"""Select a policy coefficient for one approximator, sample size, and seed."""

import argparse
import json
from pathlib import Path

from functional_fitted_q.design import build_manifest
from functional_fitted_q.run_artifacts import atomic_json
from functional_fitted_q.selection import load_tuning_result, select_lambda
from functional_fitted_q.tracking import (
    init_offline_wandb,
    finish_wandb_without_affecting_science,
)


def main():
    """Select one policy coefficient using tuning returns only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-index", type=int, required=True)
    parser.add_argument("--config-path", type=Path)
    args = parser.parse_args()
    from functional_fitted_q.runtime import require_execution_host

    require_execution_host()
    experiment = build_manifest(args.project_root, args.config_path)
    cells = experiment["selection_cells"]
    if not 0 <= args.selection_index < len(cells):
        raise IndexError(args.selection_index)
    cell = cells[args.selection_index]
    method = "krr" if cell["approximator"] == "nystrom_krr" else "adafnn"
    candidates = []
    for name, coefficient in zip(
        cell["candidate_fit_ids"], cell["candidate_lambdas"], strict=True
    ):
        directory = args.candidate_root / method / name
        marker = "_SUCCESS" if method == "adafnn" else "COMPLETE.json"
        if not (directory / marker).is_file():
            raise RuntimeError(f"Finish training {name} before selecting a coefficient")
        result = load_tuning_result(directory / "tuning_mc.npz")
        if result["fit_id"] != name or result["lambda_dimensionless"] != coefficient:
            raise ValueError(f"Tuning results do not match the settings for {name}")
        candidates.append(result)

    result = select_lambda(
        candidates,
        expected_lambdas=cell["candidate_lambdas"],
        tuning_seed=cell["tuning_seed"],
        episodes=1000,
    )
    for key in (
        "selection_id",
        "approximator",
        "master_seed",
        "n_subjects",
        "t_per_subject",
        "n_transitions",
        "reporting_seed",
    ):
        result[key] = cell[key]
    directory = args.output_root / cell["selection_id"]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "selection.json"
    if output.is_file() and (directory / "_SUCCESS").is_file():
        if json.loads(output.read_text()) != result:
            raise RuntimeError("Tuning results changed; use a new output directory")
        print(f"Already selected: {result['selected_fit_id']}")
        return
    atomic_json(output, result)
    tracker = init_offline_wandb(
        output_root=directory,
        resolved_config=result,
        name=cell["selection_id"],
        group=f"{method}-selection",
        job_type="policy-selection",
        tags=[method, "tuning"],
        notes="Select using tuning returns; report on independent episodes.",
    )
    if tracker is not None:
        tracker.log(
            {
                "selection/lambda_dimensionless": result[
                    "selected_lambda_dimensionless"
                ],
                "selection/margin": result["selection_margin_over_runner_up"],
            }
        )
    finish_wandb_without_affecting_science(tracker, directory)
    (directory / "_SUCCESS").touch()
    print(f"Selected {result['selected_fit_id']}")


if __name__ == "__main__":
    main()
