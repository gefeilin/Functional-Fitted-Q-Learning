"""Independent reporting MC for one trained oracle-lambda candidate policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from functional_fitted_q.algorithms.nystrom_ffqi import _policy_from_payload
from functional_fitted_q.evaluation import evaluate_policy
from functional_fitted_q.formal_config import load_pendulum_config
from functional_fitted_q.design import (
    bind_wandb_identity,
    build_manifest,
    load_config,
)
from functional_fitted_q.run_artifacts import (
    atomic_json,
    atomic_path,
    file_size,
    verify_run_artifact_manifest,
)
from functional_fitted_q.tracking import (
    finish_wandb_without_affecting_science,
    init_offline_wandb,
)


def _candidate_root(base: Path, approximator: str, fit_id: str) -> Path:
    return base / ("adafnn" if approximator == "adafnn" else "krr") / fit_id


def _load_m20_policy(source: Path) -> tuple[object, dict]:
    index_path = source / "checkpoint_index.json"
    index = json.loads(index_path.read_text())
    paths = index.get("full_checkpoints", [])
    if index.get("completed_iteration") != 20 or paths != [
        "recovery/full/iteration_019.pt",
        "recovery/full/iteration_020.pt",
    ]:
        raise RuntimeError("reporting requires the exact M19/M20 retained pair")
    for relative in paths:
        if file_size(source / relative) != index["file_sizes"][relative]:
            raise RuntimeError("reporting source checkpoint size mismatch")
    payload = torch.load(source / paths[-1], map_location="cpu", weights_only=False)
    return _policy_from_payload(payload["policy"]), {
        "checkpoint_index_bytes": file_size(index_path),
        "m19_bytes": index["file_sizes"][paths[0]],
        "m20_bytes": index["file_sizes"][paths[1]],
    }


def _save_npz(path: Path, **arrays) -> None:
    with atomic_path(path) as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)


def main() -> None:
    """Evaluate one fitted policy on the independent reporting stream."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--config-path", type=Path)
    args = parser.parse_args()
    from functional_fitted_q.runtime import require_execution_host

    require_execution_host()
    root = args.project_root.resolve()
    config = load_config(root, args.config_path)
    bind_wandb_identity(config)
    manifest = build_manifest(root, args.config_path)
    matches = [
        row for row in manifest["candidate_fits"] if row["fit_id"] == args.fit_id
    ]
    if len(matches) != 1:
        raise KeyError(args.fit_id)
    task = matches[0]
    cell_matches = [
        row
        for row in manifest["selection_cells"]
        if row["approximator"] == task["approximator"]
        and int(row["n_transitions"]) == int(task["n_transitions"])
        and int(row["master_seed"]) == int(task["master_seed"])
    ]
    if len(cell_matches) != 1:
        raise RuntimeError("reporting selection cell is not unique")
    cell = cell_matches[0]
    fixed_n = int(config["scope"]["fixed_lambda_sweep_sample_size"])
    selection_path = (
        args.selection_root.resolve() / cell["selection_id"] / "selection.json"
    )
    if int(task["n_transitions"]) != fixed_n:
        if not selection_path.is_file():
            raise RuntimeError("non-fixed-n reporting waits for lambda selection")
        selection = json.loads(selection_path.read_text())
        if selection["selected_fit_id"] != task["fit_id"]:
            raise RuntimeError(
                "only the selected non-fixed-n candidate may be reported"
            )

    source = _candidate_root(
        args.candidate_root.resolve(), task["approximator"], task["fit_id"]
    )
    if task["approximator"] == "adafnn":
        if not (source / "_SUCCESS").is_file():
            raise RuntimeError("AdaFNN candidate is incomplete")
        verify_run_artifact_manifest(source)
        completion_bytes = file_size(source / "run_artifact_manifest.json")
    else:
        complete = json.loads((source / "COMPLETE.json").read_text())
        for relative, expected in complete["artifacts"].items():
            if file_size(source / relative) != expected:
                raise RuntimeError(f"KRR candidate artifact changed: {relative}")
        completion_bytes = file_size(source / "COMPLETE.json")
    policy, checkpoint_provenance = _load_m20_policy(source)
    output = args.output_root.resolve() / task["fit_id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "_SUCCESS").is_file():
        summary = json.loads((output / "reporting_summary.json").read_text())
        if summary["fit_id"] != task["fit_id"]:
            raise RuntimeError("completed reporting identity mismatch")
        print(json.dumps({"state": "VERIFIED_COMPLETE", "fit_id": task["fit_id"]}))
        return

    reporting = config["monte_carlo"]["reporting"]
    if int(cell["reporting_seed"]) == int(cell["tuning_seed"]):
        raise RuntimeError("tuning and reporting streams collided")
    environment = load_pendulum_config(root)
    selection_document = (
        json.loads(selection_path.read_text()) if selection_path.is_file() else None
    )
    evaluation = evaluate_policy(
        policy,
        int(cell["reporting_seed"]),
        environment,
        episodes=int(reporting["episodes"]),
        horizon=int(reporting["horizon"]),
    )
    _save_npz(
        output / "reporting_mc.npz",
        fit_id=np.asarray(task["fit_id"]),
        lambda_dimensionless=np.asarray(task["lambda_dimensionless"], dtype=np.float64),
        evaluation_seed=np.asarray(cell["reporting_seed"], dtype=np.uint32),
        stream=np.asarray("reporting"),
        raw_returns=np.asarray(evaluation.episode_returns, dtype=np.float64),
        normalized_returns=np.asarray(evaluation.normalized_returns, dtype=np.float64),
    )
    summary = {
        "schema_version": 1,
        "fit_id": task["fit_id"],
        "selection_id": cell["selection_id"],
        "approximator": task["approximator"],
        "master_seed": int(task["master_seed"]),
        "n_transitions": int(task["n_transitions"]),
        "lambda_dimensionless": float(task["lambda_dimensionless"]),
        "fixed_n_all_lambda_view": int(task["n_transitions"]) == fixed_n,
        "selection_receipt_available_at_evaluation": selection_document is not None,
        "selected_sample_size_view": (
            selection_document["selected_fit_id"] == task["fit_id"]
            if selection_document is not None
            else None
        ),
        "evaluation_seed": int(cell["reporting_seed"]),
        "seed_namespace": reporting["seed_namespace"],
        "episodes": int(reporting["episodes"]),
        "horizon": int(reporting["horizon"]),
        "independent_of_tuning_stream": True,
        "source_completion_bytes": completion_bytes,
        **checkpoint_provenance,
        **evaluation.summary(),
    }
    atomic_json(output / "reporting_summary.json", summary)
    tracker = init_offline_wandb(
        output_root=output,
        resolved_config=summary,
        name=(
            f"oracle-report-{task['approximator']}-n{int(task['n_transitions'])}-"
            f"lam{float(task['lambda_dimensionless']):g}-s{int(task['master_seed']):02d}"
        ),
        group=f"oracle-lambda-rate-{task['approximator']}-reporting",
        job_type="independent-monte-carlo-reporting",
        tags=[
            task["approximator"],
            "oracle-lambda",
            "reporting-stream",
            f"n-{int(task['n_transitions'])}",
        ],
        notes=(
            "Independent reporting MC. Tuning episodes are excluded from the reported value."
        ),
    )
    if tracker is not None:
        tracker.log(
            {
                "reporting/J_raw": summary["J_raw_mean"],
                "reporting/J_normalized": summary["J_normalized_mean"],
                "reporting/lambda_dimensionless": float(task["lambda_dimensionless"]),
            },
            step=20,
        )
    finish_wandb_without_affecting_science(tracker, output)
    atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "artifacts": {
                "reporting_mc.npz": file_size(output / "reporting_mc.npz"),
                "reporting_summary.json": file_size(output / "reporting_summary.json"),
            },
        },
    )
    (output / "_SUCCESS").touch()
    print(json.dumps({"state": "COMPLETED", **summary}))


if __name__ == "__main__":
    main()
