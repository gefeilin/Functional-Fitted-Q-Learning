"""Checkpoint-only action-locality and secant-identification analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from functional_fitted_q.algorithms.adafnn_ffqi import (
    _critic_from_state,
    _policy_from_payload,
)
from functional_fitted_q.critics.adafnn_functional_critic import (
    AdaFNNCriticTrainingConfig,
)
from functional_fitted_q.critics.nystrom_functional_critic import (
    NystromFunctionalKRR,
    NystromKRRConfig,
    deterministic_landmark_indices,
)
from functional_fitted_q.data import (
    OfflineDataset,
    generate_offline_dataset,
    verify_offline_dataset,
)
from functional_fitted_q.diagnostics.adafnn_representation import (
    critic_values,
    penultimate_features,
)
from functional_fitted_q.formal_config import load_pendulum_config
from functional_fitted_q.identification import (
    blocked_state_neighbors,
    representation_identification,
)
from functional_fitted_q.design import (
    bind_wandb_identity,
    build_manifest,
    derived_mc_seed,
    krr_landmark_seed,
    load_config,
)
from functional_fitted_q.policies.reference_policy import AnalyticReferencePolicy
from functional_fitted_q.run_artifacts import (
    atomic_json,
    atomic_path,
    environment_manifest,
    file_size,
    verify_run_artifact_manifest,
)
from functional_fitted_q.tracking import (
    finish_wandb_without_affecting_science,
    init_offline_wandb,
)


def _subset(
    dataset: OfflineDataset, n_subjects: int, t_per_subject: int
) -> OfflineDataset:
    selected = dataset.subject_order[:n_subjects]
    mask = np.isin(dataset.subject_ids, selected) & (
        dataset.time_indices < t_per_subject
    )
    result = OfflineDataset(
        dataset.states[mask],
        dataset.action_values[mask],
        dataset.rewards[mask],
        dataset.next_states[mask],
        dataset.subject_ids[mask],
        dataset.time_indices[mask],
        dataset.subject_order.copy(),
    )
    if len(result.states) != n_subjects * t_per_subject:
        raise RuntimeError("identification nested data subset mismatch")
    return result


def _candidate_root(base: Path, approximator: str, fit_id: str) -> Path:
    return base / ("adafnn" if approximator == "adafnn" else "krr") / fit_id


def _load_pair(source: Path) -> tuple[dict, dict, dict]:
    index_path = source / "checkpoint_index.json"
    index = json.loads(index_path.read_text())
    paths = [
        "recovery/full/iteration_019.pt",
        "recovery/full/iteration_020.pt",
    ]
    if index.get("completed_iteration") != 20 or index.get("full_checkpoints") != paths:
        raise RuntimeError("identification requires exact M19/M20 full checkpoints")
    payloads = []
    for relative in paths:
        if file_size(source / relative) != index["file_sizes"][relative]:
            raise RuntimeError("identification checkpoint size mismatch")
        payloads.append(
            torch.load(source / relative, map_location="cpu", weights_only=False)
        )
    return (
        payloads[0],
        payloads[1],
        {
            "checkpoint_index_bytes": file_size(index_path),
            "m19_checkpoint_bytes": index["file_sizes"][paths[0]],
            "m20_checkpoint_bytes": index["file_sizes"][paths[1]],
        },
    )


def _adafnn_config(parent: dict) -> AdaFNNCriticTrainingConfig:
    common = parent["critic"]
    selected = parent["selected"]
    return AdaFNNCriticTrainingConfig(
        n_basis_nodes=int(selected["adaptive_basis_nodes"]),
        micro_hidden_layers=tuple(common["micro_hidden_layers"]),
        outer_hidden_layers=tuple(common["outer_hidden_layers"]),
        learning_rate=float(selected["critic_learning_rate"]),
        batch_size=int(common["batch_size"]),
        max_epochs=int(common["max_epochs"]),
        early_stopping_patience=int(common["early_stopping_patience"]),
        early_stopping_min_delta=float(common["early_stopping_min_delta"]),
        weight_decay=float(common["weight_decay"]),
        gradient_clip=float(common["gradient_clip"]),
        parameter_bound=float(selected["parameter_bound"]),
        require_early_stopping_convergence=True,
    )


def _heldout(config: dict, task: dict, root: Path) -> dict[str, np.ndarray]:
    identification = config["identification"]
    seed = derived_mc_seed(
        int(task["master_seed"]), identification["heldout_seed_namespace"]
    )
    reference_document = yaml.safe_load(
        (root / "configs/reference_policy.yaml").read_text()
    )
    reference = AnalyticReferencePolicy(
        np.asarray(reference_document["beta"], dtype=np.float64)
    )
    environment = load_pendulum_config(root)
    data = generate_offline_dataset(
        master_seed=seed,
        reference=reference,
        n_subjects=int(identification["heldout_subjects"]),
        t_per_subject=int(identification["heldout_decisions"]),
        env_config=environment,
        gp_resolution=128,
        step_gp_amplitude=float(config["scope"]["step_gp_amplitude"]),
        vectorized=True,
    )
    start, stop = [int(value) for value in identification["retained_time_indices"]]
    mask = (data.time_indices >= start) & (data.time_indices <= stop)
    states = np.asarray(data.states[mask], dtype=np.float64)
    actions = np.asarray(data.action_values[mask], dtype=np.float64)
    subjects = np.asarray(data.subject_ids[mask], dtype=np.int64)
    times = np.asarray(data.time_indices[mask], dtype=np.int64)
    order = np.lexsort((times, subjects))
    states, actions, subjects, times = (
        value[order] for value in (states, actions, subjects, times)
    )
    split_subjects = np.asarray(
        data.subject_order[: int(identification["split_a_subjects"])],
        dtype=np.int64,
    )
    splits = np.where(np.isin(subjects, split_subjects), "A", "B").astype("U1")
    action_grid = np.linspace(0.0, 1.0, actions.shape[1])
    return {
        "states": states,
        "behavior_actions": actions,
        "reference_actions": reference.values_batch(states, action_grid),
        "subject_ids": subjects,
        "time_indices": times,
        "splits": splits,
        "action_grid": action_grid,
        "derived_seed": np.asarray(seed, dtype=np.uint32),
    }


def _krr_features(critic, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(states), critic.config.prediction_block_size):
            stop = min(len(states), start + critic.config.prediction_block_size)
            outputs.append(
                critic._features(
                    torch.as_tensor(
                        states[start:stop], dtype=torch.float64, device=critic.device
                    ),
                    torch.as_tensor(
                        actions[start:stop], dtype=torch.float64, device=critic.device
                    ),
                )
                .cpu()
                .numpy()
            )
    return np.concatenate(outputs)


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    with atomic_path(path) as temporary:
        frame.to_parquet(temporary, index=False)


def _atomic_npz(path: Path, **arrays) -> None:
    with atomic_path(path) as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)


def main() -> None:
    """Analyze a retained Q19/Q20 pair on independent held-out states."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fit-id", required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--smoke-query-rows", type=int)
    args = parser.parse_args()
    from functional_fitted_q.runtime import require_execution_host

    require_execution_host()
    if not torch.cuda.is_available():
        raise RuntimeError("identification requires an allocated GPU")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
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
        raise RuntimeError("identification selection cell is not unique")
    cell = cell_matches[0]
    fixed_n = int(config["scope"]["fixed_lambda_sweep_sample_size"])
    selection_path = (
        args.selection_root.resolve() / cell["selection_id"] / "selection.json"
    )
    if int(task["n_transitions"]) != fixed_n:
        if not selection_path.is_file():
            raise RuntimeError("non-fixed-n identification waits for lambda selection")
        if json.loads(selection_path.read_text())["selected_fit_id"] != task["fit_id"]:
            raise RuntimeError("only selected non-fixed-n candidate may be analyzed")

    source = _candidate_root(
        args.candidate_root.resolve(), task["approximator"], task["fit_id"]
    )
    if task["approximator"] == "adafnn":
        if not (source / "_SUCCESS").is_file():
            raise RuntimeError("AdaFNN source fit is incomplete")
        verify_run_artifact_manifest(source)
        completion_bytes = file_size(source / "run_artifact_manifest.json")
    else:
        complete_path = source / "COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        for relative, expected in complete["artifacts"].items():
            if file_size(source / relative) != expected:
                raise RuntimeError(f"KRR source artifact changed: {relative}")
        completion_bytes = file_size(complete_path)
    output = args.output_root.resolve() / task["fit_id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "_SUCCESS").is_file():
        artifact_manifest = json.loads((output / "artifact_manifest.json").read_text())
        for relative, expected in artifact_manifest["artifacts"].items():
            if file_size(output / relative) != expected:
                raise RuntimeError(
                    f"completed identification artifact changed: {relative}"
                )
        print(json.dumps({"state": "VERIFIED_COMPLETE", "fit_id": task["fit_id"]}))
        return
    previous_payload, current_payload, checkpoint_provenance = _load_pair(source)
    previous_policy = _policy_from_payload(previous_payload["policy"])
    current_policy = _policy_from_payload(current_payload["policy"])

    data_dir = args.data_root.resolve() / task["data_pool_id"]
    data_manifest = verify_offline_dataset(data_dir)
    data = _subset(
        OfflineDataset.load(data_dir),
        int(task["n_subjects"]),
        int(task["t_per_subject"]),
    )
    heldout = _heldout(config, task, root)
    if args.smoke_query_rows is not None:
        count = min(int(args.smoke_query_rows), len(heldout["states"]))
        a = np.flatnonzero(heldout["splits"] == "A")[: count // 2]
        b = np.flatnonzero(heldout["splits"] == "B")[: count - len(a)]
        take = np.concatenate([a, b])
        for key in (
            "states",
            "behavior_actions",
            "reference_actions",
            "subject_ids",
            "time_indices",
            "splits",
        ):
            heldout[key] = heldout[key][take]
    neighbor_indices, neighbor_state_sq = blocked_state_neighbors(
        heldout["states"],
        data.states,
        tuple(float(v) for v in config["krr"]["state_lengthscales"]),
        int(config["identification"]["state_neighbor_count"]),
    )
    action_grid = heldout["action_grid"]
    actions_by_class = {
        "learned": current_policy.values_batch(heldout["states"], action_grid),
        "previous": previous_policy.values_batch(heldout["states"], action_grid),
        "behavior": heldout["behavior_actions"],
        "reference": heldout["reference_actions"],
    }

    if task["approximator"] == "adafnn":
        parent = yaml.safe_load((root / config["adafnn"]["parent_config"]).read_text())
        critic_config = _adafnn_config(parent)
        quadrature_grid = np.linspace(
            0.0, 1.0, int(parent["critic"]["quadrature_resolution_default"])
        )
        previous_critic = _critic_from_state(
            previous_payload["critic_state"],
            quadrature_grid,
            critic_config,
            float(config["ffqi"]["gamma"]),
            device,
        )
        current_critic = _critic_from_state(
            current_payload["critic_state"],
            quadrature_grid,
            critic_config,
            float(config["ffqi"]["gamma"]),
            device,
        )
        design_features = penultimate_features(
            current_critic,
            data.states,
            data.action_values,
            action_grid,
            device,
            batch_size=8192,
        )
        design_secant = critic_values(
            current_critic, data.states, data.action_values, action_grid, device
        ) - critic_values(
            previous_critic, data.states, data.action_values, action_grid, device
        )
        features_by_class = {}
        secants_by_class = {}
        for name, actions in actions_by_class.items():
            features_by_class[name] = penultimate_features(
                current_critic,
                heldout["states"],
                actions,
                action_grid,
                device,
                batch_size=8192,
            )
            secants_by_class[name] = critic_values(
                current_critic, heldout["states"], actions, action_grid, device
            ) - critic_values(
                previous_critic, heldout["states"], actions, action_grid, device
            )
        representation_definition = "M20_final_hidden_relu_plus_bias"
    else:
        bandwidth = float(
            json.loads((source / "action_bandwidth.json").read_text())[
                "action_l2_lengthscale"
            ]
        )
        critic = NystromFunctionalKRR(
            NystromKRRConfig(
                state_lengthscales=tuple(
                    float(v) for v in config["krr"]["state_lengthscales"]
                ),
                action_l2_lengthscale=bandwidth,
                ridge=1.0e-3,
                jitter=float(config["krr"]["jitter"]),
                landmark_rank=int(task["landmark_rank"]),
                eigenvalue_relative_tolerance=float(
                    config["krr"]["eigenvalue_relative_tolerance"]
                ),
                prediction_block_size=int(config["krr"]["prediction_block_size"]),
            ),
            value_max=float(config["ffqi"]["q_scale"]),
        )
        indices = deterministic_landmark_indices(
            len(data.states), int(task["landmark_rank"]), krr_landmark_seed(task)
        )
        critic.fit_design(
            data.states,
            data.action_values,
            action_grid,
            device,
            landmark_indices=indices,
        )
        design_features = critic._design_features.detach().cpu().numpy()
        previous_alpha = previous_payload["alpha"].numpy()
        current_alpha = current_payload["alpha"].numpy()
        if previous_alpha.shape != (
            design_features.shape[1],
        ) or current_alpha.shape != (design_features.shape[1],):
            raise RuntimeError("KRR checkpoint coefficient/feature dimension mismatch")
        design_secant = np.clip(
            design_features @ current_alpha, 0.0, critic.value_max
        ) - np.clip(design_features @ previous_alpha, 0.0, critic.value_max)
        features_by_class = {}
        secants_by_class = {}
        for name, actions in actions_by_class.items():
            features = _krr_features(critic, heldout["states"], actions)
            features_by_class[name] = features
            secants_by_class[name] = np.clip(
                features @ current_alpha, 0.0, critic.value_max
            ) - np.clip(features @ previous_alpha, 0.0, critic.value_max)
        representation_definition = "frozen_M20_Nystrom_whitened_landmark_features"

    metadata = {
        "fit_id": task["fit_id"],
        "approximator": task["approximator"],
        "master_seed": int(task["master_seed"]),
        "n_transitions": int(task["n_transitions"]),
        "lambda_dimensionless": float(task["lambda_dimensionless"]),
        "checkpoint_iteration": 20,
        "previous_checkpoint_iteration": 19,
    }
    query, secant, spectrum, geometry = representation_identification(
        metadata=metadata,
        splits=heldout["splits"],
        subject_ids=heldout["subject_ids"],
        time_indices=heldout["time_indices"],
        query_states=heldout["states"],
        training_actions=data.action_values,
        action_grid=action_grid,
        neighbor_indices=neighbor_indices,
        design_features=design_features,
        design_secant=design_secant,
        actions_by_class=actions_by_class,
        features_by_class=features_by_class,
        actual_secants_by_class=secants_by_class,
        radii=tuple(float(v) for v in config["identification"]["action_l2_radii"]),
    )
    _atomic_frame(query, output / "query_metrics.parquet")
    _atomic_frame(secant, output / "secant_metrics.parquet")
    _atomic_frame(spectrum, output / "representation_spectrum.parquet")
    _atomic_npz(
        output / "query_actions.npz",
        states=heldout["states"],
        behavior_actions=heldout["behavior_actions"],
        learned_actions=actions_by_class["learned"],
        previous_actions=actions_by_class["previous"],
        reference_actions=actions_by_class["reference"],
        subject_ids=heldout["subject_ids"],
        time_indices=heldout["time_indices"],
        splits=heldout["splits"],
        action_grid=action_grid,
        heldout_seed=heldout["derived_seed"],
    )
    _atomic_npz(
        output / "state_neighbors.npz",
        indices=neighbor_indices,
        squared_distances=neighbor_state_sq,
    )
    atomic_json(
        output / "analysis_metadata.json",
        {
            "schema_version": 1,
            **metadata,
            "experiment": manifest["experiment"],
            "source_completion_bytes": completion_bytes,
            "representation_definition": representation_definition,
            "adjacent_secant_definition": "clipped_Q20_minus_clipped_Q19",
            "heldout_seed_namespace": config["identification"][
                "heldout_seed_namespace"
            ],
            "heldout_seed": int(heldout["derived_seed"]),
            "paper_split": config["identification"]["paper_split"],
            "smoke_query_rows": args.smoke_query_rows,
            "fixed_n_all_lambda_view": int(task["n_transitions"]) == fixed_n,
            "selected_nonfixed_n_view": int(task["n_transitions"]) != fixed_n,
            "environment_manifest": environment_manifest(),
            "geometry": geometry,
            **checkpoint_provenance,
        },
    )
    paper_query = query[(query["split"] == "B") & (query["query_class"] == "learned")]
    paper_secant = secant[
        (secant["split"] == "B") & (secant["query_class"] == "learned")
    ].iloc[0]
    tracker = init_offline_wandb(
        output_root=output,
        resolved_config=metadata,
        name=(
            f"oracle-ident-{task['approximator']}-n{int(task['n_transitions'])}-"
            f"lam{float(task['lambda_dimensionless']):g}-s{int(task['master_seed']):02d}"
        ),
        group=f"oracle-lambda-rate-{task['approximator']}-identification",
        job_type="checkpoint-identification-analysis",
        tags=[task["approximator"], "identification", "M19-M20", "oracle-lambda"],
        notes=(
            "Read-only action-locality, representation-range/leverage, and adjacent-critic secant analysis."
        ),
    )
    if tracker is not None:
        tracker.log(
            {
                "locality/learned_d_min_mean": float(paper_query["d_min"].mean()),
                "representation/learned_relative_leverage_mean": float(
                    paper_query["relative_representation_leverage"].mean()
                ),
                "secant/actual_graph_design_ratio": float(
                    paper_secant["actual_graph_design_ratio"]
                ),
                "secant/projected_graph_design_ratio": float(
                    paper_secant["projected_graph_design_ratio"]
                ),
            },
            step=20,
        )
    finish_wandb_without_affecting_science(tracker, output)
    artifacts = {
        name: file_size(output / name)
        for name in (
            "query_metrics.parquet",
            "secant_metrics.parquet",
            "representation_spectrum.parquet",
            "query_actions.npz",
            "state_neighbors.npz",
            "analysis_metadata.json",
        )
    }
    atomic_json(
        output / "artifact_manifest.json", {"schema_version": 1, "artifacts": artifacts}
    )
    (output / "_SUCCESS").touch()
    print(json.dumps({"state": "COMPLETED", **metadata}))


if __name__ == "__main__":
    main()
