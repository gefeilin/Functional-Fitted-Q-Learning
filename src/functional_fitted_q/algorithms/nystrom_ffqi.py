"""Reusable checkpointed FQI with a fixed Nyström design.

Each iteration builds Bellman labels from the previous critic/policy, solves
for new critic coefficients, and improves the policy on logged next states.
The paper CLI uses the specialized loop in runners/train_krr.py; this module
also supports reusable optimizer and resume workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch

from ..data import OfflineDataset
from ..policies.bspline_policy import BoundedCoefficientBSplinePolicy
from ..policies.constant_policy import BoundedConstantActionPolicy
from ..run_artifacts import atomic_json, atomic_path, file_size
from ..seeding import SeedTree


@dataclass
class NystromFittedQResult:
    completed_iteration: int
    policy: object
    iteration_records: list[dict]
    policy_trace: list[dict]
    checkpoint_index: dict


def _normalize_config(document: dict) -> dict:
    """Normalize tuples and lists before comparing saved training settings."""
    return json.loads(json.dumps(document, allow_nan=False))


def _atomic_torch(path: Path, payload: dict) -> None:
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def _policy_payload(policy) -> dict:
    if isinstance(policy, BoundedCoefficientBSplinePolicy):
        return {
            "kind": "bounded_coefficient_bspline",
            "parameters": policy.parameter_matrix.copy(),
            "degree": int(policy.degree),
        }
    if isinstance(policy, BoundedConstantActionPolicy):
        return {
            "kind": "bounded_constant_function",
            "parameters": policy.parameters.copy(),
        }
    raise TypeError(type(policy))


def _policy_from_payload(payload: dict):
    if payload["kind"] == "bounded_coefficient_bspline":
        return BoundedCoefficientBSplinePolicy(
            np.asarray(payload["parameters"], dtype=np.float64), int(payload["degree"])
        )
    if payload["kind"] == "bounded_constant_function":
        return BoundedConstantActionPolicy(
            np.asarray(payload["parameters"], dtype=np.float64)
        )
    raise ValueError("unknown Nyström policy payload")


def _critic_solve_metadata(critic) -> dict:
    keys = (
        "target_solve_count",
        "latest_relative_linear_solve_residual",
        "relative_linear_solve_residual",
        "latest_target_solve_wall_time_sec",
    )
    if "ridge_selection" in critic.fit_metadata:
        keys += ("gcv_selection", "regularization_diagonal")
    return {key: critic.fit_metadata[key] for key in keys if key in critic.fit_metadata}


def _restore_critic_solve_metadata(critic, payload: dict) -> None:
    for key, value in payload.items():
        critic.fit_metadata[key] = value


def _save_index(
    run_root: Path,
    resume_config: dict,
    completed_iteration: int,
    full_paths: list[str],
    segment_path: str | None,
) -> dict:
    paths = [*full_paths, *([segment_path] if segment_path else [])]
    document = {
        "schema_version": 1,
        "resume_config": resume_config,
        "completed_iteration": int(completed_iteration),
        "full_checkpoints": full_paths,
        "segment_checkpoint": segment_path,
        "file_sizes": {path: file_size(run_root / path) for path in paths},
    }
    atomic_json(run_root / "checkpoint_index.json", document)
    return document


def _load_index(run_root: Path, resume_config: dict) -> dict:
    index_path = run_root / "checkpoint_index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise FileNotFoundError("Nyström checkpoint index is missing")
    index = json.loads(index_path.read_text())
    if index.get("resume_config") != resume_config:
        raise RuntimeError("Nyström checkpoint invariant mismatch")
    paths = [*index.get("full_checkpoints", [])]
    if index.get("segment_checkpoint"):
        paths.append(index["segment_checkpoint"])
    for relative in paths:
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Nyström checkpoint is missing: {relative}")
        if file_size(path) != index["file_sizes"][relative]:
            raise RuntimeError(f"Nyström checkpoint size mismatch: {relative}")
    return index


def run_nystrom_ffqi(
    *,
    dataset: OfflineDataset,
    dataset_identity: str,
    implementation_identity: str,
    master_seed: int,
    iterations: int,
    critic,
    policy_optimizer,
    gamma: float,
    k: int,
    policy_kind: str,
    run_root: Path,
    evaluation_checkpoints: tuple[int, ...],
    resume: bool,
    checkpoint_retention_limit: int = 2,
    segment_observer: Callable[[dict], None] | None = None,
    iteration_observer: Callable[[int, object, object | None], None] | None = None,
) -> NystromFittedQResult:
    """Run Nyström KRR fitted Q-iteration and retain resumable checkpoints.

    The critic's kernel design and ridge-selection state are fixed before this
    loop.  Each iteration solves the Bellman regression, improves the requested
    policy class, and atomically records enough state for exact continuation.
    """
    if iterations <= 0 or checkpoint_retention_limit != 2:
        raise ValueError(
            "positive iterations and two-checkpoint retention are required"
        )
    if policy_kind == "bounded_coefficient_bspline":
        policy = BoundedCoefficientBSplinePolicy(np.zeros((7, k), dtype=np.float64))
    elif policy_kind == "bounded_constant_function":
        policy = BoundedConstantActionPolicy(np.zeros(7, dtype=np.float64))
    else:
        raise ValueError(policy_kind)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    if critic.fit_metadata is None:
        raise RuntimeError("Nyström critic design must be fitted before FFQI")
    critic_identity_keys = (
        "solver",
        "approximation",
        "n",
        "requested_landmark_rank",
        "landmark_count",
        "retained_feature_rank",
        "landmark_indices",
        "dtype",
        "config",
        "regularization_diagonal",
        "factor_reuse",
        "ridge_selection",
        "svd_reuse",
    )
    critic_identity = {
        key: critic.fit_metadata[key]
        for key in critic_identity_keys
        if key in critic.fit_metadata
    }
    invariant = {
        "schema_version": 1,
        "algorithm": "deterministic_nystrom_krr_ffqi_bounded_cmaes_v1",
        "dataset_identity": dataset_identity,
        "implementation_identity": implementation_identity,
        "dataset_shape": [
            int(len(dataset.states)),
            int(dataset.action_values.shape[1]),
        ],
        "master_seed": int(master_seed),
        "iterations": int(iterations),
        "gamma": float(gamma),
        "k": int(k),
        "policy_kind": policy_kind,
        "critic_class": type(critic).__name__,
        # Timings and mutable solve counters are deliberately excluded so that
        # a refactored design factorization can resume the same scientific run.
        "critic_design_identity": critic_identity,
        "policy_optimizer_config": asdict(policy_optimizer.config),
        "evaluation_checkpoints": list(evaluation_checkpoints),
    }
    resume_config = _normalize_config(invariant)
    invariant_path = run_root / "checkpoint_invariant.json"
    if invariant_path.is_file():
        if _normalize_config(json.loads(invariant_path.read_text())) != resume_config:
            raise RuntimeError("existing Nyström invariant changed")
    else:
        atomic_json(invariant_path, invariant)
    completed_iteration = 0
    full_paths: list[str] = []
    iteration_records: list[dict] = []
    policy_trace: list[dict] = []
    segment = None
    if resume:
        index = _load_index(run_root, resume_config)
        completed_iteration = int(index["completed_iteration"])
        full_paths = list(index["full_checkpoints"])
        if full_paths:
            latest = torch.load(
                run_root / full_paths[-1], map_location="cpu", weights_only=False
            )
            critic.alpha = latest["alpha"].to(device=critic.device, dtype=torch.float64)
            _restore_critic_solve_metadata(critic, latest["critic_solve_metadata"])
            policy = _policy_from_payload(latest["policy"])
            iteration_records = list(latest["iteration_records"])
            policy_trace = list(latest["policy_trace"])
        if index.get("segment_checkpoint"):
            segment = torch.load(
                run_root / index["segment_checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
    elif (run_root / "checkpoint_index.json").exists():
        raise FileExistsError("Nyström checkpoints exist; resume is required")
    if completed_iteration > iterations:
        raise RuntimeError("checkpoint exceeds requested iteration budget")
    rng_tree = SeedTree(master_seed)
    # Preserve an existing segment pointer until the resumed optimizer has
    # either checkpointed again or completed the iteration.  Overwriting the
    # index here would make a second interruption lose the valid segment.
    if not resume:
        index = _save_index(
            run_root, resume_config, completed_iteration, full_paths, None
        )
    for iteration in range(completed_iteration + 1, iterations + 1):
        iteration_started = time.time()
        previous_policy = policy
        pending = (
            segment if segment and int(segment["iteration"]) == iteration else None
        )
        if pending is None:
            # With Q_hat_0 = 0 the first target is R; later targets use the
            # previous clipped prediction at the previous policy's next action.
            if iteration == 1:
                targets = np.asarray(dataset.rewards, dtype=np.float64)
            else:
                actions = policy.values_batch(dataset.next_states, critic.action_grid)
                targets = np.asarray(
                    dataset.rewards, dtype=np.float64
                ) + gamma * critic.predict(dataset.next_states, actions)
            critic.solve_targets(targets)
            optimizer_resume = None
        else:
            critic.alpha = pending["alpha"].to(
                device=critic.device, dtype=torch.float64
            )
            _restore_critic_solve_metadata(critic, pending["critic_solve_metadata"])
            optimizer_resume = pending["optimizer_resume_state"]

        if "ridge_selection" in critic.fit_metadata:
            selection = critic.fit_metadata.get("gcv_selection")
            if (
                selection is None
                or int(critic.fit_metadata["target_solve_count"]) != iteration
            ):
                raise RuntimeError(
                    "GCV checkpoint lacks the current iteration's lambda selection"
                )
            atomic_json(
                run_root / "gcv" / f"iteration_{iteration:03d}.json",
                {"iteration": iteration, **selection},
            )

        def optimizer_callback(state, elapsed: float) -> None:
            nonlocal index
            relative = "recovery/segment.pt"
            _atomic_torch(
                run_root / relative,
                {
                    "schema_version": 1,
                    "resume_config": resume_config,
                    "iteration": iteration,
                    "alpha": critic.alpha.detach().cpu(),
                    "critic_solve_metadata": _critic_solve_metadata(critic),
                    "optimizer_resume_state": state,
                },
            )
            index = _save_index(
                run_root, resume_config, completed_iteration, full_paths, relative
            )
            if segment_observer is not None:
                segment_observer(index)

        optimizer_rng = rng_tree.rng("policy_optimizer", iteration)
        if policy_kind == "bounded_coefficient_bspline":
            policy, trace, _ = policy_optimizer.optimize(
                critic,
                dataset.next_states,
                policy,
                optimizer_rng,
                resume_state=optimizer_resume,
                checkpoint_callback=optimizer_callback,
                checkpoint_every_generations=5,
                checkpoint_every_seconds=30.0,
            )
        else:
            policy, trace, _ = policy_optimizer.optimize_constant(
                critic,
                dataset.next_states,
                policy,
                optimizer_rng,
                resume_state=optimizer_resume,
                checkpoint_callback=optimizer_callback,
                checkpoint_every_generations=5,
                checkpoint_every_seconds=30.0,
            )
        policy_trace.extend({"iteration": iteration, **item} for item in trace)
        record = {
            "iteration": int(iteration),
            "target_solve_count": int(critic.fit_metadata["target_solve_count"]),
            "relative_linear_solve_residual": float(
                critic.fit_metadata["latest_relative_linear_solve_residual"]
            ),
            "target_solve_wall_time_sec": float(
                critic.fit_metadata["latest_target_solve_wall_time_sec"]
            ),
            "policy_optimizer_generations": int(len(trace)),
            "policy_optimizer_final_reason": trace[-1].get("termination_reason"),
            "iteration_wall_time_sec": float(time.time() - iteration_started),
        }
        if "gcv_selection" in critic.fit_metadata:
            selection = critic.fit_metadata["gcv_selection"]
            record.update(
                {
                    "ridge_selection": "gcv_each_fqi_iteration",
                    "ridge_lambda": selection["selected_lambda"],
                    "gcv_score": selection["selected_gcv"],
                    "gcv_effective_df": selection["selected_effective_df"],
                    "gcv_at_grid_boundary": selection["selected_at_grid_boundary"],
                    "gcv_candidate_count": selection["candidate_count"],
                }
            )
        iteration_records.append(record)
        if iteration_observer is not None:
            iteration_observer(iteration, policy, previous_policy)
        relative = f"recovery/full/iteration_{iteration:03d}.pt"
        _atomic_torch(
            run_root / relative,
            {
                "schema_version": 1,
                "resume_config": resume_config,
                "completed_iteration": iteration,
                "alpha": critic.alpha.detach().cpu(),
                "critic_solve_metadata": _critic_solve_metadata(critic),
                "policy": _policy_payload(policy),
                "iteration_records": iteration_records,
                "policy_trace": policy_trace,
            },
        )
        full_paths.append(relative)
        while len(full_paths) > checkpoint_retention_limit:
            (run_root / full_paths.pop(0)).unlink(missing_ok=True)
        (run_root / "recovery/segment.pt").unlink(missing_ok=True)
        completed_iteration = iteration
        index = _save_index(
            run_root, resume_config, completed_iteration, full_paths, None
        )
        if iteration in evaluation_checkpoints:
            payload = _policy_payload(policy)
            with atomic_path(
                run_root / "policy_views" / f"M{iteration:03d}.npz"
            ) as temporary:
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        kind=np.asarray(payload["kind"]),
                        parameters=payload["parameters"],
                        degree=np.asarray(payload.get("degree", -1), dtype=np.int64),
                        iteration=np.asarray(iteration, dtype=np.int64),
                    )
        segment = None
    return NystromFittedQResult(
        completed_iteration=completed_iteration,
        policy=policy,
        iteration_records=iteration_records,
        policy_trace=policy_trace,
        checkpoint_index=index,
    )


__all__ = ["NystromFittedQResult", "run_nystrom_ffqi"]
