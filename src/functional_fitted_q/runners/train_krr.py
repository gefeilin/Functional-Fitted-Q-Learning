"""Run one KRR candidate in the oracle-MC lambda/rate design.

The critic ridge is selected by grouped five-fold CV at every FQI iteration.
The separately named policy lambda uses exactly the same dimensionless,
uncentered total-curvature objective as the AdaFNN candidate runs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from functional_fitted_q.runners.krr_prediction import FastPredictionView
from functional_fitted_q.runners.krr_restart import (
    bounded_adam_with_total_curvature_restart,
)
from functional_fitted_q.runners.krr_optimizer import TheoryTotalCurvatureAdamConfig
from functional_fitted_q.action_bandwidth import (
    persist_bandwidth_receipt,
    resolve_action_bandwidth,
)
from functional_fitted_q.algorithms.dimensionless_total_curvature import (
    OMEGA_SCALE_RULE,
    PENALTY_CONVENTION,
    Q_SCALE_RULE,
)
from functional_fitted_q.algorithms.nystrom_ffqi import (
    _atomic_torch,
    _load_index,
    _policy_from_payload,
    _policy_payload,
    _save_index,
)
from functional_fitted_q.algorithms.theory_total_curvature import (
    total_curvature_matrix,
)
from functional_fitted_q.critics.nystrom_functional_critic import (
    NystromKRRConfig,
    deterministic_landmark_indices,
)
from functional_fitted_q.critics.nystrom_grouped_cv import (
    GroupedCVNystromFunctionalKRR,
    GroupedRidgeCVConfig,
)
from functional_fitted_q.data import OfflineDataset, verify_offline_dataset
from functional_fitted_q.evaluation import evaluate_policy
from functional_fitted_q.formal_config import load_pendulum_config
from functional_fitted_q.design import (
    DESIGN_ID,
    bind_wandb_identity,
    build_manifest,
    derived_mc_seed,
    krr_landmark_seed,
    load_config,
)
from functional_fitted_q.interruption import PenaltyCheckpointSignals
from functional_fitted_q.policies.bspline_policy import (
    BoundedCoefficientBSplinePolicy,
)
from functional_fitted_q.run_artifacts import (
    atomic_json,
    atomic_path,
    environment_manifest,
    file_size,
)
from functional_fitted_q.seeding import SeedTree
from functional_fitted_q.tracking import (
    finish_wandb_without_affecting_science,
    init_offline_wandb,
)


def _canonical(document):
    return json.loads(json.dumps(document, sort_keys=True, allow_nan=False))


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
        raise RuntimeError("oracle KRR nested subject/time subset cardinality mismatch")
    return result


def _save_frame(path: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows)
    with atomic_path(path) as temporary:
        frame.to_parquet(temporary, index=False)


def _critic_metadata(critic) -> dict:
    keys = (
        "target_solve_count",
        "latest_relative_linear_solve_residual",
        "relative_linear_solve_residual",
        "latest_target_solve_wall_time_sec",
        "grouped_cv_selection",
        "regularization_diagonal",
    )
    return {key: critic.fit_metadata[key] for key in keys if key in critic.fit_metadata}


def _restore_critic_metadata(critic, payload: dict) -> None:
    for key, value in payload.items():
        critic.fit_metadata[key] = value


def _dimensionless_constants(policy, action_grid, q_scale: float, policy_lambda: float):
    matrix, receipt = total_curvature_matrix(policy, action_grid)
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2.0)
    tolerance = max(1.0, float(eigenvalues[-1])) * 1.0e-12
    if float(eigenvalues[0]) < -tolerance:
        raise RuntimeError("KRR curvature matrix is not PSD within roundoff")
    omega_scale = 4.0 * policy.k * float(max(eigenvalues[-1], 0.0))
    if not np.isfinite(omega_scale) or omega_scale <= 0:
        raise RuntimeError("invalid KRR dimensionless curvature scale")
    effective_raw_lambda = float(policy_lambda) * float(q_scale) / omega_scale
    receipt.update(
        {
            "dimensionless_policy_lambda": float(policy_lambda),
            "q_scale": float(q_scale),
            "q_scale_rule": Q_SCALE_RULE,
            "omega_scale": omega_scale,
            "omega_scale_rule": OMEGA_SCALE_RULE,
            "effective_raw_lambda": effective_raw_lambda,
            "dimensionless_objective_equivalence": (
                "argmax(q_mean/q_scale-lambda_bar*Omega/omega_scale)="
                "argmax(q_mean-effective_raw_lambda*Omega)"
            ),
        }
    )
    return omega_scale, effective_raw_lambda, receipt


def _write_tuning_npz(
    path: Path,
    *,
    fit_id: str,
    policy_lambda: float,
    evaluation_seed: int,
    raw_returns: np.ndarray,
    normalized_returns: np.ndarray,
) -> None:
    with atomic_path(path) as temporary:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                fit_id=np.asarray(fit_id),
                lambda_dimensionless=np.asarray(policy_lambda, dtype=np.float64),
                evaluation_seed=np.asarray(evaluation_seed, dtype=np.uint32),
                stream=np.asarray("tuning"),
                raw_returns=np.asarray(raw_returns, dtype=np.float64),
                normalized_returns=np.asarray(normalized_returns, dtype=np.float64),
            )


def main() -> None:
    """Train one configured Nyström KRR candidate and its tuning evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--config-path", type=Path)
    args = parser.parse_args()
    from functional_fitted_q.runtime import require_execution_host

    require_execution_host()
    if (
        not torch.cuda.is_available()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise RuntimeError("oracle KRR fit requires deterministic CUDA")
    if not all(
        os.environ.get(key) == "4"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    ):
        raise RuntimeError("oracle KRR fit requires four-thread BLAS settings")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(8)

    project_root = args.project_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    config_path = (
        args.config_path or project_root / "configs/paper_simulation.yaml"
    ).resolve()
    config = load_config(project_root, config_path)
    bind_wandb_identity(config)
    manifest_path = project_root / "configs/tasks.json"
    manifest = json.loads(manifest_path.read_text())
    rebuilt = build_manifest(project_root, config_path)
    if manifest != rebuilt:
        raise RuntimeError("oracle-lambda manifest/config/source mismatch")
    matches = [
        row
        for row in manifest["candidate_fits"]
        if int(row["task_id"]) == args.task_id and row["approximator"] == "nystrom_krr"
    ]
    if len(matches) != 1:
        raise IndexError(args.task_id)
    task = matches[0]
    if int(task["master_seed"]) not in range(20) or task[
        "retained_full_checkpoints"
    ] != [19, 20]:
        raise RuntimeError("KRR oracle task escaped the frozen seed/checkpoint scope")

    run_root = output_root / task["fit_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    q_scale = float(config["ffqi"]["q_scale"])
    tuning = config["monte_carlo"]["tuning"]
    tuning_seed = derived_mc_seed(int(task["master_seed"]), tuning["seed_namespace"])
    identity = _canonical(
        {
            "schema_version": 1,
            "design_id": DESIGN_ID,
            "classification": config["classification"],
            "algorithm": "full_FFQI_nystrom_KRR_grouped5_CV_dimensionless_total_curvature",
            "task": task,
            "settings": config,
            "experiment": manifest["experiment"],
            "gamma": float(config["ffqi"]["gamma"]),
            "M": 20,
            "policy_lambda_dimensionless": float(task["lambda_dimensionless"]),
            "q_scale": q_scale,
            "q_scale_rule": Q_SCALE_RULE,
            "omega_scale_rule": OMEGA_SCALE_RULE,
            "penalty_convention": PENALTY_CONVENTION,
            "critic_ridge_selection": config["krr"]["critic_ridge_selection"],
            "tuning_mc": {
                "episodes": int(tuning["episodes"]),
                "horizon": int(tuning["horizon"]),
                "seed_namespace": tuning["seed_namespace"],
                "derived_seed": tuning_seed,
            },
        }
    )
    run_config = identity

    with (run_root / ".lock").open("a") as lock, PenaltyCheckpointSignals(
        run_root, run_config, float(task["lambda_dimensionless"])
    ) as signals:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete_path = run_root / "COMPLETE.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text())
            if complete["identity"] != identity:
                raise RuntimeError("completed oracle KRR identity mismatch")
            for relative, expected in complete["artifacts"].items():
                if file_size(run_root / relative) != expected:
                    raise RuntimeError(f"completed KRR artifact changed: {relative}")
            print(json.dumps({"state": "VERIFIED_COMPLETE", "fit_id": task["fit_id"]}))
            return
        if (run_root / "IDENTITY.json").is_file():
            if json.loads((run_root / "IDENTITY.json").read_text()) != identity:
                raise RuntimeError("oracle KRR resume identity mismatch")
        else:
            atomic_json(run_root / "IDENTITY.json", identity)

        started = time.time()
        data_dir = data_root / task["data_pool_id"]
        data_manifest = verify_offline_dataset(data_dir)
        data = _subset(
            OfflineDataset.load(data_dir),
            int(task["n_subjects"]),
            int(task["t_per_subject"]),
        )
        unique, counts = np.unique(data.subject_ids, return_counts=True)
        if len(unique) != int(task["n_subjects"]) or not np.all(
            counts == int(task["t_per_subject"])
        ):
            raise RuntimeError("oracle KRR grouped panel geometry mismatch")
        action_grid = np.linspace(0.0, 1.0, data.action_values.shape[1])
        bandwidth_kernel = {
            "action_l2_lengthscale": "median",
            "action_median_max_in_memory_pairs": int(
                config["krr"]["action_median_max_in_memory_pairs"]
            ),
        }
        bandwidth, bandwidth_receipt = resolve_action_bandwidth(
            bandwidth_kernel,
            data.action_values,
            action_grid,
            cache_dir=output_root
            / "action_bandwidth"
            / task["data_pool_id"]
            / f"n{task['n_transitions']}",
        )
        persist_bandwidth_receipt(run_root, bandwidth_receipt)
        ridge_spec = config["krr"]["grouped_cv"]
        ridge_config = GroupedRidgeCVConfig(
            lambda_grid=tuple(
                float(value)
                for value in np.logspace(
                    float(ridge_spec["log10_min"]),
                    float(ridge_spec["log10_max"]),
                    int(ridge_spec["grid_points"]),
                )
            ),
            folds=int(ridge_spec["folds"]),
            fold_seed=int(ridge_spec["fold_seed"]),
        )
        critic_config = NystromKRRConfig(
            state_lengthscales=tuple(
                float(v) for v in config["krr"]["state_lengthscales"]
            ),
            action_l2_lengthscale=bandwidth,
            ridge=None,
            jitter=float(config["krr"]["jitter"]),
            landmark_rank=int(task["landmark_rank"]),
            eigenvalue_relative_tolerance=float(
                config["krr"]["eigenvalue_relative_tolerance"]
            ),
            prediction_block_size=int(config["krr"]["prediction_block_size"]),
        )
        critic = GroupedCVNystromFunctionalKRR(
            critic_config, ridge_config, data.subject_ids, value_max=q_scale
        )
        landmark_indices = deterministic_landmark_indices(
            len(data.states), critic_config.landmark_rank, krr_landmark_seed(task)
        )
        critic.fit_design(
            data.states,
            data.action_values,
            action_grid,
            torch.device("cuda"),
            landmark_indices=landmark_indices,
        )
        atomic_json(
            run_root / "data_and_critic_provenance.json",
            {
                "data_pool_id": task["data_pool_id"],
                "n_subjects": int(task["n_subjects"]),
                "t_per_subject": int(task["t_per_subject"]),
                "n_transitions": len(data.states),
                "action_l2_lengthscale": bandwidth,
                "landmark_indices": critic.fit_metadata["landmark_indices"],
                "fold_assignment": critic.fit_metadata["ridge_selection"][
                    "fold_assignment"
                ],
            },
        )
        atomic_json(run_root / "nystrom_fit_metadata.json", critic.fit_metadata)
        (run_root / "environment_manifest.txt").write_text(environment_manifest())
        atomic_json(
            run_root / "runtime_binding.json",
            {
                "node": os.uname().nodename,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "gpu": torch.cuda.get_device_name(0),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            },
        )

        policy = BoundedCoefficientBSplinePolicy(
            np.zeros((7, int(config["scope"]["basis_dimension"])), dtype=np.float64)
        )
        omega_scale, effective_raw_lambda, curvature_receipt = _dimensionless_constants(
            policy,
            action_grid,
            q_scale,
            float(task["lambda_dimensionless"]),
        )
        atomic_json(
            run_root / "dimensionless_total_curvature_receipt.json", curvature_receipt
        )
        optimizer_config = TheoryTotalCurvatureAdamConfig(
            steps=40000, policy_lambda=effective_raw_lambda
        )
        tracker = init_offline_wandb(
            output_root=run_root,
            resolved_config={
                **identity,
                "effective_raw_policy_lambda": effective_raw_lambda,
                "omega_scale": omega_scale,
                "optimizer": asdict(optimizer_config),
            },
            name=(
                f"oracle-krr-n{int(task['n_transitions'])}-"
                f"lam{float(task['lambda_dimensionless']):g}-"
                f"seed{int(task['master_seed']):02d}"
            ),
            group="oracle-lambda-rate-krr-candidates",
            job_type="oracle-lambda-candidate-fit",
            tags=[
                "KRR",
                "functional-FFQI",
                "oracle-lambda",
                "dimensionless-total-curvature",
                f"n-{int(task['n_transitions'])}",
                f"lambda-{float(task['lambda_dimensionless']):g}",
            ],
            notes=(
                "Full M20 KRR FFQI candidate. Critic ridge uses grouped five-fold CV; "
                "policy lambda is a separate positive dimensionless total-curvature candidate."
            ),
        )

        critic.alpha = None
        critic.fit_metadata["target_solve_count"] = 0
        records: list[dict] = []
        full_paths: list[str] = []
        completed_iteration = 0
        pending = None
        index_path = run_root / "checkpoint_index.json"
        if index_path.is_file():
            index = _load_index(run_root, run_config)
            completed_iteration = int(index["completed_iteration"])
            full_paths = list(index["full_checkpoints"])
            if full_paths:
                checkpoint = torch.load(
                    run_root / full_paths[-1], weights_only=False, map_location="cpu"
                )
                critic.alpha = checkpoint["alpha"].to(critic.device)
                policy = _policy_from_payload(checkpoint["policy"])
                records = list(checkpoint["records"])
                _restore_critic_metadata(critic, checkpoint["critic_solve_metadata"])
            if index["segment_checkpoint"]:
                pending = torch.load(
                    run_root / index["segment_checkpoint"],
                    weights_only=False,
                    map_location="cpu",
                )
        else:
            _save_index(run_root, run_config, 0, [], None)

        environment = replace(load_pendulum_config(project_root), ode_substeps=128)
        if environment.gamma != float(config["ffqi"]["gamma"]):
            raise RuntimeError("oracle KRR environment gamma mismatch")

        for iteration in range(completed_iteration + 1, 21):
            iteration_started = time.time()
            initial_policy = policy
            if pending is not None:
                if int(pending["iteration"]) != iteration:
                    raise RuntimeError("oracle KRR segment iteration mismatch")
                critic.alpha = pending["alpha"].to(critic.device)
                initial_policy = _policy_from_payload(pending["initial_policy"])
                _restore_critic_metadata(critic, pending["critic_solve_metadata"])
                optimizer_resume = pending["optimizer"]
                objective_indices = pending["objective_indices"]
            else:
                if iteration == 1:
                    target = data.rewards
                else:
                    target = data.rewards + float(
                        config["ffqi"]["gamma"]
                    ) * FastPredictionView(critic).predict(
                        data.next_states,
                        policy.values_batch(data.next_states, critic.action_grid),
                    )
                with torch.no_grad():
                    critic.solve_targets(target)
                optimizer_resume = None
                rng = SeedTree(int(task["master_seed"])).rng(
                    "oracle_lambda_policy_optimizer", iteration
                )
                count = min(len(data.states), int(task["q_objective_state_count"]))
                objective_indices = np.sort(
                    rng.choice(len(data.states), count, replace=False)
                )

            selection = critic.fit_metadata["grouped_cv_selection"]
            atomic_json(
                run_root / "grouped_cv" / f"iteration_{iteration:03d}.json",
                selection,
            )

            def durable_optimizer_checkpoint(state: dict) -> None:
                relative = "recovery/segment.pt"
                _atomic_torch(
                    run_root / relative,
                    {
                        "iteration": iteration,
                        "alpha": critic.alpha.detach().cpu(),
                        "optimizer": state,
                        "initial_policy": _policy_payload(initial_policy),
                        "objective_indices": objective_indices,
                        "critic_solve_metadata": _critic_metadata(critic),
                    },
                )
                _save_index(
                    run_root,
                    run_config,
                    completed_iteration,
                    full_paths,
                    relative,
                )
                atomic_json(
                    run_root / "status.json",
                    {
                        "state": "RUNNING",
                        "iteration": iteration,
                        "optimizer_step": state["next_step"],
                        "best_objective": state["best_objective"],
                        "wall_seconds": time.time() - started,
                        "job": os.environ.get("SLURM_JOB_ID"),
                    },
                )
                signals.after_durable_checkpoint()

            policy, trace, _, optimizer_info = (
                bounded_adam_with_total_curvature_restart(
                    FastPredictionView(critic),
                    data.next_states[objective_indices],
                    initial_policy,
                    optimizer_config,
                    penalty_states=data.next_states,
                    resume=optimizer_resume,
                    checkpoint=durable_optimizer_checkpoint,
                )
            )
            _save_frame(
                run_root / "optimizer" / f"iteration_{iteration:03d}.parquet", trace
            )
            if optimizer_info["reason"] != "empirical_stability_and_finite_probe_check":
                atomic_json(
                    run_root / "status.json",
                    {
                        "state": "CONVERGENCE_NOT_REACHED",
                        "iteration": iteration,
                        **optimizer_info,
                    },
                )
                raise RuntimeError(
                    f"M{iteration}: KRR policy optimizer did not converge"
                )

            last = trace[-1]
            raw_curvature = float(last["uncentered_total_curvature"])
            q_mean = float(last["q_mean"])
            record = {
                "iteration": iteration,
                "selected_critic_ridge_lambda": selection["selected_lambda"],
                "grouped_cv_mse": selection["selected_grouped_cv_mse"],
                "critic_ridge_selected_at_grid_boundary": selection[
                    "selected_at_grid_boundary"
                ],
                "lambda_dimensionless": float(task["lambda_dimensionless"]),
                "effective_raw_policy_lambda": effective_raw_lambda,
                "q_mean": q_mean,
                "q_normalized": q_mean / q_scale,
                "raw_total_curvature": raw_curvature,
                "dimensionless_total_curvature": raw_curvature / omega_scale,
                "dimensionless_weighted_policy_penalty": float(
                    task["lambda_dimensionless"]
                )
                * raw_curvature
                / omega_scale,
                "dimensionless_objective": q_mean / q_scale
                - float(task["lambda_dimensionless"]) * raw_curvature / omega_scale,
                "optimizer_reason": optimizer_info["reason"],
                "optimizer_steps": len(trace),
                "optimizer_restart_used": bool(
                    optimizer_info.get("restart_used", False)
                ),
                "final_gradient_rms": last["gradient_rms"],
                "wall_seconds": time.time() - iteration_started,
            }

            policy_payload = _policy_payload(policy)
            if iteration in (19, 20):
                with atomic_path(
                    run_root / "policy_views" / f"M{iteration:03d}.npz"
                ) as temporary:
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            parameters=policy_payload["parameters"],
                            kind=np.asarray(policy_payload["kind"]),
                            iteration=np.asarray(iteration),
                        )
            if iteration == 20:
                evaluation = evaluate_policy(
                    policy,
                    tuning_seed,
                    environment,
                    episodes=int(tuning["episodes"]),
                    horizon=int(tuning["horizon"]),
                )
                record.update(evaluation.summary())
                record.update(
                    evaluation_master_seed=tuning_seed,
                    evaluation_seed_namespace=tuning["seed_namespace"],
                    evaluation_role="tuning_mc",
                )
                _write_tuning_npz(
                    run_root / "tuning_mc.npz",
                    fit_id=task["fit_id"],
                    policy_lambda=float(task["lambda_dimensionless"]),
                    evaluation_seed=tuning_seed,
                    raw_returns=evaluation.episode_returns,
                    normalized_returns=evaluation.normalized_returns,
                )

            records.append(record)
            full = f"recovery/full/iteration_{iteration:03d}.pt"
            _atomic_torch(
                run_root / full,
                {
                    "iteration": iteration,
                    "alpha": critic.alpha.detach().cpu(),
                    "policy": policy_payload,
                    "records": records,
                    "critic_solve_metadata": _critic_metadata(critic),
                    "resume_config": run_config,
                },
            )
            full_paths.append(full)
            while len(full_paths) > 2:
                (run_root / full_paths.pop(0)).unlink()
            (run_root / "recovery/segment.pt").unlink(missing_ok=True)
            completed_iteration = iteration
            _save_index(run_root, run_config, iteration, full_paths, None)
            _save_frame(run_root / "iteration_metrics.parquet", records)
            atomic_json(
                run_root / "status.json",
                {
                    "state": "RUNNING",
                    "completed_iteration": iteration,
                    "latest": record,
                    "job": os.environ.get("SLURM_JOB_ID"),
                },
            )
            if tracker is not None:
                tracker.log(record, step=iteration)
            signals.after_durable_checkpoint()
            print(
                json.dumps(
                    {"task_id": args.task_id, "fit_id": task["fit_id"], **record}
                ),
                flush=True,
            )
            pending = None

        tracking_state = finish_wandb_without_affecting_science(tracker, run_root)
        atomic_json(
            run_root / "status.json",
            {
                "state": "COMPLETED",
                "completed_iteration": 20,
                "tracking_state": tracking_state,
                "job": os.environ.get("SLURM_JOB_ID"),
            },
        )
        paths = [
            path
            for path in run_root.rglob("*")
            if path.is_file()
            and path.name not in (".lock", "COMPLETE.json", "wandb_sync_receipt.json")
            and "wandb" not in path.parts[len(run_root.parts) :]
        ]
        artifacts = {str(path.relative_to(run_root)): file_size(path) for path in paths}
        atomic_json(
            run_root / "COMPLETE.json", {"identity": identity, "artifacts": artifacts}
        )


if __name__ == "__main__":
    main()
