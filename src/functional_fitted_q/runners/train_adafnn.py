"""Train one functional-action or constant-action AdaFNN FQI model."""

from dataclasses import asdict
from pathlib import Path
import fcntl
import json
import signal
import numpy as np
import pandas as pd
import yaml
from functional_fitted_q.algorithms.adafnn_ffqi import run_adafnn_ffqi
from functional_fitted_q.algorithms.cmaes_policy_optimization import (
    CMAESPolicyOptimizationConfig,
    DerivativeFreeCMAES,
)
from functional_fitted_q.algorithms.dimensionless_total_curvature import (
    DimensionlessTheoryTotalCurvatureCMAES,
    DimensionlessTheoryTotalCurvatureCMAESConfig,
)
from functional_fitted_q.critics.adafnn_functional_critic import (
    AdaFNNCriticTrainingConfig,
    critic_epoch_limit,
)
from functional_fitted_q.data import OfflineDataset, verify_offline_dataset
from functional_fitted_q.design import (
    build_manifest,
    constant_fit_id,
    derived_mc_seed,
    display_name,
    load_config,
)
from functional_fitted_q.evaluation import evaluate_policy
from functional_fitted_q.formal_config import load_pendulum_config
from functional_fitted_q.run_artifacts import (
    atomic_json,
    atomic_path,
    write_run_artifact_manifest,
    verify_run_artifact_manifest,
)
from functional_fitted_q.runtime import deterministic_device


def subset(data, subjects, length):
    mask = np.isin(data.subject_ids, data.subject_order[:subjects]) & (
        data.time_indices < length
    )
    return OfflineDataset(
        data.states[mask],
        data.action_values[mask],
        data.rewards[mask],
        data.next_states[mask],
        data.subject_ids[mask],
        data.time_indices[mask],
        data.subject_order.copy(),
    )


def critic_config(parent):
    common, selected = parent["critic"], parent["selected"]
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


def optimizer(parent, task, grid, device, constant=False):
    p = parent["policy"]
    kw = dict(
        restarts=int(p["cma_restarts"]),
        max_objective_evals=int(parent["selected"]["cma_max_objective_evals"]),
        population_size=int(p["cma_population_size"]),
        initial_sigma=float(p["cma_initial_sigma"]),
        minimum_sigma=float(p["cma_minimum_sigma"]),
        objective_state_count=int(p["objective_state_count"]),
        lambda_roughness=0.0,
        candidate_batch_size=int(p["candidate_batch_size"]),
    )
    if constant:
        return DerivativeFreeCMAES(CMAESPolicyOptimizationConfig(**kw), grid, device)
    kw.update(lambda_dimensionless=float(task["lambda_dimensionless"]), q_scale=20.0)
    return DimensionlessTheoryTotalCurvatureCMAES(
        DimensionlessTheoryTotalCurvatureCMAESConfig(**kw), grid, device
    )


def run(root, data_root, output_root, task_id, *, constant=False):
    """Train one AdaFNN functional or constant-action paper task."""
    device = deterministic_device()
    design = load_config(root)
    task = build_manifest(root)["candidate_fits"][task_id]
    if task["approximator"] != "adafnn":
        raise ValueError("AdaFNN task required")
    fit_id = constant_fit_id(task) if constant else task["fit_id"]
    destination = Path(output_root) / fit_id
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / "_SUCCESS").exists():
            verify_run_artifact_manifest(destination)
            return
        parent = yaml.safe_load((root / "configs/adafnn.yaml").read_text())
        data_path = Path(data_root) / task["data_pool_id"]
        verify_offline_dataset(data_path)
        data = subset(OfflineDataset.load(data_path), int(task["n_subjects"]), 20)
        if len(data.states) != int(task["n_transitions"]):
            raise ValueError("Nested sample cardinality mismatch")
        grid = np.linspace(0, 1, 128)
        interrupted = False

        def request_stop(signum, frame):
            nonlocal interrupted
            interrupted = True

        def after_checkpoint(index):
            if interrupted:
                raise InterruptedError("Saved a resumable segment before stopping")

        signal.signal(signal.SIGUSR1, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        resolved = dict(
            task=task,
            constant=constant,
            display_name=display_name(task, "constant" if constant else "functional"),
            experiment_settings=design,
            critic_settings=parent,
        )
        identity_path = destination / "config_resolved.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != resolved:
            raise RuntimeError("Training settings changed; use a new runs directory")
        atomic_json(identity_path, resolved)
        recovery = destination / "convergence_recovery.json"
        cap = (
            json.loads(recovery.read_text())["epoch_cap"]
            if recovery.exists()
            else critic_epoch_limit(critic_config(parent))
        )
        while True:
            try:
                result = run_adafnn_ffqi(
                    dataset=data,
                    dataset_identity=task["data_pool_id"],
                    master_seed=int(task["master_seed"]),
                    iterations=20,
                    quadrature_grid=grid,
                    critic_config=critic_config(parent),
                    policy_optimizer=optimizer(parent, task, grid, device, constant),
                    device=device,
                    k=12,
                    gamma=0.95,
                    policy_kind=(
                        "bounded_constant_function"
                        if constant
                        else "bounded_coefficient_bspline"
                    ),
                    checkpoint_dir=destination,
                    policy_view_iterations=(20,),
                    resume=(destination / "checkpoint_index.json").exists(),
                    checkpoint_retention_limit=2,
                    critic_convergence_epoch_limit=cap,
                    segment_observer=after_checkpoint,
                )
                break
            except RuntimeError as error:
                if (
                    "AdaFNN critic reached the convergence hard cap before the frozen early-stopping gate"
                    not in str(error)
                    or cap >= 9600
                ):
                    raise
                cap = min(9600, cap * 2)
                atomic_json(recovery, dict(epoch_cap=cap, criterion_changed=False))
        pd.DataFrame(result.training_trace).to_parquet(
            destination / "training_metrics.parquet", index=False
        )
        pd.DataFrame(result.policy_trace).to_parquet(
            destination / "policy_optimizer_trace.parquet", index=False
        )
        role = "reporting" if constant else "tuning"
        stream = design["monte_carlo"][role]
        seed = derived_mc_seed(task["master_seed"], stream["seed_namespace"])
        evaluation = evaluate_policy(
            result.final_policy,
            seed,
            load_pendulum_config(root),
            episodes=1000,
            horizon=100,
        )
        with atomic_path(destination / (role + "_mc.npz")) as p:
            with p.open("wb") as f:
                np.savez_compressed(
                    f,
                    fit_id=np.asarray(fit_id),
                    lambda_dimensionless=np.asarray(
                        0.0 if constant else task["lambda_dimensionless"]
                    ),
                    evaluation_seed=np.asarray(seed, dtype=np.uint32),
                    stream=np.asarray(role),
                    raw_returns=evaluation.episode_returns,
                    normalized_returns=evaluation.normalized_returns,
                )
        atomic_json(
            destination / "evaluation_summary.json",
            dict(
                evaluation.summary(),
                fit_id=fit_id,
                n_transitions=int(task["n_transitions"]),
                master_seed=int(task["master_seed"]),
                evaluation_seed=seed,
                role=role,
                constant=constant,
            ),
        )
        artifacts = [
            str(p.relative_to(destination))
            for p in destination.rglob("*")
            if p.is_file()
            and p.name not in {".lock", "_SUCCESS", "run_artifact_manifest.json"}
        ]
        write_run_artifact_manifest(destination, artifacts)
        verify_run_artifact_manifest(destination)
        (destination / "_SUCCESS").touch()
        print(
            json.dumps(dict(state="COMPLETED", fit_id=fit_id, **evaluation.summary()))
        )
