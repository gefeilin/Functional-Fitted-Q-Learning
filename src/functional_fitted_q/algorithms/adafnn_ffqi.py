from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from ..critics.adafnn_functional_critic import (
    AdaFNNCriticResumeState,
    AdaFNNCriticTrainingConfig,
    AdaFNNFunctionalCritic,
    fit_adafnn_critic,
)
from ..data import OfflineDataset
from ..policies.bspline_policy import BoundedCoefficientBSplinePolicy
from ..policies.constant_policy import BoundedConstantActionPolicy
from ..run_artifacts import atomic_json, atomic_path, file_size
from ..seeding import SeedTree
from .cmaes_policy_optimization import CMAESResumeState, DerivativeFreeCMAES


@dataclass
class AdaFNNFittedQResult:
    final_critic: AdaFNNFunctionalCritic
    final_policy: BoundedCoefficientBSplinePolicy | BoundedConstantActionPolicy
    completed_iteration: int
    training_trace: list[dict]
    policy_trace: list[dict]
    checkpoint_index: dict | None


def _normalize_config(document: dict) -> dict:
    """Normalize tuples and lists before comparing saved training settings."""
    return json.loads(json.dumps(document, allow_nan=False))


def _atomic_torch_save(path: Path, payload: dict) -> None:
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
    raise TypeError(f"unsupported AdaFNN policy type: {type(policy)!r}")


def _policy_from_payload(payload: dict):
    if payload["kind"] == "bounded_coefficient_bspline":
        return BoundedCoefficientBSplinePolicy(
            np.asarray(payload["parameters"], dtype=np.float64), int(payload["degree"])
        )
    if payload["kind"] == "bounded_constant_function":
        return BoundedConstantActionPolicy(
            np.asarray(payload["parameters"], dtype=np.float64)
        )
    raise ValueError("unknown AdaFNN policy payload")


def _new_critic(
    grid: np.ndarray,
    config: AdaFNNCriticTrainingConfig,
    gamma: float,
    device: torch.device,
) -> AdaFNNFunctionalCritic:
    return AdaFNNFunctionalCritic(
        quadrature_grid=grid,
        n_basis_nodes=config.n_basis_nodes,
        micro_hidden_layers=config.micro_hidden_layers,
        outer_hidden_layers=config.outer_hidden_layers,
        value_max=1.0 / (1.0 - gamma),
        parameter_bound=config.parameter_bound,
    ).to(device)


def _critic_from_state(
    state: dict[str, torch.Tensor],
    grid: np.ndarray,
    config: AdaFNNCriticTrainingConfig,
    gamma: float,
    device: torch.device,
) -> AdaFNNFunctionalCritic:
    critic = _new_critic(grid, config, gamma, device)
    critic.load_state_dict(state)
    return critic.eval()


def _resample_action_values(values: np.ndarray, target_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    source_grid = np.linspace(0.0, 1.0, values.shape[1])
    if np.array_equal(source_grid, target_grid):
        return values.astype(np.float32, copy=True)
    result = CubicSpline(source_grid, values, axis=1, bc_type="natural")(target_grid)
    return np.clip(result, -2.0, 2.0).astype(np.float32)


@torch.no_grad()
def adafnn_policy_q_values(
    critic: AdaFNNFunctionalCritic,
    policy,
    states: np.ndarray,
    grid: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(states), batch_size):
        stop = min(len(states), start + batch_size)
        state_block = np.asarray(states[start:stop], dtype=np.float32)
        action_block = policy.values_batch(state_block, grid).astype(np.float32)
        outputs.append(
            critic(
                torch.as_tensor(state_block, dtype=torch.float32, device=device),
                torch.as_tensor(action_block, dtype=torch.float32, device=device),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs)


def _load_index(checkpoint_dir: Path, resume_config: dict) -> dict:
    index_path = checkpoint_dir / "checkpoint_index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise FileNotFoundError("AdaFNN checkpoint index is missing")
    document = json.loads(index_path.read_text())
    if document.get("resume_config") != resume_config:
        raise RuntimeError("AdaFNN checkpoint invariant mismatch")
    for relative in document.get("full_checkpoints", []):
        path = checkpoint_dir / relative
        if not path.is_file() or file_size(path) != document["file_sizes"][relative]:
            raise RuntimeError(f"AdaFNN full checkpoint integrity failure: {relative}")
    segment = document.get("segment_checkpoint")
    if segment:
        path = checkpoint_dir / segment
        if not path.is_file() or file_size(path) != document["file_sizes"][segment]:
            raise RuntimeError("AdaFNN segment checkpoint integrity failure")
    return document


def _save_index(
    checkpoint_dir: Path,
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
        "file_sizes": {
            relative: file_size(checkpoint_dir / relative) for relative in paths
        },
    }
    atomic_json(checkpoint_dir / "checkpoint_index.json", document)
    return document


def run_adafnn_ffqi(
    *,
    dataset: OfflineDataset,
    dataset_identity: str,
    master_seed: int,
    iterations: int,
    quadrature_grid: np.ndarray,
    critic_config: AdaFNNCriticTrainingConfig,
    policy_optimizer: DerivativeFreeCMAES,
    device: torch.device,
    k: int = 12,
    gamma: float = 0.95,
    policy_kind: str = "bounded_coefficient_bspline",
    checkpoint_dir: Path | None = None,
    policy_view_iterations: tuple[int, ...] = (1, 2, 4, 8, 12, 20, 30),
    resume: bool = False,
    checkpoint_retention_limit: int = 2,
    critic_convergence_epoch_limit: int | None = None,
    segment_observer: Callable[[dict], None] | None = None,
    iteration_observer: (
        Callable[
            [
                int,
                AdaFNNFunctionalCritic,
                object,
                AdaFNNFunctionalCritic | None,
                object,
            ],
            None,
        ]
        | None
    ) = None,
) -> AdaFNNFittedQResult:
    """Run AdaFNN fitted Q-iteration with resumable critic and policy steps.

    Each iteration fits a Bellman-regression critic and then improves either a
    functional B-spline policy or a constant-action comparator.  The retained
    checkpoints are sufficient for exact interruption recovery and the paper's
    adjacent-critic identification analysis.
    """
    if iterations <= 0 or checkpoint_retention_limit != 2:
        raise ValueError(
            "positive iterations and the frozen two-checkpoint limit are required"
        )
    if getattr(policy_optimizer.config, "penalty_convention", None) is not None:
        if (
            type(iterations) is not int
            or iterations > 20
            or any(m > 20 for m in policy_view_iterations)
        ):
            raise ValueError("configured FQI horizon is M20, including saved views")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0,1)")
    grid = np.asarray(quadrature_grid, dtype=np.float64)
    if not np.array_equal(grid, policy_optimizer.grid):
        raise ValueError("critic and policy-optimizer quadrature grids differ")
    if policy_kind == "bounded_coefficient_bspline":
        policy = BoundedCoefficientBSplinePolicy(np.zeros((7, k), dtype=np.float64))
    elif policy_kind == "bounded_constant_function":
        policy = BoundedConstantActionPolicy(np.zeros(7, dtype=np.float64))
    else:
        raise ValueError("unsupported AdaFNN policy kind")
    data_actions = _resample_action_values(dataset.action_values, grid)
    invariant = {
        "schema_version": 1,
        "algorithm": "v12_adafnn_ffqi",
        "dataset_identity": str(dataset_identity),
        "dataset_shape": [
            int(len(dataset.states)),
            int(dataset.action_values.shape[1]),
        ],
        "master_seed": int(master_seed),
        "iterations": int(iterations),
        "gamma": float(gamma),
        "k": int(k),
        "policy_kind": policy_kind,
        "quadrature_grid": grid.tolist(),
        "critic_config": asdict(critic_config),
        "policy_optimizer_config": asdict(policy_optimizer.config),
        "policy_view_iterations": list(policy_view_iterations),
    }
    resume_config = _normalize_config(invariant)
    tree = SeedTree(master_seed)
    previous_critic = None
    completed_iteration = 0
    training_trace: list[dict] = []
    policy_trace: list[dict] = []
    full_paths: list[str] = []
    pending_segment = None
    index = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        invariant_path = checkpoint_dir / "checkpoint_invariant.json"
        if invariant_path.exists():
            existing = json.loads(invariant_path.read_text())
            if _normalize_config(existing) != resume_config:
                raise RuntimeError("existing AdaFNN checkpoint invariant changed")
        else:
            atomic_json(invariant_path, invariant)
        index_exists = (checkpoint_dir / "checkpoint_index.json").exists()
        if resume:
            index = _load_index(checkpoint_dir, resume_config)
            completed_iteration = int(index["completed_iteration"])
            full_paths = list(index["full_checkpoints"])
            if full_paths:
                payload = torch.load(
                    checkpoint_dir / full_paths[-1],
                    map_location="cpu",
                    weights_only=False,
                )
                previous_critic = _critic_from_state(
                    payload["critic_state"], grid, critic_config, gamma, device
                )
                policy = _policy_from_payload(payload["policy"])
                training_trace = list(payload["training_trace"])
                policy_trace = list(payload["policy_trace"])
            if index.get("segment_checkpoint"):
                pending_segment = torch.load(
                    checkpoint_dir / index["segment_checkpoint"],
                    map_location="cpu",
                    weights_only=False,
                )
        elif index_exists:
            raise FileExistsError("AdaFNN checkpoints already exist; use resume=True")
    elif resume:
        raise ValueError("resume requires checkpoint_dir")
    if completed_iteration > iterations:
        raise ValueError("checkpoint is beyond requested iteration budget")

    for iteration in range(completed_iteration + 1, iterations + 1):
        prior_critic = previous_critic
        prior_policy = policy
        segment = (
            pending_segment
            if pending_segment and pending_segment["iteration"] == iteration
            else None
        )
        if previous_critic is None:
            targets = np.asarray(dataset.rewards, dtype=np.float32).copy()
        else:
            targets = np.asarray(
                dataset.rewards, dtype=np.float32
            ) + gamma * adafnn_policy_q_values(
                previous_critic, policy, dataset.next_states, grid, device
            ).astype(
                np.float32
            )
        initialization_seed = int(
            tree.seed_sequence(
                "adaptive_basis_initialization", iteration
            ).generate_state(1)[0]
        )
        torch.manual_seed(initialization_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(initialization_seed)
        critic = _new_critic(grid, critic_config, gamma, device)
        if previous_critic is not None:
            critic.load_state_dict(copy.deepcopy(previous_critic.state_dict()))
        critic_resume_state = None
        critic_initial_trace: list[dict] | None = None
        if segment and segment["phase"] == "critic":
            critic_resume_state = segment["critic_resume_state"]
            critic_initial_trace = list(segment["iteration_training_trace"])

        def save_segment(payload: dict) -> None:
            nonlocal index
            if checkpoint_dir is None:
                return
            relative = "recovery/segment.pt"
            _atomic_torch_save(checkpoint_dir / relative, payload)
            index = _save_index(
                checkpoint_dir,
                resume_config,
                completed_iteration,
                full_paths,
                relative,
            )
            if segment_observer is not None:
                segment_observer(index)

        def critic_callback(model, state: AdaFNNCriticResumeState, trace, elapsed):
            save_segment(
                {
                    "schema_version": 1,
                    "resume_config": resume_config,
                    "iteration": iteration,
                    "phase": "critic",
                    "critic_resume_state": state,
                    "iteration_training_trace": list(trace),
                }
            )

        if segment and segment["phase"] == "policy":
            critic = _critic_from_state(
                segment["critic_state"], grid, critic_config, gamma, device
            )
            iteration_training_trace = list(segment["iteration_training_trace"])
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int(
                    tree.seed_sequence(
                        "critic_minibatch_order", iteration
                    ).generate_state(1)[0]
                )
            )
            critic, iteration_training_trace, _ = fit_adafnn_critic(
                critic,
                dataset.states,
                data_actions,
                targets,
                critic_config,
                device,
                generator,
                resume_state=critic_resume_state,
                initial_trace=critic_initial_trace,
                checkpoint_callback=(
                    critic_callback if checkpoint_dir is not None else None
                ),
                checkpoint_every_updates=500,
                checkpoint_every_seconds=30.0,
                # This is intentionally not part of the estimator invariant:
                # it only expands the computation ceiling needed to meet the
                # unchanged early-stopping acceptance rule.  Every expansion
                # is bound to a per-run immutable recovery receipt by the
                # formal runner.
                convergence_epoch_limit_override=critic_convergence_epoch_limit,
            )
        training_trace.extend(
            {"iteration": iteration, **row} for row in iteration_training_trace
        )

        policy_resume_state = (
            segment["policy_resume_state"]
            if segment and segment["phase"] == "policy"
            else None
        )
        policy_initial_trace = (
            list(segment["iteration_policy_trace"])
            if segment and segment["phase"] == "policy"
            else []
        )
        optimizer_rng = tree.rng("policy_optimizer", iteration)

        def policy_callback(state: CMAESResumeState, elapsed: float):
            save_segment(
                {
                    "schema_version": 1,
                    "resume_config": resume_config,
                    "iteration": iteration,
                    "phase": "policy",
                    "critic_state": {
                        name: value.detach().cpu()
                        for name, value in critic.state_dict().items()
                    },
                    "iteration_training_trace": iteration_training_trace,
                    "policy_resume_state": state,
                    "iteration_policy_trace": list(state.trace),
                }
            )

        if policy_kind == "bounded_coefficient_bspline":
            policy, iteration_policy_trace, _ = policy_optimizer.optimize(
                critic,
                dataset.next_states,
                policy,
                optimizer_rng,
                resume_state=policy_resume_state,
                checkpoint_callback=(
                    policy_callback if checkpoint_dir is not None else None
                ),
                checkpoint_every_generations=5,
                checkpoint_every_seconds=30.0,
            )
        else:
            policy, iteration_policy_trace, _ = policy_optimizer.optimize_constant(
                critic,
                dataset.next_states,
                policy,
                optimizer_rng,
                resume_state=policy_resume_state,
                checkpoint_callback=(
                    policy_callback if checkpoint_dir is not None else None
                ),
                checkpoint_every_generations=5,
                checkpoint_every_seconds=30.0,
            )
        if (
            policy_initial_trace
            and iteration_policy_trace[: len(policy_initial_trace)]
            != policy_initial_trace
        ):
            raise RuntimeError("AdaFNN policy trace changed across resume")
        policy_trace.extend(
            {"iteration": iteration, **row} for row in iteration_policy_trace
        )
        previous_critic = critic
        completed_iteration = iteration
        if iteration_observer is not None:
            iteration_observer(iteration, critic, policy, prior_critic, prior_policy)

        if checkpoint_dir is not None:
            relative = f"recovery/full/iteration_{iteration:03d}.pt"
            _atomic_torch_save(
                checkpoint_dir / relative,
                {
                    "schema_version": 1,
                    "resume_config": resume_config,
                    "completed_iteration": iteration,
                    "critic_state": {
                        name: value.detach().cpu()
                        for name, value in critic.state_dict().items()
                    },
                    "policy": _policy_payload(policy),
                    "training_trace": training_trace,
                    "policy_trace": policy_trace,
                },
            )
            full_paths.append(relative)
            while len(full_paths) > checkpoint_retention_limit:
                obsolete = checkpoint_dir / full_paths.pop(0)
                obsolete.unlink(missing_ok=True)
            segment_path = checkpoint_dir / "recovery/segment.pt"
            segment_path.unlink(missing_ok=True)
            index = _save_index(
                checkpoint_dir,
                resume_config,
                completed_iteration,
                full_paths,
                None,
            )
            if iteration in policy_view_iterations:
                view_path = checkpoint_dir / "policy_views" / f"M{iteration:03d}.npz"
                with atomic_path(view_path) as temporary:
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            kind=np.asarray(_policy_payload(policy)["kind"]),
                            parameters=_policy_payload(policy)["parameters"],
                            degree=np.asarray(
                                _policy_payload(policy).get("degree", -1),
                                dtype=np.int64,
                            ),
                            iteration=np.asarray(iteration, dtype=np.int64),
                        )
        pending_segment = None

    if previous_critic is None:
        raise RuntimeError("AdaFNN FFQI produced no fitted critic")
    return AdaFNNFittedQResult(
        final_critic=previous_critic,
        final_policy=policy,
        completed_iteration=completed_iteration,
        training_trace=training_trace,
        policy_trace=policy_trace,
        checkpoint_index=index,
    )


__all__ = ["AdaFNNFittedQResult", "adafnn_policy_q_values", "run_adafnn_ffqi"]
