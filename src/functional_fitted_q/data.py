"""Logged functional-action trajectories and nested subject samples.

Rows flatten the paper's (i, t) indices: states and next_states have shape
(n, 3), action_values has shape (n, action_grid_points), and rewards has
shape (n,). Keep subject_ids when splitting data because n = N*T transitions
are not n independent subjects. See docs/notation.md for the symbol mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline

from .env.batched import apply_transition_noise_batch, deterministic_transition_batch
from .env.dynamics import FunctionalPendulumEnv, PendulumConfig
from .policies.behavior_gp_policy import BehaviorGPPolicy
from .policies.reference_policy import AnalyticReferencePolicy
from .seeding import SeedTree


DATASET_ARRAY_NAMES = (
    "states",
    "action_values",
    "rewards",
    "next_states",
    "subject_ids",
    "time_indices",
    "subject_order",
)


def verify_offline_dataset(directory: Path, *, require_complete: bool = True) -> dict:
    directory = Path(directory)
    if require_complete and not (directory / "_SUCCESS").is_file():
        raise RuntimeError(f"offline dataset success marker is missing: {directory}")
    manifest_path = directory / "dataset_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeError("offline dataset manifest is missing or is a symlink")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported offline dataset manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("offline dataset manifest artifacts are invalid")
    records = {
        record.get("path"): record for record in artifacts if isinstance(record, dict)
    }
    expected_paths = {f"{name}.npy" for name in DATASET_ARRAY_NAMES}
    if set(records) != expected_paths or len(records) != len(artifacts):
        raise RuntimeError(
            "offline dataset artifact inventory is incomplete or duplicated"
        )
    for relative_path, record in records.items():
        path = directory / relative_path
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"offline dataset artifact is missing or is a symlink: {relative_path}"
            )
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(
                f"offline dataset artifact byte count mismatch: {relative_path}"
            )
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != list(record["shape"]) or str(array.dtype) != str(
            record["dtype"]
        ):
            raise RuntimeError(
                f"offline dataset artifact shape/dtype mismatch: {relative_path}"
            )
    if int(manifest["n_transitions"]) != int(records["states.npy"]["shape"][0]):
        raise RuntimeError("offline dataset transition count is inconsistent")
    return manifest


@dataclass
class OfflineDataset:
    states: np.ndarray
    action_values: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    subject_ids: np.ndarray
    time_indices: np.ndarray
    subject_order: np.ndarray

    def subset_subjects(self, count: int) -> "OfflineDataset":
        """Keep a prefix of whole subjects, preserving the nested sample design."""
        selected = self.subject_order[:count]
        mask = np.isin(self.subject_ids, selected)
        return OfflineDataset(
            self.states[mask],
            self.action_values[mask],
            self.rewards[mask],
            self.next_states[mask],
            self.subject_ids[mask],
            self.time_indices[mask],
            self.subject_order.copy(),
        )

    def save(self, directory: Path, metadata: dict | None = None) -> dict:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            "states": self.states,
            "action_values": self.action_values,
            "rewards": self.rewards,
            "next_states": self.next_states,
            "subject_ids": self.subject_ids,
            "time_indices": self.time_indices,
            "subject_order": self.subject_order,
        }
        artifacts = []
        for name, array in arrays.items():
            target = directory / f"{name}.npy"
            temporary = directory / f".{name}.npy.tmp"
            with temporary.open("wb") as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(temporary, target)
            artifacts.append(
                {
                    "path": target.name,
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "bytes": target.stat().st_size,
                }
            )
        manifest = {
            "schema_version": 1,
            "n_transitions": int(len(self.states)),
            "n_subjects": int(len(np.unique(self.subject_ids))),
            "artifacts": artifacts,
            "metadata": metadata or {},
        }
        temporary = directory / ".dataset_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, directory / "dataset_manifest.json")
        return manifest

    @classmethod
    def load(cls, directory: Path, mmap_mode: str | None = "r") -> "OfflineDataset":
        directory = Path(directory)
        return cls(
            *(
                np.load(
                    directory / f"{name}.npy", mmap_mode=mmap_mode, allow_pickle=False
                )
                for name in DATASET_ARRAY_NAMES
            )
        )


def generate_offline_dataset(
    master_seed: int,
    reference: AnalyticReferencePolicy,
    n_subjects: int = 2500,
    t_per_subject: int = 20,
    env_config: PendulumConfig | None = None,
    gp_resolution: int = 128,
    step_gp_amplitude: float = 0.35,
    vectorized: bool = True,
) -> OfflineDataset:
    """Generate one seeded longitudinal offline-data pool for all sample sizes."""
    tree = SeedTree(master_seed)
    initial_rng = tree.rng("subject_initial_states")
    environment = FunctionalPendulumEnv(
        env_config or PendulumConfig(),
        seed=int(tree.seed_sequence("process_noise").generate_state(1)[0]),
    )
    behavior = BehaviorGPPolicy(
        reference,
        tree.rng("subject_behavior_gp"),
        tree.rng("step_behavior_gp"),
        resolution=gp_resolution,
        step_amplitude=step_gp_amplitude,
    )
    if vectorized:
        return _generate_offline_dataset_vectorized(
            tree=tree,
            initial_rng=initial_rng,
            environment=environment,
            behavior=behavior,
            n_subjects=n_subjects,
            t_per_subject=t_per_subject,
        )
    states, action_values, rewards, next_states, subject_ids, time_indices = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for subject_id in range(n_subjects):
        theta = initial_rng.uniform(-np.pi, np.pi)
        omega = initial_rng.uniform(-1.0, 1.0)
        state = environment.reset(theta=theta, omega=omega)
        for time_index in range(t_per_subject):
            action = behavior.sample(state, subject_id)
            next_state, reward, _ = environment.step(action)
            states.append(state)
            action_values.append(action.values)
            rewards.append(reward)
            next_states.append(next_state)
            subject_ids.append(subject_id)
            time_indices.append(time_index)
            state = next_state
    subject_order = tree.rng("dataset_subject_order").permutation(n_subjects)
    return OfflineDataset(
        np.asarray(states),
        np.asarray(action_values),
        np.asarray(rewards),
        np.asarray(next_states),
        np.asarray(subject_ids),
        np.asarray(time_indices),
        subject_order,
    )


def _generate_offline_dataset_vectorized(
    *,
    tree: SeedTree,
    initial_rng: np.random.Generator,
    environment: FunctionalPendulumEnv,
    behavior: BehaviorGPPolicy,
    n_subjects: int,
    t_per_subject: int,
) -> OfflineDataset:
    """Generate all subjects in parallel while preserving seeded RNG streams."""
    initial_uniforms = initial_rng.random((n_subjects, 2))
    theta = -np.pi + 2.0 * np.pi * initial_uniforms[:, 0]
    omega = -1.0 + 2.0 * initial_uniforms[:, 1]
    current_states = np.column_stack([np.cos(theta), np.sin(theta), omega])
    subject_effects = behavior.draw_subject_effects(n_subjects)
    step_effects = behavior.draw_step_effects(n_subjects, t_per_subject)

    process_rng = np.random.default_rng(
        int(tree.seed_sequence("process_noise").generate_state(1)[0])
    )
    theta_normals = np.empty((n_subjects, t_per_subject), dtype=np.float64)
    omega_uniforms = np.empty((n_subjects, t_per_subject), dtype=np.float64)
    for subject_id in range(n_subjects):
        for time_index in range(t_per_subject):
            theta_normals[subject_id, time_index] = process_rng.standard_normal()
            omega_uniforms[subject_id, time_index] = process_rng.random()

    states = np.empty((n_subjects, t_per_subject, 3), dtype=np.float64)
    actions = np.empty(
        (n_subjects, t_per_subject, behavior.resolution), dtype=np.float64
    )
    rewards = np.empty((n_subjects, t_per_subject), dtype=np.float64)
    next_states = np.empty((n_subjects, t_per_subject, 3), dtype=np.float64)
    integration_grid = np.linspace(0.0, 1.0, 2 * environment.config.ode_substeps + 1)
    for time_index in range(t_per_subject):
        action_values = behavior.sample_values_batch(
            current_states,
            subject_effects,
            step_effects[:, time_index],
        )
        torque_values = np.clip(
            CubicSpline(behavior.grid, action_values, axis=1, bc_type="natural")(
                integration_grid
            ),
            -environment.config.torque_limit,
            environment.config.torque_limit,
        )
        deterministic_states, step_rewards = deterministic_transition_batch(
            current_states,
            torque_values,
            environment.config,
        )
        noisy_states = apply_transition_noise_batch(
            deterministic_states,
            theta_normals[:, time_index],
            omega_uniforms[:, time_index],
            environment.config,
        )
        states[:, time_index] = current_states
        actions[:, time_index] = action_values
        rewards[:, time_index] = step_rewards
        next_states[:, time_index] = noisy_states
        current_states = noisy_states
    subject_order = tree.rng("dataset_subject_order").permutation(n_subjects)
    return OfflineDataset(
        states.reshape(-1, 3),
        actions.reshape(-1, behavior.resolution),
        rewards.reshape(-1),
        next_states.reshape(-1, 3),
        np.repeat(np.arange(n_subjects), t_per_subject),
        np.tile(np.arange(t_per_subject), n_subjects),
        subject_order,
    )
