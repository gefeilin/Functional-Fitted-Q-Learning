from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .env.batched import apply_transition_noise_batch, deterministic_transition_batch
from .env.dynamics import PendulumConfig
from .policies.behavior_gp_policy import BehaviorGPPolicy
from .policies.reference_policy import AnalyticReferencePolicy
from .seeding import SeedTree


@dataclass
class EvaluationResult:
    episode_returns: np.ndarray
    normalized_returns: np.ndarray

    def summary(self) -> dict:
        return {
            "episodes": int(len(self.episode_returns)),
            "J_raw_mean": float(np.mean(self.episode_returns)),
            "J_raw_std": float(np.std(self.episode_returns, ddof=1)),
            "J_normalized_mean": float(np.mean(self.normalized_returns)),
            "J_normalized_std": float(np.std(self.normalized_returns, ddof=1)),
        }


def evaluation_randomness(
    master_seed: int, episodes: int, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = SeedTree(master_seed)
    initial_rng = tree.rng("evaluation_initial_states")
    initial_conditions = np.column_stack(
        [
            initial_rng.uniform(-np.pi, np.pi, size=episodes),
            initial_rng.uniform(-1.0, 1.0, size=episodes),
        ]
    )
    theta_normals = np.empty((episodes, horizon), dtype=np.float64)
    omega_uniforms = np.empty((episodes, horizon), dtype=np.float64)
    for episode in range(episodes):
        rng = tree.rng("evaluation_process_noise", episode)
        theta_normals[episode] = rng.standard_normal(horizon)
        omega_uniforms[episode] = rng.random(horizon)
    return initial_conditions, theta_normals, omega_uniforms


def evaluate_policy(
    policy,
    master_seed: int,
    config: PendulumConfig,
    episodes: int = 1000,
    horizon: int = 100,
) -> EvaluationResult:
    if episodes <= 0 or horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    initial_conditions, theta_normals, omega_uniforms = evaluation_randomness(
        master_seed, episodes, horizon
    )
    states = np.column_stack(
        [
            np.cos(initial_conditions[:, 0]),
            np.sin(initial_conditions[:, 0]),
            initial_conditions[:, 1],
        ]
    )
    returns = np.zeros(episodes, dtype=np.float64)
    discount = 1.0
    action_grid = np.linspace(0.0, 1.0, 2 * config.ode_substeps + 1)
    for time_index in range(horizon):
        if hasattr(policy, "values_batch"):
            torque_values = policy.values_batch(states, action_grid)
        else:
            torque_values = np.stack(
                [policy.action(state)(action_grid) for state in states]
            )
        if (
            not np.isfinite(torque_values).all()
            or np.max(np.abs(torque_values)) > config.torque_limit + 1e-10
        ):
            raise ValueError(
                "policy produced a non-finite or out-of-bounds functional action"
            )
        deterministic_states, rewards = deterministic_transition_batch(
            states, torque_values, config
        )
        returns += discount * rewards
        discount *= config.gamma
        states = apply_transition_noise_batch(
            deterministic_states,
            theta_normals[:, time_index],
            omega_uniforms[:, time_index],
            config,
        )
    return EvaluationResult(returns, (1.0 - config.gamma) * returns)


def evaluate_behavior_policy(
    reference_policy: AnalyticReferencePolicy,
    master_seed: int,
    config: PendulumConfig,
    *,
    episodes: int = 1000,
    horizon: int = 100,
    resolution: int = 128,
    step_gp_amplitude: float = 0.35,
) -> tuple[EvaluationResult, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the stochastic behavior DGP with evaluation CRN and persistent subject effects."""
    if episodes <= 0 or horizon <= 0:
        raise ValueError("episodes and horizon must be positive")
    tree = SeedTree(master_seed)
    initial_conditions, theta_normals, omega_uniforms = evaluation_randomness(
        master_seed, episodes, horizon
    )
    states = np.column_stack(
        [
            np.cos(initial_conditions[:, 0]),
            np.sin(initial_conditions[:, 0]),
            initial_conditions[:, 1],
        ]
    )
    policy = BehaviorGPPolicy(
        reference_policy,
        tree.rng("evaluation_behavior_gp", 0),
        tree.rng("evaluation_behavior_gp", 1),
        resolution=resolution,
        step_amplitude=step_gp_amplitude,
    )
    subject_effects = policy.draw_subject_effects(episodes)
    step_effects = policy.draw_step_effects(episodes, horizon)
    returns = np.zeros(episodes, dtype=np.float64)
    discount = 1.0
    integration_grid = np.linspace(0.0, 1.0, 2 * config.ode_substeps + 1)
    example_states = None
    example_actions = None
    from scipy.interpolate import CubicSpline

    for time_index in range(horizon):
        action_values = policy.sample_values_batch(
            states, subject_effects, step_effects[:, time_index]
        )
        if time_index == 0:
            example_states = states[:32].copy()
            example_actions = action_values[:32].copy()
        torque_values = np.clip(
            CubicSpline(policy.grid, action_values, axis=1, bc_type="natural")(
                integration_grid
            ),
            -config.torque_limit,
            config.torque_limit,
        )
        deterministic_states, rewards = deterministic_transition_batch(
            states, torque_values, config
        )
        returns += discount * rewards
        discount *= config.gamma
        states = apply_transition_noise_batch(
            deterministic_states,
            theta_normals[:, time_index],
            omega_uniforms[:, time_index],
            config,
        )
    assert example_states is not None and example_actions is not None
    return (
        EvaluationResult(returns, (1.0 - config.gamma) * returns),
        example_states,
        policy.grid.copy(),
        example_actions,
    )
