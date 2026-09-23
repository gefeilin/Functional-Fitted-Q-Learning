from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import time
from typing import Callable

import numpy as np
import torch
from scipy.interpolate import BSpline

from ..policies.bspline_policy import (
    BoundedCoefficientBSplinePolicy,
    open_uniform_knots,
    policy_state_features_batch,
)
from ..policies.constant_policy import BoundedConstantActionPolicy


@dataclass(frozen=True)
class CMAESPolicyOptimizationConfig:
    restarts: int = 3
    max_objective_evals: int = 5000
    population_size: int = 64
    initial_sigma: float = 0.5
    minimum_sigma: float = 1.0e-5
    objective_state_count: int = 2048
    lambda_roughness: float = 0.0
    candidate_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.restarts <= 0 or self.max_objective_evals <= 0:
            raise ValueError("CMA-ES restart and evaluation budgets must be positive")
        if self.population_size < 4 or self.objective_state_count <= 0:
            raise ValueError("CMA-ES population must be >=4 and state count positive")
        if self.initial_sigma <= 0 or self.minimum_sigma <= 0:
            raise ValueError("CMA-ES sigma values must be positive")
        if self.lambda_roughness < 0 or self.candidate_batch_size <= 0:
            raise ValueError(
                "roughness penalty must be nonnegative and batch size positive"
            )


@dataclass
class CMAESResumeState:
    mode: str
    next_restart: int
    next_generation: int
    mean: np.ndarray | None
    sigma: float | None
    covariance: np.ndarray | None
    path_c: np.ndarray | None
    path_sigma: np.ndarray | None
    best_parameters: np.ndarray
    best_objective: float
    evaluations: int
    objective_state_indices: np.ndarray
    rng_state: dict
    trace: list[dict]


class DerivativeFreeCMAES:
    """Continuous full-covariance CMA-ES for the bounded functional policy sieve."""

    algorithm_schema = "v12_full_covariance_cmaes_v1"

    def __init__(
        self,
        config: CMAESPolicyOptimizationConfig,
        quadrature_grid: np.ndarray,
        device: torch.device,
    ) -> None:
        self.config = config
        self.grid = np.asarray(quadrature_grid, dtype=np.float64)
        if (
            self.grid.ndim != 1
            or len(self.grid) < 2
            or not np.all(np.diff(self.grid) > 0)
        ):
            raise ValueError("quadrature grid must be strictly increasing")
        self.device = device

    @staticmethod
    def _eigendecomposition(
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        covariance = (covariance + covariance.T) / 2.0
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, 1.0e-20)
        sqrt_values = np.sqrt(values)
        invsqrt = (vectors * (1.0 / sqrt_values)) @ vectors.T
        return vectors, sqrt_values, invsqrt

    def _optimize_parameters(
        self,
        *,
        initial_parameters: np.ndarray,
        states: np.ndarray,
        rng: np.random.Generator,
        mode: str,
        objective_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
        resume_state: CMAESResumeState | None,
        checkpoint_callback: Callable[[CMAESResumeState, float], None] | None,
        checkpoint_every_generations: int,
        checkpoint_every_seconds: float,
    ) -> tuple[np.ndarray, list[dict], CMAESResumeState]:
        """Run deterministic CMA-ES from a fresh or saved generation state."""
        if checkpoint_every_generations <= 0 or checkpoint_every_seconds <= 0:
            raise ValueError("CMA-ES checkpoint intervals must be positive")
        cfg = self.config
        states = np.asarray(states, dtype=np.float64)
        initial_parameters = np.asarray(initial_parameters, dtype=np.float64)
        dimension = initial_parameters.size
        population = cfg.population_size
        mu = population // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2.0) / (dimension + mu_eff + 5.0)
        d_sigma = (
            1.0
            + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (dimension + 1.0)) - 1.0)
            + c_sigma
        )
        c_c = (4.0 + mu_eff / dimension) / (dimension + 4.0 + 2.0 * mu_eff / dimension)
        c1 = 2.0 / ((dimension + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dimension + 2.0) ** 2 + mu_eff),
        )
        expected_norm = math.sqrt(dimension) * (
            1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2)
        )
        generations_per_restart = max(
            1, cfg.max_objective_evals // (cfg.restarts * population)
        )

        if resume_state is None:
            if len(states) > cfg.objective_state_count:
                objective_state_indices = np.sort(
                    rng.choice(len(states), cfg.objective_state_count, replace=False)
                )
            else:
                objective_state_indices = np.arange(len(states), dtype=np.int64)
            next_restart = 0
            next_generation = 0
            mean = None
            sigma = None
            covariance = None
            path_c = None
            path_sigma = None
            best_parameters = initial_parameters.copy()
            initial_objective = objective_function(
                initial_parameters[None, ...], states[objective_state_indices]
            )
            best_objective = float(initial_objective[0])
            evaluations = 1
            trace: list[dict] = []
        else:
            if resume_state.mode != mode:
                raise ValueError("CMA-ES resume mode mismatch")
            next_restart = int(resume_state.next_restart)
            next_generation = int(resume_state.next_generation)
            mean = None if resume_state.mean is None else resume_state.mean.copy()
            sigma = None if resume_state.sigma is None else float(resume_state.sigma)
            covariance = (
                None
                if resume_state.covariance is None
                else resume_state.covariance.copy()
            )
            path_c = None if resume_state.path_c is None else resume_state.path_c.copy()
            path_sigma = (
                None
                if resume_state.path_sigma is None
                else resume_state.path_sigma.copy()
            )
            best_parameters = resume_state.best_parameters.copy()
            best_objective = float(resume_state.best_objective)
            evaluations = int(resume_state.evaluations)
            objective_state_indices = resume_state.objective_state_indices.copy()
            trace = list(resume_state.trace)
            rng.bit_generator.state = copy.deepcopy(resume_state.rng_state)
        if best_parameters.shape != initial_parameters.shape:
            raise ValueError("CMA-ES resume parameter shape mismatch")
        if len(objective_state_indices) == 0 or int(
            objective_state_indices.max()
        ) >= len(states):
            raise ValueError("CMA-ES objective-state identity mismatch")
        objective_states = states[objective_state_indices]
        last_checkpoint_time = time.monotonic()

        def snapshot() -> CMAESResumeState:
            return CMAESResumeState(
                mode=mode,
                next_restart=next_restart,
                next_generation=next_generation,
                mean=None if mean is None else mean.copy(),
                sigma=sigma,
                covariance=None if covariance is None else covariance.copy(),
                path_c=None if path_c is None else path_c.copy(),
                path_sigma=None if path_sigma is None else path_sigma.copy(),
                best_parameters=best_parameters.copy(),
                best_objective=best_objective,
                evaluations=evaluations,
                objective_state_indices=objective_state_indices.copy(),
                rng_state=copy.deepcopy(rng.bit_generator.state),
                trace=list(trace),
            )

        while next_restart < cfg.restarts:
            if mean is None:
                mean = initial_parameters.ravel().copy()
                if next_restart > 0:
                    mean += rng.normal(scale=cfg.initial_sigma, size=dimension)
                sigma = float(cfg.initial_sigma)
                covariance = np.eye(dimension, dtype=np.float64)
                path_c = np.zeros(dimension, dtype=np.float64)
                path_sigma = np.zeros(dimension, dtype=np.float64)
                next_generation = 0

            while next_generation < generations_per_restart:
                vectors, sqrt_values, invsqrt = self._eigendecomposition(covariance)
                z = rng.standard_normal((population, dimension))
                y = (z * sqrt_values) @ vectors.T
                candidates_flat = mean[None, :] + sigma * y
                candidates = candidates_flat.reshape(
                    population, *initial_parameters.shape
                )
                objectives = objective_function(candidates, objective_states)
                if (
                    objectives.shape != (population,)
                    or not np.isfinite(objectives).all()
                ):
                    raise RuntimeError("CMA-ES objective returned invalid values")
                evaluations += population
                order = np.argsort(objectives)[::-1]
                selected_y = y[order[:mu]]
                selected_z = z[order[:mu]]
                y_w = np.sum(weights[:, None] * selected_y, axis=0)
                z_w = np.sum(weights[:, None] * selected_z, axis=0)
                old_mean = mean.copy()
                mean = mean + sigma * y_w
                path_sigma = (1.0 - c_sigma) * path_sigma + math.sqrt(
                    c_sigma * (2.0 - c_sigma) * mu_eff
                ) * (invsqrt @ ((mean - old_mean) / sigma))
                norm_path_sigma = float(np.linalg.norm(path_sigma))
                h_sigma = float(
                    norm_path_sigma
                    / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (next_generation + 1)))
                    < (1.4 + 2.0 / (dimension + 1.0)) * expected_norm
                )
                path_c = (1.0 - c_c) * path_c + h_sigma * math.sqrt(
                    c_c * (2.0 - c_c) * mu_eff
                ) * y_w
                rank_mu = sum(
                    weight * np.outer(direction, direction)
                    for weight, direction in zip(weights, selected_y)
                )
                covariance = (
                    (1.0 - c1 - c_mu) * covariance
                    + c1
                    * (
                        np.outer(path_c, path_c)
                        + (1.0 - h_sigma) * c_c * (2.0 - c_c) * covariance
                    )
                    + c_mu * rank_mu
                )
                covariance = (covariance + covariance.T) / 2.0
                sigma *= math.exp(
                    (c_sigma / d_sigma) * (norm_path_sigma / expected_norm - 1.0)
                )
                generation_best = int(order[0])
                if objectives[generation_best] > best_objective:
                    best_objective = float(objectives[generation_best])
                    best_parameters = candidates[generation_best].copy()
                next_generation += 1
                eigvals = np.linalg.eigvalsh(covariance)
                termination = "running"
                if sigma <= cfg.minimum_sigma:
                    termination = "minimum_sigma"
                elif next_generation == generations_per_restart:
                    termination = "budget"
                trace.append(
                    {
                        "restart": next_restart,
                        "generation": next_generation,
                        "objective_evaluations": evaluations,
                        "best_objective": best_objective,
                        "generation_mean_objective": float(objectives.mean()),
                        "generation_sd_objective": float(objectives.std(ddof=1)),
                        "sigma": float(sigma),
                        "covariance_condition": float(
                            eigvals[-1] / max(eigvals[0], 1.0e-20)
                        ),
                        "termination_reason": termination,
                    }
                )
                if termination != "running":
                    next_restart += 1
                    next_generation = 0
                    mean = None
                    sigma = None
                    covariance = None
                    path_c = None
                    path_sigma = None
                elapsed = time.monotonic() - last_checkpoint_time
                if checkpoint_callback is not None and (
                    len(trace) % checkpoint_every_generations == 0
                    or elapsed >= checkpoint_every_seconds
                    or mean is None
                ):
                    checkpoint_callback(snapshot(), elapsed)
                    last_checkpoint_time = time.monotonic()
                if mean is None:
                    break

        final_state = snapshot()
        return best_parameters, trace, final_state

    def _spline_operators(self, k: int, degree: int) -> tuple[np.ndarray, np.ndarray]:
        knots = open_uniform_knots(k, degree)
        basis = [
            BSpline(knots, np.eye(k)[j], degree, extrapolate=False) for j in range(k)
        ]
        basis_values = np.column_stack([item(self.grid) for item in basis])
        second_values = np.column_stack([item(self.grid, nu=2) for item in basis])
        roughness = np.trapz(
            second_values[:, :, None] * second_values[:, None, :], self.grid, axis=0
        )
        return basis_values, roughness

    @torch.no_grad()
    def _functional_objective(
        self,
        candidates: np.ndarray,
        states: np.ndarray,
        critic,
        basis_values: np.ndarray,
        roughness_operator: np.ndarray,
    ) -> np.ndarray:
        features = policy_state_features_batch(states)
        outputs: list[np.ndarray] = []
        for start in range(0, len(candidates), self.config.candidate_batch_size):
            block = candidates[start : start + self.config.candidate_batch_size]
            coefficients = 2.0 * np.tanh(np.einsum("nf,cfk->cnk", features, block))
            action_values = np.einsum("cnk,jk->cnj", coefficients, basis_values)
            repeated_states = np.broadcast_to(
                states[None, :, :], (len(block), len(states), 3)
            )
            values = critic(
                torch.as_tensor(
                    np.array(repeated_states.reshape(-1, 3), copy=True),
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    action_values.reshape(-1, len(self.grid)),
                    dtype=torch.float32,
                    device=self.device,
                ),
            ).reshape(len(block), len(states))
            q_mean = values.mean(dim=1).cpu().numpy()
            roughness = np.einsum(
                "cnk,kl,cnl->cn", coefficients, roughness_operator, coefficients
            ).mean(axis=1)
            outputs.append(q_mean - self.config.lambda_roughness * roughness)
        return np.concatenate(outputs)

    def optimize(
        self,
        critic,
        states: np.ndarray,
        initial: BoundedCoefficientBSplinePolicy,
        rng: np.random.Generator,
        *,
        resume_state: CMAESResumeState | None = None,
        checkpoint_callback: Callable[[CMAESResumeState, float], None] | None = None,
        checkpoint_every_generations: int = 5,
        checkpoint_every_seconds: float = 30.0,
    ) -> tuple[BoundedCoefficientBSplinePolicy, list[dict], CMAESResumeState]:
        basis_values, roughness_operator = self._spline_operators(
            initial.k, initial.degree
        )

        def objective(
            candidates: np.ndarray, objective_states: np.ndarray
        ) -> np.ndarray:
            return self._functional_objective(
                candidates,
                objective_states,
                critic,
                basis_values,
                roughness_operator,
            )

        parameters, trace, state = self._optimize_parameters(
            initial_parameters=initial.parameter_matrix,
            states=states,
            rng=rng,
            mode="bounded_coefficient_bspline",
            objective_function=objective,
            resume_state=resume_state,
            checkpoint_callback=checkpoint_callback,
            checkpoint_every_generations=checkpoint_every_generations,
            checkpoint_every_seconds=checkpoint_every_seconds,
        )
        return BoundedCoefficientBSplinePolicy(parameters, initial.degree), trace, state

    def optimize_constant(
        self,
        critic,
        states: np.ndarray,
        initial: BoundedConstantActionPolicy,
        rng: np.random.Generator,
        *,
        resume_state: CMAESResumeState | None = None,
        checkpoint_callback: Callable[[CMAESResumeState, float], None] | None = None,
        checkpoint_every_generations: int = 5,
        checkpoint_every_seconds: float = 30.0,
    ) -> tuple[BoundedConstantActionPolicy, list[dict], CMAESResumeState]:
        @torch.no_grad()
        def objective(
            candidates: np.ndarray, objective_states: np.ndarray
        ) -> np.ndarray:
            features = policy_state_features_batch(objective_states)
            outputs: list[np.ndarray] = []
            for start in range(0, len(candidates), self.config.candidate_batch_size):
                block = candidates[start : start + self.config.candidate_batch_size]
                scalars = 2.0 * np.tanh(np.einsum("nf,cf->cn", features, block))
                action_values = np.broadcast_to(
                    scalars[:, :, None],
                    (len(block), len(objective_states), len(self.grid)),
                )
                repeated_states = np.broadcast_to(
                    objective_states[None, :, :],
                    (len(block), len(objective_states), 3),
                )
                values = critic(
                    torch.as_tensor(
                        np.array(repeated_states.reshape(-1, 3), copy=True),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.as_tensor(
                        np.array(action_values.reshape(-1, len(self.grid)), copy=True),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                ).reshape(len(block), len(objective_states))
                outputs.append(values.mean(dim=1).cpu().numpy())
            return np.concatenate(outputs)

        parameters, trace, state = self._optimize_parameters(
            initial_parameters=initial.parameters,
            states=states,
            rng=rng,
            mode="bounded_constant_function",
            objective_function=objective,
            resume_state=resume_state,
            checkpoint_callback=checkpoint_callback,
            checkpoint_every_generations=checkpoint_every_generations,
            checkpoint_every_seconds=checkpoint_every_seconds,
        )
        return BoundedConstantActionPolicy(parameters), trace, state


__all__ = [
    "CMAESPolicyOptimizationConfig",
    "CMAESResumeState",
    "DerivativeFreeCMAES",
]
