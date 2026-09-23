from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch


@dataclass(frozen=True)
class ExactKRRConfig:
    state_lengthscales: tuple[float, float, float] = (1.0, 1.0, 2.0)
    action_l2_lengthscale: float = 0.75
    ridge: float = 1.0e-3
    jitter: float = 1.0e-10
    prediction_block_size: int = 512

    def __post_init__(self) -> None:
        if len(self.state_lengthscales) != 3 or any(
            value <= 0 for value in self.state_lengthscales
        ):
            raise ValueError("state_lengthscales must contain three positive values")
        if self.action_l2_lengthscale <= 0 or self.ridge <= 0 or self.jitter < 0:
            raise ValueError(
                "KRR lengthscale/ridge must be positive and jitter nonnegative"
            )
        if self.prediction_block_size <= 0:
            raise ValueError("prediction_block_size must be positive")


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("quadrature grid must be a strictly increasing vector")
    weights = np.empty(len(grid), dtype=np.float64)
    weights[0] = (grid[1] - grid[0]) / 2.0
    weights[-1] = (grid[-1] - grid[-2]) / 2.0
    weights[1:-1] = (grid[2:] - grid[:-2]) / 2.0
    return weights


def product_kernel_numpy(
    states_left: np.ndarray,
    actions_left: np.ndarray,
    states_right: np.ndarray,
    actions_right: np.ndarray,
    action_grid: np.ndarray,
    config: ExactKRRConfig,
) -> np.ndarray:
    states_left = np.asarray(states_left, dtype=np.float64)
    states_right = np.asarray(states_right, dtype=np.float64)
    actions_left = np.asarray(actions_left, dtype=np.float64)
    actions_right = np.asarray(actions_right, dtype=np.float64)
    if states_left.shape != (len(actions_left), 3) or states_right.shape != (
        len(actions_right),
        3,
    ):
        raise ValueError("state/action row counts are inconsistent")
    if (
        actions_left.ndim != 2
        or actions_right.ndim != 2
        or actions_left.shape[1] != actions_right.shape[1]
    ):
        raise ValueError(
            "action arrays require a shared two-dimensional grid representation"
        )
    weights = trapezoid_weights(action_grid)
    if len(weights) != actions_left.shape[1]:
        raise ValueError("action grid length mismatch")
    state_scale = np.asarray(config.state_lengthscales)
    state_delta = (states_left[:, None, :] - states_right[None, :, :]) / state_scale
    state_distance = np.sum(state_delta**2, axis=-1)
    weighted_left = actions_left * np.sqrt(weights)[None, :]
    weighted_right = actions_right * np.sqrt(weights)[None, :]
    left_norm = np.sum(weighted_left**2, axis=1)
    right_norm = np.sum(weighted_right**2, axis=1)
    action_distance = np.maximum(
        left_norm[:, None]
        + right_norm[None, :]
        - 2.0 * weighted_left @ weighted_right.T,
        0.0,
    )
    return np.exp(
        -0.5 * state_distance - 0.5 * action_distance / config.action_l2_lengthscale**2
    )


class ExactFunctionalKRR:
    """Exact product-kernel ridge regression; no low-rank/random-feature approximation."""

    def __init__(self, config: ExactKRRConfig | None = None, value_max: float = 20.0):
        self.config = config or ExactKRRConfig()
        self.value_max = float(value_max)
        self.training_states: np.ndarray | None = None
        self.training_actions: np.ndarray | None = None
        self.action_grid: np.ndarray | None = None
        self.alpha: torch.Tensor | None = None
        self.cholesky_factor: torch.Tensor | None = None
        self._training_states_tensor: torch.Tensor | None = None
        self._weighted_training_actions_tensor: torch.Tensor | None = None
        self._training_action_norms_tensor: torch.Tensor | None = None
        self._quadrature_weights_tensor: torch.Tensor | None = None
        self.device: torch.device | None = None
        self.fit_metadata: dict | None = None

    def fit_design(
        self,
        states: np.ndarray,
        action_values: np.ndarray,
        action_grid: np.ndarray,
        device: torch.device,
    ) -> "ExactFunctionalKRR":
        """Factor the fixed KRR design once for all fitted-Q target updates."""
        states = np.asarray(states, dtype=np.float64)
        action_values = np.asarray(action_values, dtype=np.float64)
        action_grid = np.asarray(action_grid, dtype=np.float64)
        if states.shape != (len(action_values), 3) or action_values.ndim != 2:
            raise ValueError("invalid exact-KRR design shapes")
        if not np.isfinite(states).all() or not np.isfinite(action_values).all():
            raise ValueError("exact-KRR design must be finite")
        weights = trapezoid_weights(action_grid)
        if len(weights) != action_values.shape[1]:
            raise ValueError("action grid length mismatch")

        started = time.time()
        state_tensor = torch.as_tensor(states, dtype=torch.float64, device=device)
        action_tensor = torch.as_tensor(
            action_values, dtype=torch.float64, device=device
        )
        weight_tensor = torch.as_tensor(weights, dtype=torch.float64, device=device)
        weighted_actions = action_tensor * torch.sqrt(weight_tensor)[None, :]
        action_norms = torch.sum(weighted_actions**2, dim=1)
        n = len(states)
        gram = torch.empty((n, n), dtype=torch.float64, device=device)
        block_size = int(self.config.prediction_block_size)
        state_scale = torch.as_tensor(
            self.config.state_lengthscales, dtype=torch.float64, device=device
        )
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            state_delta = (
                state_tensor[start:stop, None, :] - state_tensor[None, :, :]
            ) / state_scale
            state_distance = torch.sum(state_delta**2, dim=-1)
            action_distance = torch.clamp(
                action_norms[start:stop, None]
                + action_norms[None, :]
                - 2.0 * weighted_actions[start:stop] @ weighted_actions.T,
                min=0.0,
            )
            gram[start:stop] = torch.exp(
                -0.5 * state_distance
                - 0.5 * action_distance / self.config.action_l2_lengthscale**2
            )
        kernel_seconds = time.time() - started
        regularization = n * self.config.ridge + self.config.jitter
        gram.diagonal().add_(regularization)
        factor_started = time.time()
        factor = torch.linalg.cholesky(gram)
        factor_seconds = time.time() - factor_started

        self.training_states = states.copy()
        self.training_actions = action_values.copy()
        self.action_grid = action_grid.copy()
        self.device = device
        self.cholesky_factor = factor.detach()
        self._training_states_tensor = state_tensor.detach()
        self._weighted_training_actions_tensor = weighted_actions.detach()
        self._training_action_norms_tensor = action_norms.detach()
        self._quadrature_weights_tensor = weight_tensor.detach()
        self.fit_metadata = {
            "schema_version": 2,
            "solver": "exact_dense_cholesky",
            "approximation": "none",
            "n": n,
            "action_resolution": action_values.shape[1],
            "dtype": "float64",
            "device": str(device),
            "config": asdict(self.config),
            "regularization_diagonal": regularization,
            "cholesky_diagonal_min": float(torch.min(torch.diagonal(factor)).cpu()),
            "cholesky_diagonal_max": float(torch.max(torch.diagonal(factor)).cpu()),
            "kernel_wall_time_sec": kernel_seconds,
            "factor_wall_time_sec": factor_seconds,
            "factor_reuse": True,
            "target_solve_count": 0,
        }
        return self

    def solve_targets(self, targets: np.ndarray) -> "ExactFunctionalKRR":
        """Update only the KRR coefficients while retaining the fixed factor."""
        if (
            self.cholesky_factor is None
            or self.device is None
            or self.fit_metadata is None
        ):
            raise RuntimeError("exact KRR design is not factored")
        targets = np.array(targets, dtype=np.float64, copy=True)
        if (
            targets.shape != (int(self.fit_metadata["n"]),)
            or not np.isfinite(targets).all()
        ):
            raise ValueError("invalid exact-KRR targets")
        target_tensor = torch.as_tensor(
            targets, dtype=torch.float64, device=self.device
        )
        started = time.time()
        self.alpha = (
            torch.cholesky_solve(target_tensor[:, None], self.cholesky_factor)
            .squeeze(1)
            .detach()
        )
        solve_seconds = time.time() - started
        regularization = float(self.fit_metadata["regularization_diagonal"])
        # Compute the residual without retaining the dense Gram matrix.
        fitted = self.predict_torch(
            self._training_states_tensor,
            torch.as_tensor(
                self.training_actions, dtype=torch.float64, device=self.device
            ),
            clip=False,
        )
        residual = fitted + regularization * self.alpha - target_tensor
        relative_residual = float(
            torch.linalg.vector_norm(residual).cpu()
            / torch.clamp(
                torch.linalg.vector_norm(target_tensor).cpu(),
                min=torch.finfo(torch.float64).eps,
            )
        )
        self.fit_metadata["target_solve_count"] = (
            int(self.fit_metadata["target_solve_count"]) + 1
        )
        self.fit_metadata["latest_target_solve_wall_time_sec"] = solve_seconds
        self.fit_metadata["latest_relative_linear_solve_residual"] = relative_residual
        # Backward-compatible public diagnostic name used by the original
        # exact-KRR tests and downstream raw-output schema.
        self.fit_metadata["relative_linear_solve_residual"] = relative_residual
        return self

    def fit(
        self,
        states: np.ndarray,
        action_values: np.ndarray,
        action_grid: np.ndarray,
        targets: np.ndarray,
        device: torch.device,
    ) -> "ExactFunctionalKRR":
        targets = np.asarray(targets, dtype=np.float64)
        if len(targets) != len(states):
            raise ValueError("invalid exact-KRR training shapes")
        return self.fit_design(
            states, action_values, action_grid, device
        ).solve_targets(targets)

    def predict_torch(
        self,
        states: torch.Tensor,
        action_values: torch.Tensor,
        *,
        clip: bool = True,
    ) -> torch.Tensor:
        """Differentiable raw-function KRR prediction used by coefficient Adam."""
        if (
            self.alpha is None
            or self._training_states_tensor is None
            or self._weighted_training_actions_tensor is None
            or self._training_action_norms_tensor is None
            or self._quadrature_weights_tensor is None
            or self.device is None
        ):
            raise RuntimeError("exact KRR is not fitted")
        states = states.to(device=self.device, dtype=torch.float64)
        action_values = action_values.to(device=self.device, dtype=torch.float64)
        if (
            states.ndim != 2
            or states.shape[1] != 3
            or action_values.shape
            != (
                len(states),
                len(self._quadrature_weights_tensor),
            )
        ):
            raise ValueError("invalid exact-KRR prediction shapes")
        scale = torch.as_tensor(
            self.config.state_lengthscales, dtype=torch.float64, device=self.device
        )
        outputs = []
        for start in range(0, len(states), self.config.prediction_block_size):
            stop = min(len(states), start + self.config.prediction_block_size)
            weighted = (
                action_values[start:stop]
                * torch.sqrt(self._quadrature_weights_tensor)[None, :]
            )
            norms = torch.sum(weighted**2, dim=1)
            state_delta = (
                states[start:stop, None, :] - self._training_states_tensor[None, :, :]
            ) / scale
            state_distance = torch.sum(state_delta**2, dim=-1)
            action_distance = torch.clamp(
                norms[:, None]
                + self._training_action_norms_tensor[None, :]
                - 2.0 * weighted @ self._weighted_training_actions_tensor.T,
                min=0.0,
            )
            kernel = torch.exp(
                -0.5 * state_distance
                - 0.5 * action_distance / self.config.action_l2_lengthscale**2
            )
            outputs.append(kernel @ self.alpha)
        result = torch.cat(outputs)
        return torch.clamp(result, 0.0, self.value_max) if clip else result

    def predict(self, states: np.ndarray, action_values: np.ndarray) -> np.ndarray:
        if (
            self.alpha is None
            or self.training_states is None
            or self.training_actions is None
            or self.action_grid is None
            or self.device is None
        ):
            raise RuntimeError("exact KRR is not fitted")
        states = np.asarray(states, dtype=np.float64)
        action_values = np.asarray(action_values, dtype=np.float64)
        with torch.no_grad():
            output = self.predict_torch(
                torch.as_tensor(states, dtype=torch.float64, device=self.device),
                torch.as_tensor(action_values, dtype=torch.float64, device=self.device),
            )
        return output.cpu().numpy()
