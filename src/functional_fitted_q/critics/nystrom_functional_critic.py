from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from .krr_functional_critic import trapezoid_weights


@dataclass(frozen=True)
class NystromKRRConfig:
    state_lengthscales: tuple[float, float, float] = (1.0, 1.0, 2.0)
    action_l2_lengthscale: float = 0.75
    ridge: float | None = 1.0e-3
    jitter: float = 1.0e-10
    landmark_rank: int = 1024
    eigenvalue_relative_tolerance: float = 1.0e-12
    prediction_block_size: int = 512

    def __post_init__(self) -> None:
        if len(self.state_lengthscales) != 3 or any(
            value <= 0 for value in self.state_lengthscales
        ):
            raise ValueError("three positive state lengthscales are required")
        if (
            self.action_l2_lengthscale <= 0
            or (
                self.ridge is not None
                and (not np.isfinite(self.ridge) or self.ridge <= 0)
            )
            or self.jitter < 0
            or self.landmark_rank <= 0
            or self.eigenvalue_relative_tolerance <= 0
            or self.prediction_block_size <= 0
        ):
            raise ValueError("positive Nyström KRR settings are required")


class NystromFunctionalKRR:
    """Deterministic landmark Nyström product-kernel ridge regression.

    The critic uses the same state/RBF-functional-action product kernel as the
    exact KRR implementation. It is explicitly approximate whenever the
    retained landmark count is smaller than the training sample size.
    """

    solver_identity = "deterministic_landmark_nystrom_primal_cholesky"

    def __init__(self, config: NystromKRRConfig, value_max: float = 20.0):
        self.config = config
        self.value_max = float(value_max)
        self.action_grid: np.ndarray | None = None
        self.alpha: torch.Tensor | None = None
        self.device: torch.device | None = None
        self.fit_metadata: dict | None = None
        self.landmark_indices: np.ndarray | None = None
        self._landmark_states: torch.Tensor | None = None
        self._weighted_landmark_actions: torch.Tensor | None = None
        self._landmark_action_norms: torch.Tensor | None = None
        self._quadrature_weights: torch.Tensor | None = None
        self._whitener: torch.Tensor | None = None
        self._design_features: torch.Tensor | None = None
        self._normal_factor: torch.Tensor | None = None

    def _kernel_to_landmarks(
        self, states: torch.Tensor, action_values: torch.Tensor
    ) -> torch.Tensor:
        if (
            self._landmark_states is None
            or self._weighted_landmark_actions is None
            or self._landmark_action_norms is None
            or self._quadrature_weights is None
            or self.device is None
        ):
            raise RuntimeError("Nyström landmarks are not fitted")
        states = states.to(device=self.device, dtype=torch.float64)
        action_values = action_values.to(device=self.device, dtype=torch.float64)
        scale = torch.as_tensor(
            self.config.state_lengthscales, dtype=torch.float64, device=self.device
        )
        weighted = action_values * torch.sqrt(self._quadrature_weights)[None, :]
        norms = torch.sum(weighted**2, dim=1)
        state_distance = torch.sum(
            ((states[:, None, :] - self._landmark_states[None, :, :]) / scale) ** 2,
            dim=-1,
        )
        action_distance = torch.clamp(
            norms[:, None]
            + self._landmark_action_norms[None, :]
            - 2.0 * weighted @ self._weighted_landmark_actions.T,
            min=0.0,
        )
        return torch.exp(
            -0.5 * state_distance
            - 0.5 * action_distance / self.config.action_l2_lengthscale**2
        )

    def _features(
        self, states: torch.Tensor, action_values: torch.Tensor
    ) -> torch.Tensor:
        if self._whitener is None:
            raise RuntimeError("Nyström whitener is not fitted")
        return self._kernel_to_landmarks(states, action_values) @ self._whitener

    def fit_design(
        self,
        states: np.ndarray,
        action_values: np.ndarray,
        action_grid: np.ndarray,
        device: torch.device,
        *,
        landmark_indices: np.ndarray,
    ) -> "NystromFunctionalKRR":
        """Fit the deterministic Nyström feature map for a fixed offline design."""
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(action_values, dtype=np.float64)
        grid = np.asarray(action_grid, dtype=np.float64)
        indices = np.asarray(landmark_indices, dtype=np.int64)
        if states.shape != (len(actions), 3) or actions.ndim != 2:
            raise ValueError("invalid Nyström design shapes")
        if len(indices) != min(self.config.landmark_rank, len(states)):
            raise ValueError("landmark index count does not match the configured rank")
        if (
            len(np.unique(indices)) != len(indices)
            or np.any(indices < 0)
            or np.any(indices >= len(states))
        ):
            raise ValueError("landmark indices must be unique valid training rows")
        weights = trapezoid_weights(grid)
        if len(weights) != actions.shape[1]:
            raise ValueError("action grid length mismatch")
        self.device = device
        self.action_grid = grid.copy()
        self.landmark_indices = indices.copy()
        state_tensor = torch.as_tensor(states, dtype=torch.float64, device=device)
        action_tensor = torch.as_tensor(actions, dtype=torch.float64, device=device)
        self._quadrature_weights = torch.as_tensor(
            weights, dtype=torch.float64, device=device
        )
        self._landmark_states = state_tensor[indices].detach()
        weighted_landmarks = (
            action_tensor[indices] * torch.sqrt(self._quadrature_weights)[None, :]
        )
        self._weighted_landmark_actions = weighted_landmarks.detach()
        self._landmark_action_norms = torch.sum(weighted_landmarks**2, dim=1).detach()
        started = time.time()
        landmark_kernel = self._kernel_to_landmarks(
            self._landmark_states, action_tensor[indices]
        )
        landmark_kernel = (landmark_kernel + landmark_kernel.T) / 2.0
        eigenvalues, eigenvectors = torch.linalg.eigh(landmark_kernel)
        threshold = self.config.eigenvalue_relative_tolerance * torch.max(eigenvalues)
        keep = eigenvalues > threshold
        if not torch.any(keep):
            raise RuntimeError("Nyström landmark kernel has zero retained rank")
        retained_values = eigenvalues[keep]
        retained_vectors = eigenvectors[:, keep]
        self._whitener = (
            retained_vectors / torch.sqrt(retained_values + self.config.jitter)[None, :]
        ).detach()
        landmark_seconds = time.time() - started
        feature_blocks = []
        feature_started = time.time()
        for start in range(0, len(states), self.config.prediction_block_size):
            stop = min(len(states), start + self.config.prediction_block_size)
            feature_blocks.append(
                self._features(state_tensor[start:stop], action_tensor[start:stop])
            )
        self._design_features = torch.cat(feature_blocks, dim=0).detach()
        feature_seconds = time.time() - feature_started
        solver_metadata = self._prepare_target_solver()
        self.fit_metadata = {
            "schema_version": 1,
            "solver": self.solver_identity,
            "approximation": (
                "nystrom" if len(indices) < len(states) else "full_landmark_feature"
            ),
            "n": int(len(states)),
            "requested_landmark_rank": int(self.config.landmark_rank),
            "landmark_count": int(len(indices)),
            "retained_feature_rank": int(torch.sum(keep).cpu()),
            "landmark_indices": indices.tolist(),
            "dtype": "float64",
            "device": str(device),
            "config": asdict(self.config),
            "landmark_eigendecomposition_wall_time_sec": landmark_seconds,
            "design_feature_wall_time_sec": feature_seconds,
            "target_solve_count": 0,
            **solver_metadata,
        }
        return self

    def _prepare_target_solver(self) -> dict:
        if self.config.ridge is None:
            raise ValueError(
                "fixed-ridge KRR requires a numeric ridge; use the GCV critic for ridge=None"
            )
        normal = self._design_features.T @ self._design_features
        regularization = len(self._design_features) * self.config.ridge
        normal.diagonal().add_(regularization)
        started = time.time()
        self._normal_factor = torch.linalg.cholesky(normal).detach()
        return {
            "regularization_diagonal": regularization,
            "normal_factor_wall_time_sec": time.time() - started,
            "factor_reuse": True,
        }

    def solve_targets(self, targets: np.ndarray) -> "NystromFunctionalKRR":
        if (
            self._design_features is None
            or self._normal_factor is None
            or self.device is None
            or self.fit_metadata is None
        ):
            raise RuntimeError("Nyström design is not fitted")
        target = torch.as_tensor(
            np.array(targets, dtype=np.float64, copy=True),
            dtype=torch.float64,
            device=self.device,
        )
        if (
            target.shape != (self._design_features.shape[0],)
            or not torch.isfinite(target).all()
        ):
            raise ValueError("invalid Nyström targets")
        right = self._design_features.T @ target
        started = time.time()
        self.alpha = (
            torch.cholesky_solve(right[:, None], self._normal_factor)
            .squeeze(1)
            .detach()
        )
        solve_seconds = time.time() - started
        normal_residual = (
            self._normal_factor @ self._normal_factor.T @ self.alpha - right
        )
        relative_residual = float(
            torch.linalg.vector_norm(normal_residual).cpu()
            / torch.clamp(
                torch.linalg.vector_norm(right).cpu(),
                min=torch.finfo(torch.float64).eps,
            )
        )
        self.fit_metadata["target_solve_count"] = (
            int(self.fit_metadata["target_solve_count"]) + 1
        )
        self.fit_metadata["latest_target_solve_wall_time_sec"] = solve_seconds
        self.fit_metadata["latest_relative_linear_solve_residual"] = relative_residual
        self.fit_metadata["relative_linear_solve_residual"] = relative_residual
        return self

    def predict_torch(
        self, states: torch.Tensor, action_values: torch.Tensor, *, clip: bool = True
    ) -> torch.Tensor:
        if self.alpha is None or self.device is None:
            raise RuntimeError("Nyström KRR coefficients are not fitted")
        states = states.to(device=self.device, dtype=torch.float64)
        actions = action_values.to(device=self.device, dtype=torch.float64)
        outputs = []
        for start in range(0, len(states), self.config.prediction_block_size):
            stop = min(len(states), start + self.config.prediction_block_size)
            outputs.append(
                self._features(states[start:stop], actions[start:stop]) @ self.alpha
            )
        values = torch.cat(outputs)
        return torch.clamp(values, 0.0, self.value_max) if clip else values

    def __call__(
        self, states: torch.Tensor, action_values: torch.Tensor
    ) -> torch.Tensor:
        """Match the critic interface consumed by the shared CMA-ES optimizer."""
        return self.predict_torch(states, action_values)

    def predict(self, states: np.ndarray, action_values: np.ndarray) -> np.ndarray:
        if self.device is None:
            raise RuntimeError("Nyström KRR is not fitted")
        with torch.no_grad():
            values = self.predict_torch(
                torch.as_tensor(states, dtype=torch.float64, device=self.device),
                torch.as_tensor(action_values, dtype=torch.float64, device=self.device),
            )
        return values.detach().cpu().numpy()


def deterministic_landmark_indices(n: int, rank: int, seed: int) -> np.ndarray:
    if n <= 0 or rank <= 0:
        raise ValueError("positive n and landmark rank are required")
    count = min(int(n), int(rank))
    return np.random.default_rng(int(seed)).permutation(int(n))[:count].astype(np.int64)


__all__ = [
    "NystromFunctionalKRR",
    "NystromKRRConfig",
    "deterministic_landmark_indices",
]
