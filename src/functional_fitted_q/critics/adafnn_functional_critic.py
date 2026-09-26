"""Adaptive functional scores followed by a state/action regression network.

The learned basis networks approximate beta_j(u), and action_scores computes
quadrature approximations of integral a(u)*beta_j(u) du. forward_unclipped
is the raw fit; forward is the clipped critic deployed in FQI and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import time
from typing import Callable

import numpy as np
import torch
from torch import nn


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.diff(grid) > 0):
        raise ValueError("quadrature grid must be a strictly increasing vector")
    weights = np.empty_like(grid)
    weights[0] = (grid[1] - grid[0]) / 2.0
    weights[-1] = (grid[-1] - grid[-2]) / 2.0
    if len(grid) > 2:
        weights[1:-1] = (grid[2:] - grid[:-2]) / 2.0
    return weights


def _relu_mlp(dimensions: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
        layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU()])
    layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
    return nn.Sequential(*layers)


class AdaFNNFunctionalCritic(nn.Module):
    """Adaptive Basis Layer critic for continuous function-valued actions.

    Each learned global basis direction is represented by its own micro neural
    network of ``u``. Raw functional action values enter only through a fixed
    deterministic quadrature approximation of their L2 inner products with
    those learned bases. State dependence is handled by the outer critic.
    """

    def __init__(
        self,
        quadrature_grid: np.ndarray,
        n_basis_nodes: int = 8,
        micro_hidden_layers: tuple[int, ...] = (32, 32),
        outer_hidden_layers: tuple[int, ...] = (256, 256, 256),
        value_max: float = 20.0,
        parameter_bound: float = 5.0,
    ) -> None:
        super().__init__()
        if n_basis_nodes <= 0:
            raise ValueError("n_basis_nodes must be positive")
        if not micro_hidden_layers or any(width <= 0 for width in micro_hidden_layers):
            raise ValueError("micro_hidden_layers must contain positive widths")
        if not outer_hidden_layers or any(width <= 0 for width in outer_hidden_layers):
            raise ValueError("outer_hidden_layers must contain positive widths")
        if value_max <= 0 or parameter_bound <= 0:
            raise ValueError("value_max and parameter_bound must be positive")
        grid = np.asarray(quadrature_grid, dtype=np.float64)
        weights = trapezoid_weights(grid)
        self.register_buffer(
            "quadrature_grid", torch.as_tensor(grid, dtype=torch.float32)
        )
        self.register_buffer(
            "quadrature_weights", torch.as_tensor(weights, dtype=torch.float32)
        )
        self.basis_micro_networks = nn.ModuleList(
            [_relu_mlp([1, *micro_hidden_layers, 1]) for _ in range(n_basis_nodes)]
        )
        self.outer_network = _relu_mlp([3 + n_basis_nodes, *outer_hidden_layers, 1])
        self.n_basis_nodes = int(n_basis_nodes)
        self.value_max = float(value_max)
        self.parameter_bound = float(parameter_bound)
        self.project_parameters_()

    def project_parameters_(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.clamp_(-self.parameter_bound, self.parameter_bound)

    def basis_values(self) -> torch.Tensor:
        u = self.quadrature_grid[:, None]
        return torch.column_stack(
            [network(u).squeeze(-1) for network in self.basis_micro_networks]
        )

    def action_scores(self, action_values: torch.Tensor) -> torch.Tensor:
        """Map (..., grid_points) action samples to (..., learned_basis_count)."""
        if action_values.ndim < 2 or action_values.shape[-1] != len(
            self.quadrature_grid
        ):
            raise ValueError(
                "action_values last dimension must equal quadrature resolution"
            )
        # Quadrature weights retain the functional L2 inner product; an
        # unweighted dot product would change scale with grid resolution.
        weighted_basis = self.quadrature_weights[:, None] * self.basis_values()
        return action_values @ weighted_basis

    def forward_unclipped(
        self, states: torch.Tensor, action_values: torch.Tensor
    ) -> torch.Tensor:
        if states.shape[:-1] != action_values.shape[:-1] or states.shape[-1] != 3:
            raise ValueError("states and action_values batch shapes are inconsistent")
        scores = self.action_scores(action_values)
        return self.outer_network(torch.cat([states, scores], dim=-1)).squeeze(-1)

    def forward(
        self, states: torch.Tensor, action_values: torch.Tensor
    ) -> torch.Tensor:
        return torch.clamp(
            self.forward_unclipped(states, action_values), 0.0, self.value_max
        )

    @torch.no_grad()
    def basis_diagnostics(self) -> dict[str, np.ndarray | float]:
        basis = self.basis_values()
        weighted = basis * torch.sqrt(self.quadrature_weights[:, None])
        gram = weighted.T @ weighted
        # This is a reporting-only diagnostic, so do the spectral calculation
        # on the CPU in float64.  CUDA's float32 symmetric eigensolver can fail
        # for an otherwise valid learned basis when the Gram matrix has nearly
        # repeated eigenvalues.  If LAPACK's symmetric eigensolver also rejects
        # the matrix, squared singular values of the weighted basis are the
        # same nonnegative Gram eigenvalues and provide a stable fallback.
        weighted64 = weighted.detach().to(device="cpu", dtype=torch.float64)
        gram64 = weighted64.T @ weighted64
        gram64 = 0.5 * (gram64 + gram64.T)
        if not torch.isfinite(gram64).all():
            raise RuntimeError("non-finite learned-basis Gram diagnostic")
        try:
            eigvals64 = torch.linalg.eigvalsh(gram64)
        except RuntimeError:
            singular_values = torch.linalg.svdvals(weighted64)
            eigvals64 = torch.sort(singular_values.square()).values
            if eigvals64.numel() < self.n_basis_nodes:
                eigvals64 = torch.cat(
                    [
                        torch.zeros(
                            self.n_basis_nodes - eigvals64.numel(),
                            dtype=eigvals64.dtype,
                            device=eigvals64.device,
                        ),
                        eigvals64,
                    ]
                )
        eigvals = eigvals64.to(dtype=gram.dtype)
        return {
            "basis_values": basis.cpu().numpy(),
            "basis_gram_matrix": gram.cpu().numpy(),
            "basis_l2_norms": torch.sqrt(torch.clamp(torch.diag(gram), min=0.0))
            .cpu()
            .numpy(),
            "basis_gram_eigenvalues": eigvals.cpu().numpy(),
            "parameter_abs_max": max(
                float(parameter.detach().abs().max().cpu())
                for parameter in self.parameters()
            ),
        }


@dataclass(frozen=True)
class AdaFNNCriticTrainingConfig:
    n_basis_nodes: int = 8
    micro_hidden_layers: tuple[int, ...] = (32, 32)
    outer_hidden_layers: tuple[int, ...] = (256, 256, 256)
    learning_rate: float = 1.0e-3
    batch_size: int = 256
    max_epochs: int = 300
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 1.0e-5
    weight_decay: float = 1.0e-5
    gradient_clip: float = 10.0
    parameter_bound: float = 5.0
    require_early_stopping_convergence: bool = False

    def __post_init__(self) -> None:
        positive = {
            "n_basis_nodes": self.n_basis_nodes,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
        }
        if any(int(value) <= 0 for value in positive.values()):
            raise ValueError(f"positive integer critic settings required: {positive}")
        if (
            self.learning_rate <= 0
            or self.parameter_bound <= 0
            or self.gradient_clip <= 0
        ):
            raise ValueError(
                "learning rate, parameter bound, and gradient clip must be positive"
            )
        if self.early_stopping_min_delta < 0 or self.weight_decay < 0:
            raise ValueError("nonnegative training settings required")


CONVERGENCE_HARD_CAP_MULTIPLIER = 4


def critic_epoch_limit(config: AdaFNNCriticTrainingConfig) -> int:
    """Return the epoch hard cap used when early stopping is mandatory.

    ``max_epochs`` remains the nominal training budget.  A critic that has not
    yet met the frozen early-stopping gate may continue deterministically up to
    this derived cap instead of being accepted unconverged or failing exactly
    at the nominal budget.
    """

    multiplier = (
        CONVERGENCE_HARD_CAP_MULTIPLIER
        if config.require_early_stopping_convergence
        else 1
    )
    return int(config.max_epochs) * multiplier


@dataclass
class AdaFNNCriticResumeState:
    completed_epochs: int
    completed_updates: int
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict
    generator_state: torch.Tensor
    best_model_state: dict[str, torch.Tensor]
    best_epoch_loss: float
    no_improvement_epochs: int
    current_epoch_permutation: torch.Tensor
    current_cursor: int
    current_epoch_sums: dict[str, float]
    current_gradient_norm: float


def fit_adafnn_critic(
    critic: AdaFNNFunctionalCritic,
    states: np.ndarray,
    action_values: np.ndarray,
    targets: np.ndarray,
    config: AdaFNNCriticTrainingConfig,
    device: torch.device,
    minibatch_generator: torch.Generator,
    *,
    resume_state: AdaFNNCriticResumeState | None = None,
    initial_trace: list[dict] | None = None,
    checkpoint_callback: (
        Callable[
            [AdaFNNFunctionalCritic, AdaFNNCriticResumeState, list[dict], float], None
        ]
        | None
    ) = None,
    checkpoint_every_updates: int = 500,
    checkpoint_every_seconds: float = 30.0,
    restore_best: bool = True,
    convergence_epoch_limit_override: int | None = None,
) -> tuple[AdaFNNFunctionalCritic, list[dict], AdaFNNCriticResumeState]:
    """Fit one AdaFNN Bellman regression with deterministic resume support.

    The function minimizes mean squared Bellman error.  It records enough
    optimizer, minibatch, and early-stopping state to resume at the next
    minibatch without changing the resulting training trajectory.
    """
    if checkpoint_every_updates <= 0 or checkpoint_every_seconds <= 0:
        raise ValueError("critic checkpoint intervals must be positive")
    states = np.asarray(states, dtype=np.float32)
    action_values = np.asarray(action_values, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 3 or len(states) == 0:
        raise ValueError("states must have nonempty shape (n,3)")
    if action_values.shape != (len(states), len(critic.quadrature_grid)):
        raise ValueError("action_values do not match states/quadrature grid")
    if targets.shape != (len(states),) or not np.isfinite(targets).all():
        raise ValueError("targets must be finite with shape (n,)")

    model = copy.deepcopy(critic).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    trace = [] if initial_trace is None else list(initial_trace)

    if resume_state is None:
        completed_epochs = 0
        completed_updates = 0
        best_model_state = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
        best_epoch_loss = float("inf")
        no_improvement_epochs = 0
        current_epoch_permutation = torch.empty(0, dtype=torch.int64)
        current_cursor = 0
        current_epoch_sums = {
            "bellman": 0.0,
            "count": 0.0,
        }
        current_gradient_norm = float("nan")
    else:
        completed_epochs = int(resume_state.completed_epochs)
        completed_updates = int(resume_state.completed_updates)
        model.load_state_dict(resume_state.model_state)
        optimizer.load_state_dict(resume_state.optimizer_state)
        minibatch_generator.set_state(resume_state.generator_state)
        best_model_state = copy.deepcopy(resume_state.best_model_state)
        best_epoch_loss = float(resume_state.best_epoch_loss)
        no_improvement_epochs = int(resume_state.no_improvement_epochs)
        current_epoch_permutation = resume_state.current_epoch_permutation.clone()
        current_cursor = int(resume_state.current_cursor)
        current_epoch_sums = dict(resume_state.current_epoch_sums)
        current_gradient_norm = float(resume_state.current_gradient_norm)
    base_epoch_limit = critic_epoch_limit(config)
    epoch_limit = (
        base_epoch_limit
        if convergence_epoch_limit_override is None
        else int(convergence_epoch_limit_override)
    )
    if epoch_limit < base_epoch_limit:
        raise ValueError(
            "convergence epoch-limit override cannot reduce the frozen base cap"
        )
    if not 0 <= completed_epochs <= epoch_limit:
        raise ValueError("resume epoch is outside the configured budget")

    state_tensor = torch.from_numpy(states)
    action_tensor = torch.from_numpy(action_values)
    target_tensor = torch.from_numpy(targets)
    last_checkpoint_time = time.monotonic()

    def snapshot() -> AdaFNNCriticResumeState:
        return AdaFNNCriticResumeState(
            completed_epochs=completed_epochs,
            completed_updates=completed_updates,
            model_state={
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            optimizer_state=copy.deepcopy(optimizer.state_dict()),
            generator_state=minibatch_generator.get_state().clone(),
            best_model_state=copy.deepcopy(best_model_state),
            best_epoch_loss=float(best_epoch_loss),
            no_improvement_epochs=no_improvement_epochs,
            current_epoch_permutation=current_epoch_permutation.clone(),
            current_cursor=current_cursor,
            current_epoch_sums=dict(current_epoch_sums),
            current_gradient_norm=current_gradient_norm,
        )

    while completed_epochs < epoch_limit:
        if len(current_epoch_permutation) == 0:
            current_epoch_permutation = torch.randperm(
                len(states), generator=minibatch_generator
            )
            current_cursor = 0
            current_epoch_sums = {
                "bellman": 0.0,
                "count": 0.0,
            }
            current_gradient_norm = float("nan")
        while current_cursor < len(states):
            ids = current_epoch_permutation[
                current_cursor : current_cursor + config.batch_size
            ]
            current_cursor += len(ids)
            state_batch = state_tensor[ids].to(device)
            action_batch = action_tensor[ids].to(device)
            target_batch = target_tensor[ids].to(device)
            prediction = model(state_batch, action_batch)
            bellman_loss = torch.mean((prediction - target_batch) ** 2)
            optimizer.zero_grad(set_to_none=True)
            bellman_loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip
            )
            optimizer.step()
            model.project_parameters_()
            completed_updates += 1
            batch_count = len(ids)
            current_epoch_sums["bellman"] += (
                float(bellman_loss.detach().cpu()) * batch_count
            )
            current_epoch_sums["count"] += float(batch_count)
            current_gradient_norm = float(gradient_norm.detach().cpu())
            elapsed = time.monotonic() - last_checkpoint_time
            if checkpoint_callback is not None and (
                completed_updates % checkpoint_every_updates == 0
                or elapsed >= checkpoint_every_seconds
            ):
                checkpoint_callback(model, snapshot(), trace, elapsed)
                last_checkpoint_time = time.monotonic()

        completed_epochs += 1
        epoch_loss = current_epoch_sums["bellman"] / current_epoch_sums["count"]
        improved = epoch_loss < best_epoch_loss - config.early_stopping_min_delta
        if improved:
            best_epoch_loss = epoch_loss
            best_model_state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
        trace.append(
            {
                "epoch": completed_epochs,
                "updates": completed_updates,
                "bellman_loss": current_epoch_sums["bellman"]
                / current_epoch_sums["count"],
                "total_loss": epoch_loss,
                "best_total_loss": best_epoch_loss,
                "gradient_norm_last_batch": current_gradient_norm,
                "no_improvement_epochs": no_improvement_epochs,
                "termination_reason": (
                    "early_stopping"
                    if no_improvement_epochs >= config.early_stopping_patience
                    else (
                        "convergence_hard_cap"
                        if completed_epochs == epoch_limit
                        and config.require_early_stopping_convergence
                        else (
                            "max_epochs"
                            if completed_epochs == epoch_limit
                            else "running"
                        )
                    )
                ),
            }
        )
        current_epoch_permutation = torch.empty(0, dtype=torch.int64)
        current_cursor = 0
        current_epoch_sums = {
            "bellman": 0.0,
            "count": 0.0,
        }
        current_gradient_norm = float("nan")
        elapsed = time.monotonic() - last_checkpoint_time
        if checkpoint_callback is not None:
            checkpoint_callback(model, snapshot(), trace, elapsed)
            last_checkpoint_time = time.monotonic()
        if no_improvement_epochs >= config.early_stopping_patience:
            break

    if restore_best:
        model.load_state_dict(best_model_state)
    if (
        config.require_early_stopping_convergence
        and trace
        and trace[-1]["termination_reason"] != "early_stopping"
    ):
        raise RuntimeError(
            "AdaFNN critic reached the convergence hard cap before the frozen "
            "early-stopping gate"
        )
    final_state = snapshot()
    return model.eval(), trace, final_state


__all__ = [
    "AdaFNNFunctionalCritic",
    "AdaFNNCriticTrainingConfig",
    "AdaFNNCriticResumeState",
    "CONVERGENCE_HARD_CAP_MULTIPLIER",
    "critic_epoch_limit",
    "fit_adafnn_critic",
    "trapezoid_weights",
]
