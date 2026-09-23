"""Configuration and objective helpers for theory-aligned KRR policy penalty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from functional_fitted_q.runners.krr_prediction import BoundedAdamConfig
from functional_fitted_q.algorithms.theory_total_curvature import (
    PENALTY_CONVENTION,
    torch_uncentered_total_curvature,
    total_curvature_matrix,
)


@dataclass(frozen=True)
class TheoryTotalCurvatureAdamConfig(BoundedAdamConfig):
    policy_lambda: float = 0.0001
    penalty_convention: str = PENALTY_CONVENTION
    penalty_state_scope: str = "all_training_next_states"

    def __post_init__(self) -> None:
        if not np.isfinite(self.policy_lambda) or self.policy_lambda <= 0:
            raise ValueError(
                "theory total-curvature lambda must be finite and positive"
            )
        if self.penalty_convention != PENALTY_CONVENTION:
            raise ValueError("theory total-curvature convention changed")
        if self.penalty_state_scope != "all_training_next_states":
            raise ValueError("theory penalty must average all training next states")


def penalty_matrix(policy, grid, device) -> tuple[torch.Tensor, dict]:
    matrix, receipt = total_curvature_matrix(policy, grid)
    return torch.as_tensor(matrix, dtype=torch.float64, device=device), receipt


def penalized_score(raw, penalty_coefficients, value_max, policy_lambda, matrix):
    q_mean = raw.clamp(0.0, value_max).mean(dim=-1)
    total_curvature = torch_uncentered_total_curvature(penalty_coefficients, matrix)
    return q_mean - policy_lambda * total_curvature
