"""Blocked critic prediction used by the KRR policy optimizer."""

import numpy as np
import torch
from dataclasses import dataclass
from functional_fitted_q.policies.bspline_policy import (
    BoundedCoefficientBSplinePolicy,
    policy_state_features_batch,
)


class FastPredictionView:
    """Algebraically identical Nyström prediction, without query-wise rank² work.

    beta is recomputed from the current alpha on every call, so no stale cache is
    possible after a solve, checkpoint restore, or in-place coefficient update.
    """

    def __init__(self, critic, bias=0.0):
        self.critic = critic
        self.bias = bias
        self.device = critic.device
        self.action_grid = critic.action_grid
        self.value_max = critic.value_max

    def predict_torch(self, states, actions, *, clip=True):
        c = self.critic
        beta = c._whitener @ c.alpha
        values = []
        for start in range(0, len(states), c.config.prediction_block_size):
            stop = min(len(states), start + c.config.prediction_block_size)
            values.append(
                c._kernel_to_landmarks(states[start:stop], actions[start:stop]) @ beta
                + self.bias
            )
        q = torch.cat(values)
        return q.clamp(0.0, self.value_max) if clip else q

    def predict(self, states, actions):
        with torch.no_grad():
            return (
                self.predict_torch(
                    torch.as_tensor(states, device=self.device),
                    torch.as_tensor(actions, device=self.device),
                )
                .cpu()
                .numpy()
            )

    def __call__(self, states, actions):
        return self.predict_torch(states, actions)


class FixedStateSplineObjective:
    """Exact functional L2 geometry in spline coordinates for frozen critic/states.

    Created afresh for each policy optimization. No cache crosses a critic solve.
    The kernel, policy and clipped objective are unchanged.
    """

    def __init__(self, view, states, basis):
        c = view.critic
        self.view, self.states, self.basis = view, states, basis
        self.beta = (c._whitener @ c.alpha).detach().clone()
        self.bias = torch.as_tensor(view.bias, device=c.device).detach().clone()
        scale = torch.as_tensor(c.config.state_lengthscales, device=c.device)
        self.state_kernel = torch.exp(
            -0.5
            * (
                ((states[:, None, :] - c._landmark_states[None, :, :]) / scale) ** 2
            ).sum(2)
        )
        weighted_basis = basis * c._quadrature_weights.sqrt()[:, None]
        self.gram = weighted_basis.T @ weighted_basis
        self.cross = weighted_basis.T @ c._weighted_landmark_actions.T
        self.norms = c._landmark_action_norms
        self.lengthscale = c.config.action_l2_lengthscale

    def raw(self, coefficients):
        norms = (coefficients @ self.gram * coefficients).sum(1)
        distance = (
            norms[:, None] + self.norms[None, :] - 2 * coefficients @ self.cross
        ).clamp_min(0)
        k = self.state_kernel * torch.exp(-0.5 * distance / self.lengthscale**2)
        return k @ self.beta + self.bias


@dataclass(frozen=True)
class BoundedAdamConfig:
    steps: int = 2000
    learning_rate: float = 0.01
    minimum_steps: int = 100
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    minimum_lr: float = 1e-6
    window: int = 25
    objective_span_tolerance: float = 1e-5
    action_update_rms_tolerance: float = 1e-4
    gradient_rms_tolerance: float = 1e-4
    gradient_gate: bool = False
    probe_radii: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    probe_improvement_tolerance: float = 1e-5
    checkpoint_seconds: float = 30.0
