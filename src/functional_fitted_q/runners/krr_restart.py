"""Prespecified fresh-Adam restart for theory total-curvature KRR policies."""

from __future__ import annotations

from functional_fitted_q.runners.krr_restart_state import (
    TotalCurvatureRestartConfig,
    bounded_adam_with_restart,
)
from functional_fitted_q.runners.krr_optimizer import TheoryTotalCurvatureAdamConfig


def bounded_adam_with_total_curvature_restart(
    critic, states, initial, config, *, penalty_states, **kwargs
):
    if not isinstance(config, TheoryTotalCurvatureAdamConfig):
        raise TypeError("theory total-curvature optimizer config required")
    restart = TotalCurvatureRestartConfig(
        policy_lambda=config.policy_lambda,
        penalty_convention=config.penalty_convention,
        penalty_state_scope=config.penalty_state_scope,
    )
    return bounded_adam_with_restart(
        critic,
        states,
        initial,
        config,
        penalty_states=penalty_states,
        restart_config=restart,
        **kwargs,
    )
