"""Compatibility exports for the public package.

This project previously concentrated most of the implementation in a single
``utils.py`` file. The codebase is now organized into smaller modules, but we
keep this re-export layer so existing scripts can import the public symbols
from one place.
"""

from .algorithms import (
    Fitted_Q_Iteration,
    Fitted_Q_Iteration_c,
    Fitted_q_evaluation,
    QHat_c,
    max_Q_c,
)
from .common import (
    DEFAULT_X,
    DEFAULT_Y,
    calculate_reward,
    calculate_reward_c,
    compute_bspline_penalty_R,
    generate_action,
    generate_unique_seed,
    random_policy,
    safe_append,
    state_dependent_gp,
)
from .envs import (
    PendulumEnv,
    PendulumEnv_c,
    Pendulum_data_generator,
    Pendulum_data_generator_c,
    Pendulum_data_generator_for_monte,
    Pendulum_data_generator_for_monte_c,
    angle_normalize,
    monte_carlo_value_score,
    monte_carlo_value_score_c,
)
from .kernels import (
    EarlyStopping,
    KRR,
    KRRlamda_gcv,
    KRRlamda_gcv_selector,
    Maximize_QHat,
)

__all__ = [
    "DEFAULT_X",
    "DEFAULT_Y",
    "EarlyStopping",
    "Fitted_Q_Iteration",
    "Fitted_Q_Iteration_c",
    "Fitted_q_evaluation",
    "KRR",
    "KRRlamda_gcv",
    "KRRlamda_gcv_selector",
    "Maximize_QHat",
    "PendulumEnv",
    "PendulumEnv_c",
    "Pendulum_data_generator",
    "Pendulum_data_generator_c",
    "Pendulum_data_generator_for_monte",
    "Pendulum_data_generator_for_monte_c",
    "QHat_c",
    "angle_normalize",
    "calculate_reward",
    "calculate_reward_c",
    "compute_bspline_penalty_R",
    "generate_action",
    "generate_unique_seed",
    "max_Q_c",
    "monte_carlo_value_score",
    "monte_carlo_value_score_c",
    "random_policy",
    "safe_append",
    "state_dependent_gp",
]
