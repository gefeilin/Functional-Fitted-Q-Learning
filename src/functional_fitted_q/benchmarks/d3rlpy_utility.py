"""Compatibility exports for the optional benchmark package.

The actual implementations live in smaller benchmark modules.
"""

from .benchmark_common import (
    DEFAULT_X,
    DEFAULT_Y,
    angle_normalize,
    calculate_reward,
    calculate_reward_c,
    generate_unique_seed,
    random_policy,
    safe_append,
)
from .benchmark_envs import (
    PendulumEnv_c,
    PendulumEnv_functional,
    Pendulum_data_generator,
    Pendulum_data_generator_for_monte_c,
    monte_carlo_value_score_c,
)
from .benchmark_fqe import (
    EarlyStopping,
    Fitted_q_evaluation,
    KRR,
    KRRlamda_gcv,
    KRRlamda_gcv_selector,
    dataset_for_d3rlpy,
)

__all__ = [
    "DEFAULT_X",
    "DEFAULT_Y",
    "EarlyStopping",
    "Fitted_q_evaluation",
    "KRR",
    "KRRlamda_gcv",
    "KRRlamda_gcv_selector",
    "PendulumEnv_c",
    "PendulumEnv_functional",
    "Pendulum_data_generator",
    "Pendulum_data_generator_for_monte_c",
    "angle_normalize",
    "calculate_reward",
    "calculate_reward_c",
    "dataset_for_d3rlpy",
    "generate_unique_seed",
    "monte_carlo_value_score_c",
    "random_policy",
    "safe_append",
]
