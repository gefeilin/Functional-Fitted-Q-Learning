import numpy as np
import torch

from functional_fitted_q import ExperimentConfig, Fitted_Q_Iteration, Fitted_q_evaluation
from functional_fitted_q.envs import Pendulum_data_generator
from functional_fitted_q.kernels import EarlyStopping, KRR
from functional_fitted_q.runner import build_bspline_basis


def _tiny_config():
    return ExperimentConfig(
        horizon=2,
        gamma=0.8,
        size=3,
        iteration=0,
        num_basis_knots=5,
        spline_degree=3,
        num_action_grid_points=10,
        fqi_iterations=1,
        training_iterations=2,
        early_stopping_patience=1,
    )


def test_tiny_fqi_and_fqe_return_valid_values():
    np.random.seed(123)
    torch.manual_seed(123)
    config = _tiny_config()
    device = torch.device("cpu")
    dataset = Pendulum_data_generator(config.size, config.horizon, config.num_action_grid_points)
    basis, penalty = build_bspline_basis(config, device)

    fqi = Fitted_Q_Iteration(
        dataset,
        discount=config.discount,
        bspline_basis=basis,
        R_matrix=penalty,
        model=KRR,
        max_iterations=config.training_iterations,
        max_fqi_iteration=config.fqi_iterations,
        early_stop=EarlyStopping(1, 0.0),
        device=device,
        lambda_spline=1e-3,
        q_hat_upper_bound=config.q_hat_upper,
    )
    fqi.update_q_function()

    initial_states = dataset[dataset["Time"] == 1]["State"].to_numpy()
    states = torch.tensor(np.stack(initial_states), dtype=torch.float32)
    actions = fqi.get_policy(states)
    assert actions.shape == (config.size, config.num_action_grid_points)
    assert torch.isfinite(actions).all()
    assert torch.max(torch.abs(actions)) <= 2.0

    fqe = Fitted_q_evaluation(
        dataset,
        discount=config.discount,
        model=KRR,
        device=device,
        policy=fqi.get_policy,
        bound=[0, config.q_hat_upper],
        num_iterations=1,
        gcv_max_iterations=2,
    )
    fqe.update_q_function()
    values = fqe.predict(states, actions)
    assert values.shape[0] == config.size
    assert torch.isfinite(values).all()
