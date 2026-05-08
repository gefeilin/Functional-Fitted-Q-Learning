from pathlib import Path
import logging

import numpy as np
import torch
from filelock import FileLock
from scipy.interpolate import BSpline
from sklearn.model_selection import KFold
from tqdm import tqdm

from .algorithms import Fitted_Q_Iteration, Fitted_q_evaluation
from .common import compute_bspline_penalty_R, generate_action, generate_unique_seed, safe_append
from .config import ExperimentConfig
from .envs import Pendulum_data_generator, monte_carlo_policy_value_score, monte_carlo_value_score
from .kernels import EarlyStopping, KRR
from .paths import output_paths


logger = logging.getLogger(__name__)


def build_bspline_basis(config: ExperimentConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Precompute the spline basis and roughness penalty once per experiment, not once per CV fold.
    spline_degree = config.spline_degree
    knots = np.linspace(0, 1, config.num_basis_knots)
    knots = np.concatenate(([knots[0]] * spline_degree, knots, [knots[-1]] * spline_degree))
    coefficients = np.eye(len(knots) - spline_degree - 1)
    basis_functions = [BSpline(knots, coefficients[i], spline_degree) for i in range(len(coefficients))]

    x_grid = np.linspace(0, 1, config.num_action_grid_points)
    basis_values = np.array([basis(x_grid) for basis in basis_functions])
    penalty_matrix = compute_bspline_penalty_R(basis_values, x_grid)

    basis_tensor = torch.tensor(basis_values, dtype=torch.float32, device=device)
    penalty_tensor = torch.tensor(penalty_matrix, dtype=torch.float32, device=device)
    return basis_tensor, penalty_tensor


def sample_policy_actions(
    states,
    num_samples: int,
    device: torch.device,
    desc: str,
    x_length: int = 100,
) -> torch.Tensor:
    x_index = np.linspace(0, 1, x_length)
    sampled_actions = []
    for state in tqdm(states, desc=desc, unit="state"):
        sampled_actions.append(
            np.stack([generate_action(state, x_index=x_index) for _ in range(num_samples)])
        )
    return torch.tensor(np.stack(sampled_actions), dtype=torch.float32, device=device)


def get_or_compute_fqe_ground_truth(
    result_path: Path,
    gamma: float,
    discount: float,
    x_length: int,
) -> float:
    ground_truth_file = result_path / f"fqe_ground_truth_g{gamma}.txt"
    lock = FileLock(str(ground_truth_file) + ".lock")

    with lock:
        if ground_truth_file.exists():
            cached_value = ground_truth_file.read_text().strip()
            if cached_value:
                value = float(cached_value)
                logger.info("Loaded cached ground truth for gamma=%s: %.6f", gamma, value)
                return value

        logger.info("Computing ground truth for gamma=%s with 1000 simulations and 20 cycles", gamma)
        x_index = np.linspace(0, 1, x_length)
        value = monte_carlo_policy_value_score(
            sim_num=1000,
            cycle_num=20,
            x_length=x_length,
            policy=lambda state: generate_action(state, x_index=x_index),
            discount=discount,
        )
        ground_truth_file.write_text(f"{value:.12f}\n")
        logger.info("Saved ground truth for gamma=%s: %.6f", gamma, value)
        return value


def run_experiment(config: ExperimentConfig, base_dir: Path) -> None:
    logger.info(
        "Running functional FQI: size=%s, horizon=%s, gamma=%s, iteration=%s",
        config.size,
        config.horizon,
        config.gamma,
        config.iteration,
    )

    save_path, result_path = output_paths(
        base_dir=base_dir,
        run_tag=config.run_tag,
        horizon=config.horizon,
        gamma=config.gamma,
        size=config.size,
    )

    seed = generate_unique_seed(420, config.size, config.iteration, config.gpu_id)
    np.random.seed(seed)
    if torch.cuda.is_available():
        gpu_index = config.gpu_id % torch.cuda.device_count()
        torch.cuda.set_device(gpu_index)
        current_device = torch.device(f"cuda:{gpu_index}")
    else:
        current_device = torch.device("cpu")

    basis_tensor, penalty_tensor = build_bspline_basis(config, current_device)
    dataset = Pendulum_data_generator(
        episode_num=config.size,
        cycle_num=config.horizon,
        x_length=config.num_action_grid_points,
    )

    lambda_grid = np.logspace(
        np.log10(config.lambda_min),
        np.log10(config.lambda_max),
        num=config.lambda_grid_size,
    ).tolist()
    subjects = dataset["Subject"].unique()
    kfold = KFold(n_splits=config.num_cv_splits, shuffle=True, random_state=42)
    cv_scores = []

    for lambda_spline in lambda_grid:
        fold_scores = []
        logger.info("Evaluating lambda_spline=%.5f", lambda_spline)

        for fold_id, (train_idx, val_idx) in enumerate(kfold.split(subjects), start=1):
            logger.info("Fold %s/%s", fold_id, config.num_cv_splits)
            train_subjects = subjects[train_idx]
            val_subjects = subjects[val_idx]

            train_df = dataset[dataset["Subject"].isin(train_subjects)].reset_index(drop=True)
            val_df = dataset[dataset["Subject"].isin(val_subjects)].reset_index(drop=True)
            val_df_initial = val_df[val_df["Time"] == 1].reset_index(drop=True)

            # Cross-validation only tunes the spline smoothness penalty; the inner FQI loop stays fixed.
            fitted_q = Fitted_Q_Iteration(
                train_df,
                discount=config.discount,
                bspline_basis=basis_tensor,
                R_matrix=penalty_tensor,
                lr=config.learning_rate,
                model=KRR,
                max_iterations=config.training_iterations,
                max_fqi_iteration=config.fqi_iterations,
                early_stop=EarlyStopping(config.early_stopping_patience, config.early_stopping_delta),
                device=current_device,
                lambda_spline=lambda_spline,
                q_hat_upper_bound=config.q_hat_upper,
            )
            fitted_q.update_q_function()

            # Reuse the same nonparametric Q machinery in evaluation mode to score the frozen policy.
            fqe = Fitted_q_evaluation(
                val_df,
                config.discount,
                KRR,
                current_device,
                fitted_q.get_policy,
                [0, config.q_hat_upper] if config.fqe_clamp_targets else None,
                gcv_max_iterations=min(1000, config.training_iterations),
            )
            fqe.update_q_function()

            state = torch.tensor(val_df_initial["State"].tolist(), dtype=torch.float32, device=current_device)
            fqe_value = fqe.predict(state, fitted_q.get_policy(state)).cpu().numpy()
            avg_score = np.mean(np.array(fqe_value).flatten())
            fold_scores.append(avg_score)
            logger.info("Fold %s FQE score: %.4f", fold_id, avg_score)

        minimal_score = np.min(fold_scores)
        cv_scores.append(minimal_score)
        logger.info("Minimum FQE score for lambda_spline=%.5f: %.4f", lambda_spline, minimal_score)

    best_lambda_spline = lambda_grid[np.argmax(cv_scores)]
    logger.info("Best lambda_spline=%.5f with score %.4f", best_lambda_spline, np.max(cv_scores))

    # Refit on the full dataset with the best penalty selected by cross-validation.
    fitted_q_whole = Fitted_Q_Iteration(
        dataset,
        discount=config.discount,
        bspline_basis=basis_tensor,
        R_matrix=penalty_tensor,
        lr=config.learning_rate,
        model=KRR,
        max_iterations=config.training_iterations,
        max_fqi_iteration=config.fqi_iterations,
        early_stop=EarlyStopping(config.early_stopping_patience, config.early_stopping_delta),
        device=current_device,
        lambda_spline=best_lambda_spline,
        q_hat_upper_bound=config.q_hat_upper,
    )
    fitted_q_whole.experiment_config = config
    fitted_q_whole.update_q_function()
    fitted_q_whole.save(str(save_path / f"h{config.horizon}_g{config.gamma}_s{config.size}_i{config.iteration}.pth"))

    score_sim = monte_carlo_value_score(
        model=fitted_q_whole,
        sim_num=config.fqe_simulations,
        cycle_num=config.fqe_cycles,
        device=current_device,
        discount=config.discount,
    )

    result_file = result_path / f"seed420_cv_g{config.gamma}_h{config.horizon}.txt"
    safe_append(str(result_file), f"Data size {config.size}, Batch {config.iteration}: scores = {score_sim:.4f}\n")

    cv_file = result_path / f"cv_logs_g{config.gamma}_h{config.horizon}.txt"
    safe_append(
        str(cv_file),
        (
            f"Data size {config.size}, Batch {config.iteration}: "
            f"CV scores = {cv_scores}, best spline lambda = {best_lambda_spline}\n"
        ),
    )


def run_fqe_simulation(config: ExperimentConfig, base_dir: Path) -> None:
    logger.info(
        "Running functional FQE: size=%s, horizon=%s, gamma=%s, iteration=%s",
        config.size,
        config.horizon,
        config.gamma,
        config.iteration,
    )

    _, result_path = output_paths(
        base_dir=base_dir,
        run_tag=config.run_tag,
        horizon=config.horizon,
        gamma=config.gamma,
        size=config.size,
    )

    seed = generate_unique_seed(420, config.size, config.iteration, config.gpu_id)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        gpu_index = config.gpu_id % torch.cuda.device_count()
        torch.cuda.set_device(gpu_index)
        current_device = torch.device(f"cuda:{gpu_index}")
    else:
        current_device = torch.device("cpu")

    x_length = config.num_action_grid_points
    true_value = get_or_compute_fqe_ground_truth(
        result_path=result_path,
        gamma=config.gamma,
        discount=config.discount,
        x_length=x_length,
    )

    train_expectation_samples = config.fqe_train_expectation_samples
    eval_expectation_samples = config.fqe_eval_expectation_samples
    policy_x_index = np.linspace(0, 1, x_length)
    dataset = Pendulum_data_generator(
        episode_num=config.size,
        cycle_num=config.horizon,
        x_length=x_length,
    )
    cached_policy_actions = sample_policy_actions(
        dataset["Next_State"].tolist(),
        num_samples=train_expectation_samples,
        device=current_device,
        desc=f"Sampling train actions (n={train_expectation_samples})",
        x_length=x_length,
    )

    fqe = Fitted_q_evaluation(
        dataset=dataset,
        discount=config.discount,
        model=KRR,
        device=current_device,
        policy=lambda state_batch: torch.tensor(
            np.stack(
                [
                    np.stack(
                        [
                            generate_action(state, x_index=policy_x_index)
                            for _ in range(train_expectation_samples)
                        ]
                    )
                    for state in state_batch.detach().cpu().numpy()
                ]
            ),
            dtype=torch.float32,
            device=current_device,
        ),
        bound=[0, config.q_hat_upper] if config.fqe_clamp_targets else None,
        policy_actions=cached_policy_actions,
        num_iterations=config.fqe_cycles,
        gcv_max_iterations=min(1000, config.training_iterations),
    )
    fqe.update_q_function()

    initial_state_df = dataset[dataset["Time"] == 1]
    eval_states = np.array(initial_state_df["State"].tolist())
    if config.fqe_eval_states > 0 and config.fqe_eval_states < len(eval_states):
        eval_states = eval_states[: config.fqe_eval_states]
    eval_state_tensor = torch.tensor(eval_states, dtype=torch.float32, device=current_device)
    eval_action_tensor = sample_policy_actions(
        eval_states,
        num_samples=eval_expectation_samples,
        device=current_device,
        desc=f"Sampling eval actions (n={eval_expectation_samples})",
        x_length=x_length,
    )
    predicted_values = fqe.predict(eval_state_tensor, eval_action_tensor).detach().cpu().numpy().reshape(-1)
    estimate = float(np.mean(predicted_values))
    squared_error = float((estimate - true_value) ** 2)
    logger.info("Estimated value: %.6f", estimate)

    result_file = result_path / f"fqe_seed420_g{config.gamma}_h{config.horizon}.txt"
    safe_append(
        str(result_file),
        (
            f"Data size {config.size}, Batch {config.iteration}: "
            f"Score = {estimate:.6f}, GroundTruth = {true_value:.6f}, SquaredError = {squared_error:.6f}\n"
        ),
    )
    logger.info("Ground truth: %.6f", true_value)
    logger.info("Squared error: %.6f", squared_error)
