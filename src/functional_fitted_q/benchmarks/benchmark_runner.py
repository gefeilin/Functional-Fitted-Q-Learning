from dataclasses import dataclass
from itertools import product
import logging
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.model_selection import KFold

from .benchmark_common import (
    generate_unique_seed,
    safe_append,
)
from .benchmark_envs import (
    Pendulum_data_generator,
    monte_carlo_value_score_c,
)
from .benchmark_fqe import (
    Fitted_q_evaluation,
    KRR,
    dataset_for_d3rlpy,
)


logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class BenchmarkSpec:
    algo_name: str
    run_tag: str
    result_tag: str
    batch_size: int
    param_grid: dict[str, np.ndarray]
    sample_size: int = 10
    use_noop_logger: bool = False
    cv_splits: int = 3
    train_steps_multiplier: int = 10
    eval_sim_num: int = 1000


def sample_parameter_dicts(param_grid: dict[str, np.ndarray], seed: int, sample_size: int) -> list[dict]:
    keys = list(param_grid.keys())
    grid = list(product(*(param_grid[key] for key in keys)))
    random.seed(seed)
    sampled_grid = random.sample(grid, k=min(sample_size, len(grid)))
    return [dict(zip(keys, combo)) for combo in sampled_grid]


def fit_kwargs(experiment_name: str, use_noop_logger: bool) -> dict:
    kwargs = {"experiment_name": experiment_name}
    if use_noop_logger:
        try:
            from d3rlpy.logging import NoopAdapterFactory

            kwargs["logger_adapter"] = NoopAdapterFactory()
            kwargs["save_interval"] = 10**9
        except ImportError:
            # Some d3rlpy versions do not expose the logging adapter API.
            pass
    return kwargs


def evaluate_policy_with_fqe(val_df, val_df_initial, policy, gamma, q_hat_upper, current_device):
    fqe = Fitted_q_evaluation(
        dataset=val_df,
        discount=gamma,
        model=KRR,
        device=current_device,
        policy=policy,
        bound=(0, q_hat_upper),
    )
    fqe.update_q_function()

    state = torch.tensor(val_df_initial["State"].tolist(), dtype=torch.float32, device=current_device)
    n_states = state.shape[0]
    n_action_samples = 100
    state_expanded = state.unsqueeze(1).repeat(1, n_action_samples, 1).reshape(-1, state.shape[1])
    sampled_actions_matrix = policy(state_expanded).view(n_states, n_action_samples)

    expectation_q = []
    for idx in range(n_states):
        sampled_actions = sampled_actions_matrix[idx, :].unsqueeze(1)
        expectation = fqe.predict(state[idx].unsqueeze(0).repeat(n_action_samples, 1), sampled_actions)
        expectation_q.append(expectation.mean(dim=0).cpu().numpy())

    return np.mean(np.array(expectation_q).flatten())


def run_benchmark_experiment(args, spec: BenchmarkSpec, create_algo):
    horizon = args.horizon
    gamma = args.gamma
    q_hat_upper = 1 / (1 - gamma)
    size = args.size
    iteration = args.iteration
    gpu_id = iteration % 4

    logger.info("Running %s benchmark: size=%s, horizon=%s, gamma=%s, iteration=%s", spec.algo_name, size, horizon, gamma, iteration)

    output_root = Path(getattr(args, "output_dir", "outputs/benchmarks")).resolve()
    script_dir = ensure_dir(output_root / spec.algo_name)
    save_path = ensure_dir(script_dir / "saved_models" / spec.run_tag / f"h{horizon}_g{gamma}_s{size}")
    save_directory = ensure_dir(script_dir / "simulation_results_test" / spec.result_tag)

    seed = generate_unique_seed(420, size, iteration, gpu_id)
    np.random.seed(seed)
    df = Pendulum_data_generator(size, horizon, 100)

    sampled_param_dicts = sample_parameter_dicts(spec.param_grid, seed=seed, sample_size=spec.sample_size)
    subjects = df["Subject"].unique()
    n_splits = min(spec.cv_splits, len(subjects))
    if n_splits < 2:
        raise ValueError("Benchmark cross-validation requires at least 2 subjects.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_scores = []

    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        current_device_d3rlpy = f"cuda:{gpu_id % torch.cuda.device_count()}"
    else:
        current_device_d3rlpy = "cpu"

    algo_slug = spec.algo_name.lower()

    for index, param_dict in enumerate(sampled_param_dicts, start=1):
        param_text = ", ".join(f"{key}={float(value):.5e}" for key, value in param_dict.items())
        logger.info("Parameter set %s: %s", index, param_text)

        fqe_scores = []
        for fold_id, (train_idx, val_idx) in enumerate(kf.split(subjects), start=1):
            logger.info("Fold %s/%s", fold_id, n_splits)
            train_subjects = subjects[train_idx]
            val_subjects = subjects[val_idx]
            train_df = df[df["Subject"].isin(train_subjects)].reset_index(drop=True)
            val_df = df[df["Subject"].isin(val_subjects)].reset_index(drop=True)
            val_df_initial = val_df[val_df["Time"] == 1].reset_index(drop=True)

            dataset_offline = dataset_for_d3rlpy(train_df)
            algo = create_algo(param_dict, gamma=gamma, batch_size=spec.batch_size, device=current_device_d3rlpy)

            n_steps = len(train_df) * spec.train_steps_multiplier
            n_steps_per_epoch = max(1, len(train_df) // 10)
            algo.fit(
                dataset_offline,
                n_steps=n_steps,
                n_steps_per_epoch=n_steps_per_epoch,
                **fit_kwargs(
                    experiment_name=f"{algo_slug}_cv_h{horizon}_g{gamma}_s{size}_i{iteration}_fold{fold_id}",
                    use_noop_logger=spec.use_noop_logger,
                ),
            )

            def d3rlpy_policy(state):
                if isinstance(state, torch.Tensor):
                    state = state.detach().cpu().numpy()
                action = algo.predict(state) * 2
                return torch.tensor(action, dtype=torch.float32)

            avg_score = evaluate_policy_with_fqe(
                val_df=val_df,
                val_df_initial=val_df_initial,
                policy=d3rlpy_policy,
                gamma=gamma,
                q_hat_upper=q_hat_upper,
                current_device=current_device,
            )
            fqe_scores.append(avg_score)
            logger.info("Fold %s average score: %.4f", fold_id, avg_score)

        minimal_score = np.min(fqe_scores)
        cv_scores.append(minimal_score)
        logger.info("Minimum FQE score for %s: %.4f", param_dict, minimal_score)

    best_index = int(np.argmax(cv_scores))
    best_params = sampled_param_dicts[best_index]
    logger.info("Best parameters: %s with score %.4f", best_params, np.max(cv_scores))

    dataset_offline_whole = dataset_for_d3rlpy(df)
    algo_whole = create_algo(best_params, gamma=gamma, batch_size=spec.batch_size, device=current_device_d3rlpy)

    n_steps_whole = len(df) * spec.train_steps_multiplier
    n_steps_per_epoch_whole = max(1, len(df) // 10)
    algo_whole.fit(
        dataset_offline_whole,
        n_steps=n_steps_whole,
        n_steps_per_epoch=n_steps_per_epoch_whole,
        **fit_kwargs(
            experiment_name=f"{algo_slug}_cv_h{horizon}_g{gamma}_s{size}_i{iteration}_whole",
            use_noop_logger=spec.use_noop_logger,
        ),
    )

    algo_whole.save_model(str(save_path / f"h{horizon}_g{gamma}_s{size}_i{iteration}.pt"))

    avg_score_whole = monte_carlo_value_score_c(
        sim_num=spec.eval_sim_num,
        cycle_num=20,
        model=algo_whole,
        device=current_device,
        discount=gamma,
    )
    logger.info("Average score on whole dataset: %.4f", avg_score_whole)

    main_result_file = save_directory / f"seed420_cv_g{gamma}_h{horizon}.txt"
    safe_append(
        str(main_result_file),
        f"Data size {size}, Batch {iteration}: scores = {avg_score_whole:.4f}\n",
    )

    cv_result_file = save_directory / f"cv_logs_g{gamma}_h{horizon}.txt"
    safe_append(
        str(cv_result_file),
        f"Data size {size}, Batch {iteration}: CV scores = {cv_scores}, best parameter = {best_params}\n",
    )
