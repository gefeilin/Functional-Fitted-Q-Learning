"""End-to-end toy example for functional fitted Q-learning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from functional_fitted_q import ExperimentConfig, Fitted_Q_Iteration, Fitted_q_evaluation
from functional_fitted_q.envs import Pendulum_data_generator
from functional_fitted_q.kernels import EarlyStopping, KRR
from functional_fitted_q.runner import build_bspline_basis


def serialize_dataset(dataset: pd.DataFrame) -> list[dict]:
    rows = []
    for row in dataset.itertuples(index=False):
        rows.append(
            {
                "subject": int(row.Subject),
                "time": int(row.Time),
                "state": np.asarray(row.State, dtype=float).tolist(),
                "functional_action": np.asarray(row.FunctionalData, dtype=float).tolist(),
                "reward": float(row.Reward),
                "next_state": np.asarray(row.Next_State, dtype=float).tolist(),
                "terminal": bool(row.Status),
            }
        )
    return rows


def main() -> None:
    np.random.seed(123)
    torch.manual_seed(123)

    config = ExperimentConfig(
        horizon=2,
        gamma=0.8,
        size=4,
        iteration=0,
        run_tag="quickstart",
        num_basis_knots=6,
        spline_degree=3,
        fqi_iterations=1,
        training_iterations=5,
        learning_rate=0.02,
        early_stopping_patience=2,
        early_stopping_delta=1e-4,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Quickstart requires a CUDA-capable PyTorch install.")
    gpu_index = config.gpu_id % torch.cuda.device_count()
    torch.cuda.set_device(gpu_index)
    torch.cuda.manual_seed_all(123)
    device = torch.device(f"cuda:{gpu_index}")

    output_dir = REPO_ROOT / "outputs" / "quickstart"
    data_dir = REPO_ROOT / "data" / "example"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = Pendulum_data_generator(
        episode_num=config.size,
        cycle_num=config.horizon,
        x_length=config.num_action_grid_points,
    )
    (data_dir / "toy_pendulum_transitions.json").write_text(json.dumps(serialize_dataset(dataset), indent=2))

    basis, penalty = build_bspline_basis(config, device)
    fqi = Fitted_Q_Iteration(
        dataset,
        discount=config.discount,
        bspline_basis=basis,
        R_matrix=penalty,
        lr=config.learning_rate,
        model=KRR,
        max_iterations=config.training_iterations,
        max_fqi_iteration=config.fqi_iterations,
        early_stop=EarlyStopping(config.early_stopping_patience, config.early_stopping_delta),
        device=device,
        lambda_spline=1e-3,
        q_hat_upper_bound=config.q_hat_upper,
    )
    fqi.update_q_function()

    initial_df = dataset[dataset["Time"] == 1].reset_index(drop=True)
    initial_states = torch.tensor(np.stack(initial_df["State"].to_numpy()), dtype=torch.float32, device=device)
    learned_actions = fqi.get_policy(initial_states).detach().cpu().numpy()

    fqe = Fitted_q_evaluation(
        dataset=dataset,
        discount=config.discount,
        model=KRR,
        device=device,
        policy=fqi.get_policy,
        bound=[0, config.q_hat_upper],
        num_iterations=1,
        gcv_max_iterations=config.training_iterations,
    )
    fqe.update_q_function()
    value_estimates = (
        fqe.predict(initial_states, torch.tensor(learned_actions, dtype=torch.float32, device=device))
        .detach()
        .cpu()
        .numpy()
    )

    summary = pd.DataFrame(
        [
            {
                "n_transitions": len(dataset),
                "n_subjects": dataset["Subject"].nunique(),
                "horizon": config.horizon,
                "gamma": config.gamma,
                "num_action_grid_points": config.num_action_grid_points,
                "device": str(device),
                "mean_reward": float(dataset["Reward"].mean()),
                "mean_initial_fqe_value": float(value_estimates.mean()),
            }
        ]
    )
    summary.to_csv(output_dir / "quickstart_summary.csv", index=False)

    grid = np.linspace(0, 1, config.num_action_grid_points)
    fig, axis = plt.subplots(figsize=(6, 4))
    for idx, action in enumerate(learned_actions):
        axis.plot(grid, action, alpha=0.8, label=f"subject {idx}")
    axis.set_xlabel("Functional action grid")
    axis.set_ylabel("Torque")
    axis.set_title("Learned toy functional actions")
    axis.set_ylim(-2.1, 2.1)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "learned_toy_actions.png", dpi=160)

    print(summary.to_string(index=False))
    print("Wrote toy data to data/example/toy_pendulum_transitions.json")
    print("Wrote outputs to outputs/quickstart/")


if __name__ == "__main__":
    main()
