"""Train and evaluate a small functional-action AdaFNN FQI model on CPU."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs/quickstart.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "outputs/quickstart"
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/example")
    args = parser.parse_args()

    # Establish execution and deterministic-runtime settings before loading torch.
    from functional_fitted_q.runtime import deterministic_device

    device = deterministic_device("cpu")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml
    from functional_fitted_q.algorithms.adafnn_ffqi import run_adafnn_ffqi
    from functional_fitted_q.algorithms.dimensionless_total_curvature import (
        DimensionlessTheoryTotalCurvatureCMAES,
        DimensionlessTheoryTotalCurvatureCMAESConfig,
    )
    from functional_fitted_q.critics.adafnn_functional_critic import (
        AdaFNNCriticTrainingConfig,
    )
    from functional_fitted_q.data import generate_offline_dataset
    from functional_fitted_q.evaluation import evaluate_policy
    from functional_fitted_q.formal_config import load_pendulum_config
    from functional_fitted_q.policies.reference_policy import AnalyticReferencePolicy

    config = yaml.safe_load(args.config.read_text())
    reference_config = yaml.safe_load(
        (REPO_ROOT / "configs/reference_policy.yaml").read_text()
    )
    reference = AnalyticReferencePolicy(np.array(reference_config["beta"]))
    environment = replace(
        load_pendulum_config(REPO_ROOT), ode_substeps=config["ode_substeps"]
    )
    grid = np.linspace(0, 1, config["action_grid_points"])

    # Simulate a small dataset with the same behavior family as the paper.
    data = generate_offline_dataset(
        master_seed=config["master_seed"],
        reference=reference,
        n_subjects=config["n_subjects"],
        t_per_subject=config["t_per_subject"],
        env_config=environment,
        gp_resolution=len(grid),
        step_gp_amplitude=0.35,
        vectorized=True,
    )
    args.data_dir.mkdir(parents=True, exist_ok=True)
    records = [
        dict(
            state=s.tolist(),
            functional_action=a.tolist(),
            reward=float(r),
            next_state=sp.tolist(),
            subject_id=int(subject),
            time_index=int(time_index),
            terminal=False,
        )
        for s, a, r, sp, subject, time_index in zip(
            data.states,
            data.action_values,
            data.rewards,
            data.next_states,
            data.subject_ids,
            data.time_indices,
        )
    ]
    (args.data_dir / "toy_pendulum_transitions.json").write_text(
        json.dumps(dict(action_grid=grid.tolist(), transitions=records), indent=2)
        + "\n"
    )

    # Fit two small FQI iterations; the reduced settings do not reproduce paper results.
    critic_config = dict(config["critic"])
    for key in ["micro_hidden_layers", "outer_hidden_layers"]:
        critic_config[key] = tuple(critic_config[key])
    optimizer_config = DimensionlessTheoryTotalCurvatureCMAESConfig(
        **config["policy"], lambda_dimensionless=config["policy_coefficient"]
    )
    optimizer = DimensionlessTheoryTotalCurvatureCMAES(optimizer_config, grid, device)
    fit = run_adafnn_ffqi(
        dataset=data,
        dataset_identity="quickstart-example",
        master_seed=config["master_seed"],
        iterations=config["fqi_iterations"],
        quadrature_grid=grid,
        critic_config=AdaFNNCriticTrainingConfig(**critic_config),
        device=device,
        k=config["policy_basis_dimension"],
        gamma=environment.gamma,
        policy_view_iterations=(config["fqi_iterations"],),
        policy_optimizer=optimizer,
    )

    # Save a small independent policy evaluation and a functional-action plot.
    evaluation = evaluate_policy(
        fit.final_policy, config["evaluation_seed"], environment, **config["evaluation"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            dict(
                n_transitions=len(data.states),
                n_subjects=config["n_subjects"],
                t_per_subject=config["t_per_subject"],
                fqi_iterations=config["fqi_iterations"],
                gamma=environment.gamma,
                device="cpu",
                mean_reward=float(data.rewards.mean()),
                normalized_return=float(evaluation.normalized_returns.mean()),
            )
        ]
    )
    summary.to_csv(args.output_dir / "quickstart_summary.csv", index=False)
    actions = fit.final_policy.values_batch(data.states[:4], grid)
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, action in enumerate(actions):
        ax.plot(grid, action, label=f"State {i + 1}")
    ax.set(xlabel="Within-action time, u", ylabel="Torque", ylim=(-2.1, 2.1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "learned_toy_actions.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
