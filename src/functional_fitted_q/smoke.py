"""Small real computations: data, FQI, grouped KRR CV and interrupted resume.

The deliberately small CPU configuration tests execution, not paper performance.
"""

from dataclasses import replace
import importlib
import json
import pkgutil
import tempfile
import time
import numpy as np
import yaml
import functional_fitted_q
from .runtime import deterministic_device


def run(root, output):
    """Exercise data generation, both critics, and interrupted resume on CPU."""
    started = time.time()
    device = deterministic_device("cpu")
    import torch
    from .data import generate_offline_dataset
    from .formal_config import load_pendulum_config
    from .policies.reference_policy import AnalyticReferencePolicy
    from .algorithms.adafnn_ffqi import run_adafnn_ffqi
    from .algorithms.dimensionless_total_curvature import (
        DimensionlessTheoryTotalCurvatureCMAES,
        DimensionlessTheoryTotalCurvatureCMAESConfig,
    )
    from .algorithms.cmaes_policy_optimization import (
        CMAESPolicyOptimizationConfig,
        DerivativeFreeCMAES,
    )
    from .critics.adafnn_functional_critic import AdaFNNCriticTrainingConfig
    from .critics.nystrom_grouped_cv import (
        GroupedCVNystromFunctionalKRR,
        GroupedRidgeCVConfig,
    )
    from .critics.nystrom_functional_critic import NystromKRRConfig
    from .runners.krr_prediction import FastPredictionView
    from .runners.krr_gradient import bounded_adam_batched
    from .runners.krr_optimizer import TheoryTotalCurvatureAdamConfig
    from .policies.bspline_policy import BoundedCoefficientBSplinePolicy
    from .algorithms.theory_total_curvature import total_curvature_matrix
    from .identification import blocked_state_neighbors
    from .evaluation import evaluate_policy
    from .run_artifacts import atomic_json

    for module in pkgutil.walk_packages(
        functional_fitted_q.__path__, functional_fitted_q.__name__ + "."
    ):
        if module.name != "functional_fitted_q.__main__":
            importlib.import_module(module.name)
    reference = AnalyticReferencePolicy(
        np.array(
            yaml.safe_load((root / "configs/reference_policy.yaml").read_text())["beta"]
        )
    )
    environment = replace(load_pendulum_config(root), ode_substeps=8)
    data = generate_offline_dataset(
        master_seed=200000,
        reference=reference,
        n_subjects=10,
        t_per_subject=3,
        env_config=environment,
        gp_resolution=16,
        step_gp_amplitude=0.35,
        vectorized=True,
    )
    again = generate_offline_dataset(
        master_seed=200000,
        reference=reference,
        n_subjects=10,
        t_per_subject=3,
        env_config=environment,
        gp_resolution=16,
        step_gp_amplitude=0.35,
        vectorized=True,
    )
    np.testing.assert_array_equal(data.states, again.states)
    grid = np.linspace(0, 1, 16)
    critic = AdaFNNCriticTrainingConfig(
        n_basis_nodes=2,
        micro_hidden_layers=(4,),
        outer_hidden_layers=(8,),
        batch_size=16,
        max_epochs=4,
        early_stopping_patience=2,
        require_early_stopping_convergence=False,
    )
    pcfg = DimensionlessTheoryTotalCurvatureCMAESConfig(
        restarts=1,
        max_objective_evals=8,
        population_size=4,
        objective_state_count=8,
        candidate_batch_size=2,
        lambda_dimensionless=0.001,
    )
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="execution-", dir=output) as tmp:
        from pathlib import Path

        tmp = Path(tmp)
        common = dict(
            dataset=data,
            dataset_identity="cpu-smoke",
            master_seed=200000,
            iterations=2,
            quadrature_grid=grid,
            critic_config=critic,
            device=device,
            k=4,
            gamma=0.95,
            policy_view_iterations=(2,),
        )
        opt = lambda: DimensionlessTheoryTotalCurvatureCMAES(pcfg, grid, device)
        uninterrupted = run_adafnn_ffqi(
            **common, policy_optimizer=opt(), checkpoint_dir=tmp / "uninterrupted"
        )

        def interrupt(index):
            raise InterruptedError(
                "intentional smoke interruption after durable segment"
            )

        try:
            run_adafnn_ffqi(
                **common,
                policy_optimizer=opt(),
                checkpoint_dir=tmp / "resume",
                segment_observer=interrupt
            )
        except InterruptedError:
            pass
        else:
            raise AssertionError("Interruption hook did not run")
        resumed = run_adafnn_ffqi(
            **common, policy_optimizer=opt(), checkpoint_dir=tmp / "resume", resume=True
        )
        np.testing.assert_array_equal(
            uninterrupted.final_policy.parameter_matrix,
            resumed.final_policy.parameter_matrix,
        )
        for key, value in uninterrupted.final_critic.state_dict().items():
            torch.testing.assert_close(
                value, resumed.final_critic.state_dict()[key], rtol=0, atol=0
            )
        const = run_adafnn_ffqi(
            **common,
            policy_kind="bounded_constant_function",
            policy_optimizer=DerivativeFreeCMAES(
                CMAESPolicyOptimizationConfig(
                    restarts=1,
                    max_objective_evals=8,
                    population_size=4,
                    objective_state_count=8,
                    candidate_batch_size=2,
                ),
                grid,
                device,
            ),
            checkpoint_dir=tmp / "constant"
        )
        actions = const.final_policy.values_batch(data.states, grid)
        np.testing.assert_allclose(
            actions, np.repeat(actions[:, :1], len(grid), axis=1), rtol=0, atol=0
        )
        evaluation = evaluate_policy(
            resumed.final_policy, 200001, environment, episodes=4, horizon=5
        )
        assert np.isfinite(evaluation.normalized_returns).all()
    krr = GroupedCVNystromFunctionalKRR(
        NystromKRRConfig(landmark_rank=6, ridge=None, action_l2_lengthscale=0.75),
        GroupedRidgeCVConfig((1e-4, 1e-2, 1.0), folds=5),
        data.subject_ids,
        value_max=20.0,
    )
    krr.fit_design(
        data.states, data.action_values, grid, device, landmark_indices=np.arange(6)
    )
    krr.solve_targets(data.rewards)
    pred = FastPredictionView(krr).predict(data.states, data.action_values)
    np.testing.assert_allclose(
        pred, krr.predict(data.states, data.action_values), rtol=1e-10, atol=1e-10
    )
    initial = BoundedCoefficientBSplinePolicy(np.zeros((7, 4)))
    acfg = TheoryTotalCurvatureAdamConfig(
        steps=6, minimum_steps=100, policy_lambda=1e-7
    )
    full = bounded_adam_batched(
        FastPredictionView(krr),
        data.next_states[:8],
        initial,
        acfg,
        penalty_states=data.next_states,
    )
    short = bounded_adam_batched(
        FastPredictionView(krr),
        data.next_states[:8],
        initial,
        acfg,
        penalty_states=data.next_states,
        stop_after=3,
    )
    restored = bounded_adam_batched(
        FastPredictionView(krr),
        data.next_states[:8],
        initial,
        acfg,
        penalty_states=data.next_states,
        resume=short[2],
    )
    np.testing.assert_array_equal(
        full[0].parameter_matrix, restored[0].parameter_matrix
    )
    matrix, _ = total_curvature_matrix(initial, grid)
    np.testing.assert_allclose(np.ones(4) @ matrix @ np.ones(4), 0, atol=1e-8)
    idx, _ = blocked_state_neighbors(
        data.states[:2], data.states, (1.0, 1.0, 2.0), count=4
    )
    assert idx.shape == (2, 4)
    report = dict(
        status="PASS",
        device="cpu",
        paper_training_rerun=False,
        transitions=30,
        iterations=2,
        adafnn_interrupted_resume="bitwise equal critic and policy",
        krr_optimizer_resume="bitwise equal policy",
        constant_action="constant in u",
        krr_grouped_cv_folds=5,
        seconds=time.time() - started,
        versions=dict(torch=torch.__version__, numpy=np.__version__),
    )
    atomic_json(output / "validation.json", report)
    print(json.dumps(report, indent=2))
