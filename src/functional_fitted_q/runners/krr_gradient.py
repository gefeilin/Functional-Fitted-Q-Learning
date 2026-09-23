"""Gradient policy improvement with raw uncentered total curvature.

The clipped-Q average uses the frozen objective-state subset.  The curvature
term is evaluated on every training next state, exactly matching the empirical
sample-averaged penalty in the manuscript.  Every finite stopping probe uses
the same two state sets and the same penalized objective.
"""

from __future__ import annotations

from dataclasses import asdict
import time

import numpy as np
import torch

from functional_fitted_q.runners.krr_prediction import (
    BoundedCoefficientBSplinePolicy,
    FastPredictionView,
    FixedStateSplineObjective,
    policy_state_features_batch,
)
from functional_fitted_q.runners.krr_optimizer import (
    TheoryTotalCurvatureAdamConfig,
    penalty_matrix,
    penalized_score,
)
from functional_fitted_q.algorithms.theory_total_curvature import (
    torch_uncentered_total_curvature,
)


def _candidate_scores(
    objective,
    q_features,
    penalty_features,
    parameters,
    *,
    block_size=8,
):
    if block_size < 1:
        raise ValueError("positive probe block size required")
    scores = []
    with torch.no_grad():
        for start in range(0, len(parameters), block_size):
            candidate = parameters[start : start + block_size]
            q_coefficients = 2.0 * torch.tanh(
                torch.matmul(q_features.unsqueeze(0), candidate)
            )
            penalty_coefficients = 2.0 * torch.tanh(
                torch.matmul(penalty_features.unsqueeze(0), candidate)
            )
            norms = (torch.matmul(q_coefficients, objective.gram) * q_coefficients).sum(
                -1
            )
            distance = (
                norms[:, :, None]
                + objective.norms[None, None, :]
                - 2.0 * torch.matmul(q_coefficients, objective.cross)
            ).clamp_min(0.0)
            kernel = objective.state_kernel[None, :, :] * torch.exp(
                -0.5 * distance / objective.lengthscale**2
            )
            raw = torch.matmul(kernel, objective.beta) + objective.bias
            scores.append(
                penalized_score(
                    raw,
                    penalty_coefficients,
                    objective.view.value_max,
                    objective.policy_lambda,
                    objective.policy_penalty_matrix,
                )
            )
    return torch.cat(scores)


def _probe_improvements(
    objective,
    q_features,
    penalty_features,
    best,
    directions,
    radii,
    best_value,
    tolerance,
):
    parameters = torch.stack(
        [
            best + sign * radius * direction
            for direction in directions
            for radius in radii
            for sign in (-1, 1)
        ]
    )
    improvements = (
        (
            _candidate_scores(objective, q_features, penalty_features, parameters)
            - best_value
        )
        .cpu()
        .tolist()
    )
    scalar_fallback = abs(max(improvements) - tolerance) <= 1.0e-9
    if scalar_fallback:
        improvements = []
        for parameters_one in parameters:
            q_coefficients = 2.0 * torch.tanh(q_features @ parameters_one)
            penalty_coefficients = 2.0 * torch.tanh(penalty_features @ parameters_one)
            raw = objective.raw(q_coefficients)
            score = penalized_score(
                raw,
                penalty_coefficients,
                objective.view.value_max,
                objective.policy_lambda,
                objective.policy_penalty_matrix,
            )
            improvements.append(float(score) - best_value)
    return improvements, scalar_fallback


def _witnessed_probe_improvements(
    objective,
    q_features,
    penalty_features,
    best,
    directions,
    radii,
    best_value,
    tolerance,
    previous_index,
):
    count = len(directions) * len(radii) * 2
    if previous_index is not None and 0 <= previous_index < count:
        direction_index, remainder = divmod(previous_index, len(radii) * 2)
        radius_index, sign_index = divmod(remainder, 2)
        parameters = (
            best
            + (-1 if sign_index == 0 else 1)
            * radii[radius_index]
            * directions[direction_index]
        )
        q_coefficients = 2.0 * torch.tanh(q_features @ parameters)
        penalty_coefficients = 2.0 * torch.tanh(penalty_features @ parameters)
        score = float(
            penalized_score(
                objective.raw(q_coefficients),
                penalty_coefficients,
                objective.view.value_max,
                objective.policy_lambda,
                objective.policy_penalty_matrix,
            )
        )
        gain = score - best_value
        if gain > tolerance + 1.0e-9:
            return [gain], False, previous_index, True
    values, fallback = _probe_improvements(
        objective,
        q_features,
        penalty_features,
        best,
        directions,
        radii,
        best_value,
        tolerance,
    )
    index = max(range(len(values)), key=values.__getitem__)
    return values, fallback, index, False


def bounded_adam_batched(
    critic,
    states,
    initial,
    config,
    *,
    penalty_states,
    resume=None,
    checkpoint=None,
    stop_after=None,
    allow_failing_witness=False,
):
    """Optimize bounded B-spline coefficients with resumable batched Adam.

    Q values are averaged on ``states`` while the manuscript's uncentered
    curvature is averaged on ``penalty_states``.  The returned state contains
    the optimizer and stopping-window history needed for exact continuation.
    """
    if not isinstance(config, TheoryTotalCurvatureAdamConfig):
        raise TypeError("theory total-curvature Adam configuration required")
    if not isinstance(initial, BoundedCoefficientBSplinePolicy):
        raise TypeError("positive total-curvature tasks require a functional policy")
    if config.steps < 1 or config.minimum_steps < 1 or config.window < 2:
        raise ValueError("invalid optimizer budget")
    states = np.ascontiguousarray(states, dtype=np.float64)
    penalty_states = np.ascontiguousarray(penalty_states, dtype=np.float64)
    if (
        states.ndim != 2
        or penalty_states.ndim != 2
        or states.shape[1:] != (3,)
        or penalty_states.shape[1:] != (3,)
        or len(penalty_states) < len(states)
    ):
        raise ValueError("invalid Q-objective or penalty state arrays")
    device = critic.device
    q_states = torch.as_tensor(states, dtype=torch.float64, device=device)
    q_features = torch.as_tensor(
        policy_state_features_batch(states), dtype=torch.float64, device=device
    )
    penalty_features = torch.as_tensor(
        policy_state_features_batch(penalty_states),
        dtype=torch.float64,
        device=device,
    )
    basis = torch.as_tensor(
        initial.basis_matrix(critic.action_grid), dtype=torch.float64, device=device
    )
    matrix, matrix_receipt = penalty_matrix(initial, critic.action_grid, device)
    view = (
        critic if isinstance(critic, FastPredictionView) else FastPredictionView(critic)
    )
    objective = FixedStateSplineObjective(view, q_states, basis)
    objective.policy_penalty_matrix = matrix
    objective.policy_lambda = config.policy_lambda

    initial_parameters = initial.parameter_matrix
    identity = {
        "config": asdict(config),
        "q_state_count": len(states),
        "penalty_state_count": len(penalty_states),
        "action_grid": np.asarray(critic.action_grid).tolist(),
        "shape": list(initial_parameters.shape),
        "curvature_matrix_receipt": matrix_receipt,
    }
    parameters = torch.nn.Parameter(
        torch.as_tensor(initial_parameters.copy(), dtype=torch.float64, device=device)
    )
    optimizer = torch.optim.Adam([parameters], lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=getattr(config, "scheduler_threshold", 1.0e-5),
        threshold_mode="abs",
        min_lr=config.minimum_lr,
    )
    best = parameters.detach().clone()
    best_objective = -float("inf")
    trace = []
    start = 0
    previous_actions = None
    failing_probe_index = None
    if resume is not None:
        if not np.array_equal(resume["q_states"], states) or not np.array_equal(
            resume["penalty_states"], penalty_states
        ):
            raise RuntimeError("Cannot resume the optimizer with different states")
        if resume["identity"] != identity:
            raise RuntimeError("total-curvature optimizer resume identity mismatch")
        with torch.no_grad():
            parameters.copy_(resume["parameters"].to(device))
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        best = resume["best"].to(device)
        best_objective = float(resume["best_objective"])
        trace = list(resume["trace"])
        start = int(resume["next_step"])
        if resume["previous_actions"] is not None:
            previous_actions = resume["previous_actions"].to(device)
        if allow_failing_witness:
            failing_probe_index = resume.get("failing_probe_index")
    last_checkpoint = time.monotonic()
    reason = "budget_exhausted"

    def snapshot(next_step):
        state = {
            "identity": identity,
            "q_states": states.copy(),
            "penalty_states": penalty_states.copy(),
            "parameters": parameters.detach().cpu().clone(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best": best.detach().cpu().clone(),
            "best_objective": best_objective,
            "trace": list(trace),
            "next_step": next_step,
            "previous_actions": (
                None if previous_actions is None else previous_actions.detach().cpu()
            ),
        }
        if allow_failing_witness:
            state["failing_probe_index"] = failing_probe_index
        return state

    for step in range(start, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        q_coefficients = 2.0 * torch.tanh(q_features @ parameters)
        penalty_coefficients = 2.0 * torch.tanh(penalty_features @ parameters)
        actions = q_coefficients @ basis.T
        raw = objective.raw(q_coefficients)
        q_mean = raw.clamp(0.0, critic.value_max).mean()
        total_curvature = torch_uncentered_total_curvature(penalty_coefficients, matrix)
        value_tensor = q_mean - config.policy_lambda * total_curvature
        if not torch.isfinite(value_tensor):
            raise RuntimeError("nonfinite total-curvature policy objective")
        value_tensor.backward()
        if not torch.isfinite(parameters.grad).all():
            raise RuntimeError("nonfinite total-curvature policy gradient")
        value = float(value_tensor.detach())
        gradient_rms = float(parameters.grad.square().mean().sqrt())
        update_rms = (
            None
            if previous_actions is None
            else float((actions.detach() - previous_actions).square().mean().sqrt())
        )
        if value > best_objective:
            best_objective = value
            best = parameters.detach().clone()
        row = {
            "step": step,
            "objective": value,
            "best_objective": best_objective,
            "q_mean": float(q_mean.detach()),
            "uncentered_total_curvature": float(total_curvature.detach()),
            "weighted_policy_penalty": float(
                config.policy_lambda * total_curvature.detach()
            ),
            "policy_lambda": config.policy_lambda,
            "penalty_convention": config.penalty_convention,
            "penalty_state_count": int(len(penalty_states)),
            "gradient_rms": gradient_rms,
            "action_update_rms": update_rms,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "upper_clipping_fraction": float((raw >= critic.value_max).double().mean()),
            "lower_clipping_fraction": float((raw <= 0).double().mean()),
            "tanh_saturated_coefficient_fraction": float(
                (q_coefficients.detach().abs() >= 1.999).double().mean()
            ),
        }
        trace.append(row)
        window = trace[-config.window :]
        stationary = (
            step >= config.minimum_steps
            and len(window) == config.window
            and max(item["objective"] for item in window)
            - min(item["objective"] for item in window)
            <= config.objective_span_tolerance
            and all(
                item["action_update_rms"] is not None
                and item["action_update_rms"] <= config.action_update_rms_tolerance
                for item in window
            )
        )
        if stationary:
            with torch.no_grad():
                directions = [
                    value.reshape(best.shape)
                    for value in torch.eye(
                        best.numel(), dtype=best.dtype, device=device
                    )
                ]
                gradient_norm = torch.linalg.vector_norm(parameters.grad)
                if float(gradient_norm) > 0:
                    directions.append(parameters.grad / gradient_norm)
                early_rejection = False
                if allow_failing_witness:
                    (
                        improvements,
                        used_scalar_fallback,
                        failing_probe_index,
                        early_rejection,
                    ) = _witnessed_probe_improvements(
                        objective,
                        q_features,
                        penalty_features,
                        best,
                        directions,
                        config.probe_radii,
                        best_objective,
                        config.probe_improvement_tolerance,
                        failing_probe_index,
                    )
                    row["finite_probe_early_rejection"] = early_rejection
                else:
                    improvements, used_scalar_fallback = _probe_improvements(
                        objective,
                        q_features,
                        penalty_features,
                        best,
                        directions,
                        config.probe_radii,
                        best_objective,
                        config.probe_improvement_tolerance,
                    )
                row["probe_scalar_fallback"] = used_scalar_fallback
                if early_rejection:
                    row["finite_probe_improvement_lower_bound"] = max(improvements)
                else:
                    row["finite_probe_max_improvement"] = max(improvements)
                row["finite_probe_count"] = len(improvements)
                stationary = max(improvements) <= config.probe_improvement_tolerance
        if stationary or step == config.steps:
            reason = (
                "empirical_stability_and_finite_probe_check"
                if stationary
                else "budget_exhausted"
            )
            state = snapshot(step + 1)
            break
        parameters.grad.neg_()
        optimizer.step()
        scheduler.step(-value)
        previous_actions = actions.detach()
        due = (
            checkpoint is not None
            and time.monotonic() - last_checkpoint >= config.checkpoint_seconds
        )
        interrupt = stop_after is not None and step + 1 >= stop_after
        if due or interrupt:
            state = snapshot(step + 1)
            if due:
                checkpoint(state)
                last_checkpoint = time.monotonic()
            if interrupt:
                reason = "test_interruption"
                break
    else:
        state = snapshot(start)
        reason = "already_scored_all_steps"
    if checkpoint is not None:
        checkpoint(state)
    result = BoundedCoefficientBSplinePolicy(best.cpu().numpy(), initial.degree)
    return (
        result,
        trace,
        state,
        {
            "reason": reason,
            "best_objective": best_objective,
            "global_convergence_certified": False,
            "scored_iterates": len(trace),
            "penalty_matrix_receipt": matrix_receipt,
        },
    )
