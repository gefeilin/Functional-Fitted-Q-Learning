"""One identity-preserving Adam restart for theory total-curvature tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

from functional_fitted_q.runners.krr_prediction import BoundedCoefficientBSplinePolicy
from functional_fitted_q.runners.krr_gradient import bounded_adam_batched
from functional_fitted_q.runners.krr_optimizer import TheoryTotalCurvatureAdamConfig


@dataclass(frozen=True)
class TotalCurvatureRestartConfig(TheoryTotalCurvatureAdamConfig):
    steps: int = 20000
    learning_rate: float = 0.001
    scheduler_patience: int = 50
    scheduler_threshold: float = 0.0


def policy_at(state, initial):
    if not isinstance(initial, BoundedCoefficientBSplinePolicy):
        raise TypeError("total-curvature restart requires a functional policy")
    return BoundedCoefficientBSplinePolicy(
        state["best"].detach().cpu().numpy(), initial.degree
    )


def accepted_saved_state(state, config):
    trace = state["trace"]
    if not trace or config.gradient_gate:
        return False
    last = trace[-1]
    window = trace[-config.window :]
    return (
        state["next_step"] == last["step"] + 1
        and last["step"] >= config.minimum_steps
        and len(window) == config.window
        and max(row["objective"] for row in window)
        - min(row["objective"] for row in window)
        <= config.objective_span_tolerance
        and all(
            row["action_update_rms"] is not None
            and row["action_update_rms"] <= config.action_update_rms_tolerance
            for row in window
        )
        and not last.get("finite_probe_early_rejection", False)
        and math.isfinite(last.get("finite_probe_max_improvement", float("nan")))
        and last["finite_probe_max_improvement"] <= config.probe_improvement_tolerance
        and last.get("finite_probe_count", 0) >= 6 * state["best"].numel()
    )


def phase(
    critic,
    states,
    penalty_states,
    initial,
    config,
    resume,
    checkpoint,
    stop_after,
):
    if resume is not None:
        reason = (
            "empirical_stability_and_finite_probe_check"
            if accepted_saved_state(resume, config)
            else None
        )
        if reason is None and resume["next_step"] == config.steps + 1:
            reason = "budget_exhausted"
        if reason:
            state = copy.deepcopy(resume)
            state["termination_reason"] = reason
            return (
                policy_at(state, initial),
                state["trace"],
                state,
                {
                    "reason": reason,
                    "best_objective": state["best_objective"],
                    "global_convergence_certified": False,
                    "scored_iterates": len(state["trace"]),
                },
            )
    policy, trace, state, info = bounded_adam_batched(
        critic,
        states,
        initial,
        config,
        penalty_states=penalty_states,
        resume=resume,
        checkpoint=checkpoint,
        stop_after=stop_after,
        allow_failing_witness=True,
    )
    state["termination_reason"] = info["reason"]
    checkpoint(state)
    return policy, trace, state, info


def bounded_adam_with_restart(
    critic,
    states,
    initial,
    config,
    *,
    penalty_states,
    resume=None,
    checkpoint=None,
    stop_after=None,
    restart_config=None,
):
    """Run the base bounded-Adam phase and, when needed, its frozen restart."""
    if restart_config is None:
        raise ValueError("an explicit total-curvature restart config is required")
    state = {
        "schema": "bounded_adam_theory_total_curvature_restart_v1",
        "phase": "base",
        "base_state": None,
        "restart_state": None,
        "base_config": asdict(config),
        "restart_config": asdict(restart_config),
    }
    if resume is not None:
        if resume.get("schema") == state["schema"]:
            state = copy.deepcopy(resume)
            if state["base_config"] != asdict(config) or state[
                "restart_config"
            ] != asdict(restart_config):
                raise RuntimeError("total-curvature restart resume config mismatch")
        else:
            state["base_state"] = resume

    def save_base(value):
        state.update(
            phase="base",
            base_state=value,
            next_step=value["next_step"],
            best_objective=value["best_objective"],
        )
        if checkpoint is not None:
            checkpoint(state)

    def save_restart(value):
        state.update(
            phase="restart",
            restart_state=value,
            next_step=len(state["base_state"]["trace"]) + value["next_step"],
            best_objective=value["best_objective"],
        )
        if checkpoint is not None:
            checkpoint(state)

    if state["phase"] == "base":
        policy, base_trace, base_state, info = phase(
            critic,
            states,
            penalty_states,
            initial,
            config,
            state["base_state"],
            save_base,
            stop_after,
        )
        save_base(base_state)
        if info["reason"] != "budget_exhausted":
            return (
                policy,
                base_trace,
                state,
                {
                    **info,
                    "restart_used": False,
                    "base_scored_iterates": len(base_trace),
                    "restart_scored_iterates": 0,
                },
            )
        state["phase"] = "restart"
        if checkpoint is not None:
            checkpoint(state)
    else:
        base_state = state["base_state"]
        base_trace = base_state["trace"]
        if base_state["next_step"] != config.steps + 1 or accepted_saved_state(
            base_state, config
        ):
            raise RuntimeError("invalid base state before total-curvature restart")
        policy = policy_at(base_state, initial)

    policy, extra, extra_state, info = phase(
        critic,
        states,
        penalty_states,
        policy,
        restart_config,
        state["restart_state"],
        save_restart,
        stop_after,
    )
    save_restart(extra_state)
    trace = [
        {**row, "optimizer_phase": "base", "phase_step": row["step"]}
        for row in base_trace
    ]
    trace.extend(
        {
            **row,
            "optimizer_phase": "restart",
            "phase_step": row["step"],
            "step": len(base_trace) + row["step"],
        }
        for row in extra
    )
    if [row["step"] for row in trace] != list(range(len(trace))):
        raise RuntimeError("total-curvature combined optimizer trace is not contiguous")
    if info["best_objective"] < base_state["best_objective"] - 1.0e-9:
        raise RuntimeError("total-curvature restart lost the base best objective")
    return (
        policy,
        trace,
        state,
        {
            **info,
            "scored_iterates": len(trace),
            "restart_used": True,
            "base_scored_iterates": len(base_trace),
            "restart_scored_iterates": len(extra),
            "restart_config": asdict(restart_config),
        },
    )
