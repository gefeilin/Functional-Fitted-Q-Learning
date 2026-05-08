# Tutorial

This tutorial describes the synthetic quickstart workflow.

## 1. Generate Toy Functional-Action Data

```bash
python examples/quickstart_example.py
```

The example simulates short pendulum trajectories. Each transition contains:

- a three-dimensional state vector,
- a smooth torque trajectory on a grid over `[0, 1]`,
- a normalized reward,
- the next state, and
- a terminal indicator.

The generated toy dataset is written to `data/example/toy_pendulum_transitions.json`.

## 2. Represent Functional Actions

Functional actions are stored as grid-valued vectors. The FQI policy class maps a state vector into B-spline coefficients, then evaluates the spline basis on the action grid. The quickstart uses a small cubic B-spline basis; paper-scale simulations use a larger basis.

## 3. Fit Functional FQI

The quickstart calls `Fitted_Q_Iteration` with:

- kernel ridge regression for the Q-function,
- one fitted-Q update,
- a smoothness penalty on the learned functional policy, and
- fixed random seeds for reproducibility.

Paper-scale runs use `scripts/run_functional_fqi.py` with `configs/paper_simulation.yaml`.

## 4. Evaluate A Policy

The example evaluates the learned policy with `Fitted_q_evaluation`, which keeps the policy fixed and iterates the Bellman evaluation update. For stochastic policies, the implementation can average over sampled functional actions.

## 5. Inspect Outputs

The quickstart writes:

- `outputs/quickstart/quickstart_summary.csv`
- `outputs/quickstart/learned_toy_actions.png`

The figure plots learned functional torque trajectories for the initial toy states.
