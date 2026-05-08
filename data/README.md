# Data

This public release does not include restricted real-data records. The quickstart generates a synthetic toy dataset in `data/example/toy_pendulum_transitions.json`.

## Expected Transition Format

Each row or record should contain:

- `subject`: trajectory or subject identifier
- `time`: decision time within the trajectory
- `state`: numeric state vector at the decision time
- `functional_action`: action values on a common grid over `[0, 1]`
- `reward`: scalar reward
- `next_state`: numeric next-state vector
- `terminal`: whether the row is the final transition in the trajectory

The core code expects equivalent pandas columns named `Subject`, `Time`, `State`, `FunctionalData`, `Reward`, `Next_State`, and `Status`.

## Public Toy Data

The toy data are generated from a modified pendulum simulator. Functional actions are smooth torque trajectories. They are intended for testing software behavior and explaining the workflow, not for reproducing paper-scale numerical results.

## User Data

To use restricted or external data, transform each trajectory into the transition format above and keep non-public files outside the repository or under ignored directories such as `data/private/` or `data/raw/`.
