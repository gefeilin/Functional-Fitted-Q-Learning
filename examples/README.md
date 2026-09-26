# Examples

## Quick Start

```bash
python examples/quickstart_example.py
```

This CPU example generates 30 transitions, fits two small AdaFNN FQI iterations,
and evaluates the learned functional policy. PyTorch is required. Settings are
in `configs/quickstart.yaml`; the paper configuration is separate.

Outputs:

- `data/example/toy_pendulum_transitions.json`: states, functional actions on a
  shared grid, rewards, next states, subject identifiers, time indices, and
  terminal indicators (always false for this continuing MDP).
- `outputs/quickstart/quickstart_summary.csv`: dataset and evaluation summary.
- `outputs/quickstart/learned_toy_actions.png`: example learned torque functions.

The reduced integration grid, network sizes, and optimization budgets make this
a tutorial, not a paper-scale replication. Its sample sizes and evaluation
horizon differ from the paper.

For a guided reading of the implementation, see the
[source-code guide](../docs/code-guide.md) and [notation guide](../docs/notation.md).

## Paper Analysis Notebook

`paper_results.ipynb` explains coefficient selection, bootstrap intervals, action
locality, and adjacent-critic energy ratios using the supplied simulation results.
Execute it with:

```bash
python scripts/run_notebook.py
```

The executed copy is saved to `outputs/paper_results.executed.ipynb`, preserving
the example in this directory. Open the executed copy to read its tables and
figures. `make notebook` runs the same command.

The notebook requires the analysis environment, not PyTorch or CUDA. Full
experiment commands are in [the reproduction guide](../docs/reproducibility.md).

## Slurm Submission

`slurm/functional_fqi.sbatch` is a single-GPU example for one functional FQI task.
It accepts the approximator (`adafnn` or `krr`) and task index as arguments.
See [the submission instructions](../docs/slurm.md) for prerequisites and usage.
