# Compute requirements

## Figures and notebook

The saved-result analysis runs on CPU with Python 3.12. It does not need PyTorch
or CUDA. It reads the released per-fit summaries rather than the full query-level
archive. `results/source/` occupies about 0.65 MiB, and reference figures and
summaries together occupy about 0.95 MiB in this release. Git history,
the Python environment, and newly generated outputs require additional space.
See the [result guide](../results/README.md) for the included tables.

## CPU examples

The following wall times were measured once on an Apple M2 with 16 GiB memory,
macOS 26.5.2, and Python 3.12.3. The environment was created from
`requirements-analysis.lock.txt`; the training examples additionally used
PyTorch 2.5.1 on CPU, with four PyTorch threads. These are small-workflow timings,
not estimates for the full experiment.

| Command | Measured wall time | Work performed |
| --- | --- | --- |
| `make check` | 37.2 s | Tests without PyTorch, six figures, numeric and pixel comparisons, and the seven-cell notebook |
| `make quickstart` | 3.0 s | Generate 30 transitions, fit two AdaFNN FQI iterations, and evaluate the policy |
| `make smoke` | 2.8 s | Small AdaFNN and KRR computations, constant-action fitting, and interrupted-resume checks |

Times include process startup and writing outputs, but exclude dependency
installation. The analysis run used a fresh Python environment with an existing
Matplotlib cache. The training timings followed environment installation and
unit tests. Hardware, caches, and concurrent processes affect wall time.

To measure wall time on your own machine:

```bash
time make check
time make quickstart
```

For peak resident memory as well, use `/usr/bin/time -v make check` on Linux
(GNU time), or `/usr/bin/time -l make check` on macOS. The reported memory and
time cover this CPU workflow; use a GPU profiler or scheduler accounting for
full-training resource measurements.

## Training

The [Slurm example](../examples/slurm/functional_fqi.sbatch) requests one GPU,
8 CPUs, 64 GB RAM, and 24 hours. Adjust these for your cluster. A100 is not
required; compatible V100-class hardware can also be used.

The full experiment consists of 20 offline data pools, 500 functional fits per
approximator, 100 constant-action fits, and the subsequent evaluations. Runtime
depends on sample size, GPU, and optimizer stopping times. In particular, KRR
rank increases with sample size. The original scheduler wall-time receipts are
not part of the scientific result tables, so this package does not claim a
measured per-fit training time. Time one small and one large fit on the target
hardware before scheduling the full grid; the example's 24-hour limit is a
scheduler ceiling, not a runtime estimate.

Data pools are shared across coefficients. Each fit keeps two full checkpoints
and one in-progress checkpoint, together with training traces and Monte Carlo
returns. The action-bandwidth cache also keeps its input arrays so that it can
check that a cached value belongs to the same data. Measure checkpoint sizes
for both approximators before choosing how many fits to run concurrently.
