# Compute requirements

## Figures and notebook

The saved-result analysis runs on CPU with Python 3.12. It does not need PyTorch
or CUDA. It reads the released per-fit summaries rather than the full query-level
archive. `results/source/` occupies about 0.65 MiB, and reference figures and
summaries together occupy about 0.95 MiB in this release. Git history,
the Python environment, and newly generated outputs require additional space.
See the [result guide](../results/README.md) for the included tables.

Use the following command to measure the complete CPU artifact workflow on the
target system:

```bash
/usr/bin/time -v make test reproduce verify notebook
```

Wall time is hardware dependent; the repository does not present one machine's
analysis time as a portable benchmark.

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
