# Slurm submission example

[functional_fqi.sbatch](../examples/slurm/functional_fqi.sbatch) submits one
functional-action FQI fit. It requests one GPU, 8 CPUs, 64 GB of host memory,
and a 24-hour time limit. Adjust the partition and resource directives for your
cluster; no specific GPU model is required by the example.

Before submission, install and activate the [training environment](../README.md#installation)
and generate the data pool required by the chosen task. Run data generation and
training on compute nodes. First prepare the scheduler task lists:

```bash
python scripts/plan_experiments.py
```

This writes `runs/tasks/adafnn.json` and `runs/tasks/krr.json`. Each file contains
the 500 functional fits for that approximator: five sample sizes, 20 seeds, and
five policy-curvature coefficients.

From the repository root, submit an AdaFNN fit:

```bash
sbatch examples/slurm/functional_fqi.sbatch adafnn 0
```

For KRR, use the same example with a different approximator:

```bash
sbatch examples/slurm/functional_fqi.sbatch krr 0
```

The arguments are the approximator (`adafnn` or `krr`) and its zero-based task
index. The index selects the corresponding row of that approximator's task-list
JSON file; inspect that row to see the sample size, seed, and coefficient used
by a job. This index is only a scheduler convenience and is not part of the
scientific experiment definition. The example inherits the active environment
and writes its log to `slurm-<job-id>.out`. Training outputs are written under
`runs/`.

To resume an interrupted fit, resubmit the same command with the same
configuration and run directory. Resume starts from the last durable checkpoint.
Avoid submitting the same task twice concurrently. This example submits only
the requested fit; data generation and post-training analysis use the commands
in the reproduction guide.
