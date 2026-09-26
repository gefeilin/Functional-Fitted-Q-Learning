# Running the experiments

If you only want the paper plots, run:

```bash
python scripts/generate_simulation_figures.py
```

This uses the tables in `results/source/` and writes six figures to
`outputs/figures/`. It does not need a GPU or fitted model weights.
`python scripts/run_notebook.py` runs the same analysis with intermediate tables
and explanations. To compare the generated numbers with the supplied paper
summaries, run `python scripts/verify_results.py` after plotting.

The remaining sections describe training from scratch. Use the full environment
from the README and an allocated GPU for fitting.

## 1. Generate the offline data

```bash
python scripts/generate_offline_data.py --seed 0
```

Repeat for seeds 0–19. Each seed produces a pool of 2,500 subjects with 20
decisions each, saved under `data/pendulum-seed00/`, and so on. Training takes
nested prefixes of 100, 200, 400, 800, or 1,600 subjects to obtain the five sample
sizes. Generate the full pool even if you only need a small prefix: changing
the pool size changes the order in which the simulator draws random numbers.

## 2. Fit the policies

For example, these commands fit the two functional-action models and the
constant-action comparator at $`n=8{,}000`$, seed 0:

```bash
python scripts/run_functional_fqi.py --approximator adafnn --n 8000 --seed 0 --policy-lambda 0.001
python scripts/run_functional_fqi.py --approximator krr --n 8000 --seed 0 --policy-lambda 0.001
python scripts/run_constant_fqi.py --n 8000 --seed 0
```

The `--policy-lambda` argument is the dimensionless ratio
$`\lambda_{\Omega,n}/s_\Omega`$; see [coefficient units](notation.md#policy-objective-and-coefficient-units).

For each approximator, train all five policy coefficients (0.0001, 0.001, 0.01,
0.1, 1) at each sample size and seed. This gives 500 fits per approximator.
The 100 constant-action fits use AdaFNN and do not have a curvature coefficient.
All three methods share the corresponding seed's offline data.

Each functional fit includes a tuning evaluation. Each constant fit includes
its reporting evaluation. Model checkpoints and training traces are saved in
the run directory; for example,
`runs/candidates/adafnn/adafnn-n8000-lambda0.001-seed00/`.

Independent fits can run in parallel. Use [the Slurm example](slurm.md) if
submitting to a cluster. `python scripts/plan_experiments.py` can prepare task
lists for a batch scheduler; the explicit commands above remain the clearest
way to reproduce an individual fit.

## 3. Select the coefficient and evaluate

Once all five candidates for a sample size and seed finish, select the one with
the highest tuning return:

```bash
python scripts/select_policy_coefficient.py --approximator adafnn --n 8000 --seed 0
```

The chosen run and the five tuning means are saved in
`runs/selection/adafnn-n8000-seed00/selection.json`. Exact ties favor the smaller
coefficient. Reporting uses a separate set of 1,000 simulated episodes.

At $`n=8{,}000`$, evaluate every coefficient for the saved coefficient-sweep tables:

```bash
python scripts/evaluate_policy.py --approximator adafnn --n 8000 --seed 0 --policy-lambda 0.001
python scripts/analyze_identification.py --approximator adafnn --n 8000 --seed 0 --policy-lambda 0.001
```

Repeat these commands for each coefficient, seed, and approximator. At the other
sample sizes, evaluate and analyze only the selected coefficient. The analysis
evaluates $`\widehat Q_{19}`$ and $`\widehat Q_{20}`$ on an independent cohort. A cell with $`n=NT`$
transitions uses $`N`$ new trajectories of $`T=20`$ decisions, giving $`q=n`$ evaluation
states. The learned and observed behavior actions are evaluated at each state.
Across both approximators this requires 360 unique reporting runs and 360
identification analyses. The selected $`n=8{,}000`$ run is reused in the sample-size
curve rather than evaluated twice.

## 4. Collect results and draw figures

After completing the grid:

```bash
python scripts/aggregate_results.py --output outputs/new_training
python scripts/generate_simulation_figures.py --source outputs/new_training/source --output outputs/new_training
```

The aggregation combines tuning, reporting, constant-action, and diagnostic
outputs. It keeps your new results separate from the supplied paper results.
See [the result guide](../results/README.md) for the tables and metric definitions.

## Resuming and changing settings

To resume an interrupted fit, repeat its command using the same data and run
directory. The checkpoint contains the model, optimizer, and random-number state.
Only the latest two full checkpoints and one in-progress checkpoint are kept;
a finished run retains Q19 and Q20.

Use `--runs runs/my_experiment` when changing training settings so that new
outputs do not overwrite an existing fit. The resume checks compare the saved
settings and file sizes; keep the original data and code when continuing a run.
Numerical results from a new fit can differ across GPUs and library versions.
