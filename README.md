# Functional-Action Fitted Q-Iteration

Simulation code for **Finite-Sample Theory for Fitted Q-Iteration When Actions
Are Functions**.

The experiments use a stochastic Pendulum in which each action is a torque
function. We fit functional-action policies with AdaFNN and Nyström KRR critics,
compare them with separately trained constant-action policies, and examine
critic differences at the learned actions.

You can reproduce the paper figures from the included results, or generate data
and train the models yourself. All data are synthetic.

## Installation

For training and the quickstart:

```bash
conda env create -f environment.yml
conda activate functional-fitted-q-theory
```

For figures and the analysis notebook only, a CPU environment is enough:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .
```

Use a separate environment if another installation provides the
`functional_fitted_q` package. The training environment includes PyTorch;
paper-scale training uses an NVIDIA GPU.

## Quick start

Run a small CPU example with the training environment:

```bash
python examples/quickstart_example.py
```

It generates 30 transitions, fits two FQI iterations, and saves a return summary
and example torque functions in `outputs/quickstart/`. These reduced settings
demonstrate the method; the paper settings are in `configs/paper_simulation.yaml`.

To draw the paper figures from the saved results:

```bash
python scripts/generate_simulation_figures.py
```

The six figures are saved as PDF and PNG files in `outputs/figures/`, with their
numerical summaries in `outputs/tables/`. See [results/README.md](results/README.md)
for the figure list and data columns.

For a clean end-to-end check of the supplied artifact, run:

```bash
make test reproduce verify notebook
```

## Training

Generate seed 0's offline data, then fit an AdaFNN policy using 2,000 transitions
and policy coefficient 0.001:

```bash
python scripts/generate_offline_data.py --seed 0
python scripts/run_functional_fqi.py --approximator adafnn --n 2000 --seed 0 --policy-lambda 0.001
```

For KRR or the constant-action comparator:

```bash
python scripts/run_functional_fqi.py --approximator krr --n 2000 --seed 0 --policy-lambda 0.001
python scripts/run_constant_fqi.py --n 2000 --seed 0
```

Run directories describe the experiment, for example:

```text
data/pendulum-seed00/
runs/candidates/adafnn/adafnn-n2000-lambda0.001-seed00/
runs/candidates/krr/krr-n2000-lambda0.001-seed00/
runs/constant/adafnn-constant-n2000-seed00/
```

The [reproduction guide](docs/reproducibility.md) covers the full grid, coefficient
selection, evaluation, and checkpoint analysis. A [Slurm example](docs/slurm.md)
is provided for submitting one fit. If a run is interrupted, repeat its command
with the same settings to resume from the last saved checkpoint.

## Repository structure

```text
configs/                  Experiment settings and task lists
src/functional_fitted_q/   Environment, policies, critics, FQI, and analysis
scripts/                  Commands for training, evaluation, and plotting
examples/                 Small example, analysis notebook, and Slurm example
data/                     Generated offline trajectories
results/source/           Per-run paper results and fit-level diagnostics
results/reference/        Paper figures and numerical summaries
docs/                     Experimental details and running instructions
tests/                    Tests for settings, selection, statistics, and resume
outputs/                  Generated figures and tables
```

## Experiments

The paper uses 2,000, 4,000, 8,000, 16,000, and 32,000 transitions, with 20 decisions
per subject and seeds 0–19. Each fit runs 20 FQI iterations at discount 0.95.
Policy coefficients are 0.0001, 0.001, 0.01, 0.1, and 1. Policy tuning and reporting
use separate Monte Carlo episodes; KRR critic ridge is selected by subject-grouped
cross-validation.

Details of the reward, architectures, curvature penalty, optimization, and
identification statistics are in [docs/experiments.md](docs/experiments.md).
[Compute requirements](docs/compute.md) and [data availability](data/README.md)
are documented separately. Fitted paper checkpoints are not included; the saved
results are sufficient for plotting, while checkpoint-level analysis requires
retraining. Identification evaluates Q19 and Q20 on an independent cohort of
`q = n = N x 20` states. The learned and observed behavior actions are evaluated
at each state in this cohort.

## Tests and analysis notebook

```bash
python -m unittest discover -s tests -v
python scripts/verify_results.py
python scripts/run_notebook.py
```

Run the figure command before `verify_results.py`. The notebook walks through
coefficient selection, bootstrap intervals, and critic diagnostics. For a short
training and interrupted-resume test, use the training environment:

```bash
python -m functional_fitted_q smoke
```

## Citation and license

Please cite the accompanying manuscript. `CITATION.cff` contains the software
citation information for anonymous review.

The code is released under the [MIT License](LICENSE).
