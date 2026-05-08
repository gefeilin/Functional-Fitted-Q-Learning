# Functional-Action Fitted Q-Learning

Companion code for a paper on offline reinforcement learning with functional actions. The method represents each action as a function over a grid, estimates Q-functions with kernel ridge regression, and performs fitted Q-evaluation and fitted Q-iteration in the functional-action setting.

This anonymized review release contains the implementation, configuration files,
synthetic examples, lightweight released simulation summaries, and documentation
needed to regenerate the public figures and tables. Restricted real-data
materials are not included.

## Installation

```bash
# Clone from the anonymous review URL supplied by the review system,
# or download and unpack the anonymized review archive.
git clone <anonymous-review-repository-url>
cd Functional-Fitted-Q-Learning

conda env create -f environment.yml
conda activate functional-fitted-q
pip install -e .
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The scalar-action benchmark scripts require the optional `d3rlpy` package:

```bash
pip install -e ".[benchmarks]"
```

## Quick Start

Run a small synthetic example from the repository root. The quickstart uses the
default 100-point functional-action grid and requires a CUDA-capable PyTorch
install:

```bash
python examples/quickstart_example.py
```

This creates:

- `data/example/toy_pendulum_transitions.json`
- `outputs/quickstart/quickstart_summary.csv`
- `outputs/quickstart/learned_toy_actions.png`

The quickstart uses deliberately tiny episode counts and training iterations so
it can run quickly. It is a smoke test and tutorial, not a paper-scale
replication.

## Components

| Component | Path | Purpose |
| --- | --- | --- |
| Core package | `src/functional_fitted_q/` | Implements environments, kernels, fitted Q-iteration, fitted Q-evaluation, runners, and visualization helpers. |
| Configurations | `configs/` | Stores reproducible quickstart and paper-scale parameter settings. |
| Command-line workflows | `scripts/` | Runs functional FQI/FQE jobs, scalar-action baselines, released-result summaries, and figure/table generation. |
| Quickstart example | `examples/quickstart_example.py` | Runs a small GPU smoke test with synthetic functional-action data. |
| Public toy data | `data/example/` | Contains the generated synthetic example transition data used by the quickstart. |
| Released summaries | `results/` | Contains lightweight public simulation and FQE summaries used to regenerate paper figures and tables. |
| Documentation | `docs/` | Maps paper results to public code and explains reproducibility details. |
| Tests | `tests/` | Provides regression and smoke tests for data generation, kernels, algorithms, and quickstart outputs. |

## Repository Structure

```text
configs/                  YAML configuration files
src/functional_fitted_q/   Core algorithms, environments, kernels, and helpers
scripts/                  Command-line workflows for simulations and summaries
examples/                 End-to-end toy examples
data/                     Public toy-data notes and generated example data
docs/                     Tutorial, reproducibility notes, and result mapping
results/                  Lightweight released result summaries
outputs/                  Generated local outputs, ignored by git
tests/                    Lightweight regression and smoke tests
```

## Main Commands

Run a functional fitted Q-iteration job:

```bash
python scripts/run_functional_fqi.py \
  --config configs/paper_simulation.yaml \
  --horizon 5 --gamma 0.8 --size 100 --iteration 0
```

Run a functional fitted Q-evaluation job:

```bash
python scripts/run_functional_fqe.py \
  --config configs/paper_simulation.yaml \
  --horizon 5 --gamma 0.8 --size 100 --iteration 0
```

Generate simulation summaries and a comparison figure from released score files:

```bash
python scripts/generate_simulation_figures.py --gamma 0.8
```

Summarize released FQE RMSE tables:

```bash
python scripts/summarize_fqe_results.py
```

Run an optional scalar benchmark:

```bash
python scripts/run_d3rlpy_baseline.py DDPG \
  --horizon 5 --gamma 0.8 --size 100 --iteration 0
```

## Key Parameters

The main configuration files are `configs/quickstart.yaml` for a small smoke
test and `configs/paper_simulation.yaml` for paper-scale simulation settings.

| Parameter | Meaning |
| --- | --- |
| `horizon` | Number of decision cycles per simulated trajectory. |
| `gamma` | Discount factor used by FQI/FQE and Monte Carlo value scoring. |
| `size` | Number of simulated trajectories or subjects in the training dataset. |
| `iteration` | Batch index used for seeding and output naming. |
| `run_tag` | Label used to organize saved models and result directories. |
| `num_action_grid_points` | Number of grid points used to represent each functional action. |
| `num_basis_knots` | Number of knots used to construct the B-spline action basis. |
| `spline_degree` | Degree of the B-spline basis for functional action optimization. |
| `lambda_grid_size`, `lambda_min`, `lambda_max` | Grid used to select the spline smoothness penalty by cross-validation. |
| `num_cv_splits` | Number of cross-validation folds for penalty selection. |
| `fqi_iterations` | Number of fitted Q-iteration updates. |
| `training_iterations` | Maximum optimization iterations for GCV and policy optimization subproblems. |
| `fqe_simulations`, `fqe_cycles` | Monte Carlo settings used when scoring learned policies. |
| `fqe_eval_states` | Number of initial states used for FQE value summaries. |
| `fqe_train_expectation_samples`, `fqe_eval_expectation_samples` | Number of policy-action samples used for FQE expectation estimates. |
| `fqe_clamp_targets` | Whether FQE targets are clamped to the configured value bounds. |
| `learning_rate` | Optimizer learning rate for GCV and policy optimization. |
| `early_stopping_patience`, `early_stopping_delta` | Early-stopping controls for iterative optimization. |

## Data

The public repository includes only synthetic toy data generated by the quickstart. For user-supplied studies, each transition should contain a state vector, functional action values on a common grid, reward, next-state vector, time index, subject or trajectory identifier, and terminal indicator. See `data/README.md`.

## Paper Results

The released code maps to the simulation and FQE workflows in the paper. Lightweight simulation score files are included under `results/` for figure/table regeneration. Restricted real-data application outputs are documented but not included or reproduced publicly. See `docs/paper_results_mapping.md`.

## Testing

```bash
pytest
```

The tests check toy data generation, basis construction, kernel ridge regression, a tiny FQE/FQI run, and the quickstart output contract.

## Troubleshooting

- If `gym` emits a deprecation warning, the simulations can still run; the code preserves the original paper environment.
- On V100-class GPUs, install a PyTorch release below 2.6; newer wheels may not include kernels for compute capability 7.0.
- If benchmark imports fail, install the optional benchmark dependencies.
- If a paper-scale run is slow, reduce `training_iterations`, `fqi_iterations`, or sample size for a diagnostic run, then restore the paper config.
- Generated outputs are written under `outputs/` and ignored by git.

## Citation

For peer review, citation metadata are anonymized. Final citation information will be added after peer review.

## License

For peer review, license-holder information is anonymized. Final license metadata will be added after peer review.

## Contact

Maintainer contact information will be added after peer review.
