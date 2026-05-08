# Paper Results Mapping

This file maps paper components to the public code when a clear mapping exists.

## Method Implementation

- Functional action simulator: `src/functional_fitted_q/envs.py`
- Rewards and behavior policy: `src/functional_fitted_q/common.py`
- Kernel ridge regression and GCV: `src/functional_fitted_q/kernels.py`
- Functional FQI and FQE: `src/functional_fitted_q/algorithms.py`
- Paper-scale runners: `src/functional_fitted_q/runner.py`

## Main Simulation Figures

The policy-learning figures for discount factors `0.7`, `0.8`, `0.9`, and `0.99` correspond to:

- Functional method: `scripts/run_functional_fqi.py`
- Scalar baselines: `scripts/run_d3rlpy_baseline.py`
- Released summaries: `results/functional-fqi/` and `results/d3rlpy/`
- Public figure regeneration: `scripts/generate_simulation_figures.py`

The main manuscript figure uses the `gamma=0.8` setting. Supplementary figures use the other released gamma settings.

## FQE Supplement

The FQE simulation table corresponds to:

- FQE runner: `scripts/run_functional_fqe.py`
- Released RMSE summary: `results/fqe/fqe_rmse_by_size_setting_recomputed_ground_truth.csv`
- Released ground-truth summary: `results/fqe/fqe_ground_truth_final_summary.csv`
- Public summary script: `scripts/summarize_fqe_results.py`

The public release keeps the final recomputed FQE summaries used for reporting and excludes intermediate ground-truth files.

## Restricted Real-Data Application

The manuscript includes a real-data application with functional distributional actions. The underlying study data are restricted, so those figures and tables are documented as non-public and are not reproduced from this repository.

## Unmapped Or Excluded Material

- Manuscript drafts and TeX build artifacts are excluded.
- Cluster submission scripts are excluded or represented only by command-line Python workflows.
- Large logs, checkpoints, and training artifacts are excluded.
- Duplicated exploratory utilities are excluded from the public workflow.
