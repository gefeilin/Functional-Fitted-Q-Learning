# Results

The tables in `source/` contain the per-run results behind the paper figures.
Runs have readable names such as `adafnn-n8000-lambda0.001-seed00`. This name is
stored in the `fit_id` column, which links the return and diagnostic tables.
Functional and constant-action results are paired by sample size and master seed.

| File | Scientific unit / use |
|---|---|
| `tuning_candidate_values.csv` | 1,000 candidates; `tuning_J_normalized_mean` selects lambda |
| `selected_lambdas.csv` | 200 approximator/n/seed cells; selected fit IDs |
| `reporting_values.csv` | 360 unique fits; independent `J_normalized_mean` |
| `constant_returns.csv` | 100 separately trained AdaFNN constants |
| `identification_proxy_per_fit.csv` | locality, leverage, and adjacent-critic summaries for 360 fits |
| `neighbor_sensitivity_per_fit.csv` | AdaFNN action-distance summaries for k=16, 32, 64, and 128 |
| `identification_design.json` | checkpoint pair and independent q=n evaluation design |
| `roughness.csv` | returned AdaFNN policy curvature and optimizer floor |

Key columns:

- `approximator`: `adafnn` or `nystrom_krr`.
- `n_transitions`: N times T, not the number of subjects.
- `master_seed`: training/data replicate, integers 0–19.
- `lambda_dimensionless`: policy-curvature grid coefficient, not critic ridge.
- `selected_sample_size_view`: selected positive coefficient for that cell.
- `fixed_n_all_lambda_view`: all five candidates at n=8,000, including the selected
  one. These flags overlap; do not concatenate both views without deduplication.
- `evaluation_seed`: independent reporting stream identity for pairing.
- `d_min_median_learned`, `d_min_median_behavior`: within-fit median functional
  action L2 distances among the 32 nearest training states.
- `relative_representation_leverage_median_learned`: within-fit median
  learned-action leverage normalized by the same fit's median held-out behavior
  leverage.
- `D_actual_sq`, `G_actual_sq`: actual clipped-critic difference squared energies
  on logged and policy designs.
- `actual_graph_design_ratio`: G²/D², with no denominator regularization.
- `raw_total_curvature_penalty`: uncentered integral of squared second derivative,
  averaged over training next states; not multiplied by lambda.

Every fit uses one independent cohort with `q = n = N x 20` evaluation states.
Each listed action class is evaluated at these states. Query rows are summarized
within each fit and are not treated as independent training replicates. Raw query
tables remain with the full experiment archive; this public package includes the
fit-level quantities needed to reproduce the reported figures.

## Figures

Run `python scripts/generate_simulation_figures.py` from the repository root.
Each figure is written to `outputs/figures/` in PDF and PNG format.

| Filename | Content |
| --- | --- |
| `main_figure` | AdaFNN value, action locality, and adjacent-critic energy ratio |
| `krr_value` | KRR return versus sample size |
| `krr_identification` | KRR counterparts of the main identification diagnostics |
| `neighbor_sensitivity` | AdaFNN action locality for k=16, 32, 64, and 128 |
| `paired_value_difference` | Within-seed functional minus constant-action return |
| `critic_update_energies` | Separate graph and logged-design critic energies |

The full n=8,000 coefficient sweep is available in the per-run tables. Numeric
figure summaries are written to `outputs/tables/`; `reference/` holds the paper
figures and summaries for comparison. `figures.json` lists the input files for
each plot.

The figure script recomputes coefficient selection and bootstrap intervals from
the supplied per-run data. It reads independent reporting returns for the plots;
the tuning means are used only for selecting the coefficient.

## Results from new training

Training saves episode returns in `.npz` files and critic/optimizer traces in
Parquet files. `python scripts/aggregate_results.py` collects these into the same
table format. Keep new outputs separate from `source/` when comparing a new run
with the paper results. Full checkpoints and episode-level returns are not part
of the supplied result bundle.
