# Results

The tables in `source/` contain the per-run results behind the paper figures.
Runs have readable names such as `adafnn-n8000-lambda0.001-seed00`. This name is
stored in the `fit_id` column, which links the return and diagnostic tables.
Functional and constant-action results are paired by sample size and master seed.

| File | Scientific unit / use |
|---|---|
| `tuning_candidate_values.csv` | 1,000 candidates; `tuning_J_normalized_mean` selects $`\lambda_{\Omega,n}/s_\Omega`$ |
| `selected_lambdas.csv` | 200 approximator/n/seed cells; selected fit IDs |
| `reporting_values.csv` | 360 unique fits; independent `J_normalized_mean` |
| `constant_returns.csv` | 100 separately trained AdaFNN constants |
| `identification_proxy_per_fit.csv` | locality, leverage, and adjacent-critic summaries for 360 fits |
| `neighbor_sensitivity_per_fit.csv` | AdaFNN action-distance summaries for $`k\in\{16,32,64,128\}`$ |
| `identification_design.json` | checkpoint pair and independent $`q=n`$ evaluation design |
| `roughness.csv` | returned AdaFNN policy curvature and optimizer floor |

## Column definitions

Symbols follow the [experiment description](../docs/experiments.md).
The [source-code guide](../docs/code-guide.md) maps them to Python names.

- `approximator`: `adafnn` or `nystrom_krr`.
- `n_transitions`: $`n=NT`$, the number of transitions; $`N`$ counts subjects.
- `master_seed`: training/data replicate, integers 0–19.
- `lambda_dimensionless`: $`\lambda_{\Omega,n}/s_\Omega`$, the policy-curvature grid coefficient; this is separate from critic ridge.
- `selected_sample_size_view`: selected positive coefficient for that cell.
- `fixed_n_all_lambda_view`: all five candidates at $`n=8{,}000`$, including the selected
  one. These flags overlap; do not concatenate both views without deduplication.
- `evaluation_seed`: independent reporting stream identity for pairing.
- `d_min_median_learned`, `d_min_median_behavior`: within-fit median functional
  action $`L^2`$ distances among the 32 nearest training states.
- `relative_representation_leverage_median_learned`: within-fit median
  learned-action leverage normalized by the same fit's median held-out behavior
  leverage.
- `D_actual_sq`, `G_actual_sq`: $`D_h^2`$ and $`G_h^2`$, the squared energies of
  $`h=\widehat Q_{20}-\widehat Q_{19}`$ on logged and learned-policy designs.
- `actual_graph_design_ratio`: $`\mathsf{ER}_h=G_h^2/D_h^2`$, with no denominator regularization.
- `raw_total_curvature_penalty`: $`\widehat\Omega_{\mathcal D}(\pi_C)`$, the uncentered integral of squared second derivative,
  averaged over training next states; the coefficient has not been applied.

Every fit uses one independent cohort with $`q=n=NT`$ evaluation states, where $`T=20`$.
Each listed action class is evaluated at these states. Query rows are summarized
within each fit and are not treated as independent training replicates. Raw query
tables remain with the full experiment archive; this public package includes the
fit-level quantities needed to reproduce the reported figures.

## Figures

Run `python scripts/generate_simulation_figures.py` from the repository root.
The repository contains exactly the six experimental figures used in the paper.
Each figure is written to `outputs/figures/` in PDF and PNG format; the PDF is
the manuscript asset and the PNG supports browser preview and pixel validation.

| Repository artifact | Manuscript file | Content |
| --- | --- | --- |
| `main_figure` | `fig01_adafnn_main.pdf` | AdaFNN value, action locality, and adjacent-critic energy ratio |
| `krr_value` | `figS01_krr_value.pdf` | KRR return versus sample size |
| `paired_value_difference` | `figS06_paired_return_difference.pdf` | Within-seed functional minus constant-action return |
| `neighbor_sensitivity` | `figS08_adafnn_neighbor_sensitivity.pdf` | AdaFNN action locality for $`k\in\{16,32,64,128\}`$ |
| `krr_identification` | `figS03_krr_identification.pdf` | KRR counterparts of the main identification diagnostics |
| `critic_update_energies` | `figS07_critic_update_energies.pdf` | Separate graph and logged-design critic energies |

The mapping is also recorded in `results/figures.json`. Files in
`results/reference/figures/` are the reference renderings used by the automated
result check; no additional experimental figure is included.

The full $`n=8{,}000`$ coefficient sweep is available in the per-run tables. Numeric
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
