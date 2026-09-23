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
| `identification_query_metrics.parquet` | one held-out query/class/split/fit |
| `identification_secant_metrics.parquet` | aggregate energies per fit/class/split |
| `identification_spectrum.parquet` | representation eigenvalue/ridge records |
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
- `split`: A or B; the paper uses B.
- `query_class`: learned, previous, behavior, or reference action at held-out states.
- `d_min`: minimum functional action L2 distance among 32 nearest training states.
- `relative_representation_leverage`: leverage divided by the corresponding
  fit/split's median held-out behavior leverage.
- `D_actual_sq`, `G_actual_sq`: actual clipped-critic difference squared energies
  on logged and policy designs.
- `actual_graph_design_ratio`: G²/D², with no denominator regularization.
- `raw_total_curvature_penalty`: uncentered integral of squared second derivative,
  averaged over training next states; not multiplied by lambda.

Spectrum/projection/ridge fields beyond those used in the paper remain available
as recorded diagnostics. Their presence is not a claim that all are theorem
constants or that all should appear as manuscript plots.

## Figures

Run `python scripts/generate_simulation_figures.py` from the repository root.
Each figure is written to `outputs/figures/` in PDF and PNG format.

| Filename | Content |
| --- | --- |
| `main_value` | AdaFNN functional and constant-action returns versus sample size |
| `main_identification` | Action locality and adjacent-critic energy ratio for AdaFNN |
| `krr_value` | KRR return versus sample size |
| `krr_identification` | KRR counterparts of the main identification diagnostics |
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
