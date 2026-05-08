# Reproducibility

## Fast Smoke Test

Use the quickstart to verify that installation and the core workflow work:

```bash
python examples/quickstart_example.py
pytest
```

This does not reproduce paper-scale numerical results. It intentionally uses a tiny dataset and few optimizer iterations.

## Functional FQI Simulations

Paper-scale functional FQI jobs use:

```bash
python scripts/run_functional_fqi.py \
  --config configs/paper_simulation.yaml \
  --horizon <5|10|20> \
  --gamma <0.7|0.8|0.9|0.99> \
  --size <100|200|400> \
  --iteration <replicate-index>
```

The output directory defaults to `outputs/functional_fqi/`.

## Functional FQE Simulations

Functional FQE jobs use:

```bash
python scripts/run_functional_fqe.py \
  --config configs/paper_simulation.yaml \
  --horizon <5|10|20> \
  --gamma <0.7|0.8|0.9|0.99> \
  --size <100|200|400> \
  --iteration <replicate-index>
```

The released FQE summaries under `results/fqe/` were copied as lightweight public summaries. Full intermediate ground-truth chunks and logs are excluded.

## Scalar Benchmarks

Scalar-action benchmarks require optional d3rlpy dependencies:

```bash
pip install -e ".[benchmarks]"
python scripts/run_d3rlpy_baseline.py DDPG \
  --horizon 5 --gamma 0.8 --size 100 --iteration 0
```

The scalar benchmarks compress each functional action into a one-dimensional action summary and are included only for simulation comparison.

## Figures And Tables

Create public simulation summaries:

```bash
python scripts/generate_simulation_figures.py --gamma 0.8
python scripts/summarize_fqe_results.py
```

Generated files are written to `outputs/figures/` and `outputs/tables/`.

## Randomness

Simulation jobs derive replicate seeds from the base seed, sample size, replicate index, and device index. The quickstart sets NumPy and PyTorch seeds directly.
