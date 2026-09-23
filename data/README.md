# Data and models

All experiments use synthetic trajectories from the included Pendulum
simulator. The supplied results in `results/source/` are enough to reproduce
the paper figures. Fitted model weights and episode-level Monte Carlo returns
are not included.

To generate the offline trajectories for one seed:

```bash
python scripts/generate_offline_data.py --seed 0
```

The output directory is `data/pendulum-seed00/`. It contains NumPy arrays for
states, functional actions, rewards, next states, subject identifiers, time
indices, and the subject ordering used for nested samples.

Generate the full pool of 2,500 subjects with 20 decisions even when fitting a
smaller sample. Training takes the required subject prefix; changing the initial
pool size would change random-number consumption.

To obtain fitted models, follow [the training instructions](../docs/reproducibility.md).
Each completed fit retains Q19 and Q20, including policy parameters. Interrupted
runs also keep the optimizer and random-number state needed to resume. Keep
the data and settings unchanged when resuming, and only load checkpoints from
sources you trust.
