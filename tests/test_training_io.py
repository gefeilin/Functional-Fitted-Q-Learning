"""Small tests for data creation, bandwidth caching, and checkpoint loading."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Requires the training environment"
)
class TrainingIOTests(unittest.TestCase):
    def test_data_generation_completes_before_reader_loads(self):
        from functional_fitted_q.data import OfflineDataset, verify_offline_dataset
        from functional_fitted_q.runners import generate_data
        from functional_fitted_q.runtime import project_root

        data = OfflineDataset(
            np.zeros((6, 3)),
            np.zeros((6, 4)),
            np.ones(6),
            np.zeros((6, 3)),
            np.repeat([0, 1], 3),
            np.tile(np.arange(3), 2),
            np.arange(2),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(
                generate_data, "generate_offline_dataset", return_value=data
            ):
                generate_data.run(project_root(), Path(temporary), 0)
            directory = Path(temporary) / "pendulum-seed00"
            self.assertTrue((directory / "_SUCCESS").is_file())
            self.assertEqual(verify_offline_dataset(directory)["n_transitions"], 6)

    def test_bandwidth_cache_checks_actual_inputs(self):
        from functional_fitted_q.action_bandwidth import resolve_action_bandwidth

        actions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
        grid = np.linspace(0.0, 1.0, 3)
        with tempfile.TemporaryDirectory() as temporary:
            options = dict(cache_dir=Path(temporary))
            first = resolve_action_bandwidth(
                {"action_l2_lengthscale": "median"}, actions, grid, **options
            )
            again = resolve_action_bandwidth(
                {"action_l2_lengthscale": "median"}, actions, grid, **options
            )
            self.assertEqual(first, again)
            self.assertEqual(first[0], 1.0)
            changed = actions.copy()
            changed[0, 0] = 0.25
            with self.assertRaises(RuntimeError):
                resolve_action_bandwidth(
                    {"action_l2_lengthscale": "median"}, changed, grid, **options
                )

    def test_checkpoint_settings_and_truncation(self):
        import torch
        from functional_fitted_q.algorithms.nystrom_ffqi import (
            _atomic_torch,
            _save_index,
            _load_index,
        )

        settings = {"seed": 0, "gamma": 0.95, "rank": 6}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            relative = "recovery/full/iteration_001.pt"
            _atomic_torch(directory / relative, {"alpha": torch.ones(6)})
            _save_index(directory, settings, 1, [relative], None)
            self.assertEqual(_load_index(directory, settings)["completed_iteration"], 1)
            with self.assertRaises(RuntimeError):
                _load_index(directory, {**settings, "gamma": 0.9})
            with (directory / relative).open("r+b") as handle:
                handle.truncate(8)
            with self.assertRaises(RuntimeError):
                _load_index(directory, settings)


if __name__ == "__main__":
    unittest.main()
