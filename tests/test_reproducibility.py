"""Fast CPU contracts: paper grid, tuning isolation, and seed-level statistics."""

import unittest
import numpy as np
import pandas as pd
from functional_fitted_q.runtime import project_root
from functional_fitted_q.design import build_manifest, derived_mc_seed, constant_fit_id
from functional_fitted_q.cli import grid
from functional_fitted_q.selection import select_lambda
from functional_fitted_q.analysis.statistics import (
    bootstrap_interval,
    grid_check,
    N_GRID,
    LAMBDAS,
)


class PaperGridTests(unittest.TestCase):
    def setUp(self):
        self.root = project_root()
        self.manifest = build_manifest(self.root)

    def test_task_counts(self):
        expected = {
            "data": 20,
            "adafnn": 500,
            "krr": 500,
            "constant": 100,
            "select": 200,
        }
        self.assertEqual({k: len(grid(self.root, k)) for k in expected}, expected)

    def test_unique_fit_and_selection_ids(self):
        self.assertEqual(
            len({t["fit_id"] for t in self.manifest["candidate_fits"]}), 1000
        )
        self.assertEqual(
            len({c["selection_id"] for c in self.manifest["selection_cells"]}), 200
        )

    def test_paired_data_design(self):
        tasks = self.manifest["candidate_fits"]
        for seed in range(20):
            self.assertEqual(
                len({t["data_pool_id"] for t in tasks if t["master_seed"] == seed}), 1
            )

    def test_constant_ids_unique(self):
        self.assertEqual(
            len({constant_fit_id(t) for t in grid(self.root, "constant")}), 100
        )

    def test_readable_run_names(self):
        first = self.manifest["candidate_fits"][0]
        self.assertEqual(first["fit_id"], "adafnn-n2000-lambda0.0001-seed00")
        self.assertEqual(first["data_pool_id"], "pendulum-seed00")
        self.assertEqual(constant_fit_id(first), "adafnn-constant-n2000-seed00")
        self.assertEqual(
            self.manifest["selection_cells"][0]["selection_id"], "adafnn-n2000-seed00"
        )

    def test_tuning_reporting_streams_distinct_and_repeatable(self):
        for seed in range(20):
            tune = derived_mc_seed(seed, "oracle_lambda_tuning_v1")
            report = derived_mc_seed(seed, "oracle_lambda_reporting_v1")
            self.assertNotEqual(tune, report)
            self.assertEqual(tune, derived_mc_seed(seed, "oracle_lambda_tuning_v1"))


class TuningTests(unittest.TestCase):
    def candidates(self):
        return [
            dict(
                fit_id=f"fit-{i}",
                lambda_dimensionless=lam,
                normalized_returns=np.full(20, 0.5),
                evaluation_seed=17,
                stream="tuning",
            )
            for i, lam in enumerate(LAMBDAS)
        ]

    def select(self, rows):
        return select_lambda(
            rows, expected_lambdas=LAMBDAS, tuning_seed=17, episodes=20
        )

    def test_exact_tie_uses_smallest_lambda(self):
        self.assertEqual(
            self.select(self.candidates()[::-1])["selected_lambda_dimensionless"], 1e-4
        )

    def test_largest_tuning_mean_wins(self):
        rows = self.candidates()
        rows[2]["normalized_returns"] = np.full(20, 0.7)
        self.assertEqual(self.select(rows)["selected_fit_id"], "fit-2")

    def test_reporting_cannot_select_lambda(self):
        rows = self.candidates()
        rows[0]["stream"] = "reporting"
        with self.assertRaises(ValueError):
            self.select(rows)

    def test_bad_mc_seed_rejected(self):
        rows = self.candidates()
        rows[0]["evaluation_seed"] = 18
        with self.assertRaises(ValueError):
            self.select(rows)

    def test_missing_candidate_rejected(self):
        with self.assertRaises(ValueError):
            self.select(self.candidates()[:-1])


class StatisticTests(unittest.TestCase):
    def test_constant_bootstrap(self):
        self.assertEqual(bootstrap_interval(np.full(20, 0.5)), (0.5, 0.5, 0.5))

    def test_bootstrap_requires_training_replicates(self):
        with self.assertRaises(ValueError):
            bootstrap_interval(np.ones(1000))

    def test_grid_rejects_duplicate_seed(self):
        frame = pd.DataFrame(
            [dict(n=n, master_seed=s) for n in N_GRID for s in range(20)]
        )
        grid_check(frame, "n", N_GRID)
        frame.loc[1, "master_seed"] = 0
        with self.assertRaises(ValueError):
            grid_check(frame, "n", N_GRID)


if __name__ == "__main__":
    unittest.main()
