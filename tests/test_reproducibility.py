"""Fast CPU contracts: paper grid, tuning isolation, and seed-level statistics."""

import json
import unittest
import yaml
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

    def test_identification_queries_match_training_sample_size(self):
        config = yaml.safe_load(
            (self.root / "configs/paper_simulation.yaml").read_text()
        )["identification"]
        self.assertEqual(config["heldout_subjects_rule"], "n_subjects")
        self.assertEqual(config["heldout_decisions"], 20)
        self.assertEqual(config["query_count_rule"], "n_transitions")
        self.assertEqual(config["paper_split"], "evaluation")
        self.assertEqual(config["state_neighbor_count"], 32)
        self.assertEqual(
            config["state_neighbor_sensitivity_counts"], [16, 32, 64, 128]
        )
        for task in self.manifest["candidate_fits"]:
            self.assertEqual(task["n_transitions"], task["n_subjects"] * 20)


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


class ReleasedIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.root = project_root()
        self.proxy = pd.read_csv(
            self.root / "results/source/identification_proxy_per_fit.csv"
        )
        self.neighbor = pd.read_csv(
            self.root / "results/source/neighbor_sensitivity_per_fit.csv"
        )

    def test_fit_level_grid_and_energy_identity(self):
        self.assertEqual(len(self.proxy), 360)
        self.assertEqual(int(self.proxy.selected_sample_size_view.sum()), 200)
        self.assertEqual(int(self.proxy.fixed_n_all_lambda_view.sum()), 200)
        np.testing.assert_allclose(
            self.proxy.actual_graph_design_ratio,
            self.proxy.G_actual_sq / self.proxy.D_actual_sq,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_neighbor_queries_match_each_training_cell(self):
        self.assertEqual(len(self.neighbor), 800)
        self.assertEqual(set(self.neighbor.state_neighbor_count), {16, 32, 64, 128})
        self.assertEqual(set(self.neighbor.query_class), {"learned", "behavior"})
        self.assertTrue(
            self.neighbor.query_count.eq(self.neighbor.n_transitions).all()
        )


class PaperFigureRegistryTests(unittest.TestCase):
    def test_registry_contains_only_manuscript_figures(self):
        root = project_root()
        registry = json.loads((root / "results/figures.json").read_text())
        expected = {
            "main_figure": "fig01_adafnn_main.pdf",
            "krr_value": "figS01_krr_value.pdf",
            "paired_value_difference": "figS06_paired_return_difference.pdf",
            "neighbor_sensitivity": "figS08_adafnn_neighbor_sensitivity.pdf",
            "krr_identification": "figS03_krr_identification.pdf",
            "critic_update_energies": "figS07_critic_update_energies.pdf",
        }
        observed = {
            item["artifact_id"]: item["manuscript_filename"] for item in registry
        }
        self.assertEqual(observed, expected)
        expected_reference_files = {
            f"{artifact_id}.{suffix}"
            for artifact_id in expected
            for suffix in ("pdf", "png")
        }
        reference_files = {
            path.name
            for path in (root / "results/reference/figures").iterdir()
            if path.is_file()
        }
        self.assertEqual(reference_files, expected_reference_files)


if __name__ == "__main__":
    unittest.main()
