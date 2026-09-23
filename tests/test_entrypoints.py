"""Standalone command routing and repository-relative path tests."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from functional_fitted_q import cli
from functional_fitted_q.runtime import project_root


class EntrypointTests(unittest.TestCase):
    def test_functional_critic_selection(self):
        root = project_root()
        for critic in ["adafnn", "krr"]:
            with self.subTest(critic=critic), patch.object(cli, "execute") as execute:
                cli.main(
                    ["--root", str(root), "--approximator", critic, "--index", "7"],
                    stage="functional",
                )
                args, resolved = execute.call_args.args
                self.assertEqual(args.stage, critic)
                self.assertEqual(args.index, 7)
                self.assertEqual(resolved, root)

    def test_constant_routes_to_separate_fit(self):
        with patch.object(cli, "execute") as execute:
            cli.main(["--index", "3"], stage="constant")
            self.assertEqual(execute.call_args.args[0].stage, "constant")

    def test_invalid_critic_fails_before_execution(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as error:
            cli.main(["--approximator", "unknown"], stage="functional")
        self.assertEqual(error.exception.code, 2)

    def test_script_help_outside_repository(self):
        root = project_root()
        names = [
            "run_functional_fqi",
            "run_constant_fqi",
            "generate_offline_data",
            "generate_simulation_figures",
            "verify_results",
            "plan_experiments",
            "select_policy_coefficient",
            "evaluate_policy",
            "analyze_identification",
            "aggregate_results",
        ]
        with tempfile.TemporaryDirectory() as directory:
            for name in names:
                with self.subTest(script=name):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(root / "scripts" / f"{name}.py"),
                            "--help",
                        ],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("--index", result.stdout)

    def test_explicit_training_settings(self):
        for critic in ["adafnn", "krr"]:
            with self.subTest(critic=critic), patch.object(cli, "execute") as execute:
                cli.main(
                    [
                        "--approximator",
                        critic,
                        "--n",
                        "8000",
                        "--seed",
                        "3",
                        "--policy-lambda",
                        "0.001",
                    ],
                    stage="functional",
                )
                args, root = execute.call_args.args
                task = cli.grid(root, args.stage)[args.index]
                self.assertEqual(task["fit_id"], f"{critic}-n8000-lambda0.001-seed03")

    def test_explicit_data_and_constant_settings(self):
        for stage, arguments in [
            ("data", ["--seed", "7"]),
            ("constant", ["--n", "8000", "--seed", "7"]),
        ]:
            with self.subTest(stage=stage), patch.object(cli, "execute") as execute:
                cli.main(arguments, stage=stage)
                args, root = execute.call_args.args
                task = cli.grid(root, args.stage)[args.index]
                self.assertEqual(task.get("master_seed", task.get("seed")), 7)

    def test_explicit_post_training_settings(self):
        for stage in ["select", "evaluate", "identify"]:
            arguments = ["--approximator", "krr", "--n", "8000", "--seed", "2"]
            if stage != "select":
                arguments += ["--policy-lambda", "0.01"]
            with self.subTest(stage=stage), patch.object(cli, "execute") as execute:
                cli.main(arguments, stage=stage)
                args, root = execute.call_args.args
                task = cli.grid(root, args.stage)[args.index]
                self.assertEqual(task["approximator"], "nystrom_krr")
                self.assertEqual(task["n_transitions"], 8000)
                self.assertEqual(task["master_seed"], 2)

    def test_ambiguous_or_conflicting_settings_are_rejected(self):
        for arguments in [
            ["--n", "8000"],
            ["--n", "8000", "--index", "0"],
            ["--n", "1234", "--seed", "0", "--policy-lambda", "0.001"],
        ]:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(
                io.StringIO()
            ):
                with self.assertRaises(SystemExit) as error:
                    cli.main(arguments, stage="functional")
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
