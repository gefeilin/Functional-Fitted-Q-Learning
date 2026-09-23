"""One command-line interface for every experiment stage."""

import argparse
import importlib
import json
from pathlib import Path
import sys
from .runtime import project_root, configure_runtime


def call(module, arguments):
    sys.argv = [module, *map(str, arguments)]
    importlib.import_module("functional_fitted_q.runners." + module).main()


def grid(root, stage):
    from .design import build_manifest, display_name

    m = build_manifest(root)
    if stage == "data":
        return [{"seed": s} for s in range(20)]
    if stage in ["adafnn", "krr", "constant"]:
        critic = "nystrom_krr" if stage == "krr" else "adafnn"
        return [
            dict(
                t,
                run_name=display_name(
                    t, "constant" if stage == "constant" else "functional"
                ),
            )
            for t in m["candidate_fits"]
            if t["approximator"] == critic
            and (stage != "constant" or t["lambda_dimensionless"] == 0.0001)
        ]
    if stage == "select":
        return m["selection_cells"]
    return m["candidate_fits"]


def execute(args, root):
    """Resolve one CLI task and dispatch it to the corresponding runner."""
    from .design import build_manifest
    from .run_artifacts import atomic_json

    work = root / args.runs
    data = root / args.data
    if args.stage == "plan":
        counts = {
            s: len(grid(root, s))
            for s in ["data", "adafnn", "krr", "constant", "select"]
        }
        document = dict(
            counts=counts,
            reporting_unique=360,
            identification_unique=360,
            data_subjects_per_pool=2500,
            no_jobs_submitted=True,
        )
        for stage in counts:
            atomic_json(work / "tasks" / f"{stage}.json", grid(root, stage))
        print(json.dumps(document, indent=2))
        return
    tasks = grid(root, args.stage)
    if not 0 <= args.index < len(tasks):
        raise ValueError("Task index outside stage grid")
    t = tasks[args.index]
    if args.stage == "data":
        from .runners.generate_data import run

        run(root, data, t["seed"])
    elif args.stage in ["adafnn", "constant"]:
        from .runners.train_adafnn import run

        run(
            root,
            data,
            work / ("constant" if args.stage == "constant" else "candidates/adafnn"),
            t["task_id"],
            constant=args.stage == "constant",
        )
    elif args.stage == "krr":
        call(
            "train_krr",
            [
                "--project-root",
                root,
                "--data-root",
                data,
                "--output-root",
                work / "candidates/krr",
                "--task-id",
                t["task_id"],
            ],
        )
    elif args.stage == "select":
        call(
            "select",
            [
                "--project-root",
                root,
                "--candidate-root",
                work / "candidates",
                "--output-root",
                work / "selection",
                "--selection-index",
                args.index,
            ],
        )
    else:
        # All 1,000 indices exist. Exactly 360 execute; the rest are explicit
        # non-selected non-fixed-n no-ops after the selection stage completes.
        if t["n_transitions"] != 8000:
            cell = next(
                c
                for c in build_manifest(root)["selection_cells"]
                if t["fit_id"] in c["candidate_fit_ids"]
            )
            selected = json.loads(
                (
                    work / "selection" / cell["selection_id"] / "selection.json"
                ).read_text()
            )
            if selected["selected_fit_id"] != t["fit_id"]:
                print(json.dumps(dict(state="NOT_REQUESTED", fit_id=t["fit_id"])))
                return
        argv = [
            "--project-root",
            root,
            "--candidate-root",
            work / "candidates",
            "--selection-root",
            work / "selection",
            "--output-root",
            work / args.stage,
            "--fit-id",
            t["fit_id"],
        ]
        if args.stage == "identify":
            argv += ["--data-root", data]
        call(args.stage, argv)


def main(argv=None, *, stage=None, description=None):
    """Parse the unified command line used by the public wrapper scripts."""
    parser = argparse.ArgumentParser(description=description or __doc__)
    if stage is None:
        parser.add_argument(
            "stage",
            choices=[
                "plan",
                "data",
                "adafnn",
                "krr",
                "constant",
                "select",
                "evaluate",
                "identify",
                "aggregate",
                "reproduce",
                "verify",
                "smoke",
            ],
        )
    elif stage == "functional":
        parser.add_argument(
            "--approximator",
            choices=["adafnn", "krr"],
            default="adafnn",
            help="Critic family for the functional-action FQI task",
        )
        parser.set_defaults(stage="adafnn")
    else:
        parser.set_defaults(stage=stage)
    if stage != "functional":
        parser.add_argument(
            "--approximator",
            choices=["adafnn", "krr"],
            help="Critic family for selection, evaluation, or analysis",
        )
    parser.add_argument(
        "--index", type=int, help="Task index, as an alternative to explicit settings"
    )
    parser.add_argument("--n", type=int, help="Number of offline transitions")
    parser.add_argument("--seed", type=int, help="Master seed")
    parser.add_argument(
        "--policy-lambda", type=float, help="Policy-curvature coefficient"
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--source", type=Path, default=Path("results/source"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if stage == "functional":
        args.stage = args.approximator
    root = project_root(args.root)
    settings_given = any(
        value is not None for value in (args.n, args.seed, args.policy_lambda)
    )
    if settings_given:
        if args.index is not None:
            parser.error(
                "Choose either --index or explicit --n/--seed/--policy-lambda settings"
            )
        if args.stage not in (
            "data",
            "adafnn",
            "krr",
            "constant",
            "select",
            "evaluate",
            "identify",
        ):
            parser.error("This command does not select an individual experiment")
        candidates = grid(root, args.stage)
        matches = []
        for index, task in enumerate(candidates):
            if args.n is not None and task.get("n_transitions") != args.n:
                continue
            if (
                args.seed is not None
                and task.get("master_seed", task.get("seed")) != args.seed
            ):
                continue
            if (
                args.policy_lambda is not None
                and task.get("lambda_dimensionless") != args.policy_lambda
            ):
                continue
            if args.approximator is not None:
                family = "nystrom_krr" if args.approximator == "krr" else "adafnn"
                if task.get("approximator") != family:
                    continue
            matches.append(index)
        if len(matches) != 1:
            parser.error(
                f"Settings match {len(matches)} tasks; specify the sample size, seed, critic, and coefficient as applicable"
            )
        args.index = matches[0]
    elif args.index is None:
        args.index = 0
    from .runtime import require_execution_host

    require_execution_host()
    configure_runtime()
    if args.stage == "reproduce":
        from .analysis.reproduce import reproduce

        reproduce(root, root / args.source, root / args.output)
    elif args.stage == "verify":
        from .analysis.verify import verify

        verify(root, root / args.output)
    elif args.stage == "aggregate":
        from .analysis.aggregate import aggregate

        aggregate(root, root / args.runs, root / args.output / "source")
    elif args.stage == "smoke":
        from .smoke import run

        run(root, root / args.output / "smoke")
    else:
        execute(args, root)
