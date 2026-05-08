"""Run the functional fitted Q-iteration simulation workflow."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from functional_fitted_q import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "paper_simulation.yaml")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--size", type=int)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--run-tag")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "functional_fqi")
    return parser


def load_config(path: Path, overrides: argparse.Namespace) -> ExperimentConfig:
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    for key in ["horizon", "gamma", "size", "iteration", "run_tag"]:
        value = getattr(overrides, key)
        if value is not None:
            data[key] = value
    missing = [key for key in ["horizon", "gamma", "size", "iteration"] if key not in data]
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")
    return ExperimentConfig.from_dict(data)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = build_parser().parse_args()
    config = load_config(args.config, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_experiment(config=config, base_dir=args.output_dir)


if __name__ == "__main__":
    main()
