"""Run an optional scalar-action d3rlpy benchmark."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULES = {
    "BCQ": "functional_fitted_q.benchmarks.bcq",
    "CQL": "functional_fitted_q.benchmarks.cql",
    "DDPG": "functional_fitted_q.benchmarks.ddpg",
    "SAC": "functional_fitted_q.benchmarks.sac",
    "TD3": "functional_fitted_q.benchmarks.td3",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(MODULES))
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "benchmarks"))
    args = parser.parse_args()

    try:
        module = importlib.import_module(MODULES[args.method])
    except ModuleNotFoundError as exc:
        if exc.name == "d3rlpy":
            raise SystemExit("d3rlpy is required for benchmarks. Install with: pip install -e '.[benchmarks]'") from exc
        raise
    sys.argv = [
        sys.argv[0],
        "--horizon",
        str(args.horizon),
        "--gamma",
        str(args.gamma),
        "--size",
        str(args.size),
        "--iteration",
        str(args.iteration),
        "--output-dir",
        args.output_dir,
    ]
    module.main()


if __name__ == "__main__":
    main()
