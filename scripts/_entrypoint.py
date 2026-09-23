"""Resolve repository paths for the standalone experiment scripts."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_stage(stage: str, description: str | None = None) -> None:
    """Run a shared workflow stage, resolving default paths from this repository."""
    from functional_fitted_q.cli import main

    main(
        ["--root", str(REPO_ROOT), *sys.argv[1:]], stage=stage, description=description
    )
