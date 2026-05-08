import subprocess
import sys
from pathlib import Path


def test_quickstart_script_creates_expected_outputs():
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "examples/quickstart_example.py"],
        cwd=repo_root,
        check=True,
        timeout=180,
    )

    assert (repo_root / "data" / "example" / "toy_pendulum_transitions.json").exists()
    assert (repo_root / "outputs" / "quickstart" / "quickstart_summary.csv").exists()
    assert (repo_root / "outputs" / "quickstart" / "learned_toy_actions.png").exists()
