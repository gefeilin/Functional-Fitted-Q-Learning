"""Training-action-only, exact median bandwidth for the functional RBF kernel.

No outcomes, states, landmarks, policies or random seeds enter this calculation.
Large condensed distance arrays are disk-backed. A per-dataset cache avoids
repeating the calculation across regularization coefficients.
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.distance import pdist

from functional_fitted_q.critics.krr_functional_critic import trapezoid_weights
from functional_fitted_q.run_artifacts import atomic_json, atomic_path


MEDIAN_RULE = "exact_training_pairwise_functional_l2_median_v1"


def _exact_median(
    weighted: np.ndarray, temporary: Path, memory_pair_limit: int
) -> float:
    count = len(weighted) * (len(weighted) - 1) // 2
    # Lock is held by the caller; clear a killed writer's exact-key scratch file
    # before checking available space or switching back to in-memory execution.
    temporary.unlink(missing_ok=True)
    if count <= memory_pair_limit:
        distances = pdist(weighted, metric="euclidean")
        return float(np.median(distances, overwrite_input=True))
    required_bytes = count * np.dtype(np.float64).itemsize
    if shutil.disk_usage(temporary.parent).free < required_bytes + 64 * 1024**2:
        raise RuntimeError(
            f"exact action median needs {required_bytes} temporary bytes"
        )
    distances = None
    try:
        distances = np.memmap(temporary, mode="w+", dtype=np.float64, shape=(count,))
        pdist(weighted, metric="euclidean", out=distances)
        # In-place selection, not a sort/copy of the full condensed distance array.
        return float(np.median(distances, overwrite_input=True))
    finally:
        if distances is not None:
            distances._mmap.close()
        temporary.unlink(missing_ok=True)


def resolve_action_bandwidth(
    kernel: dict,
    action_values: np.ndarray,
    action_grid: np.ndarray,
    *,
    cache_dir: Path,
) -> tuple[float, dict | None]:
    """Resolve a fixed numeric bandwidth or the training-action ``median`` rule.

    The median includes every unordered off-diagonal pair (including duplicate
    actions at distinct rows). For an even pair count, average the two middle
    *distances*, not their squares. Degenerate/nonfinite medians fail explicitly.
    """
    requested = kernel["action_l2_lengthscale"]
    if requested != "median":
        value = float(requested)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("action lengthscale must be finite and positive")
        return value, None
    actions = np.asarray(action_values, dtype=np.float64)
    grid = np.asarray(action_grid, dtype=np.float64)
    if actions.ndim != 2 or len(actions) < 2:
        raise ValueError("action median requires at least two training action curves")
    if grid.ndim != 1 or actions.shape[1] != len(grid):
        raise ValueError("training action width and quadrature grid do not match")
    if not np.isfinite(actions).all() or not np.isfinite(grid).all():
        raise ValueError("action median inputs must be finite")
    weights = trapezoid_weights(grid)
    if grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("functional Pendulum action grid must span [0, 1]")
    memory_pair_limit = int(kernel.get("action_median_max_in_memory_pairs", 2_000_000))
    if memory_pair_limit < 0:
        raise ValueError("action median memory pair limit must be nonnegative")
    identity = {
        "schema_version": 1,
        "rule": MEDIAN_RULE,
        "n_training_actions": len(actions),
        "n_action_grid_points": len(grid),
        "pair_count": len(actions) * (len(actions) - 1) // 2,
        "quadrature": "unnormalized_trapezoid_weights_on_unit_interval",
        "kernel_convention": "exp(-functional_l2_distance_squared/(2*lengthscale**2))",
        "duplicate_row_zero_distances_included": True,
        "diagonal_self_pairs_included": False,
        "pair_subsampling": False,
        "even_count_rule": "arithmetic_mean_of_two_middle_distances",
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = cache_dir / "bandwidth.json"
    with (cache_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if any(receipt.get(k) != v for k, v in identity.items()):
                raise RuntimeError(
                    "Action bandwidth settings changed; use a new cache directory"
                )
            for name, expected in (("actions", actions), ("grid", grid)):
                saved = np.load(
                    cache_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False
                )
                if not np.array_equal(saved, expected):
                    raise RuntimeError(
                        "Cached bandwidth belongs to different data; use a new cache directory"
                    )
        else:
            weighted = np.ascontiguousarray(actions * np.sqrt(weights)[None, :])
            value = _exact_median(
                weighted, cache_dir / "distances.tmp", memory_pair_limit
            )
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "training action median is nonpositive/nonfinite; no silent fallback"
                )
            receipt = {**identity, "action_l2_lengthscale": value}
            for name, values in (("actions", actions), ("grid", grid)):
                with atomic_path(cache_dir / f"{name}.npy") as temporary:
                    with temporary.open("wb") as handle:
                        np.save(handle, values, allow_pickle=False)
            atomic_json(receipt_path, receipt)
        value = float(receipt["action_l2_lengthscale"])
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError("cached action median is invalid")
        return value, receipt


def validate_median_run_request(
    config: dict, task: dict, views: list[dict], output_root: Path, *, smoke_mode: bool
) -> None:
    """Require task horizons, checkpoint views, and output namespaces to agree."""
    if config["kernel"]["action_l2_lengthscale"] != "median":
        return
    expected_name = Path(config["output_root"]).name
    actual_name = Path(output_root).name
    if actual_name != expected_name and not (
        smoke_mode and actual_name.startswith(expected_name + "_smoke")
    ):
        raise ValueError(
            "median bandwidth requires its separate configured output namespace"
        )
    iterations = int(config["training"]["max_iterations"])
    if int(task["max_iterations"]) != iterations:
        raise ValueError("median task does not match the configured training horizon")
    checkpoints = [int(v) for v in str(task["evaluation_checkpoints"]).split("|")]
    if not checkpoints or max(checkpoints) != iterations or min(checkpoints) < 1:
        raise ValueError(
            "median task evaluation checkpoints must end at the final iteration"
        )
    if any(not 1 <= int(v["checkpoint_iteration"]) <= iterations for v in views):
        raise ValueError("median task views exceed the configured training horizon")


def persist_bandwidth_receipt(run_root: Path, receipt: dict | None) -> None:
    if receipt is None:
        return
    path = Path(run_root) / "action_bandwidth.json"
    if path.is_file() and json.loads(path.read_text()) != receipt:
        raise RuntimeError(
            "action bandwidth changed inside an existing run; cannot resume"
        )
    atomic_json(path, receipt)
