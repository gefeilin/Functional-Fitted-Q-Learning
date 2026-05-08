"""Small utilities for summarizing paper simulation result files."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


RESULT_RE = re.compile(
    r"Data size (?P<size>\d+), Batch (?P<batch>\d+): scores = (?P<score>[-+]?\d*\.?\d+)"
)


def parse_score_file(path: Path, method: str, gamma: float, horizon: int) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        match = RESULT_RE.search(line)
        if match is None:
            continue
        rows.append(
            {
                "method": method,
                "gamma": gamma,
                "horizon": horizon,
                "size": int(match.group("size")),
                "batch": int(match.group("batch")),
                "score": float(match.group("score")),
                "source_file": str(path),
            }
        )
    return pd.DataFrame(rows)


def collect_score_results(results_root: Path) -> pd.DataFrame:
    """Collect functional FQI and scalar benchmark score files."""
    results_root = results_root.resolve()
    frames = []

    functional_dir = results_root / "functional-fqi" / "CV_20250719_result"
    for path in sorted(functional_dir.glob("seed420_cv_g*_h*.txt")):
        gamma, horizon = _parse_gamma_horizon(path)
        frame = parse_score_file(path, "Functional FQI", gamma, horizon)
        if not frame.empty:
            frame["source_file"] = str(path.relative_to(results_root))
        frames.append(frame)

    for method in ["DDPG", "TD3", "SAC", "CQL", "BCQ"]:
        method_dir = results_root / "d3rlpy" / method
        for path in sorted(method_dir.glob("CV_*_result*/seed420_cv_g*_h*.txt")):
            gamma, horizon = _parse_gamma_horizon(path)
            frame = parse_score_file(path, method, gamma, horizon)
            if not frame.empty:
                frame["source_file"] = str(path.relative_to(results_root))
            frames.append(frame)

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["method", "gamma", "horizon", "size", "batch", "score", "source_file"])
    return pd.concat(frames, ignore_index=True)


def _parse_gamma_horizon(path: Path) -> tuple[float, int]:
    name = path.stem
    match = re.search(r"_g(?P<gamma>[0-9.]+)_h(?P<horizon>\d+)", name)
    if match is None:
        raise ValueError(f"Could not parse gamma/horizon from {path}")
    return float(match.group("gamma")), int(match.group("horizon"))


__all__ = ["collect_score_results", "parse_score_file"]
