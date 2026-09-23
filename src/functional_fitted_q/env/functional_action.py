from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import CubicSpline


@runtime_checkable
class FunctionalAction(Protocol):
    def __call__(self, u: float | np.ndarray) -> float | np.ndarray: ...


@dataclass(frozen=True)
class GridFunctionalAction:
    """A numerical interpolant of a continuous action; not an action catalogue."""

    grid: np.ndarray
    values: np.ndarray
    torque_limit: float = 2.0

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if grid.ndim != 1 or values.shape != grid.shape or len(grid) < 4:
            raise ValueError("grid and values must be matching one-dimensional arrays")
        if not np.all(np.diff(grid) > 0) or grid[0] != 0.0 or grid[-1] != 1.0:
            raise ValueError("functional grid must increase from exactly 0 to 1")
        if (
            not np.isfinite(values).all()
            or np.max(np.abs(values)) > self.torque_limit + 1e-12
        ):
            raise ValueError("action values violate finite torque bounds")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self, "_spline", CubicSpline(grid, values, bc_type="natural")
        )

    def __call__(self, u: float | np.ndarray) -> float | np.ndarray:
        x = np.asarray(u, dtype=np.float64)
        if np.any((x < -1e-14) | (x > 1.0 + 1e-14)):
            raise ValueError("normalized within-action time must lie in [0,1]")
        y = np.clip(
            self._spline(np.clip(x, 0.0, 1.0)), -self.torque_limit, self.torque_limit
        )
        return float(y) if x.ndim == 0 else y


def evaluate_action(
    action: FunctionalAction, grid: np.ndarray, torque_limit: float = 2.0
) -> np.ndarray:
    values = np.asarray(action(grid), dtype=np.float64)
    if values.shape != grid.shape:
        values = np.asarray([action(float(u)) for u in grid], dtype=np.float64)
    if not np.isfinite(values).all() or np.max(np.abs(values)) > torque_limit + 1e-10:
        raise ValueError("functional action produced invalid torque")
    return values
