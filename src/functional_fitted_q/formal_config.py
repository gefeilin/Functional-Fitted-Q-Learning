"""Load the Pendulum parameters used by data generation and evaluation."""

from __future__ import annotations
from pathlib import Path
import yaml

from .env.dynamics import PendulumConfig


def load_pendulum_config(project_root: Path) -> PendulumConfig:
    root = Path(project_root)
    env = yaml.safe_load((root / "configs" / "env.yaml").read_text())
    return PendulumConfig(
        g=float(env["g"]),
        m=float(env["m"]),
        l=float(env["l"]),
        torque_limit=float(env["torque_limit"]),
        omega_limit=float(env["omega_limit"]),
        duration=float(env["functional_action_duration_sec"]),
        ode_substeps=int(env["ode_substeps"]),
        sigma_theta=float(env["sigma_theta"]),
        sigma_omega=float(env["sigma_omega"]),
        gamma=float(env["gamma"]),
    )
