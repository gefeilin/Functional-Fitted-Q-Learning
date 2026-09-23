"""Experiment settings, readable run names, and reproducible random seeds."""

import hashlib
import json
from pathlib import Path
import yaml

DESIGN_ID = "functional_pendulum_oracle_lambda_rate_identification_v1"
CONFIG_RELATIVE = "configs/paper_simulation.yaml"


def load_config(project_root, path=None):
    root = Path(project_root)
    config = yaml.safe_load(
        (Path(path) if path else root / CONFIG_RELATIVE).read_text()
    )
    expected = json.loads((root / "project_config.json").read_text())
    scope = config["scope"]
    if (
        scope["sample_sizes"] != expected["sample_sizes"]
        or scope["seeds"] != expected["seeds"]
    ):
        raise ValueError("Paper sample-size/seed grid changed")
    if (
        scope["positive_dimensionless_policy_lambdas"]
        != expected["policy_coefficients"]
    ):
        raise ValueError("Paper policy-coefficient grid changed")
    if config["ffqi"]["max_iterations"] != 20 or config["ffqi"]["gamma"] != 0.95:
        raise ValueError("Paper M/gamma changed")
    return config


def build_manifest(project_root, path=None):
    load_config(project_root, path)
    manifest = json.loads((Path(project_root) / "configs/tasks.json").read_text())
    tasks = manifest["candidate_fits"]
    if len({task["fit_id"] for task in tasks}) != len(tasks):
        raise ValueError("Experiment list contains duplicate run names")
    return manifest


def bind_wandb_identity(config):
    # Tracking identity is configured through the caller's environment.
    return None


def derived_mc_seed(master_seed, namespace):
    payload = f"{int(master_seed)}:{namespace}:0".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def krr_landmark_seed(task):
    return int(task["landmark_seed"])


def constant_fit_id(task):
    return f"adafnn-constant-n{int(task['n_transitions'])}-seed{int(task['master_seed']):02d}"


def display_name(task, method="functional"):
    critic = "krr" if task["approximator"] == "nystrom_krr" else "adafnn"
    suffix = (
        ""
        if method == "constant"
        else f"-lambda{float(task['lambda_dimensionless']):g}"
    )
    return f"{critic}-{method}-n{int(task['n_transitions'])}-seed{int(task['master_seed']):02d}{suffix}"
