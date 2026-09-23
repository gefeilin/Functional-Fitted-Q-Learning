"""Generate the 2,500-subject offline pool before taking nested prefixes."""

from pathlib import Path
import fcntl
import json
import numpy as np
import yaml
from functional_fitted_q.data import generate_offline_dataset, verify_offline_dataset
from functional_fitted_q.design import build_manifest
from functional_fitted_q.formal_config import load_pendulum_config
from functional_fitted_q.policies.reference_policy import AnalyticReferencePolicy
from functional_fitted_q.run_artifacts import atomic_json
from functional_fitted_q.runtime import require_execution_host


def run(root, output, seed):
    require_execution_host()
    rows = build_manifest(root)["candidate_fits"]
    task = next(t for t in rows if t["master_seed"] == seed)
    destination = Path(output) / task["data_pool_id"]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (destination / "_SUCCESS").exists():
            verify_offline_dataset(destination)
            return
        env = load_pendulum_config(root)
        ref = yaml.safe_load((root / "configs/reference_policy.yaml").read_text())
        data = generate_offline_dataset(
            master_seed=seed,
            reference=AnalyticReferencePolicy(
                np.asarray(ref["beta"], dtype=np.float64)
            ),
            n_subjects=2500,
            t_per_subject=20,
            env_config=env,
            gp_resolution=128,
            step_gp_amplitude=0.35,
            vectorized=True,
        )
        data.save(
            destination,
            metadata=dict(
                data_pool_id=task["data_pool_id"],
                master_seed=seed,
                n_subjects=2500,
                t_per_subject=20,
                nested_subject_and_time_prefixes=True,
            ),
        )
        verify_offline_dataset(destination, require_complete=False)
        atomic_json(
            destination / "status.json",
            dict(state="COMPLETED", transitions=len(data.states)),
        )
        (destination / "_SUCCESS").touch()
    print(
        json.dumps(
            dict(
                seed=seed,
                data_pool_id=task["data_pool_id"],
                transitions=len(data.states),
            )
        )
    )
