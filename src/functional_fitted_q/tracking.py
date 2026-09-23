"""Optional offline W&B tracking, disabled by default for independent readers."""

import os
from .run_artifacts import atomic_json


def init_offline_wandb(*, output_root, **kwargs):
    if os.environ.get("FFQI_WANDB") != "1":
        return None
    import wandb

    config = kwargs.pop("resolved_config")
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "ffqi-reproduction"),
        entity=os.environ.get("WANDB_ENTITY"),
        mode="offline",
        dir=str(output_root),
        config=config,
        **kwargs
    )


def finish_wandb_without_affecting_science(run, output_root, exit_code=0):
    if run is None:
        return "DISABLED"
    try:
        run.finish(exit_code=exit_code)
        return "OFFLINE_FINISHED"
    except Exception as error:
        atomic_json(output_root / "tracking_error.json", {"error": str(error)})
        return "TRACKING_ERROR"
