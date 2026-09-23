"""Generate one reproducible offline functional-action Pendulum data pool."""

from _entrypoint import run_stage


if __name__ == "__main__":
    run_stage("data", __doc__)
