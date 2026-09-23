"""Train one state-dependent constant-action AdaFNN FQI comparator."""

from _entrypoint import run_stage


if __name__ == "__main__":
    run_stage("constant", __doc__)
