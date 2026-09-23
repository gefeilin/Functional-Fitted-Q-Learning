"""Train one functional-action FQI candidate with an AdaFNN or KRR critic."""

from _entrypoint import run_stage


if __name__ == "__main__":
    run_stage("functional", __doc__)
