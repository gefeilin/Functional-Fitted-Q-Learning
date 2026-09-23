"""Write the experiment task grid without submitting jobs."""

from _entrypoint import run_stage


if __name__ == "__main__":
    run_stage("plan", __doc__)
