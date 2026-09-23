"""Defer Slurm termination until a KRR checkpoint is durable."""

import signal
import time

from .algorithms.nystrom_ffqi import _load_index
from .run_artifacts import atomic_json


class PenaltyCheckpointSignals:
    def __init__(self, root, resume_config, policy_lambda):
        self.root = root
        self.resume_config = resume_config
        self.policy_lambda = policy_lambda
        self.requested = None
        self.previous = {}

    def __enter__(self):
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            self.previous[sig] = signal.signal(sig, self._request)
        return self

    def _request(self, number, frame):
        self.requested = number

    def after_durable_checkpoint(self):
        if self.requested is None:
            return
        # Do not claim recoverability from a status string or a half-written file.
        index = _load_index(self.root, self.resume_config)
        if not index["full_checkpoints"] and index["segment_checkpoint"] is None:
            raise RuntimeError("scheduler interruption has no durable checkpoint")
        atomic_json(
            self.root / "status.json",
            dict(
                state="INTERRUPTED_RECOVERABLE",
                signal=signal.Signals(self.requested).name,
                policy_lambda=self.policy_lambda,
                completed_iteration=index["completed_iteration"],
                checkpoint_resume_config=self.resume_config,
                observed_unix=time.time(),
            ),
        )
        raise SystemExit(99)

    def __exit__(self, exc_type, exc, traceback):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
