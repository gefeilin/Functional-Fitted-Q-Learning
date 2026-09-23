from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

import numpy as np


REQUIRED_STREAMS = (
    "subject_initial_states",
    "subject_behavior_gp",
    "step_behavior_gp",
    "process_noise",
    "dataset_subject_order",
    "critic_initialization",
    "adaptive_basis_initialization",
    "critic_minibatch_order",
    "policy_optimizer",
    "oracle_lambda_policy_optimizer",
    "evaluation_initial_states",
    "evaluation_process_noise",
    "evaluation_behavior_gp",
    "diagnostic_states",
    "diagnostic_transition_mc",
)


def _name_words(name: str) -> tuple[int, ...]:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4))


@dataclass(frozen=True)
class SeedTree:
    master_seed: int

    def seed_sequence(self, stream: str, subindex: int = 0) -> np.random.SeedSequence:
        if stream not in REQUIRED_STREAMS:
            raise KeyError(f"unknown RNG stream: {stream}")
        if subindex < 0:
            raise ValueError("subindex must be nonnegative")
        return np.random.SeedSequence(
            (int(self.master_seed), *_name_words(stream), int(subindex))
        )

    def rng(self, stream: str, subindex: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.seed_sequence(stream, subindex))

    def manifest(self, streams: Iterable[str] = REQUIRED_STREAMS) -> dict:
        result = {"master_seed": int(self.master_seed), "streams": {}}
        for stream in streams:
            ss = self.seed_sequence(stream)
            result["streams"][stream] = {
                "entropy": (
                    list(ss.entropy) if not isinstance(ss.entropy, int) else ss.entropy
                ),
                "spawn_key": list(ss.spawn_key),
                "state_u32": ss.generate_state(4).tolist(),
            }
        return result


def assert_disjoint_namespaces(*groups: Iterable[int]) -> None:
    seen: set[int] = set()
    for group in groups:
        values = [int(x) for x in group]
        if len(values) != len(set(values)) or seen.intersection(values):
            raise ValueError("seed namespaces overlap or contain duplicates")
        seen.update(values)
