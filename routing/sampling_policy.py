
"""Build explicit, reproducible per-case rollout sampling plans."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from agent.rollout_store import make_rollout_id


@dataclass(frozen=True)
class RolloutSample:
    """One requested sample for a case."""

    case_id: str
    rollout_index: int
    rollout_id: str
    seed: int
    sampling_config: dict[str, Any]


def build_sampling_plan(
    *,
    case_id: str,
    start_index: int,
    count: int,
    base_seed: int,
    sampling_config: dict[str, Any] | None = None,
) -> list[RolloutSample]:
    """Return count unique samples, numbered from start_index.

    The derived seed is stable across processes and independent of Python's
    randomized hash implementation. Provider backends receive it in their
    sampling configuration; backends that support seed control can reproduce
    the same case/index sample.
    """

    if not case_id:
        raise ValueError("case_id must not be empty")
    if start_index < 1:
        raise ValueError("start_index must be at least 1")
    if count < 0:
        raise ValueError("count must not be negative")

    base_config = dict(sampling_config or {})
    plan: list[RolloutSample] = []
    for index in range(start_index, start_index + count):
        seed_material = f"{base_seed}:{case_id}:{index}".encode("utf-8")
        seed = int.from_bytes(sha256(seed_material).digest()[:4], "big")
        config = {**base_config, "seed": seed}
        plan.append(
            RolloutSample(
                case_id=case_id,
                rollout_index=index,
                rollout_id=make_rollout_id(index),
                seed=seed,
                sampling_config=config,
            )
        )
    return plan
