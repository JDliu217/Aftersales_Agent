
"""Bounded follow-up probing for cases whose route is still ambiguous."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromotionPlan:
    """Action to take after a case-level route decision."""

    action: str
    additional_rollouts: int
    reason: str


def plan_promotion(*, route: str, current_k: int, max_k: int) -> PromotionPlan:
    """Probe more only for more_probe, doubling the sample count up to max_k.

    A resolved route is final immediately. A more_probe route receives another
    batch equal to the samples already collected (at least one), bounded by the
    explicit per-case maximum. When the budget is exhausted the case remains in
    more_probe for manual or later review.
    """

    if current_k < 0:
        raise ValueError("current_k must not be negative")
    if max_k < 0:
        raise ValueError("max_k must not be negative")
    if current_k > max_k:
        raise ValueError("current_k cannot exceed max_k")
    if route != "more_probe":
        return PromotionPlan("finalize", 0, f"route resolved to {route}")
    if current_k >= max_k:
        return PromotionPlan("budget_exhausted", 0, "more_probe budget exhausted")

    additional = min(max(1, current_k), max_k - current_k)
    return PromotionPlan(
        "probe_more",
        additional,
        f"route remains more_probe; add {additional} rollout(s), up to max_k={max_k}",
    )
