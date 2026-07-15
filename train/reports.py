
"""Reporting helpers for verl rollout results."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any



def summarize_verl_scores(*, run_dir: Path, out_dir: Path, high_reward: float = 0.80) -> dict[str, Any]:
    """Summarize rollout rewards with separate rollout and case metrics.

    pass_at_1 is the macro-average across cases of successful sampled rollouts
    divided by that case's rollout count. rollout_success_rate is the pooled
    micro-average across all rollout rows. any_success_case_rate_at_sampled_k
    is the share of cases with at least one successful observed sample; it is
    not the Pass@1 metric.
    """

    if not 0.0 <= high_reward <= 1.0:
        raise ValueError("high_reward must be within [0, 1]")
    scores_path = run_dir / "scores.jsonl"
    rows = _read_jsonl(scores_path) if scores_path.exists() else []
    rewards = [float(row.get("reward") or 0.0) for row in rows]
    case_counter = Counter(row.get("case_id") for row in rows)
    success_counts: Counter[Any] = Counter()
    for row in rows:
        if float(row.get("reward") or 0.0) >= high_reward:
            success_counts[row.get("case_id")] += 1
    per_case_success_rates = [
        success_counts[case_id] / count
        for case_id, count in case_counter.items()
        if count
    ]
    successful_cases = sum(1 for case_id in case_counter if success_counts[case_id] > 0)
    rollout_success_rate = sum(success_counts.values()) / len(rows) if rows else 0.0
    summary = {
        "run_dir": str(run_dir),
        "scores_path": str(scores_path),
        "rollout_count": len(rows),
        "case_count": len(case_counter),
        "success_threshold": high_reward,
        "mean_reward": round(mean(rewards), 6) if rewards else 0.0,
        "reward_std": round(pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "rollout_success_rate": round(rollout_success_rate, 6),
        "pass_at_1": round(mean(per_case_success_rates), 6) if per_case_success_rates else 0.0,
        "any_success_case_rate_at_sampled_k": round(
            successful_cases / len(case_counter), 6
        ) if case_counter else 0.0,
        "cases": dict(case_counter),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "summary.json", summary)
    _write_csv(out_dir / "scores.csv", rows)
    _write_markdown(out_dir / "report.md", title=f"verl Rollout Report: {run_dir.name}", summary=summary)
    return summary


def _write_markdown(path: Path, *, title: str, summary: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in summary.items():
        if isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
