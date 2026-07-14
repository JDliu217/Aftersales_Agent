
"""Run a baseline benchmark batch and route each case from real rollout artifacts."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from agent.providers.factory import provider_from_env, verifier_provider_from_env
from agent.rollout_store import make_run_id, write_rollout_artifacts
from agent.runtime import run_agent_loop
from agent.verifier import score_trajectory
from envs.toolfactory import ToolFactory
from routing.metrics import compute_group_metrics
from routing.pool_writer import write_pools
from routing.promotion import plan_promotion
from routing.route_case import Thresholds, route_case
from routing.sampling_policy import RolloutSample, build_sampling_plan
from schemas.route_schema import RouteDecisionSchema
from train.verl_reward_adapter import reward_metadata_from_score

_AUTO_JUDGE = object()
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_benchmark_batch(
    *,
    batch_dir: Path,
    rollout_root: Path,
    pools_root: Path,
    rollouts_per_case: int,
    max_rollouts_per_case: int | None = None,
    run_id: str | None = None,
    base_seed: int = 0,
    sampling_config: dict[str, Any] | None = None,
    max_steps: int | None = None,
    thresholds: Thresholds | None = None,
    provider: Any | None = None,
    verifier_provider: Any = _AUTO_JUDGE,
    rollout_runner: Callable[..., dict[str, Any]] = run_agent_loop,
    score_fn: Callable[..., dict[str, Any]] = score_trajectory,
    artifact_writer: Callable[..., Path] = write_rollout_artifacts,
) -> dict[str, Any]:
    """Sample, score, aggregate, route, and write pool manifests for one batch.

    The batch manifest must reference case, env_snapshot, verifier_spec, and
    gold JSON files. The gold file is retained as provenance; it is never used
    as a rollout target. Extra probing is bounded by max_rollouts_per_case and
    happens only while route_case returns more_probe.

    Providers and the rollout/scoring/writer functions are injectable so tests
    can exercise the complete flow without a model, network, or paid API.
    """

    if rollouts_per_case < 1:
        raise ValueError("rollouts_per_case must be at least 1")
    maximum = max_rollouts_per_case if max_rollouts_per_case is not None else rollouts_per_case
    if maximum < rollouts_per_case:
        raise ValueError("max_rollouts_per_case must be at least rollouts_per_case")
    selected_run_id = run_id or make_run_id("baseline")
    if not selected_run_id or selected_run_id in {".", ".."} or any(
        separator in selected_run_id for separator in ("/", "\\")
    ):
        raise ValueError("run_id must be a non-empty path-safe single component")

    batch_dir = Path(batch_dir).resolve()
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"batch manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    entries = sorted(manifest.get("entries") or [], key=lambda item: item.get("id", ""))
    if not entries:
        raise ValueError(f"batch manifest has no entries: {manifest_path}")

    case_ids = [entry.get("case_id") for entry in entries]
    if any(not case_id for case_id in case_ids):
        raise ValueError("every batch manifest entry must have a case_id")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("batch manifest contains duplicate case_id values")

    active_thresholds = thresholds or Thresholds()
    if not 0.0 <= active_thresholds.high_reward <= 1.0:
        raise ValueError("high_reward must be within [0, 1]")
    provider = provider if provider is not None else provider_from_env()
    judge = verifier_provider_from_env() if verifier_provider is _AUTO_JUDGE else verifier_provider
    tool_factory = ToolFactory()
    registry_snapshot = tool_factory.tool_registry_snapshot()
    tool_registry = {tool["name"] for tool in registry_snapshot}
    rollout_root = Path(rollout_root)
    pools_root = Path(pools_root)

    case_results: list[dict[str, Any]] = []
    source_refs_by_case: dict[str, dict[str, Any]] = {}
    rollout_refs_by_case: dict[str, str | None] = {}
    total_rollouts = 0

    for entry in entries:
        entry_id = str(entry["id"])
        case_id = str(entry["case_id"])
        files = entry.get("files") or {}
        source_paths = {
            name: _resolve_batch_file(batch_dir, files, name)
            for name in ("case", "env_snapshot", "verifier_spec", "gold")
        }
        case = _read_json(source_paths["case"])
        env_snapshot = _read_json(source_paths["env_snapshot"])
        verifier_spec = _read_json(source_paths["verifier_spec"])
        # Read and parse gold to catch a broken provenance reference, but never
        # pass it to the model or include it in the GRPO target.
        _read_json(source_paths["gold"])

        identity_mismatch = (
            case.get("case_id") != case_id
            or env_snapshot.get("case_id") != case_id
        )
        source_refs = {
            "source_batch": _portable_path(batch_dir),
            "source_manifest_id": manifest.get("manifest_id"),
            "source_entry_id": entry_id,
            "case_file_case_id": case.get("case_id"),
            "env_snapshot_case_id": env_snapshot.get("case_id"),
            **{
                f"{name}_ref": _batch_relative_path(batch_dir, path)
                for name, path in source_paths.items()
            },
        }
        source_refs_by_case[case_id] = source_refs
        routing_case = {**case, "case_id": case_id}

        case_rollouts: list[dict[str, Any]] = []
        probe_history: list[dict[str, Any]] = []
        final_metrics: dict[str, Any] = {}
        decision = None
        current_k = 0
        pending_rollouts = rollouts_per_case
        case_artifact_dir: Path | None = None

        # A mismatched case/env pair is quarantined without asking a model to
        # act on state that does not belong to the manifest entry.
        if identity_mismatch:
            final_metrics = compute_group_metrics([], high_reward=active_thresholds.high_reward)
            decision = route_case(
                case=routing_case,
                verifier_spec=verifier_spec,
                metrics=final_metrics,
                tool_registry=tool_registry,
                gold_reward=entry.get("gold_reward"),
                integrity={"id_mismatch": True},
                thresholds=active_thresholds,
            )
            probe_history.append({"k": 0, "route": decision.route, "additional_rollouts": 0})
        else:
            while True:
                samples = build_sampling_plan(
                    case_id=case_id,
                    start_index=current_k + 1,
                    count=pending_rollouts,
                    base_seed=base_seed,
                    sampling_config=sampling_config,
                )
                for sample in samples:
                    trajectory = rollout_runner(
                        case=case,
                        env_snapshot=env_snapshot,
                        provider=provider,
                        run_id=selected_run_id,
                        rollout_id=sample.rollout_id,
                        max_steps=max_steps,
                        sampling_config=sample.sampling_config,
                    )
                    _validate_trajectory_identity(
                        trajectory, run_id=selected_run_id, case_id=case_id, sample=sample
                    )
                    score = score_fn(
                        case=case,
                        env_snapshot=env_snapshot,
                        verifier_spec=verifier_spec,
                        executed_trajectory=trajectory,
                        sandbox_final_state=trajectory.get("sandbox_final_state"),
                        tool_registry_snapshot=registry_snapshot,
                        verifier_provider=judge,
                    )
                    artifact_dir = artifact_writer(
                        trajectory=trajectory,
                        root=rollout_root,
                        case=case,
                        env_snapshot=env_snapshot,
                        verifier_spec=verifier_spec,
                        score=score,
                        extra_metadata={
                            "entry_id": entry_id,
                            "source_manifest_id": manifest.get("manifest_id"),
                            "rollout_index": sample.rollout_index,
                            "sample_seed": sample.seed,
                            **reward_metadata_from_score(score),
                        },
                        overwrite=False,
                    )
                    case_artifact_dir = Path(artifact_dir).parent
                    _append_run_score(
                        rollout_root=rollout_root,
                        run_id=selected_run_id,
                        case_id=case_id,
                        trajectory=trajectory,
                        score=score,
                        artifact_dir=Path(artifact_dir),
                    )
                    case_rollouts.append(_rollout_metrics(trajectory, score))
                    total_rollouts += 1

                current_k = len(case_rollouts)
                final_metrics = compute_group_metrics(
                    case_rollouts, high_reward=active_thresholds.high_reward
                )
                decision = route_case(
                    case=routing_case,
                    verifier_spec=verifier_spec,
                    metrics=final_metrics,
                    tool_registry=tool_registry,
                    gold_reward=entry.get("gold_reward"),
                    thresholds=active_thresholds,
                )
                promotion = plan_promotion(
                    route=decision.route,
                    current_k=current_k,
                    max_k=maximum,
                )
                probe_history.append(
                    {
                        "k": current_k,
                        "route": decision.route,
                        "additional_rollouts": promotion.additional_rollouts,
                        "promotion_action": promotion.action,
                    }
                )
                if promotion.action != "probe_more":
                    break
                pending_rollouts = promotion.additional_rollouts

        if case_artifact_dir is not None:
            rollout_refs_by_case[case_id] = _portable_path(case_artifact_dir)
            _write_json(
                case_artifact_dir / "group_metrics.json",
                {
                    "case_id": case_id,
                    "metrics": final_metrics,
                    "probe_history": probe_history,
                    "route": decision.model_dump(mode="json"),
                    "source_refs": source_refs,
                },
            )
        else:
            rollout_refs_by_case[case_id] = None

        case_results.append(
            {
                "case_id": case_id,
                "metrics": final_metrics,
                "probe_history": probe_history,
                "decision": decision.model_dump(mode="json"),
                "source_refs": source_refs,
            }
        )

    run_dir = rollout_root / selected_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "case_metrics.jsonl"
    metrics_path.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": item["case_id"],
                    "metrics": item["metrics"],
                    "probe_history": item["probe_history"],
                    "route": item["decision"]["route"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for item in case_results
        ),
        encoding="utf-8",
    )
    routes_path = run_dir / "routes.json"
    _write_json(routes_path, [item["decision"] for item in case_results])
    decisions = [item["decision"] for item in case_results]
    pool_counts = write_pools(
        [RouteDecisionSchema.model_validate(item) for item in decisions],
        pools_dir=pools_root / selected_run_id,
        batch_id=selected_run_id,
        rollouts_root=_portable_path(rollout_root),
        source_manifest_id=manifest.get("manifest_id"),
        source_refs_by_case=source_refs_by_case,
        rollout_refs_by_case=rollout_refs_by_case,
    )

    route_counts = dict(Counter(item["decision"]["route"] for item in case_results))
    summary = {
        "run_id": selected_run_id,
        "source_batch": _portable_path(batch_dir),
        "source_manifest_id": manifest.get("manifest_id"),
        "case_count": len(case_results),
        "rollout_count": total_rollouts,
        "initial_rollouts_per_case": rollouts_per_case,
        "max_rollouts_per_case": maximum,
        "success_threshold": active_thresholds.high_reward,
        "route_thresholds": asdict(active_thresholds),
        "route_counts": route_counts,
        "pool_counts": pool_counts,
        "scores_path": _portable_path(run_dir / "scores.jsonl"),
        "metrics_path": _portable_path(metrics_path),
        "routes_path": _portable_path(routes_path),
        "pools_dir": _portable_path(pools_root / selected_run_id),
        "model": {
            "provider": getattr(provider, "provider_name", getattr(provider, "provider", None)),
            "model_name": getattr(provider, "model", getattr(provider, "model_name", None)),
        },
    }
    _write_json(run_dir / "benchmark_summary.json", summary)
    _write_json(
        run_dir / "benchmark_manifest.json",
        {
            **summary,
            "sampling_config": sampling_config or {},
            "base_seed": base_seed,
            "case_results": case_results,
        },
    )
    return summary


def _rollout_metrics(trajectory: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    tool_errors = trajectory.get("tool_errors") or []
    return {
        "rollout_id": trajectory.get("rollout_id"),
        "reward": score.get("reward", 0.0),
        "active_caps": score.get("active_caps") or [],
        "parse_error": any(error.get("error") == "parse_error" for error in tool_errors),
        "tool_error_llm": sum(1 for error in tool_errors if error.get("source") == "llm"),
        "max_step_hit": not bool((trajectory.get("final_text") or "").strip()),
        "stale": bool(trajectory.get("stale"))
        or any(error.get("error") == "stale" for error in tool_errors),
    }


def _append_run_score(
    *,
    rollout_root: Path,
    run_id: str,
    case_id: str,
    trajectory: dict[str, Any],
    score: dict[str, Any],
    artifact_dir: Path,
) -> None:
    """Append one score row compatible with the existing run report command."""

    run_dir = rollout_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_id,
        "case_id": case_id,
        "rollout_id": trajectory.get("rollout_id"),
        "reward": score.get("reward"),
        "raw_reward": score.get("raw_reward"),
        "subscores": score.get("subscores"),
        "active_caps": score.get("active_caps"),
        "artifact_dir": _portable_path(artifact_dir),
    }
    with (run_dir / "scores.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_trajectory_identity(
    trajectory: dict[str, Any],
    *,
    run_id: str,
    case_id: str,
    sample: RolloutSample,
) -> None:
    expected = {
        "run_id": run_id,
        "case_id": case_id,
        "rollout_id": sample.rollout_id,
    }
    mismatches = {
        field: (trajectory.get(field), value)
        for field, value in expected.items()
        if trajectory.get(field) != value
    }
    if mismatches:
        raise ValueError(f"rollout trajectory identity mismatch: {mismatches}")


def _resolve_batch_file(batch_dir: Path, files: dict[str, Any], name: str) -> Path:
    relative = files.get(name)
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"manifest entry is missing files.{name}")
    path = (batch_dir / relative).resolve()
    try:
        path.relative_to(batch_dir)
    except ValueError as exc:
        raise ValueError(f"manifest files.{name} escapes the batch directory") from exc
    if not path.is_file():
        raise FileNotFoundError(f"manifest files.{name} not found: {path}")
    return path


def _batch_relative_path(batch_dir: Path, path: Path) -> str:
    return path.relative_to(batch_dir).as_posix()


def _portable_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
