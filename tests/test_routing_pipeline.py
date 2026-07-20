
"""Synthetic checks for metrics, routing, adaptive probes, and the full batch flow."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from routing.benchmark import run_benchmark_batch
from routing.metrics import compute_group_metrics
from routing.promotion import plan_promotion
from routing.route_case import route_case
from routing.sampling_policy import build_sampling_plan
from train.reports import summarize_verl_scores
from tests.fixtures import write_synthetic_batch


def test_group_metrics_distinguish_rate_from_any_success() -> None:
    metrics = compute_group_metrics(
        [{"reward": 0.2}, {"reward": 0.9}, {"reward": 0.8}],
        high_reward=0.8,
    )
    assert metrics["k"] == 3
    assert metrics["success_count"] == 2
    assert metrics["rollout_success_rate"] == round(2 / 3, 6)
    assert metrics["pass_at_1_estimate"] == round(2 / 3, 6)
    assert metrics["any_success_at_sampled_k"] is True
    assert metrics["success_threshold"] == 0.8
    assert "success_at_k" not in metrics


def test_route_case_sends_mixed_success_to_grpo() -> None:
    decision = route_case(
        case={"case_id": "CASE-1"},
        verifier_spec={"required_side_effects": []},
        metrics={
            "parse_error_rate": 0.0,
            "tool_error_rate": 0.0,
            "min_reward": 0.2,
            "max_reward": 0.9,
            "reward_spread": 0.7,
            "max_step_hit_rate": 0.0,
            "stale_rate": 0.0,
        },
        tool_registry=set(),
    )
    assert decision.route == "rl_main"


def test_probe_promotion_is_bounded_and_route_sensitive() -> None:
    plan = plan_promotion(route="more_probe", current_k=3, max_k=8)
    assert plan.action == "probe_more"
    assert plan.additional_rollouts == 3
    assert plan_promotion(route="rl_main", current_k=3, max_k=8).additional_rollouts == 0
    assert plan_promotion(route="more_probe", current_k=8, max_k=8).action == "budget_exhausted"


def test_sampling_plan_uses_stable_unique_per_case_seeds() -> None:
    first = build_sampling_plan(
        case_id="CASE-1",
        start_index=1,
        count=3,
        base_seed=42,
        sampling_config={"temperature": 0.6},
    )
    repeated = build_sampling_plan(
        case_id="CASE-1",
        start_index=1,
        count=3,
        base_seed=42,
        sampling_config={"temperature": 0.6},
    )
    assert [item.seed for item in first] == [item.seed for item in repeated]
    assert len({item.seed for item in first}) == 3
    assert [item.rollout_id for item in first] == [
        "rollout_0001",
        "rollout_0002",
        "rollout_0003",
    ]


def test_benchmark_runs_rollouts_routes_case_and_writes_linked_pool_manifest() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        batch_dir = write_synthetic_batch(root, count=1)
        reward_by_index = {1: 0.2, 2: 0.6, 3: 0.4, 4: 0.95}

        def fake_rollout_runner(
            *, case, env_snapshot, provider, run_id, rollout_id, max_steps, sampling_config
        ):
            return {
                "run_id": run_id,
                "case_id": case["case_id"],
                "rollout_id": rollout_id,
                "namespace_id": f"{run_id}:{case['case_id']}:{rollout_id}",
                "parsed_actions": [],
                "tool_errors": [],
                "final_text": "I checked the request.",
                "sandbox_final_state": {},
                "model_metadata": {"provider": "synthetic_test_fixture"},
            }

        def fake_score_fn(*, executed_trajectory, **kwargs):
            rollout_index = int(executed_trajectory["rollout_id"].rsplit("_", 1)[1])
            return {
                "reward": reward_by_index[rollout_index],
                "active_caps": [],
                "diagnostics": {"judge": {"source": "heuristic"}},
            }

        result = run_benchmark_batch(
            batch_dir=batch_dir,
            rollout_root=root / "rollouts",
            pools_root=root / "pools",
            rollouts_per_case=2,
            max_rollouts_per_case=4,
            run_id="synthetic-run",
            provider=object(),
            verifier_provider=None,
            rollout_runner=fake_rollout_runner,
            score_fn=fake_score_fn,
        )

        assert result["case_count"] == 1
        assert result["rollout_count"] == 4
        assert result["route_counts"] == {"rl_main": 1}
        assert (root / "rollouts" / "synthetic-run" / "scores.jsonl").is_file()
        run_dir = root / "rollouts" / "synthetic-run"
        metrics_rows = [
            json.loads(line)
            for line in (run_dir / "case_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert metrics_rows[0]["metrics"]["k"] == 4
        assert metrics_rows[0]["probe_history"][0]["route"] == "more_probe"
        assert metrics_rows[0]["route"] == "rl_main"

        pool_manifest = json.loads(
            (root / "pools" / "synthetic-run" / "pool_rl_main" / "manifest.json")
            .read_text(encoding="utf-8")
        )
        entry = pool_manifest["entries"][0]
        assert entry["source_refs"]["case_ref"] == "cases/TEST_CASE_001.json"
        assert entry["source_refs"]["env_snapshot_ref"] == "env/TEST_CASE_001.json"
        assert entry["source_refs"]["verifier_spec_ref"] == "verifier/TEST_CASE_001.json"
        assert entry["source_refs"]["gold_ref"] == "gold/TEST_CASE_001.json"
        assert Path(entry["rollouts_ref"]).is_dir()


def test_run_report_separates_macro_pass_at_1_from_pooled_rollout_rate() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "scores.jsonl").write_text(
            '{"case_id":"A","reward":0.9}\n'
            '{"case_id":"A","reward":0.1}\n'
            '{"case_id":"B","reward":0.9}\n',
            encoding="utf-8",
        )
        summary = summarize_verl_scores(
            run_dir=run_dir,
            out_dir=root / "report",
            high_reward=0.8,
        )
        assert summary["rollout_success_rate"] == round(2 / 3, 6)
        assert summary["pass_at_1"] == 0.75
        assert summary["any_success_case_rate_at_sampled_k"] == 1.0
