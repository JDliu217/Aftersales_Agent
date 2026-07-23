
"""Synthetic checks for verifier reward caps and provider-aware metadata."""

from agent.verifier import score_trajectory
from envs.sandbox_state import SandboxState
from schemas.reward_schema import ACTIVE_CAP_VALUES
from train.verl_reward_adapter import reward_metadata_from_score


def test_multi_tool_protocol_violation_caps_reward_at_zero() -> None:
    score = score_trajectory(
        case={"case_id": "TEST_CASE"},
        env_snapshot={},
        verifier_spec={},
        executed_trajectory={
            "parsed_actions": [
                {"step": 1, "name": "oms.get_order"},
                {"step": 1, "name": "policy.search"},
            ]
        },
    )
    assert score["active_caps"] == ["multi_tool_per_step_cap"]
    assert score["reward"] == ACTIVE_CAP_VALUES["multi_tool_per_step_cap"] == 0.0


def test_unauthorized_write_is_capped_without_a_live_judge() -> None:
    namespace_id = "run-test:TEST_CASE:rollout_0001"
    sandbox = SandboxState(namespace_id=namespace_id)
    sandbox.write_record(
        "finance.issue_refund",
        {"order_id": "ORDER_1", "amount": 10},
        namespace_id=namespace_id,
        run_id="run-test",
        case_id="TEST_CASE",
        rollout_id="rollout_0001",
        tool_call_id="call-1",
    )
    score = score_trajectory(
        case={"case_id": "TEST_CASE", "entities": {"order_id": "ORDER_1"}},
        env_snapshot={"case_id": "TEST_CASE", "readonly_tables": {}, "policies": []},
        verifier_spec={
            "allowed_write_tools": [],
            "required_response_points": [
                {"id": "status", "description": "State the result."}
            ],
        },
        executed_trajectory={
            "case_id": "TEST_CASE",
            "namespace_id": namespace_id,
            "parsed_actions": [
                {
                    "name": "finance.issue_refund",
                    "arguments": {"order_id": "ORDER_1", "amount": 10},
                    "tool_call_id": "call-1",
                    "step": 1,
                }
            ],
            "tool_observations": [
                {"tool_call_id": "call-1", "ok": True, "result": {"ok": True}}
            ],
            "sandbox_final_state": sandbox.export(),
            "final_text": "The refund is complete.",
        },
        sandbox_final_state=sandbox.export(),
        verifier_provider=None,
    )
    assert "unauthorized_action_cap" in score["active_caps"]
    assert score["reward"] <= ACTIVE_CAP_VALUES["unauthorized_action_cap"]
    assert score["diagnostics"]["judge"]["source"] == "heuristic"


def test_reward_metadata_records_actual_judge_source() -> None:
    metadata = reward_metadata_from_score(
        {
            "diagnostics": {
                "judge": {
                    "source": "provider",
                    "provider": "qwen",
                    "model_name": "qwen-test",
                    "parse_error": "invalid json",
                }
            }
        }
    )
    assert metadata["reward_model"] == "industrial_posttrain_verifier"
    assert metadata["judge_source"] == "provider"
    assert metadata["judge_effective_source"] == "heuristic_fallback"
    assert metadata["judge_provider"] == "qwen"

