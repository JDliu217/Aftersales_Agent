
"""Synthetic checks for rollout isolation and tool-call parsing."""

from agent.runtime import parse_tool_calls
from envs.namespace import build_namespace_id
from envs.sandbox_state import SandboxState


def test_rollout_namespaces_are_isolated() -> None:
    first = build_namespace_id("run-a", "case-a", "rollout-1")
    second = build_namespace_id("run-a", "case-a", "rollout-2")
    other_case = build_namespace_id("run-a", "case-b", "rollout-1")
    assert len({first, second, other_case}) == 3


def test_sandbox_records_are_filtered_by_rollout_namespace() -> None:
    first = build_namespace_id("run-a", "case-a", "rollout-1")
    second = build_namespace_id("run-a", "case-a", "rollout-2")
    sandbox = SandboxState(namespace_id=first)
    sandbox.write_record(
        "finance.issue_refund",
        {"amount": 10},
        namespace_id=first,
        run_id="run-a",
        case_id="case-a",
        rollout_id="rollout-1",
        tool_call_id="call-1",
    )
    assert len(sandbox.records_for_tool("finance.issue_refund", first)) == 1
    assert sandbox.records_for_tool("finance.issue_refund", second) == []


def test_tool_call_parser_handles_valid_and_invalid_blocks() -> None:
    valid = '<tool_call>{"name":"policy.search","arguments":{"topic":"refund"}}</tool_call>'
    calls, error = parse_tool_calls(valid)
    assert error is None
    assert calls == [{"name": "policy.search", "arguments": {"topic": "refund"}}]

    calls, error = parse_tool_calls("<tool_call>{bad json}</tool_call>")
    assert calls == []
    assert error is not None and error.startswith("invalid_tool_call_json")

