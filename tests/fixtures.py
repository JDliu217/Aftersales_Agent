
"""Small synthetic fixtures used only by the local unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_synthetic_batch(root: Path, count: int = 2) -> Path:
    """Create a minimal Case/env/verifier/gold batch under a test temp directory."""

    batch_dir = root / "synthetic_batch"
    entries: list[dict[str, Any]] = []
    for index in range(count):
        case_id = f"TEST_CASE_{index + 1:03d}"
        files = {
            "case": f"cases/{case_id}.json",
            "env_snapshot": f"env/{case_id}.json",
            "verifier_spec": f"verifier/{case_id}.json",
            "gold": f"gold/{case_id}.json",
        }
        payloads = {
            "case": {
                "case_id": case_id,
                "market": "US",
                "customer_message": "Please check my order.",
                "entities": {"order_id": f"ORDER_{index + 1:03d}"},
            },
            "env_snapshot": {
                "case_id": case_id,
                "reference_now": "2026-10-20T00:00:00Z",
                "readonly_tables": {},
                "policies": [],
            },
            "verifier_spec": {
                "required_response_points": [
                    {"id": "status", "description": "Explain the order status."}
                ]
            },
            "gold": {
                "gold_trajectory": {
                    "parsed_actions": [
                        {
                            "name": "oms.get_order",
                            "arguments": {"order_id": f"ORDER_{index + 1:03d}"},
                            "tool_call_id": f"call_{index + 1}",
                            "step": 1,
                        }
                    ],
                    "tool_observations": [
                        {
                            "tool_name": "oms.get_order",
                            "tool_call_id": f"call_{index + 1}",
                            "ok": True,
                            "result": {"status": "shipped"},
                            "namespace_id": f"gold:{case_id}:rollout_0001",
                        }
                    ],
                },
                "final_text": "I checked the order status.",
            },
        }
        for name, relative_path in files.items():
            path = batch_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payloads[name], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        entries.append(
            {
                "id": f"entry_{index + 1:03d}",
                "case_id": case_id,
                "gold_reward": 1.0,
                "files": files,
                "metadata": {"topic": "synthetic_test_fixture"},
            }
        )

    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "manifest.json").write_text(
        json.dumps(
            {"manifest_id": "synthetic-fixture-v1", "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return batch_dir
