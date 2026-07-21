
"""Synthetic-fixture checks for SFT and GRPO dataset row construction."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.fixtures import write_synthetic_batch
from train.grpo_builder import build_grpo_dataset
from train.sft_builder import build_sft_dataset


def test_sft_replays_gold_actions_and_grpo_keeps_prompt_only_target() -> None:
    with TemporaryDirectory() as temp:
        root = Path(temp)
        batch_dir = write_synthetic_batch(root, count=2)

        sft = build_sft_dataset(
            batch_dir=batch_dir,
            out_dir=root / "sft",
            val_every=2,
            write_parquet=False,
        )
        sft_rows = [
            json.loads(line)
            for line in sft.jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        sft_messages = sft_rows[0]["messages"]
        assert any(message.get("tool_calls") for message in sft_messages)
        assert any(message.get("role") == "tool" for message in sft_messages)
        assert sft_messages[-1] == {
            "role": "assistant",
            "content": "I checked the order status.",
        }

        grpo = build_grpo_dataset(
            batch_dir=batch_dir,
            out_dir=root / "grpo",
            rollout_root=root / "rollouts",
            val_every=2,
            write_parquet=False,
        )
        grpo_rows = [
            json.loads(line)
            for line in grpo.jsonl_path.read_text(encoding="utf-8").splitlines()
        ]
        row = grpo_rows[0]
        assert all(message["role"] in {"system", "user"} for message in row["prompt"])
        assert row["reward_model"]["style"] == "industrial_posttrain_verifier"
        assert row["extra_info"]["case_path"].endswith("TEST_CASE_001.json")
        assert row["extra_info"]["env_snapshot_path"].endswith("TEST_CASE_001.json")
        assert row["extra_info"]["verifier_spec_path"].endswith("TEST_CASE_001.json")
        assert row["extra_info"]["gold_path"].endswith("TEST_CASE_001.json")
        assert grpo.train_count == 1 and grpo.val_count == 1
