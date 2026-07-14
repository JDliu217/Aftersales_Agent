
"""Run baseline multi-rollout evaluation and route each case into a pool."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.providers.factory import provider_from_env, verifier_provider_from_env  # noqa: E402
from agent.rollout_store import make_run_id  # noqa: E402
from routing.benchmark import run_benchmark_batch  # noqa: E402
from routing.route_case import Thresholds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run K baseline rollouts per case, aggregate case metrics, route cases, "
            "and write pool manifests."
        )
    )
    parser.add_argument("--batch", default="data/batches/benchmark")
    parser.add_argument("--rollouts-per-case", type=int, required=True)
    parser.add_argument(
        "--max-rollouts-per-case",
        type=int,
        help="Optional probe budget; only more_probe cases receive additional samples.",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--high-reward", type=float, default=0.80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--provider", choices=["vllm", "local_hf", "qwen", "deepseek"])
    parser.add_argument(
        "--judge-provider",
        choices=["qwen", "deepseek", "vllm", "local_hf", "none", "heuristic"],
        help="Defaults to VERIFIER_PROVIDER (which defaults to qwen).",
    )
    parser.add_argument("--rollout-root", default="data/rollouts_baseline")
    parser.add_argument("--pools-root", default="data/pools")
    args = parser.parse_args()

    batch_dir = ROOT / args.batch
    if not (batch_dir / "manifest.json").is_file():
        parser.error(f"batch manifest not found: {batch_dir / 'manifest.json'}")

    try:
        model_provider = provider_from_env(args.provider)
        if args.judge_provider is None:
            judge_provider = verifier_provider_from_env()
        elif args.judge_provider in {"none", "heuristic"}:
            judge_provider = None
        else:
            judge_provider = provider_from_env(args.judge_provider)

        run_id = args.run_id or make_run_id("baseline")
        result = run_benchmark_batch(
            batch_dir=batch_dir,
            rollout_root=ROOT / args.rollout_root,
            pools_root=ROOT / args.pools_root,
            rollouts_per_case=args.rollouts_per_case,
            max_rollouts_per_case=args.max_rollouts_per_case,
            run_id=run_id,
            base_seed=args.base_seed,
            sampling_config={"temperature": args.temperature, "top_p": args.top_p},
            max_steps=args.max_steps,
            thresholds=Thresholds(high_reward=args.high_reward),
            provider=model_provider,
            verifier_provider=judge_provider,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] benchmark failed: {type(exc).__name__}: {exc}")
        return 1

    print(
        f"[OK] benchmark run={result['run_id']} cases={result['case_count']} "
        f"rollouts={result['rollout_count']}"
    )
    print(f"     metrics={result['metrics_path']}")
    print(f"     routes={result['routes_path']}")
    print(f"     pools={result['pools_dir']}")
    for route, count in sorted(result["route_counts"].items()):
        print(f"     route {route}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
