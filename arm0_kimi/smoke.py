"""CLI for the representation baseline.

Default cases are synthetic plumbing. ``--cases official`` is the scientific
path and does not fall back to ``make_cases``. ``--out`` is required so a run
cannot write into the repository tree by accident. This module does not log
in to a cluster, submit a job, or open a tunnel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .prompts import ARMS
from .runner import RunConfig, run_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "GN representation baseline (chrono | ledger control | ledger_belief candidate). "
            "Not a context-evolution method and not a live Engaging launcher."
        )
    )
    parser.add_argument("--arm", choices=("all",) + ARMS, default="all")
    parser.add_argument("--backend", choices=("mock", "kimi", "vllm"), default="mock")
    parser.add_argument(
        "--cases",
        choices=("synthetic", "official"),
        default="synthetic",
        help="synthetic is plumbing and resamples per seed; official is fixed test.json[:n] and must be explicit",
    )
    parser.add_argument("--n", type=int, default=2, help="Prefix length of the case list. Smoke recommendation 2.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default=None, help="Comma-separated decoding seeds, e.g. 0,1,2")
    parser.add_argument("--model", default=None, help="Default Qwen/Qwen2.5-7B-Instruct on vllm; kimi-k3.1-pro on kimi")
    parser.add_argument("--base-url", default=None, help="vllm default http://127.0.0.1:18000/v1")
    parser.add_argument("--gn-json", type=Path, default=None, help="Override path to official data/gn/test.json")
    parser.add_argument("--temperature", type=float, default=None, help="vllm default 0.7; sent on the request, not only the receipt")
    parser.add_argument("--max-turns", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=None, help="vllm default 800; kimi default 4096")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request seconds; must be >= 600")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default="low",
        help="Kimi only. Omitted on the vllm/Qwen request.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Receipt directory. Required. Use a path outside this repository.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out is None:
        print("--out is required so receipts are not written into the repository.", file=sys.stderr)
        return 2
    if args.n > 3 and args.seeds is None:
        print("n>3 requires an explicit --seeds protocol (e.g. --seeds 0,1,2).", file=sys.stderr)
        return 2
    seeds = (args.seed,) if args.seeds is None else tuple(int(item) for item in args.seeds.split(",") if item != "")
    if not seeds or len(set(seeds)) != len(seeds):
        print("Seeds must be non-empty and unique", file=sys.stderr)
        return 2
    arms = ARMS if args.arm == "all" else (args.arm,)
    try:
        configs = [
            RunConfig(
                arm=arm,
                n=args.n,
                seeds=seeds,
                backend=args.backend,
                case_source=args.cases,
                model=args.model,
                max_turns=args.max_turns,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                base_url=args.base_url,
                gn_json=args.gn_json,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
            )
            for arm in arms
        ]
        if args.backend == "kimi":
            from .client import KimiActor
            KimiActor(
                model=configs[0].model,
                base_url=configs[0].base_url,
                timeout=configs[0].timeout,
                max_tokens=configs[0].max_tokens,
                reasoning_effort=configs[0].reasoning_effort,
            )
        if args.backend == "vllm":
            from .client import QwenActor
            # Construct only. chat() is what opens a socket.
            QwenActor(
                model=configs[0].model,
                base_url=configs[0].base_url,
                timeout=configs[0].timeout,
                max_tokens=configs[0].max_tokens,
                temperature=configs[0].temperature,
            )
        result = run_matrix(configs, args.out.resolve())
    except (ValueError, RuntimeError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "study_label": result["study_label"],
        "context_evolution": False,
        "scientific_lift_claimed": False,
        "out_dir": str(args.out.resolve()),
        "backend": result["backend"],
        "model": result["model"],
        "case_source": result["case_source"],
        "official_sha256": result["official_sha256"],
        "case_ids": result["case_ids"],
        "matched_seeds": result["matched_seeds"],
        "decoding_seeds_only": result["decoding_seeds_only"],
        "temperature_sent": result["temperature_sent"],
        "max_tokens_sent": result["max_tokens_sent"],
        "reasoning_effort_sent": result["reasoning_effort_sent"],
        "top_p_sent": result["top_p_sent"],
        "resumed_episodes": result["resumed_episodes"],
        "executed_episodes": result["executed_episodes"],
        "by_arm": result["by_arm"],
        "readout_passes": result["readout"]["passes"],
        "fail_closed_reasons": result["readout"]["fail_closed_reasons"],
        "usage": result["usage"],
        "pricing_estimate_usd": None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
