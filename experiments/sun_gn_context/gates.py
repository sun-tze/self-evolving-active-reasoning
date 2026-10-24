"""Frozen development-only gates and deterministic admission plans."""
from __future__ import annotations

from math import ceil

from .core import IntegrityError
from .freeze import CONDITIONS, condition_order


def evaluation_plan(manifest: dict, N: int):
    if N not in (50, 100):
        raise ValueError("Only the prespecified 100/50 alternatives are permitted")
    rows = [r for r in manifest["rows"] if r["stage"] == "evaluation"][:N]
    if [r["row"] for r in rows] != list(range(N)):
        raise IntegrityError("Evaluation rows changed")
    return [(row, condition) for row in rows for condition in condition_order(row["row"])]


def development_gate(assets, episodes: list[dict], scores: list[dict], canaries: list[dict], *,
                     dev_elapsed_seconds: float, startup_save_seconds: float) -> dict:
    expected = {(r["case_id"],c) for r in assets.manifest["rows"] if r["stage"]=="development" for c in r["conditions"]}
    observed = {(r["case_id"],r["condition"]) for r in episodes}
    scored = {(r["case_id"],r["condition"]) for r in scores}
    if len(episodes)!=24 or len(scores)!=24 or observed != expected or scored != expected or len(canaries)!=2:
        raise IntegrityError("Incomplete development/canary coverage")
    if any(not r.get("completed") or len(r["slots"])!=26 for r in episodes):
        raise IntegrityError("Development infrastructure failure")
    actor_calls = 24*26
    invalid = sum(r["invalid_outputs"] for r in scores)
    truncated = sum(r["truncated_outputs"] for r in scores)
    b0 = [r for r in scores if r["condition"]=="B0"]
    reasons = []
    if invalid/actor_calls > .10:
        reasons.append("parse_failure_rate")
    if truncated/actor_calls > .05:
        reasons.append("truncation_rate")
    if all(not r["validation"]["accepted"] for r in canaries):
        reasons.append("both_canaries_invalid")
    if all(r["Q"]==0 for r in b0) or sum(r["Q"]>=.98 for r in b0)>=18:
        reasons.append("insufficient_Q_range")
    if min(dev_elapsed_seconds,startup_save_seconds) < 0:
        raise IntegrityError("Negative elapsed time")
    actor_seconds = 2*max(r["elapsed_seconds"] for r in episodes)
    curator_seconds = 2*max(r["elapsed_seconds"] for r in canaries)
    allowance = max(600.,2*startup_save_seconds)
    stage_one = allowance+dev_elapsed_seconds+20*(actor_seconds+curator_seconds)
    stage_two = {N:allowance+ceil(4*N/8)*actor_seconds for N in (100,50)}
    if stage_one>7200:
        reasons.append("stage_one_over_120_minutes")
    N = 100 if stage_two[100]<=5400 else 50 if stage_two[50]<=5400 else None
    if N is None:
        reasons.append("stage_two_over_90_minutes")
    return dict(passed=not reasons, reasons=reasons, selected_N=N if not reasons else None,
                actor_calls=actor_calls, invalid_outputs=invalid,truncated_outputs=truncated,
                parse_failure_rate=invalid/actor_calls,truncation_rate=truncated/actor_calls,
                floor_bound=sum(r["final_EM"] for r in b0)<2,
                actor_conservative_seconds=actor_seconds,curator_conservative_seconds=curator_seconds,
                allowance_seconds=allowance,stage_one_estimated_seconds=stage_one,
                stage_two_estimated_seconds={str(N):seconds for N,seconds in stage_two.items()},
                frozen_before_adaptation_and_test=True)
