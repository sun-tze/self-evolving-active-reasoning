"""Success bar for the representation baseline.

The candidate is ``ledger_belief``. The control is ``ledger``.
Diagnostics are reported and are not themselves success.
A gate pass on supplied aggregates is not a context-evolution result
and is not, by itself, a claim that this repository observed a lift.
"""
from __future__ import annotations

from typing import Any

CONTROL_ARM = "ledger"
CANDIDATE_ARM = "ledger_belief"
REQUIRED_SEEDS = (0, 1, 2)
STUDY_LABEL = "representation_baseline"


def _mean_ready(block: dict | None) -> bool:
    return bool(block) and int(block.get("episodes") or 0) > 0 and block.get("EM") is not None


def _moved(candidate: dict, control: dict) -> bool:
    """Helpful direction only: consistency up, or repeat rate down."""
    consistency_up = _strict(candidate.get("feedback_consistency"), control.get("feedback_consistency"), higher=True)
    repeat_down = _strict(candidate.get("repeat_guess"), control.get("repeat_guess"), higher=False)
    return consistency_up or repeat_down


def _strict(new: Any, old: Any, *, higher: bool) -> bool:
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return False
    if isinstance(new, bool) or isinstance(old, bool):
        return False
    return new > old if higher else new < old


def _seed_ids(per_seed: dict | None) -> set[int]:
    if not per_seed:
        return set()
    found = set()
    for key in per_seed:
        try:
            found.add(int(key))
        except (TypeError, ValueError):
            continue
    return found


def representation_baseline_readout(
    by_arm: dict[str, dict],
    per_seed: dict | None = None,
    required_seeds: tuple[int, ...] = REQUIRED_SEEDS,
) -> dict[str, Any]:
    """Evaluate the locked bar on aggregates the caller already computed.

    ``passes`` is true only when the candidate's pooled final-guess EM is
    strictly above the ledger control, a consistency diagnostic moves with
    that EM, and seeds ``required_seeds`` are all present. Beating ``chrono``
    alone, a flat EM with a higher repeat rate, and an EM lift with no
    consistency move all fail closed.
    """
    by_arm = by_arm or {}
    candidate = by_arm.get(CANDIDATE_ARM) or {}
    control = by_arm.get(CONTROL_ARM) or {}
    chrono = by_arm.get("chrono")
    reasons: list[str] = []

    if not _mean_ready(control) or not _mean_ready(candidate):
        reasons.append("missing_control_or_candidate_em")
    else:
        if int(candidate.get("episodes") or 0) != int(control.get("episodes") or 0):
            reasons.append("unmatched_episode_counts")
        if chrono is not None and (
            not _mean_ready(chrono) or int(chrono.get("episodes") or 0) != int(control.get("episodes") or 0)
        ):
            reasons.append("unmatched_episode_counts")

    cand_em = candidate.get("EM") if _mean_ready(candidate) else None
    ctrl_em = control.get("EM") if _mean_ready(control) else None
    chrono_em = chrono.get("EM") if chrono is not None and _mean_ready(chrono) else None
    beats_ledger = (
        isinstance(cand_em, (int, float)) and not isinstance(cand_em, bool)
        and isinstance(ctrl_em, (int, float)) and not isinstance(ctrl_em, bool)
        and cand_em > ctrl_em
    )
    beats_chrono = (
        isinstance(cand_em, (int, float)) and not isinstance(cand_em, bool)
        and isinstance(chrono_em, (int, float)) and not isinstance(chrono_em, bool)
        and cand_em > chrono_em
    )
    if cand_em is not None and ctrl_em is not None and not beats_ledger:
        reasons.append("candidate_does_not_beat_ledger_control")
    if beats_chrono and not beats_ledger:
        reasons.append("beats_chrono_alone_is_not_success")
    if (
        isinstance(cand_em, (int, float)) and isinstance(ctrl_em, (int, float))
        and not isinstance(cand_em, bool) and not isinstance(ctrl_em, bool)
        and cand_em == ctrl_em
        and _strict(candidate.get("repeat_guess"), control.get("repeat_guess"), higher=True)
    ):
        reasons.append("em_flat_while_repeat_rises")
    consistency_moved = _mean_ready(candidate) and _mean_ready(control) and _moved(candidate, control)
    if beats_ledger and not consistency_moved:
        reasons.append("em_lift_without_consistency_move")
    if (
        isinstance(cand_em, (int, float)) and isinstance(ctrl_em, (int, float))
        and isinstance(chrono_em, (int, float))
        and not any(isinstance(v, bool) for v in (cand_em, ctrl_em, chrono_em))
        and (cand_em - chrono_em) < (ctrl_em - chrono_em)
    ):
        reasons.append("lift_vs_chrono_loses_to_ledger")

    present_seeds = _seed_ids(per_seed)
    if any(seed not in present_seeds for seed in required_seeds):
        reasons.append("multi_seed_coverage_incomplete")

    # Deduplicate while keeping a stable order.
    ordered: list[str] = []
    for reason in reasons:
        if reason not in ordered:
            ordered.append(reason)

    blocking = set(ordered)
    passes = beats_ledger and consistency_moved and not blocking

    per_seed_beats = []
    for seed in sorted(present_seeds):
        block = per_seed.get(seed) if per_seed and seed in per_seed else None
        if block is None and per_seed is not None:
            block = per_seed.get(str(seed))
        if not isinstance(block, dict):
            per_seed_beats.append({"seed": seed, "candidate_beats_ledger": None})
            continue
        per_seed_beats.append({
            "seed": seed,
            "candidate_beats_ledger": _em_beats(block.get(CANDIDATE_ARM), block.get(CONTROL_ARM)),
            "EM": {
                arm: (block.get(arm) or {}).get("EM")
                for arm in ("chrono", CONTROL_ARM, CANDIDATE_ARM)
                if arm in block
            },
        })

    return {
        "study_label": STUDY_LABEL,
        "context_evolution": False,
        "scientific_lift_claimed": False,
        "diagnostics_are_success": False,
        "control": CONTROL_ARM,
        "candidate": CANDIDATE_ARM,
        "primary_metric": "final_guess_exact_match",
        "singleton_substitution": False,
        "beats_ledger": beats_ledger,
        "beats_chrono": beats_chrono,
        "consistency_moved_with_em": consistency_moved,
        "passes": passes,
        "fail_closed_reasons": ordered,
        "required_seeds": list(required_seeds),
        "observed_seeds": sorted(present_seeds),
        "per_seed": per_seed_beats,
        "diagnostics": {
            arm: {
                "episodes": (by_arm.get(arm) or {}).get("episodes"),
                "EM": (by_arm.get(arm) or {}).get("EM"),
                "repeat_guess": (by_arm.get(arm) or {}).get("repeat_guess"),
                "feedback_consistency": (by_arm.get(arm) or {}).get("feedback_consistency"),
            }
            for arm in ("chrono", CONTROL_ARM, CANDIDATE_ARM)
            if arm in by_arm
        },
        "interpretation": (
            "Representation-baseline gate on these aggregates. "
            "Diagnostics are not success. A gate pass is not a context-evolution result."
        ),
    }


def _em_beats(candidate: dict | None, control: dict | None) -> bool | None:
    if not _mean_ready(candidate) or not _mean_ready(control):
        return None
    return candidate["EM"] > control["EM"]
