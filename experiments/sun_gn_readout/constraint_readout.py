"""Exact readout of public four-distinct-digit guessing feedback.

The resolver accepts no target answer. It emits an answer only when the public
records leave exactly one candidate, otherwise it abstains. Standard library only.
"""
from collections import Counter
from itertools import permutations
import json
import sys


def feedback(candidate, guess):
    same = sum(a == b for a, b in zip(candidate, guess))
    common = sum((Counter(candidate) & Counter(guess)).values())
    return same, common - same


def feasible_candidates(records):
    checked = []
    for row in records:
        if set(row) != {"turn", "guess", "same", "different"}:
            raise ValueError("Only public turn/guess/same/different fields are accepted")
        guess = row["guess"]
        if not isinstance(guess, str) or len(guess) != 4 or any(c not in "0123456789" for c in guess):
            raise ValueError("A guess must contain four ASCII digits")
        same, different, turn = row["same"], row["different"], row["turn"]
        if any(type(v) is not int for v in (same, different, turn)) or turn < 1:
            raise ValueError("Counts and turn must be integers")
        if min(same, different) < 0 or same + different > 4:
            raise ValueError("Invalid feedback counts")
        checked.append((guess, same, different))
    candidates = []
    for digits in permutations("0123456789", 4):
        candidate = "".join(digits)
        if all(feedback(candidate, guess) == (same, different) for guess, same, different in checked):
            candidates.append(candidate)
    return candidates


def resolve_public(records):
    candidates = feasible_candidates(records)
    count = len(candidates)
    return {
        "status": "unique" if count == 1 else ("inconsistent" if count == 0 else "ambiguous"),
        "candidate_count": count,
        "answer": candidates[0] if count == 1 else None,
    }


def public_state(records, list_cap=64):
    """A fixed size-limited state for model input, using public evidence only."""
    if type(list_cap) is not int or list_cap < 1:
        raise ValueError("A positive integer listing limit is required")
    candidates = feasible_candidates(records)
    if not candidates:
        raise ValueError("Public records are inconsistent")
    return {"candidate_count": len(candidates),
            "representation": "complete_list" if len(candidates) <= list_cap else "count_only",
            "candidates": candidates if len(candidates) <= list_cap else None,
            "listing_limit": list_cap,
            "source": "exact filtering of public guesses and feedback"}


if __name__ == "__main__":
    # One JSON list of public records on stdin. No labels or model access.
    print(json.dumps(resolve_public(json.load(sys.stdin)), sort_keys=True))
