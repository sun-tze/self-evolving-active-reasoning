#!/usr/bin/env python3
"""Exact public-universe properties of the two frozen learned query schedules.

Standard library only. Reads only learned-policies.json beside this file.
No private case data, labels, trajectories, model, network, or experiment runner.
Run: python public_schedule_analysis.py
"""
from collections import Counter
from itertools import permutations
from pathlib import Path
import argparse
import hashlib
import json
import time

EXPECTED_NOTE_SHA256 = {
    "E0": "14e2eea7b43d2f4cfad75665dd37c1b3c740ba33e1af561ed19ed19a34ba1231",
    "E1": "1d56f8611c3c277d9fac9486c8c6edd44339439ad3eb767f6d90f4fc1c334a2d",
}


def mask(word):
    return sum(1 << int(char) for char in word)


def feedback_code(secret, query, secret_mask, query_mask):
    bulls = sum(a == b for a, b in zip(secret, query))
    cows = (secret_mask & query_mask).bit_count() - bulls
    return 5 * bulls + cows


def partition_summary(counts, prefix_length, population):
    sizes = Counter(counts.values())
    singleton = sizes.get(1, 0)
    ambiguous = [size for size in sizes if size > 1]
    return {
        "prefix_length": prefix_length,
        "hypotheses": population,
        "group_count": len(counts),
        "singleton_hypothesis_count": singleton,
        "singleton_fraction": singleton / population,
        "ambiguous_hypothesis_count": population - singleton,
        "maximum_ambiguous_group_size": max(ambiguous, default=0),
        "group_size_distribution": {str(size): sizes[size] for size in sorted(sizes)},
    }


def analyze(universe, queries, keep_memberships=False):
    universe = list(universe)
    if not universe or len(set(universe)) != len(universe):
        raise ValueError("Universe must be nonempty with distinct hypotheses")
    width = len(universe[0])
    if width > 4 or any(len(s) != width or len(set(s)) != width or
                       any(c not in "0123456789" for c in s) for s in universe + queries):
        raise ValueError("Require equal-width distinct-digit strings of at most four digits")
    masks = [mask(s) for s in universe]
    signatures = [0] * len(universe)
    first = [None] * len(universe)
    counts = Counter(signatures)
    stages = [partition_summary(counts, 0, len(universe))]
    memberships = [list(signatures)] if keep_memberships else None
    if len(universe) == 1:
        first[0] = 0
    for step, query in enumerate(queries, 1):
        query_mask = mask(query)
        for index, secret in enumerate(universe):
            code = feedback_code(secret, query, masks[index], query_mask)
            signatures[index] = signatures[index] * 25 + code
        counts = Counter(signatures)
        stages.append(partition_summary(counts, step, len(universe)))
        for index, signature in enumerate(signatures):
            if first[index] is None and counts[signature] == 1:
                first[index] = step
        if keep_memberships:
            memberships.append(list(signatures))
    found = [cost for cost in first if cost is not None]
    histogram = Counter(found)
    result = {
        "query_count": len(queries),
        "queries": queries,
        "prefix_partitions": stages,
        "first_singleton_within_list": {
            "identified_hypotheses": len(found),
            "unresolved_after_list": len(universe) - len(found),
            "count_by_first_singleton_query": {str(k): histogram.get(k, 0)
                                                for k in range(len(queries) + 1)},
            "sum_query_cost_for_identified_hypotheses": sum(found),
            "mean_queries_conditional_on_identification_within_list":
                sum(found) / len(found) if found else None,
            "unresolved_future_query_cost": None,
            "unconditional_eventual_mean_query_cost": None,
            "meaning": "Unresolved costs beyond the listed sequence are unknown; no capped25 extrapolation.",
        },
    }
    return result, memberships


def reference_feedback(secret, query):
    # Deliberately independent, direct count-based definition for the tiny check.
    bulls = sum(secret[index] == query[index] for index in range(len(secret)))
    shared = sum(min(secret.count(char), query.count(char)) for char in "0123456789")
    return bulls, shared - bulls


def self_check():
    universe = ["".join(p) for p in permutations("0123", 3)]
    queries = ["012", "120", "301"]
    result, memberships = analyze(universe, queries, keep_memberships=True)
    equivalence_checks = 0
    for k, labels in enumerate(memberships):
        for i, secret in enumerate(universe):
            expected_class = [j for j, candidate in enumerate(universe)
                              if all(reference_feedback(secret, q) == reference_feedback(candidate, q)
                                     for q in queries[:k])]
            actual_class = [j for j, label in enumerate(labels) if label == labels[i]]
            if expected_class != actual_class:
                raise AssertionError("Feedback equivalence partition mismatch")
            equivalence_checks += 1
        stage = result["prefix_partitions"][k]
        sizes = {int(size): count for size, count in stage["group_size_distribution"].items()}
        if sum(size * count for size, count in sizes.items()) != len(universe):
            raise AssertionError("Partition mass mismatch")
    reference_first = []
    for secret in universe:
        first = None
        for k in range(len(queries) + 1):
            remaining = sum(all(reference_feedback(secret, q) == reference_feedback(candidate, q)
                                for q in queries[:k]) for candidate in universe)
            if remaining == 1:
                first = k
                break
        reference_first.append(first)
    found = [x for x in reference_first if x is not None]
    costs = result["first_singleton_within_list"]
    if costs["identified_hypotheses"] != len(found) or costs["sum_query_cost_for_identified_hypotheses"] != sum(found):
        raise AssertionError("First-singleton cost mismatch")
    # Verify both learned schedules' individual feedback codes against the
    # independent definition on a few public four-digit hypothetical secrets.
    sample_secrets = ["0123", "9876", "3049", "2468"]
    all_queries = ["1234", "5678", "9012", "3490", "6789", "4390", "2879",
                   "3456", "7890", "1357", "2468", "0379", "1478"]
    for secret in sample_secrets:
        for query in all_queries:
            bulls, cows = reference_feedback(secret, query)
            if feedback_code(secret, query, mask(secret), mask(query)) != 5 * bulls + cows:
                raise AssertionError("Public four-digit feedback mismatch")
    return {"status": "PASS", "tiny_hypotheses": len(universe),
            "tiny_query_prefixes_including_zero": 4,
            "exact_equivalence_class_checks": equivalence_checks,
            "four_digit_feedback_checks": len(sample_secrets) * len(all_queries),
            "first_singleton_cost_checked_against_independent_bruteforce": True}


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().with_name("public-schedule-partitions.json"))
    args = parser.parse_args()
    policy_path = Path(__file__).resolve().with_name("learned-policies.json")
    policy_raw = policy_path.read_bytes()
    policy_document = json.loads(policy_raw)
    correctness = self_check()
    universe = ["".join(p) for p in permutations("0123456789", 4)]
    schedules = {}
    for arm, expected in EXPECTED_NOTE_SHA256.items():
        note = policy_document["policies"][arm]
        if hashlib.sha256(note.encode()).hexdigest() != expected:
            raise ValueError("Frozen selected schedule changed: " + arm)
        lines = note.splitlines()
        if any(not line.startswith("- ") for line in lines):
            raise ValueError("Unexpected learned note format")
        queries = [line[2:] for line in lines]
        schedules[arm], _ = analyze(universe, queries)
        schedules[arm]["frozen_note_sha256"] = expected
    document = {
        "format": "gn_public_selected_schedule_partitions_v1",
        "universe": {"definition": "All ordered four-distinct-digit permutations from 0..9; leading zero allowed",
                     "hypotheses": len(universe), "weighting": "Each public hypothetical receives equal weight"},
        "method": "Exact bulls/cows feedback-signature partitions at every query prefix. No optimizer or actor execution.",
        "source": {"file": "learned-policies.json", "sha256": hashlib.sha256(policy_raw).hexdigest()},
        "correctness_check": correctness,
        "definitions": {
            "singleton_fraction": "Singleton hypothesis count divided by5040, not fraction of groups.",
            "group_size_distribution": "Keys are group sizes; values are numbers of feedback-equivalence groups of that size.",
            "maximum_ambiguous_group_size": "Largest group with size>1; zero if all groups are singleton.",
            "first_singleton": "First listed query prefix that uniquely identifies a hypothetical among the full public universe.",
        },
        "schedules": schedules,
        "limits": [
            "Post-hoc mathematical properties of two selected schedules; not independent optimizer replication or evidence of global optimality.",
            "Assumes exact faithful execution of the listed queries and exact bulls/cows feedback; actor adherence is not guaranteed.",
            "Uses no private labels, evaluation case IDs, raw trajectories, model calls, network or additional GPU allocation.",
            "Unresolved hypotheses remain unknown beyond the list; no invented continuation, capped25 outcome or eventual query cost.",
            "These mathematical properties do not causally attribute the observed E0 failure or observed differences between arms.",
            "Uniform full-universe fractions are not the private evaluation sample's empirical rates.",
        ],
    }
    raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    args.output.write_bytes(raw)
    print(json.dumps({"status": "COMPLETE", "elapsed_seconds": round(time.perf_counter() - start, 6),
                      "output": args.output.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "correctness_check": correctness,
                      "schedule_endpoints": {a: {"identified": s["first_singleton_within_list"]["identified_hypotheses"],
                                                 "unresolved": s["first_singleton_within_list"]["unresolved_after_list"]}
                                             for a, s in schedules.items()}}, sort_keys=True))


if __name__ == "__main__":
    main()
