"""Aggregate a private diagnose_saved_scores_v2.py output into anonymous numbers.

Reads the private per-branch/per-decision diagnostic and writes only counts,
quantiles and per-decision numeric rows. Case identities, state/candidate hashes,
call ids, prompts, targets and guesses are not exported. Standard library only;
no model, network or GPU use. Only --output is written; it must not exist.

Every comparison is on the same recorded proposals and states. Nothing here is a
counterfactual trajectory, a success rate or a question-count saving.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


EPS = 1e-9  # exact scores are float sums; differences below this are ties


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    def at(q):
        return values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]
    return {"n": len(values), "min": values[0], "p25": at(0.25), "median": at(0.5),
            "p75": at(0.75), "max": values[-1]}


def summarize(report):
    if (report.get("status") != "OFFLINE_RECORDED_DECISION_DIAGNOSTIC_COMPLETE" or
            report.get("diagnostic_version") != 2):
        raise ValueError("Expected a completed version-2 diagnostic")
    branches = [b for b in report["branches"] if b["response_present"]]
    missing = [b for b in report["branches"] if not b["response_present"]]

    cells = Counter((b["state_larger_than_50_input_sample"], b["finish_length"]) for b in branches)
    small = [b for b in branches if not b["state_larger_than_50_input_sample"]]
    branch_summary = {
        "recorded": len(branches),
        "missing_excluded_not_empty": len(missing),
        "state_gt_50_and_truncated": cells[(True, True)],
        "state_gt_50_not_truncated": cells[(True, False)],
        "state_le_50_and_truncated": cells[(False, True)],
        "state_le_50_not_truncated": cells[(False, False)],
        "finish_length_equals_512_cap": all(b["finish_length"] == b["at_512_output_cap"]
                                            for b in branches),
        "native_count_minus_exact": quantiles([b["native_count_minus_exact"] for b in branches]),
        "native_count_below_exact": sum(b["native_count_minus_exact"] < 0 for b in branches),
        "native_count_equal_exact": sum(b["native_count_minus_exact"] == 0 for b in branches),
        "native_count_above_exact": sum(b["native_count_minus_exact"] > 0 for b in branches),
        "exact_bucket_count": quantiles([b["exact_bucket_count"] for b in branches]),
        "native_verified_count": quantiles([b["native_verified_count"] for b in branches]),
        "branches_with_verified_duplicates": sum(b["verified_duplicate_occurrences"] > 0
                                                 for b in branches),
        "verified_duplicate_occurrences": sum(b["verified_duplicate_occurrences"] for b in branches),
        "verified_nonmember_occurrences": sum(b["verified_nonmember_occurrences"] for b in branches),
        "verified_valid_but_not_displayed_occurrences":
            sum(b["verified_not_shown_occurrences"] for b in branches),
        "exact_survivors_not_returned": sum(b["exact_survivors_not_returned"] for b in branches),
        "state_le_50_exact_survivors": sum(b["exact_bucket_count"] for b in small),
        "state_le_50_unique_survivors_returned": sum(b["unique_correct_survivors_returned"]
                                                     for b in small),
    }

    # Hashes stay inside this function: they only link a candidate to the
    # already-asked guesses of its own episode.
    asked, rows, candidate_rows = {}, [], []
    for d in sorted(report["decisions"], key=lambda d: (d["episode_index"], d["decision_index"])):
        if not d["included_in_rank_comparison"]:
            continue
        cands = d["candidates"]
        native = [c["native_expected_candidate_count_reduction"] for c in cands]
        exact = [c["exact_expected_candidate_count_reduction"] for c in cands]
        top = d["native_rank_positions"][0] - 1
        previous = asked.setdefault(d["episode_index"], [])
        repeats = [c["candidate_sha256"] in previous for c in cands]
        zero = [e <= EPS for e in exact]
        previous.append(cands[top]["candidate_sha256"])
        for c, n, e, rep in zip(cands, native, exact, repeats):
            counts = [k for _, k in c["branch_order_and_counts"]]
            candidate_rows.append({"state_count": d["state_count"], "native": n, "exact": e,
                                   "largest_bucket_share": max(counts) / d["state_count"],
                                   "zero_information": e <= EPS, "repeat_of_asked": rep})
        rows.append({
            "episode": d["episode_index"], "decision": d["decision_index"],
            "state_count": d["state_count"], "proposals": d["proposals_count"],
            "rank_changed": d["rank_changed"],
            "selected_guess_changed": d["selected_guess_changed"],
            "exact_top_tied_candidates": sum(abs(e - max(exact)) <= EPS for e in exact),
            "native_pick_is_exact_maximizer": d["one_step_expected_count_reduction_gap"] <= EPS,
            "zero_information_candidates": sum(zero),
            "repeat_of_asked_candidates": sum(repeats),
            "native_pick_zero_information": zero[top],
            "native_selected_exact_reduction": d["native_selected_exact_count_reduction"],
            "exact_best_reduction": d["exact_best_count_reduction"],
            "one_step_gap": d["one_step_expected_count_reduction_gap"],
            "one_step_gap_fraction_of_state": d["one_step_expected_count_reduction_gap"] /
                                              d["state_count"],
            "native_top_score_tied": sum(n == native[top] for n in native) > 1,
            "native_minus_exact_score_per_candidate": [n - e for n, e in zip(native, exact)],
            "entropy_selected_guess_changed_vs_native":
                d["separate_entropy_objective"]["selected_guess_changed_vs_native"],
            "entropy_gap_bits": d["separate_entropy_objective"]["entropy_best_bits"] -
                                d["separate_entropy_objective"]["native_selected_entropy_bits"],
        })

    def decision_block(subset):
        return {
            "decisions": len(subset),
            "rank_changed": sum(r["rank_changed"] for r in subset),
            "selected_guess_changed": sum(r["selected_guess_changed"] for r in subset),
            "selected_guess_changed_among_exact_ties":
                sum(r["selected_guess_changed"] and r["native_pick_is_exact_maximizer"]
                    for r in subset),
            "native_pick_strictly_worse_exact": sum(not r["native_pick_is_exact_maximizer"]
                                                    for r in subset),
            "native_pick_zero_information": sum(r["native_pick_zero_information"] for r in subset),
            "all_candidates_zero_information":
                sum(r["zero_information_candidates"] == r["proposals"] for r in subset),
            "zero_information_pick_with_informative_alternative":
                sum(r["native_pick_zero_information"] and
                    r["zero_information_candidates"] < r["proposals"] for r in subset),
            "one_step_gap": quantiles([r["one_step_gap"] for r in subset]),
            "one_step_gap_fraction_of_state":
                quantiles([r["one_step_gap_fraction_of_state"] for r in subset]),
            "entropy_selected_guess_changed_vs_native":
                sum(r["entropy_selected_guess_changed_vs_native"] for r in subset),
            "native_top_score_tied": sum(r["native_top_score_tied"] for r in subset),
        }

    return {
        "status": "ANONYMOUS_AGGREGATE_OF_OFFLINE_DIAGNOSTIC",
        "source_diagnostic_version": 2,
        "scope": report["scope"],
        "comparison_scope": "same recorded proposals/state; not a counterfactual trajectory",
        "objective": report["objective"],
        "separate_objective": report["separate_objective"],
        "coverage": {k: v for k, v in report["coverage"].items() if k != "missing_response_call_ids"},
        "episodes": [{k: e[k] for k in ("episode_index", "replay_status",
                                        "intents_replayed", "responses_replayed")} |
                     {"official_result_reproduced_exactly":
                      e.get("official_result_reproduced_exactly", False)}
                     for e in report["episodes"]],
        "branches": branch_summary,
        "decisions_all": decision_block(rows),
        "decisions_state_gt_50": decision_block([r for r in rows if r["state_count"] > 50]),
        "decisions_state_le_50": decision_block([r for r in rows if r["state_count"] <= 50]),
        "candidates": {
            "proposed": len(candidate_rows),
            "zero_information": sum(c["zero_information"] for c in candidate_rows),
            "zero_information_repeat_of_asked":
                sum(c["zero_information"] and c["repeat_of_asked"] for c in candidate_rows),
            "informative_repeat_of_asked":
                sum(not c["zero_information"] and c["repeat_of_asked"] for c in candidate_rows),
            "zero_information_native_score_fraction_of_state":
                quantiles([c["native"] / c["state_count"] for c in candidate_rows
                           if c["zero_information"]]),
            "spearman_largest_bucket_share_vs_native_inflation_state_gt_50": spearman(
                [c["largest_bucket_share"] for c in candidate_rows if c["state_count"] > 50],
                [(c["native"] - c["exact"]) / c["state_count"] for c in candidate_rows
                 if c["state_count"] > 50]),
        },
        "decision_rows": rows,
        "limitations": report["limitations"] + [
            "Three complete cases and one prefix from a single cancelled run; not a 20-case result.",
            "A larger one-step expected reduction does not imply fewer questions over a trajectory.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite " + str(args.output))
    summary = summarize(json.loads(args.diagnostic.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: summary[k] for k in ("branches", "decisions_all")}, indent=1))


if __name__ == "__main__":
    main()
