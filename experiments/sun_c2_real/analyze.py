"""Offline official char-F1 scoring and paired fixed-dialogue factorial contrasts."""
import argparse
import math
from pathlib import Path
import random
from common import (char_f1, digest, load_bundle, read_json, read_jsonl, require, sha, write_json)
from runner import journal_state, parse_completion, schedule

CONTRASTS = {
    "writer_given_bare_B-A": {"A": -1, "B": 1},
    "writer_given_first_D-C": {"C": -1, "D": 1},
    "dialogue_given_plain_C-A": {"A": -1, "C": 1},
    "dialogue_given_evidence_D-B": {"B": -1, "D": 1},
    "writer_marginal": {"A": -.5, "B": .5, "C": -.5, "D": .5},
    "dialogue_marginal": {"A": -.5, "B": -.5, "C": .5, "D": .5},
    "interaction_(D-C)-(B-A)": {"A": 1, "B": -1, "C": -1, "D": 1},
}


def paired_stats(scores, draws=10000):
    require(bool(scores), "empty case design")
    values = {name: [] for name in CONTRASTS}
    case_contrasts = []
    for case in scores:
        require(set(case) == set("ABCD"), "missing design cells")
        for v in case.values():
            require(v is None or type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1, "invalid F1 score")
        contrast = {}
        for name, weights in CONTRASTS.items():
            val = None if any(case[c] is None for c in weights) else math.fsum(case[c]*w for c, w in weights.items())
            contrast[name] = val
            values[name].append(val)
        case_contrasts.append(contrast)
    n = len(scores)
    rng = random.Random(62026)
    resamples = [[rng.randrange(n) for _ in range(n)] for _ in range(draws)]
    summary = {}
    for name, vals in values.items():
        available = sum(v is not None for v in vals)
        complete = available == n
        ci = None
        if complete and n >= 2:
            boot = sorted(math.fsum(vals[j] for j in indices) / n for indices in resamples)
            ci = [boot[int(.025 * draws)], boot[int(.975 * draws)]]
        summary[name] = {"mean": math.fsum(vals) / n if complete else None,
                         "ci95": ci, "expected": n, "available": available, "missing": n-available}
    return case_contrasts, summary


def analyze(bundle, manifest_sha, run_dir, out):
    root, run_dir = Path(bundle), Path(run_dir)
    manifest, requests = load_bundle(root, manifest_sha)
    raw_targets = (root / "private/targets.jsonl").read_bytes()
    require(sha(raw_targets) == manifest["targets_sha256"], "target file hash mismatch")
    target_rows = read_jsonl(root / "private/targets.jsonl")
    targets = {}
    for t in target_rows:
        i = t["index"]
        require(type(i) is int and i in manifest["unseen_indices"] and i not in targets, "duplicate/foreign target")
        require(sha(t["bottom"].encode()) == t["target_sha256"] and
                t["dataset_record_sha256"] == manifest["case_bindings"][str(i)]["bare"]["dataset_record_sha256"], "target identity mismatch")
        targets[i] = t["bottom"]
    require(set(targets) == set(manifest["unseen_indices"]), "incomplete targets")
    run = read_json(run_dir / "run.json")
    plan = run["plan"]
    require(run["plan_sha256"] == digest(plan) and plan["manifest_sha256"] == manifest_sha, "run binding mismatch")
    require(plan["scope"] in ("pilot", "all"), "invalid scope")
    indices = manifest["pilot_indices"] if plan["scope"] == "pilot" else manifest["unseen_indices"]
    selected = schedule(requests, indices)
    require(plan["indices"] == sorted(indices) and plan["schedule"] == [r["request_id"] for r in selected], "run design/schedule mismatch")
    events = read_jsonl(run_dir / "journal.jsonl") if (run_dir / "journal.jsonl").exists() else []
    started, finished = journal_state(events, selected)
    by_id = {e["request_id"]: e for e in events if e["event"] == "result"}
    cells, result_rows, usage, durations = {}, [], [], []
    for r in selected:
        e = by_id.get(r["request_id"])
        score, parsed = None, None
        status = "ambiguous" if r["request_id"] in started else "missing"
        if e:
            status = e["status"]
            require(type(e["elapsed_s"]) in (int, float) and math.isfinite(e["elapsed_s"]) and e["elapsed_s"] >= 0, "invalid duration")
            durations.append(e["elapsed_s"])
            if status == "completed":
                require(e["error"] is None, "completed response carries error")
                parsed = parse_completion(e["raw"], r)
                require(parsed == e["parsed"], "stored/raw completion mismatch")
                score = char_f1(parsed["text"].strip(), targets[r["index"]])
                usage.append(parsed["usage"])
            else:
                require(type(e["error"]) is str and bool(e["error"]), "missing failure diagnostic")
        cells.setdefault(r["index"], {})[r["cell"]] = score
        result_rows.append({"index": r["index"], "cell": r["cell"], "request_id": r["request_id"],
                            "status": status, "score": score, "response": parsed,
                            "error": e["error"] if e else status})
    contrasts, stats = paired_stats([cells[i] for i in sorted(indices)])
    token_cost = {}
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        known = [u[field] for u in usage if u[field] is not None]
        token_cost[field] = {"known": len(known), "unknown": len(selected)-len(known),
                             "known_subtotal": sum(known),
                             "total_if_complete": sum(known) if len(known) == len(selected) else None}
    costs_raw = (root / "private/historical_costs.json").read_bytes()
    require(sha(costs_raw) == manifest["historical_costs_sha256"], "cost source hash mismatch")
    costs = read_json(root / "private/historical_costs.json")
    historical = {}
    for role in ("bare", "first"):
        rs = [costs[str(i)][role] for i in indices]
        historical[role] = {"episodes": len(rs), "questioner_input_tokens": sum(r["tokens"]["in"] for r in rs),
                           "questioner_output_tokens": sum(r["tokens"]["out"] for r in rs),
                           "questioner_calls": sum(r["solver_calls"] for r in rs),
                           "referee_and_keypoint_judge_calls": sum(r["answerer_calls"] for r in rs),
                           "referee_tokens": None,
                           "meaning": "Sunk whole historical episodes, including original finalizers; acquisition-only cost unavailable."}
    cell_means = {}
    for c in "ABCD":
        vals = [cells[i][c] for i in indices]
        cell_means[c] = {"mean": math.fsum(vals)/len(vals) if all(v is not None for v in vals) else None,
                         "expected": len(vals), "available": sum(v is not None for v in vals)}
    report = {"format": "c2_real_analysis_v1", "scope": plan["scope"], "planned_cases": len(indices),
              "planned_requests": len(selected), "cell_means": cell_means, "contrasts": stats,
              "per_case": [{"index": i, "scores": cells[i], "contrasts": c} for i,c in zip(sorted(indices), contrasts)],
              "responses": result_rows, "new_inference_cost": {"attempted_calls": len(started),
                  "returned_calls": len(finished), "ambiguous_calls": len(started-finished),
                  "token_usage": token_cost, "returned_elapsed_s": math.fsum(durations)},
              "historical_cost": historical,
              "truncated_completions": sum(bool(r["response"] and r["response"]["truncated"]) for r in result_rows),
              "provenance": {"manifest_sha256": manifest_sha, "targets_sha256": manifest["targets_sha256"],
                   "run_sha256": sha((run_dir/"run.json").read_bytes()), "journal_sha256": sha((run_dir/"journal.jsonl").read_bytes()) if events else None,
                   "scorer_source": manifest["binding"]["source_lock"]["arbench_files"]["scorer"],
                   "analyzer_sha256": sha(Path(__file__).read_bytes())},
              "limitations": ["Dialogue contrasts measure utility of these fixed histories, not equal-compute acquisition effects.",
                  "Writer contrasts concern the two standardized instructions, not the original evolved finalizer.",
                  "Interaction is difference of paired writer effects for these histories.",
                  "Character F1 is lexical overlap, not semantic correctness or faithfulness; no causal or benchmark superiority claim.",
                  "One greedy generation per cell; matching seeds do not guarantee bitwise GPU reproducibility.",
                  "Paired case bootstrap: 10000 draws, seed 62026. Descriptive intervals, no multiple-comparison-adjusted tests.",
                  "Missing/failed cases suppress affected means and intervals. Length-capped outputs retained and flagged.",
                  "Pilot is exploratory. If used for development, reserve remaining 65 for later confirmation.",
                  "No monetary rate assumed; Slurm allocation cost requires institutional accounting."]}
    write_json(out, report)
    print(f"Scored {sum(r['score'] is not None for r in result_rows)}/{len(selected)} planned responses.")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    analyze(a.bundle, a.manifest_sha256, a.run, a.out)
