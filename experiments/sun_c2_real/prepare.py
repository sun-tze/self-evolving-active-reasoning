"""Pinned Experiment 10 -> 70 x four safe writer requests. CPU only."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path

from common import (HERE, CELLS, PROTOCOL, char_f1, decode, digest, observation, payload,
                    pilot_indices, read_json, read_jsonl, request_id, require, sha,
                    validate_requests, write_json, write_jsonl)

FIRST = "cand_sp_strict1_b10g3r2_b0r1g2"


def checked(path, expected):
    raw = Path(path).read_bytes()
    require(sha(raw) == expected, "source hash mismatch: " + str(path))
    return raw


def identify_unseen(evaluation, evolution):
    import numpy as np
    require(evaluation["task"] == evolution["task"] == "sp", "wrong task")
    require(evaluation["split"] == evolution["split"] == "test", "wrong split")
    require(evaluation["solver_model"] == evolution["solver_model"] == PROTOCOL["model"], "wrong historical model")
    require(evaluation["budget"] == evolution["budget"] == 25, "wrong history budget")
    require(evaluation["exclude_stream_seed"] == evolution["stream_seed"] == 0, "wrong stream seed")
    require(evaluation["exclude_stream_limit"] == evolution["limit"] == 30, "wrong stream length")
    require(evaluation["instances"] == list(range(100)), "historical evaluation cases changed")
    stream = [int(i) for i in np.random.default_rng(0).permutation(100)[:30]]
    require(evolution["stream"] == stream and evaluation["excluded_seen"] == sorted(stream), "historical stream disagreement")
    require(not evaluation["solver_mock"] and not evaluation["answerer_mock"], "mock historical source")
    require(not evolution["solver_mock"] and not evolution["answerer_mock"], "mock evolution source")
    require(evolution["signals"] == "strict" and evolution["run_name"] == "sp_strict1_b10g3r2", "wrong evolution")
    return sorted(set(range(100)) - set(stream))


def source_rows(raw, role, dataset, official_f1):
    rows, identities, audit = {}, {}, {"reply_counts": Counter(), "score_replay_count": 0}
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        require(line.strip(), "blank source line")
        r = decode(line)
        i = r["index"]
        require(type(i) is int and 0 <= i < 100 and i not in rows, "duplicate/invalid positional index")
        require(r["task"] == "sp" and r["harness"] == ("bare" if role == "bare" else FIRST), "source role mismatch")
        require(r["status"] == "ok" and r["error"] is None, "historical source failure")
        d = dataset[i]
        obs = observation(d["surface"], r["history"])
        # Verify surface at its historical prompt boundary, not arbitrary substring in a response.
        if role == "bare":
            marker = "\nPuzzle: " + d["surface"] + "\n"
            require(marker in r["trace"][0]["messages"][0]["content"], "historical surface mismatch")
        else:
            require(r["trace"][0]["messages"][1]["content"].startswith("Puzzle: " + d["surface"] + "\n\n"), "historical surface mismatch")
        pred = r["gold"]["prediction"]
        require(len(pred) < 1000, "historical prediction may be truncated; cannot replay")
        score = official_f1(pred, d["bottom"])
        require(abs(score - r["gold"]["f1_char"]) < 1e-12, "historical scorer/target mismatch")
        require(abs(char_f1(pred, d["bottom"]) - score) < 1e-12, "local/official scorer mismatch")
        # Structural allowlisting is primary. Full target text in observations is an additional hard stop.
        public = "\n".join([obs["surface"]] + [s for p in r["history"] for s in p])
        require(d["bottom"] not in public, "full hidden story appears in observed input")
        rows[i] = r
        identities[i] = {"source": role, "harness": r["harness"], "position": i,
                         "line_number": line_number, "line_sha256": sha(line),
                         "record_sha256": digest(r), "dataset_record_sha256": digest(d),
                         "surface_sha256": sha(d["surface"].encode()), "observation_sha256": digest(obs)}
        audit["reply_counts"].update(a for _, a in r["history"])
        audit["score_replay_count"] += 1
    require(set(rows) == set(range(100)), "source must cover all 100 positions exactly once")
    audit["reply_counts"] = dict(audit["reply_counts"])
    return rows, identities, audit


def prepare(raw_root, arbench_root, output):
    raw_root, arbench_root, output = map(Path, (raw_root, arbench_root, output))
    require(not output.exists(), "refusing to overwrite an existing bundle")
    lock = read_json(HERE / "source_lock.json")
    raw = {k: checked(raw_root / v["path"], v["sha256"]) for k, v in lock["members"].items()}
    external = {k: checked(arbench_root / v["path"], v["sha256"]) for k, v in lock["arbench_files"].items()}
    dataset = decode(external["dataset"])
    require(len(dataset) == 100, "dataset must contain 100 rows")
    require(all(type(d["bottom"]) is str and d["bottom"] for d in dataset), "missing scoring target")
    unseen = identify_unseen(decode(raw["evaluation_config"]), decode(raw["evolution_config"]))
    # Only execute the separately pinned, inspected official CPU scorer module.
    spec = importlib.util.spec_from_file_location("c2_official_scorer", arbench_root / lock["arbench_files"]["scorer"]["path"])
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    sources, identities, audits = {}, {}, {}
    for role in ("bare", "first"):
        sources[role], identities[role], audits[role] = source_rows(raw[role], role, dataset, official.f1_score)
    # Cross-check published per-source means on the exact unseen subset.
    scores = decode(raw["historical_scores"])
    for role, name in [("bare", "bare"), ("first", FIRST)]:
        mean = sum(sources[role][i]["gold"]["f1_char"] for i in unseen) / 70
        require(abs(mean - scores[name]["unseen"]["f1_char"]) < 1e-12, "historical unseen aggregate mismatch")
        audits[role]["historical_unseen_f1"] = mean
    builder = {p: sha((HERE / p).read_bytes()) for p in ("prepare.py", "common.py")}
    binding = {"source_lock": lock, "builder": builder, "protocol_sha256": digest(PROTOCOL)}
    binding_sha = digest(binding)
    bindings = {str(i): {role: dict(identities[role][i], member=lock["members"][role])
                         for role in ("bare", "first")} for i in unseen}
    manifest = {"format": "c2_real_manifest_v1", "data_kind": "historical-real",
                "binding": binding, "binding_sha256": binding_sha,
                "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
                "unseen_indices": unseen, "pilot_indices": pilot_indices(unseen), "case_bindings": bindings}
    requests = []
    for i in unseen:
        for cell, (role, writer) in CELLS.items():
            obs = observation(dataset[i]["surface"], sources[role][i]["history"])
            body = payload(obs, writer, i)
            r = {"format": "c2_real_request_v1", "index": i, "case_id": f"arbench-sp-position-{i:03d}",
                 "cell": cell, "source": role, "writer": writer, "binding_sha256": binding_sha,
                 "source_identity": bindings[str(i)][role], "observation": obs,
                 "payload": body, "payload_sha256": digest(body)}
            r["request_id"] = request_id(r)
            requests.append(r)
    validate_requests(requests, manifest)
    # Targets and historical costs have a separate offline path; runner never opens these.
    targets = [{"index": i, "bottom": dataset[i]["bottom"], "target_sha256": sha(dataset[i]["bottom"].encode()),
                "dataset_record_sha256": digest(dataset[i])} for i in unseen]
    costs = {str(i): {role: {"solver_calls": sources[role][i]["solver_calls"],
                            "tokens": sources[role][i]["tokens"],
                            "answerer_calls": sources[role][i]["answerer_calls"],
                            "duration_s": sources[role][i]["duration_s"]} for role in sources} for i in unseen}
    write_jsonl(output / "requests.jsonl", requests)
    write_jsonl(output / "private/targets.jsonl", targets)
    write_json(output / "private/historical_costs.json", costs)
    manifest["requests_sha256"] = sha((output / "requests.jsonl").read_bytes())
    manifest["targets_sha256"] = sha((output / "private/targets.jsonl").read_bytes())
    manifest["historical_costs_sha256"] = sha((output / "private/historical_costs.json").read_bytes())
    write_json(output / "manifest.json", manifest)
    manifest_sha = sha((output / "manifest.json").read_bytes())
    report = {"format": "c2_real_audit_v1", "manifest_sha256": manifest_sha, "requests": len(requests),
              "unseen_indices": unseen, "pilot_indices": manifest["pilot_indices"],
              "excluded_seen": decode(raw["evaluation_config"])["excluded_seen"],
              "dataset_index_unique_count": len({d["index"] for d in dataset}),
              "identity_rule": "zero-based dataset position, not the dataset index field",
              "historical_audits": audits, "source_lock": lock,
              "field_allowlist": ["surface", "history[].question", "history[].reply"],
              "target_full_text_in_observation": 0,
              "limitations": ["Hashes prove consistency of these pinned supplied artifacts, not signed historical authenticity.",
                              "Observed questions and replies may contain answer-relevant facts; no semantic decontamination claimed.",
                              "Pilot cases become development cases if results inform prompt or method changes.",
                              "No new inference has run. Independent review evidence and GPU authorization must be recorded separately."]}
    write_json(output / "audit.json", report)
    print("Prepared 70 cases / 280 requests. Manifest SHA-256: " + manifest_sha)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--arbench-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    prepare(args.raw_root, args.arbench_root, args.out)
