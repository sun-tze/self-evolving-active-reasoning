"""Second-path verification of generated requests against raw records, without prepare imports."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import random
from common import (HERE, canonical, load_bundle, read_json, read_jsonl, require, sha, write_json)


def audit(bundle, manifest_sha, raw_root, arbench_root, out):
    manifest, requests = load_bundle(bundle, manifest_sha)
    lock = read_json(HERE/"source_lock.json")
    raw_root, arbench_root = Path(raw_root), Path(arbench_root)
    for entry in lock["members"].values():
        require(sha((raw_root/entry["path"]).read_bytes()) == entry["sha256"], "raw source mismatch")
    for entry in lock["arbench_files"].values():
        require(sha((arbench_root/entry["path"]).read_bytes()) == entry["sha256"], "AR-Bench source mismatch")
    dataset = read_json(arbench_root/"data/sp/test.json")
    records = {role: {r["index"]: r for r in read_jsonl(raw_root/lock["members"][role]["path"])} for role in ("bare", "first")}
    evaluation = read_json(raw_root/lock["members"]["evaluation_config"]["path"])
    expected = sorted(set(evaluation["instances"])-set(evaluation["excluded_seen"]))
    require(sorted({r["index"] for r in requests}) == expected, "wrong unseen set")
    require(Counter(r["index"] for r in requests) == Counter({i:4 for i in expected}), "not 4 cells per case")
    for r in requests:
        original = records[r["source"]][r["index"]]
        expected_obs = {"surface": dataset[r["index"]]["surface"],
                        "history": [{"question": p[0], "reply": p[1]} for p in original["history"]]}
        # Compare exact independently constructed observation and actual outgoing message bytes.
        require(r["observation"] == expected_obs, "projected source mismatch")
        require(r["payload"]["messages"][1]["content"].encode() == canonical(expected_obs), "wire observation mismatch")
        body = canonical(r["payload"]).decode()
        require(dataset[r["index"]]["bottom"] not in body, "full target on wire")
        require(set(r["payload"]) == {"model","messages","temperature","top_p","max_tokens","seed"}, "wire envelope leak")
    # Independent official/local differential check, including Unicode and empty strings.
    spec=importlib.util.spec_from_file_location("audit_official_f1",arbench_root/lock["arbench_files"]["scorer"]["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from common import char_f1
    rng=random.Random(62026)
    examples=[("", ""),("aab","abb"),("ABC","abc"),(" "," "),("猫é!","é猫!")]
    alphabet="aabbCc !!猫é\n"
    for _ in range(1000):
        examples.append(("".join(rng.choices(alphabet,k=rng.randrange(80))),"".join(rng.choices(alphabet,k=rng.randrange(80)))))
    for pred,target in examples:
        require(char_f1(pred,target) == module.f1_score(pred,target), "scorer differential mismatch")
    report={"format":"c2_second_path_audit_v1","manifest_sha256":manifest_sha,
            "requests_compared_to_raw":len(requests),"scorer_differential_cases":len(examples),
            "all_passed":True,"reviewer_independence":"Separate verification path, same implementing agent; not independent human/agent sign-off."}
    write_json(out,report)
    print(report)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",required=True);p.add_argument("--manifest-sha256",required=True)
    p.add_argument("--raw-root",required=True);p.add_argument("--arbench-root",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();audit(a.bundle,a.manifest_sha256,a.raw_root,a.arbench_root,a.out)
