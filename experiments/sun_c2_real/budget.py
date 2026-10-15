"""CPU-only exact chat-template token accounting; never loads model weights."""
import argparse
from pathlib import Path
from common import MODEL, digest, load_bundle, require, sha, write_json

# Qwen revision is populated in model_lock.json from the public model repository.
def make_budget(bundle, manifest_sha, tokenizer_path, out):
    from transformers import AutoTokenizer
    from common import HERE, read_json
    lock = read_json(HERE / "model_lock.json")
    manifest, rows = load_bundle(bundle, manifest_sha)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False,
                    revision=lock["revision"] if tokenizer_path == MODEL else None)
    # Bind actual tokenizer files as well as the upstream revision.
    if tokenizer_path == MODEL:
        from huggingface_hub import snapshot_download
        tokenizer_path = snapshot_download(MODEL, revision=lock["revision"],
                            allow_patterns=list(lock["files"]))
    for name, expected in lock["files"].items():
        require(sha((Path(tokenizer_path) / name).read_bytes()) == expected, "tokenizer file mismatch: " + name)
    counts = {}
    for r in rows:
        messages = r["payload"]["messages"]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                               add_generation_prompt=True)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        reference = tokenizer.encode(rendered, add_special_tokens=False)
        require(type(tokens) is list and tokens == reference and all(type(t) is int for t in tokens),
                "tokenizer API returned unexpected IDs")
        counts[r["request_id"]] = len(tokens)
    require(max(counts.values()) + 400 <= 16384, "context overflow; never truncate history")
    report = {"format": "c2_token_budget_v1", "manifest_sha256": manifest_sha,
              "model_lock": lock, "request_input_tokens": counts, "scopes": {}}
    for scope, indices in [("pilot", manifest["pilot_indices"]), ("all", manifest["unseen_indices"])]:
        selected = [r for r in rows if r["index"] in indices]
        report["scopes"][scope] = {"cases": len(indices), "requests": len(selected),
              "input_tokens": sum(counts[r["request_id"]] for r in selected),
              "max_input_per_request": max(counts[r["request_id"]] for r in selected),
              "output_token_cap": 400 * len(selected)}
    write_json(out, report)
    print(report["scopes"])
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--tokenizer", default=MODEL)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    make_budget(a.bundle, a.manifest_sha256, a.tokenizer, a.out)
