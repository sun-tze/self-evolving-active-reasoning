"""One CPU command to fetch, prepare, tokenize, and cross-audit a new pilot bundle."""
import argparse
from pathlib import Path
from common import MODEL, read_json, require, sha, write_json
from fetch import fetch
from prepare import prepare
from budget import make_budget
from audit_real import audit
from runner import plan


def stage(inputs, bundle, tokenizer):
    inputs, bundle = Path(inputs), Path(bundle)
    require(not bundle.exists(), "use a fresh bundle path; existing artifacts are immutable")
    fetch(inputs)
    report = prepare(inputs/"raw", inputs/"AR-Bench", bundle)
    msha = report["manifest_sha256"]
    make_budget(bundle, msha, tokenizer, bundle/"budget.json")
    audit(bundle, msha, inputs/"raw", inputs/"AR-Bench", bundle/"second_path_audit.json")
    doc, _, _ = plan(bundle, msha, bundle/"budget.json", "pilot", "http://127.0.0.1:18000/v1", 20, 15210, 8000)
    write_json(bundle/"pilot_plan.json", doc)
    # This file supplies arguments; it is not experiment authorization.
    write_json(bundle/"READY.json", {"bundle": str(bundle.resolve()), "manifest_sha256": msha,
               "pilot_input_tokens": doc["totals"]["input_tokens"], "pilot_output_cap": 8000,
               "inference_authorized": False, "gpu_calls_made": 0})
    print("CPU preparation complete. No GPU job submitted or inference performed.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs",default=".c2-local")
    p.add_argument("--bundle",required=True)
    p.add_argument("--tokenizer",default=MODEL)
    a=p.parse_args();stage(a.inputs,a.bundle,a.tokenizer)
