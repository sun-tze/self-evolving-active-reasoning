"""CPU-only installed-runtime/cache audit; never loads weights or starts a server."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

from budget import make_budget
from common import HERE, MODEL, read_json, require, sha, write_json


def verify_snapshot(snapshot, lock, weights):
    require(weights["model"] == lock["model"] == MODEL and
            weights["revision"] == lock["revision"], "model lock identity mismatch")
    inventory = {}
    for name, expected in weights["files"].items():
        require(Path(name).name == name, "invalid model filename")
        path = snapshot/name
        require(path.is_file() and path.stat().st_size == expected["bytes"], "model size mismatch: " + name)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        require(checksum == expected["sha256"], "model hash mismatch: " + name)
        inventory[name] = {"bytes": path.stat().st_size, "sha256": checksum}
    for name, expected in lock["files"].items():
        require(inventory[name]["sha256"] == expected, "tokenizer lock mismatch: " + name)
    # Parse only after the index itself has matched the external lock.
    index = read_json(snapshot/"model.safetensors.index.json")
    shards = set(index["weight_map"].values())
    require(shards and all(type(n) is str and Path(n).name == n and n.endswith(".safetensors")
                           for n in shards), "invalid weight shard names")
    expected_names = {"config.json", "generation_config.json", "model.safetensors.index.json"} | shards | set(lock["files"])
    require(set(inventory) == expected_names, "incomplete model file lock")
    return inventory


def preflight(bundle, manifest_sha, cache, out):
    lock = read_json(HERE / "model_lock.json")
    root = Path(sys.prefix)
    cuda = root / "lib/python3.12/site-packages/nvidia/cu13"
    versions = {n: importlib.metadata.version(n) for n in
                ["vllm", "torch", "transformers", "tokenizers", "llguidance", "cuda-toolkit"]}
    require(sys.version_info[:2] == (3, 12), "expected the existing Python 3.12 runtime layout")
    require(versions["vllm"] == "0.31.0", "unreviewed vLLM version")
    require(versions["torch"].split("+")[0] == "2.13.0", "unreviewed torch version")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    for p in [Path(sysconfig.get_path("include"))/"Python.h", root/"bin/vllm",
              root/"bin/ninja", cuda/"bin/nvcc", cuda/"include/cuda_runtime.h"]:
        require(p.is_file(), "missing runtime tool/header: " + str(p))
    require(shutil.which("g++"), "missing g++")
    nvcc = subprocess.check_output([str(cuda/"bin/nvcc"), "--version"], text=True)
    compiler = subprocess.check_output(["g++", "--version"], text=True)
    import torch
    # Inspect the pinned snapshot directly. No remote calls and no model loading.
    snapshot = Path(cache)/"hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"/lock["revision"]
    weights = read_json(HERE/"weights_lock.json")
    inventory = verify_snapshot(snapshot, lock, weights)
    # The vLLM environment intentionally has a different supported Transformers
    # version from the CPU staging environment. Compare every frozen token count.
    frozen = read_json(Path(bundle)/"budget.json")
    with tempfile.TemporaryDirectory() as tmp:
        actual = make_budget(bundle, manifest_sha, str(snapshot), Path(tmp)/"budget.json")
    require(actual == frozen, "runtime tokenizer differs from frozen budget; do not run")
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    report = {"format": "c2_runtime_cpu_preflight_v1", "passed": True,
              "manifest_sha256": manifest_sha, "model": MODEL, "revision": lock["revision"],
              "versions": versions, "packages": dict(sorted(packages.items())),
              "nvcc": nvcc, "compiler": compiler, "torch_cuda": torch.version.cuda,
              "cuda_home_checked": str(cuda), "model_files": inventory,
              "weights_lock_sha256": sha((HERE/"weights_lock.json").read_bytes()),
              "weights_integrity_verified": True, "token_counts_compared": 280,
              "token_budget_equal": True, "gpu_startup_tested": False,
              "gpu_authorization_granted": False,
              "limitations": ["File hashes match pinned official revision metadata; not a signed provenance claim.",
                              "Target GPU driver, CUDA/JIT startup, and served /tokenize remain untested."]}
    write_json(out, report)
    print("Runtime/cache CPU preflight passed. No GPU allocation or model inference occurred.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--cache", default=os.environ.get("HF_HOME"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    require(args.cache, "--cache or HF_HOME required")
    preflight(args.bundle, args.manifest_sha256, args.cache, args.out)
