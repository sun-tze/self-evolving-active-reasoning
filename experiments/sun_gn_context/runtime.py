"""Pinned offline tokenizer plus loopback-only, no-retry serving transport.

Importing this module does not contact a service or load model weights.
"""
from __future__ import annotations

import importlib.metadata
import json
import re
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .core import CONTRACT_SHA, PROTOCOL_SHA, ADDENDUM_SHA, IntegrityError
from .freeze import BENCHMARK, MODEL_REVISION, ROOT, sha
from .sampling import A2_SHA, POLICY_SHA, verified_probe


class PinnedTokenizer:
    def __init__(self, directory: Path):
        contract = json.loads((ROOT/"tokenizer_contract.json").read_bytes())
        for file, digest in contract["assets"].items():
            if sha((directory/file).read_bytes()) != digest:
                raise IntegrityError(f"Tokenizer asset mismatch: {file}")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.lock = __import__("threading").Lock()
        self.metadata = dict(revision=MODEL_REVISION, assets=contract["assets"],
                             transformers=importlib.metadata.version("transformers"),
                             tokenizers=importlib.metadata.version("tokenizers"))

    def count(self, text: str) -> int:
        with self.lock:
            return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chat_count(self, messages: list[dict]) -> int:
        with self.lock:
            ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                                   return_dict=False,return_tensors=None,padding=False,truncation=False)
            if not isinstance(ids,list) or any(type(token) is not int for token in ids):
                raise IntegrityError("Expected a flat token-ID list from the pinned chat template")
            return len(ids)


def validate_release(release: dict, runtime: dict, stage: str, implementation_sha: str):
    # This is an audit interlock, not a substitute for the parent's authorization.
    required = dict(protocol_sha256=PROTOCOL_SHA,protocol_addendum_sha256=ADDENDUM_SHA,contract_sha256=CONTRACT_SHA,
                    benchmark_commit=BENCHMARK,model_revision=MODEL_REVISION,
                    tokenizer_revision=MODEL_REVISION,implementation_sha256=implementation_sha,
                    sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA)
    for document in (release,runtime):
        for key,value in required.items():
            if document.get(key)!=value:
                raise IntegrityError(f"Release/runtime mismatch: {key}")
    if release.get("independent_correctness_gate") is not True or release.get("released_stage")!=stage:
        raise IntegrityError("Independent gate/stage release missing")
    if runtime.get("weights_verified") is not True or runtime.get("existing_access_verified") is not True:
        raise IntegrityError("Model cache/access not verified")
    for name in ("vllm_version","torch_version","cuda_version","weights_manifest_sha256","server_chat_template_sha256"):
        if not isinstance(runtime.get(name),str) or not runtime[name]:
            raise IntegrityError(f"Unpinned serving preflight field: {name}")
    for key in ("sampling_probe_sha256","independent_tokenizer_evidence_sha256"):
        if not isinstance(runtime.get(key),str) or not re.fullmatch(r"[0-9a-f]{64}",runtime[key]) or release.get(key)!=runtime[key]:
            raise IntegrityError("Missing shared release/runtime evidence binding: " + key)
    if release.get("independent_tokenizer_gate") is not True:
        raise IntegrityError("Independent genuine-tokenizer gate missing")
    probe=verified_probe(runtime,implementation_sha)
    if any(probe["fingerprint"]["versions"][name]!=runtime[name+"_version"] for name in ("vllm","torch")):
        raise IntegrityError("CPU sampling probe/runtime version mismatch")
    return probe


class LoopbackTransport:
    def __init__(self, base_url: str, timeout: float = 120):
        parsed = urlparse(base_url)
        if parsed.scheme!="http" or parsed.hostname not in ("127.0.0.1","localhost","::1") or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise IntegrityError("Only an existing local/tunneled, credential-free HTTP service is permitted")
        if parsed.path.rstrip("/")!="/v1":
            raise IntegrityError("Expected local /v1 endpoint")
        self.url = base_url.rstrip("/")+"/chat/completions"
        self.timeout = timeout

    def __call__(self, request: dict) -> dict:
        body = json.dumps(request,ensure_ascii=True,separators=(",", ":")).encode()
        # urllib performs no application-level retries. A failed/uncertain request is never resent.
        response = urlopen(Request(self.url, data=body, headers={"Content-Type":"application/json"}, method="POST"), timeout=self.timeout)
        with response:
            raw = json.load(response)
        if len(raw.get("choices",[]))!=1:
            raise IntegrityError("Expected one completion")
        choice = raw["choices"][0]
        return dict(text=choice["message"]["content"],finish_reason=choice["finish_reason"],
                    prompt_tokens=raw["usage"]["prompt_tokens"],completion_tokens=raw["usage"]["completion_tokens"],
                    response_id=raw["id"])
