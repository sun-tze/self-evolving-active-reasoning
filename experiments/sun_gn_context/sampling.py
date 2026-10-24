"""Approved effective policy and fail-closed provenance; never a wire payload.

This module imports no serving framework, loads no weights and opens no socket.
The separate sampling_probe CLI is the only installed CPU resolver.
"""
from __future__ import annotations

import importlib.metadata
from enum import Enum
import json
import os
from pathlib import Path
import platform
import re
import sys

from .core import IntegrityError
from .freeze import ROOT, MODEL_REVISION, seed, sha

A2_SHA = "a2c6e9e7ed108cd04d7bd69fdc37638f5d0c853ed586b1ce0f924d9c99776a68"
POLICY_SHA = "d9f80f6064963092a529d5c4b62f529706dd63c634aeb289c9b0b13f85d8e69a"
SERVING_SOURCE_SHA = "f924767d38bbdeb3c25c823873756ecfdc4de329e18f48340bc671d262bebb62"
WIRE_KEYS = {"model", "messages", "temperature", "top_p", "max_tokens", "seed"}
OVERRIDE_WARNING = "Default vLLM sampling parameters have been overridden"


def policy():
    if sha((ROOT/"PROTOCOL_ADDENDUM_2.md").read_bytes()) != A2_SHA:
        raise IntegrityError("Sampling addendum changed")
    raw = (ROOT/"sampling_policy.json").read_bytes()
    if sha(raw) != POLICY_SHA:
        raise IntegrityError("Effective sampling policy changed")
    return json.loads(raw)


def validate_environment(env=None):
    env = os.environ if env is None else env
    for name, value in env.items():
        if name.startswith("VLLM_") and not (name == "VLLM_LOGGING_LEVEL" and value.upper() in ("DEBUG", "INFO", "WARNING")):
            # Report the name, never a potentially sensitive environment value.
            raise IntegrityError("Unreviewed serving environment key: " + name)


def reviewed_source_pins():
    """Bind the four mirrored serving-path files in addition to policy sources."""
    spec=policy()
    raw=(ROOT/"serving_source_contract.json").read_bytes()
    if sha(raw)!=SERVING_SOURCE_SHA:
        raise IntegrityError("Serving-path source contract changed")
    contract=json.loads(raw)
    if contract["version"] not in spec["supported_adapter_versions"] or set(contract["files"]) & set(spec["reviewed_sources"]):
        raise IntegrityError("Ambiguous serving-path source contract")
    return dict(spec["reviewed_sources"],**contract["files"])


def installed_fingerprint(vllm):
    """Metadata and source bytes only. Missing distributions fail closed."""
    spec = policy()
    names = ("vllm", "torch", "transformers", "tokenizers", "jinja2")
    versions = {name: importlib.metadata.version(name) for name in names}
    if versions["vllm"] not in spec["supported_adapter_versions"]:
        raise IntegrityError("Installed vLLM version has no reviewed CPU resolver adapter")
    dist = importlib.metadata.distribution("vllm")
    records = [name for name in (dist.files or []) if str(name).replace("\\","/").endswith(".dist-info/RECORD")]
    if len(records)!=1:
        raise IntegrityError("Installed vLLM RECORD unavailable")
    record = Path(dist.locate_file(records[0])).read_bytes()
    if not record:
        raise IntegrityError("Installed vLLM RECORD empty")
    sources = {}
    for name, expected in reviewed_source_pins().items():
        actual = sha(Path(dist.locate_file(name)).read_bytes())
        if actual != expected:
            raise IntegrityError("Installed vLLM source differs from reviewed adapter: " + name)
        sources[name] = actual
    executable = Path(vllm).resolve()
    if not executable.is_file() or executable.parent != Path(sys.executable).resolve().parent:
        raise IntegrityError("vLLM executable is not in this existing Python runtime")
    return dict(versions=versions, python=platform.python_version(),
                python_executable=str(Path(sys.executable).resolve()),
                vllm_executable=str(executable), vllm_executable_sha256=sha(executable.read_bytes()),
                vllm_RECORD_sha256=sha(record), sources=sources,serving_source_contract_sha256=SERVING_SOURCE_SHA)


def wire_fixtures(assets):
    fixtures = []
    case = "cpu-resolution-fixture"
    for condition in ("B0", "B1", "S", "E"):
        messages = [{"role": "system", "content": assets.system(condition, assets.m0 if condition in ("S", "E") else "")},
                    {"role": "user", "content": assets.prompts["guess_prompt"].format(turn=1)}]
        fixtures.append(dict(label=condition, role="actor", request=dict(model="Qwen/Qwen2.5-7B-Instruct", messages=messages,
            temperature=.7, top_p=.7, max_tokens=128, seed=seed(case, 1))))
    for purpose in ("curator", "canary"):
        fixtures.append(dict(label=purpose, role="curator", request=dict(model="Qwen/Qwen2.5-7B-Instruct",
            messages=assets.curator_messages(assets.m0, []), temperature=0, top_p=1, max_tokens=768,
            seed=seed(case, 0, purpose))))
    return fixtures


def validate_wire(request, role):
    spec = policy()
    if set(request) != WIRE_KEYS or role not in ("actor", "curator"):
        raise IntegrityError("Request differs from frozen six-key sampling wire")
    if any(request.get(key) != value for key, value in spec[role].items()):
        raise IntegrityError("Frozen role sampling value mismatch")
    if (request["model"] != "Qwen/Qwen2.5-7B-Instruct" or type(request["seed"]) is not int or
            not 0 <= request["seed"] < 2147483647 or not isinstance(request["messages"], list)):
        raise IntegrityError("Invalid model/messages/integer seed binding")


def plain(value):
    """Lossless JSON-compatible framework objects, never repr-based coercion."""
    if isinstance(value,Enum):
        return plain(value.value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return sorted(plain(v) for v in value)
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    raise IntegrityError("Unknown resolved sampling value type")


def check_resolved(request_object, params, wire, role):
    """Shared assertions only; using stand-ins does not create a probe receipt."""
    validate_wire(wire, role)
    spec = policy()
    expected = dict(spec["common"], **spec[role], seed=wire["seed"])
    values = {}
    for name, wanted in expected.items():
        if not hasattr(params, name):
            raise IntegrityError("Installed sampler lacks required effective field: " + name)
        observed = plain(getattr(params, name))
        if observed != wanted:
            raise IntegrityError("Resolved sampling mismatch: " + name)
        values[name] = observed
    for name, wanted in spec["request_defaults"].items():
        if not hasattr(request_object, name) or plain(getattr(request_object, name)) != wanted:
            raise IntegrityError("Resolved request default mismatch: " + name)
    if plain(params.stop_token_ids) != [151643]:
        raise IntegrityError("Unexpected secondary EOS stop IDs")
    if params.sampling_type.name != ("RANDOM_SEED" if role == "actor" else "GREEDY"):
        raise IntegrityError("Resolved actor/curator decoding type mismatch")
    if params.output_kind.name != "FINAL_ONLY":
        raise IntegrityError("Resolved completion output mode mismatch")
    return dict(effective=values, stop_token_ids=plain(params.stop_token_ids),
                sampling_type=params.sampling_type.name,
                output_kind=params.output_kind.name,
                request_defaults={k:plain(getattr(request_object,k)) for k in spec["request_defaults"]})


def verify_snapshot_config(snapshot):
    path = Path(snapshot)/"generation_config.json"
    if sha(path.read_bytes()) != policy()["generation_config_sha256"]:
        raise IntegrityError("Pinned model generation config mismatch")


def check_server_log(path):
    path = Path(path)
    raw = path.read_bytes()
    if OVERRIDE_WARNING.encode() in raw:
        raise IntegrityError("Server inherited model sampling defaults")
    return sha(raw)


def verified_probe(runtime, implementation_sha):
    """Rebind a real CPU receipt to existing files/runtime; never trust pass alone."""
    policy()
    path = Path(runtime.get("sampling_probe_path", ""))
    digest = runtime.get("sampling_probe_sha256")
    if not path.is_absolute() or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise IntegrityError("Missing absolute installed-runtime probe binding")
    raw = path.read_bytes()
    if sha(raw) != digest:
        raise IntegrityError("Installed-runtime probe bytes changed")
    document = json.loads(raw)
    required = dict(format="sun_gn_sampling_probe_v1",passed=True,
        verification="installed_runtime_cpu_only",outside_allocation=True,weights_loaded=False,
        server_started=False,generation_requests=0,model_revision=MODEL_REVISION,
        sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA,
        implementation_sha256=implementation_sha)
    if any(document.get(k) != v or type(document.get(k)) is not type(v) for k,v in required.items()):
        raise IntegrityError("Probe is not successful installed-runtime evidence")
    from .allocation import server_command
    from .core import Assets
    expected_argv = server_command(document["fingerprint"]["vllm_executable"],document["snapshot"],document["tokenizer"],document["port"])
    if document["server_argv"] != expected_argv:
        raise IntegrityError("Probe launch arguments changed")
    if document["fingerprint"] != installed_fingerprint(document["fingerprint"]["vllm_executable"]):
        raise IntegrityError("Installed runtime changed after CPU resolution probe")
    validate_environment()
    verify_snapshot_config(document["snapshot"])
    if (document.get("generation_config_sha256")!=policy()["generation_config_sha256"] or
            set(document.get("generation_config_fields",{}).get("eos_token_id",[]))!={151643,151645}):
        raise IntegrityError("Probe generation-config provenance mismatch")
    contract = json.loads((ROOT/"tokenizer_contract.json").read_bytes())
    for name, expected in contract["assets"].items():
        if sha((Path(document["tokenizer"])/name).read_bytes()) != expected:
            raise IntegrityError("Tokenizer changed after CPU probe")
    if document["tokenizer_assets"] != contract["assets"] or document["chat_template_sha256"] != runtime.get("server_chat_template_sha256"):
        raise IntegrityError("Probe tokenizer provenance mismatch")
    expected = wire_fixtures(Assets.load())
    if document["wire_fixtures"] != expected or len(document["resolved"]) != len(expected):
        raise IntegrityError("CPU resolver fixture coverage mismatch")
    for item, result in zip(expected, document["resolved"]):
        wanted = dict(policy()["common"], **policy()[item["role"]], seed=item["request"]["seed"])
        if (result.get("label") != item["label"] or result.get("effective") != wanted or
                result.get("sampling_type") != ("RANDOM_SEED" if item["role"]=="actor" else "GREEDY") or
                result.get("output_kind") != "FINAL_ONLY" or
                result.get("stop_token_ids") != [151643] or result.get("request_defaults") != policy()["request_defaults"]):
            raise IntegrityError("CPU receipt effective parameters differ from approved policy")
        if (not isinstance(result.get("complete_sampling_params"),dict) or
                not isinstance(result.get("complete_request"),dict) or
                any(result["complete_request"].get(k)!=v for k,v in item["request"].items())):
            raise IntegrityError("CPU receipt lacks complete resolved provenance")
    if document.get("server_sampling_overrides") != {} or document.get("watermark_config") is not None:
        raise IntegrityError("CPU receipt contains unapproved serving defaults")
    return document
