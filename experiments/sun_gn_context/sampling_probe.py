"""CPU-only installed vLLM configuration probe, before allocation or claim.

No model weights, engine, server or generation request is used. Dependencies
must already exist. Failed/unsupported resolution emits a failed receipt and
exits nonzero; test stand-ins cannot call a success-receipt constructor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .core import Assets, IntegrityError
from .freeze import MODEL_REVISION, sha
from .sampling import (A2_SHA, POLICY_SHA, WIRE_KEYS, policy, installed_fingerprint,
                       validate_environment, verify_snapshot_config, wire_fixtures,
                       check_resolved, plain)


@contextmanager
def cpu_only_environment():
    # Process-local offline/CPU guard, not a security or installed-file change.
    values = dict(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    previous = {key:os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key,value in previous.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value


@contextmanager
def forbid_cuda_initialization(torch):
    if torch.cuda.is_initialized():
        raise IntegrityError("CPU probe must start without an initialized CUDA runtime")
    original = torch.cuda._lazy_init
    def refused(*args,**kwargs):
        raise IntegrityError("GPU initialization forbidden in CPU resolution probe")
    torch.cuda._lazy_init = refused
    try:
        yield
        if torch.cuda.is_initialized():
            raise IntegrityError("CPU resolution unexpectedly initialized CUDA")
    finally:
        torch.cuda._lazy_init = original


def complete_cli_help(vllm, server_argv):
    """v0.31.0's plain --help is a group summary; request every flag once."""
    command=[str(vllm),"serve","--help=all"]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30,
                          env=dict(os.environ),stdin=subprocess.DEVNULL)
    if result.returncode:
        raise IntegrityError("Installed vLLM CPU full CLI help failed")
    required={arg for arg in server_argv if arg.startswith("--")}
    # A similarly named option must not satisfy the check (e.g. --port-legacy).
    options=set(re.findall(r"(?<![\w-])--[A-Za-z0-9][A-Za-z0-9-]*(?![\w-])",result.stdout))
    missing=sorted(required-options)
    if missing:
        raise IntegrityError("Installed vLLM full CLI help lacks required flags: " + ", ".join(missing))
    return dict(command=command,required_flags=sorted(required),sha256=sha(result.stdout))


def resolve_installed(args):
    """Reviewed v0.31.0 adapter only; no synthetic/injected backend argument."""
    if os.environ.get("SLURM_JOB_ID"):
        raise IntegrityError("Run the CPU resolution probe before allocation, not in a Slurm job")
    validate_environment()
    fingerprint = installed_fingerprint(args.vllm) # Fails before serving imports if absent/unsupported.
    verify_snapshot_config(args.snapshot)
    assets = Assets.load()
    from .runtime import PinnedTokenizer
    tokenizer = PinnedTokenizer(args.tokenizer)
    if tokenizer.tokenizer.eos_token_id != 151645:
        raise IntegrityError("Pinned tokenizer primary EOS mismatch")
    from .allocation import server_command
    argv = server_command(args.vllm,args.snapshot,args.tokenizer,args.port)
    fixtures = wire_fixtures(assets)
    with cpu_only_environment():
        torch = importlib.import_module("torch")
        with forbid_cuda_initialization(torch):
            # Full help exits in argument parsing; it does not serve or load weights.
            help_evidence=complete_cli_help(args.vllm,argv)
            model_module = importlib.import_module("vllm.config.model")
            request_module = importlib.import_module("vllm.entrypoints.openai.chat_completion.protocol")
            config_module = importlib.import_module("vllm.config")
            for module,relative in ((model_module,"vllm/config/model.py"),
                    (request_module,"vllm/entrypoints/openai/chat_completion/protocol.py"),
                    (importlib.import_module("vllm.sampling_params"),"vllm/sampling_params.py")):
                if sha(Path(module.__file__).read_bytes()) != fingerprint["sources"][relative]:
                    raise IntegrityError("Imported serving module differs from its installed distribution")
            config = model_module.ModelConfig(model=str(Path(args.snapshot).resolve()),
                tokenizer=str(Path(args.tokenizer).resolve()),revision=MODEL_REVISION,
                tokenizer_revision=MODEL_REVISION,dtype="bfloat16",max_model_len=8192,
                generation_config="vllm",override_generation_config={},trust_remote_code=False)
            if config.generation_config != "vllm" or config.override_generation_config != {}:
                raise IntegrityError("Installed model configuration lost neutral mode")
            defaults = config.get_diff_sampling_param()
            if defaults != {}:
                raise IntegrityError("Installed server has non-neutral sampling overrides")
            # Read declared engine defaults without constructing VllmConfig or an engine.
            import dataclasses
            fields = {field.name:field for field in dataclasses.fields(config_module.VllmConfig)}
            if "watermark_config" not in fields or fields["watermark_config"].default is not None:
                raise IntegrityError("Installed engine watermark configuration is not disabled")
            generation = config.try_get_generation_config()
            if set(generation.get("eos_token_id",[])) != {151643,151645}:
                raise IntegrityError("Installed generation config did not resolve both pinned EOS IDs")
            request_class = request_module.ChatCompletionRequest
            if not WIRE_KEYS <= request_class.model_fields.keys():
                raise IntegrityError("Installed request schema lacks a frozen wire key")
            resolved=[]
            for fixture in fixtures:
                wire = fixture["request"]
                request = request_class(**wire)
                params = request.to_sampling_params(wire["max_tokens"],defaults)
                params.update_from_generation_config(generation,tokenizer.tokenizer.eos_token_id)
                checked = check_resolved(request,params,wire,fixture["role"])
                # Complete serializable request + sampler state as well as audited controls.
                checked.update(label=fixture["label"], complete_request=plain(request.model_dump()),
                               complete_sampling_params={name:plain(getattr(params,name)) for name in params.__struct_fields__})
                resolved.append(checked)
    from .controller import implementation_digest
    return dict(format="sun_gn_sampling_probe_v1",passed=True,verification="installed_runtime_cpu_only",
        outside_allocation=True,weights_loaded=False,server_started=False,generation_requests=0,
        sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA,
        implementation_sha256=implementation_digest(),model_revision=MODEL_REVISION,
        snapshot=str(Path(args.snapshot).resolve()),tokenizer=str(Path(args.tokenizer).resolve()),port=args.port,
        fingerprint=fingerprint,server_argv=argv,server_sampling_overrides=defaults,watermark_config=None,
        generation_config_sha256=policy()["generation_config_sha256"],generation_config_fields=plain(generation),
        tokenizer_assets=tokenizer.metadata["assets"],chat_template_sha256=sha(tokenizer.tokenizer.chat_template),
        wire_fixtures=fixtures,resolved=resolved,cli_help_sha256=help_evidence["sha256"],
        cli_help_command=help_evidence["command"],cli_help_required_flags=help_evidence["required_flags"],
        independent_tokenizer_verification="separate_required_parent_gate")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot","tokenizer","vllm","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--port",type=int,default=18000)
    args=parser.parse_args(argv)
    # Never overwrite an uncertain or earlier receipt, including failure receipts.
    with args.output.open("x",encoding="utf-8",newline="\n") as stream:
        try:
            result=resolve_installed(args)
        except Exception as error:
            result=dict(format="sun_gn_sampling_probe_v1",passed=False,
                verification="UNVERIFIED",error_type=type(error).__name__,
                reason=str(error) if isinstance(error,IntegrityError) else "Existing CPU runtime dependency or local file unavailable",
                sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA,
                weights_loaded=False,server_started=False,generation_requests=0)
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(passed=result["passed"],verification=result["verification"],
                          receipt_sha256=sha(args.output.read_bytes()))))
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
