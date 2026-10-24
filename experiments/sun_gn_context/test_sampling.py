"""Offline A2 assertions; stand-ins never count as installed-runtime evidence."""
from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import allocation, runtime, sampling, sampling_probe
from .allocation_budget import AllocationContext
from .core import Assets,IntegrityError,PROTOCOL_SHA,ADDENDUM_SHA,CONTRACT_SHA
from .freeze import MODEL_REVISION,BENCHMARK,sha
from . import test_allocation as helpers


class SamplingTests(unittest.TestCase):
    def standins(self,role="actor"):
        fixture=next(f for f in sampling.wire_fixtures(Assets.load()) if f["role"]==role)
        spec=sampling.policy()
        request=SimpleNamespace(**spec["request_defaults"])
        params=SimpleNamespace(**dict(spec["common"],**spec[role],seed=fixture["request"]["seed"]))
        params.stop_token_ids=[151643]
        params.sampling_type=SimpleNamespace(name="RANDOM_SEED" if role=="actor" else "GREEDY")
        params.output_kind=SimpleNamespace(name="FINAL_ONLY")
        return fixture,request,params

    def test_policy_exact_hashes_and_server_mode_only_delta(self):
        sampling.policy()
        command=allocation.server_command("vllm",Path("snapshot"),Path("tokenizer"),18000)
        self.assertEqual(command[command.index("--generation-config")+1],"vllm")
        self.assertEqual(command.count("--generation-config"),1)
        for flag in ("--override-generation-config","--watermark-config","--logits-processors","--speculative-config"):
            self.assertNotIn(flag,command)
        self.assertEqual(command[command.index("--revision")+1],MODEL_REVISION)

    def help_fixture(self):
        fixture=json.loads((sampling.ROOT/"testdata/vllm-v0.31.0-help-display.json").read_bytes())
        argv=allocation.server_command("synthetic-vllm",Path("snapshot"),Path("tokenizer"),18000)
        self.assertEqual(fixture["required_flags"],sorted(arg for arg in argv if arg.startswith("--")))
        self.assertEqual(len(fixture["required_flags"]),13)
        return fixture,argv

    def test_full_cli_help_acquisition_requests_all_once(self):
        fixture,argv=self.help_fixture()
        self.assertTrue(all(flag not in fixture["abbreviated_help"] for flag in fixture["required_flags"]))
        with patch.object(sampling_probe.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout=fixture["full_help"])) as run:
            result=sampling_probe.complete_cli_help(Path("synthetic-vllm"),argv)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0],["synthetic-vllm","serve","--help=all"])
        self.assertEqual(run.call_args.kwargs["timeout"],30)
        self.assertEqual(result["command"],run.call_args.args[0])
        self.assertEqual(result["required_flags"],fixture["required_flags"])
        self.assertEqual(result["sha256"],sha(fixture["full_help"]))

    def test_abbreviated_help_failure_or_missing_flag_still_fails_closed_no_retry(self):
        fixture,argv=self.help_fixture()
        missing_port=fixture["full_help"].replace("--port PORT","--port-legacy PORT")
        missing_tokenizer=fixture["full_help"].replace("--tokenizer TOKENIZER","--tokenizer-mode TOKENIZER_MODE")
        for code,output in ((0,fixture["abbreviated_help"]),(1,fixture["full_help"]),(0,missing_port),(0,missing_tokenizer),(0,"")):
            with self.subTest(code=code,output=output[:20]), \
                    patch.object(sampling_probe.subprocess,"run",return_value=SimpleNamespace(returncode=code,stdout=output)) as run:
                with self.assertRaises(IntegrityError):
                    sampling_probe.complete_cli_help(Path("synthetic-vllm"),argv)
                run.assert_called_once()
        with patch.object(sampling_probe.subprocess,"run",side_effect=__import__("subprocess").TimeoutExpired("synthetic",30)) as run:
            with self.assertRaises(__import__("subprocess").TimeoutExpired):
                sampling_probe.complete_cli_help(Path("synthetic-vllm"),argv)
            run.assert_called_once()

    def test_actual_resolver_help_gate_uses_full_display_standin_without_serving(self):
        # Actual resolver control flow, simulated CLI/dependencies; no runtime claim.
        fixture,_=self.help_fixture()
        class GatePassed(Exception): pass
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False,_lazy_init=lambda:None))
        def import_until_gate(name):
            if name=="torch":
                return fake_torch
            if name=="vllm.config.model":
                raise GatePassed("stopped before configuration construction")
            raise AssertionError(name)
        from . import runtime as runtime_module
        args=SimpleNamespace(vllm=Path("synthetic-vllm"),snapshot=Path("snapshot"),tokenizer=Path("tokenizer"),port=18000)
        for display,error_type in ((fixture["abbreviated_help"],IntegrityError),(fixture["full_help"],GatePassed)):
            with patch.object(sampling_probe,"installed_fingerprint",return_value={"synthetic":True}), \
                    patch.object(sampling_probe,"verify_snapshot_config"), \
                    patch.object(runtime_module,"PinnedTokenizer",return_value=SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=151645))), \
                    patch.object(sampling_probe.importlib,"import_module",side_effect=import_until_gate), \
                    patch.object(sampling_probe.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout=display)) as run:
                with self.assertRaises(error_type):
                    sampling_probe.resolve_installed(args)
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0],["synthetic-vllm","serve","--help=all"])

    def test_six_key_fixtures_cover_every_condition_and_both_curator_purposes(self):
        fixtures=sampling.wire_fixtures(Assets.load())
        self.assertEqual([x["label"] for x in fixtures],["B0","B1","S","E","curator","canary"])
        for fixture in fixtures:
            sampling.validate_wire(fixture["request"],fixture["role"])
            self.assertEqual(set(fixture["request"]),sampling.WIRE_KEYS)
            for bad in (dict(fixture["request"],top_k=0),dict(fixture["request"],seed="formula"),
                        dict(fixture["request"],seed=True),dict(fixture["request"],max_tokens=999)):
                with self.assertRaises(IntegrityError):
                    sampling.validate_wire(bad,fixture["role"])

    def test_mock_resolution_accepts_neutral_rejects_non_neutral_or_missing(self):
        for role in ("actor","curator"):
            fixture,request,params=self.standins(role)
            checked=sampling.check_resolved(request,params,fixture["request"],role)
            self.assertEqual(checked["effective"]["repetition_penalty"],1)
            for key,value in (("top_k",20),("top_k",-1),("repetition_penalty",1.05),
                    ("all_stop_token_ids",[151645]),("stop_token_ids",[151643,151645]),
                    ("ignore_eos",True),("max_tokens",1),("trace_decode_token_ids",[123])):
                bad=deepcopy(params)
                setattr(bad,key,value)
                with self.subTest(role=role,key=key,value=value),self.assertRaises(IntegrityError):
                    sampling.check_resolved(request,bad,fixture["request"],role)
            del params.top_k
            with self.assertRaises(IntegrityError):
                sampling.check_resolved(request,params,fixture["request"],role)
            request.stream=True
            with self.assertRaises(IntegrityError):
                sampling.check_resolved(request,self.standins(role)[2],fixture["request"],role)

    def test_environment_rejects_hidden_overrides_and_suppressed_warnings(self):
        sampling.validate_environment({})
        sampling.validate_environment({"VLLM_LOGGING_LEVEL":"INFO"})
        for env in ({"VLLM_LOGGING_LEVEL":"ERROR"},{"VLLM_ATTENTION_BACKEND":"FLASH_ATTN"},
                    {"VLLM_SAMPLER_OVERRIDE":"secret-not-logged"}):
            with self.assertRaises(IntegrityError) as found:
                sampling.validate_environment(env)
            self.assertNotIn("secret-not-logged",str(found.exception))

    def test_unavailable_or_unreviewed_runtime_fails_before_serving_imports(self):
        with patch.object(sampling.importlib.metadata,"version",side_effect=importlib.metadata.PackageNotFoundError("vllm")):
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                sampling.installed_fingerprint(Path("unused"))
        with patch.object(sampling.importlib.metadata,"version",return_value="0.19.0"):
            with self.assertRaisesRegex(IntegrityError,"no reviewed"):
                sampling.installed_fingerprint(Path("unused"))

    def test_serving_path_pins_extend_policy_without_changing_sampling_bytes(self):
        pins=sampling.reviewed_source_pins()
        self.assertEqual(len(pins),7)
        self.assertEqual({k:pins[k] for k in sampling.policy()["reviewed_sources"]},sampling.policy()["reviewed_sources"])
        expected={"vllm/v1/engine/input_processor.py","vllm/entrypoints/openai/chat_completion/serving.py",
                  "vllm/entrypoints/serve/utils/api_utils.py","vllm/entrypoints/serve/engine/protocol.py"}
        self.assertEqual(set(pins)-set(sampling.policy()["reviewed_sources"]),expected)
        self.assertEqual(sha((sampling.ROOT/"serving_source_contract.json").read_bytes()),sampling.SERVING_SOURCE_SHA)
        self.assertEqual(sha((sampling.ROOT/"sampling_policy.json").read_bytes()),sampling.POLICY_SHA)

    def test_fingerprint_detects_each_extra_serving_source_mutation_with_mock_distribution(self):
        # Fake distribution files exercise hashing only, not an installed runtime.
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            paths=list(sampling.reviewed_source_pins())
            pins={}
            for name in paths:
                file=root/name
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_bytes(("synthetic "+name).encode())
                pins[name]=sha(file.read_bytes())
            record=Path("vllm-0.31.0.dist-info/RECORD")
            (root/record).parent.mkdir()
            (root/record).write_bytes(b"synthetic record")
            executable=root/"synthetic-vllm"
            executable.write_bytes(b"not executable; hash fixture only")
            dist=SimpleNamespace(files=[record],locate_file=lambda name:root/name)
            with patch.object(sampling,"reviewed_source_pins",return_value=pins), \
                    patch.object(sampling.importlib.metadata,"version",side_effect=lambda name:"0.31.0" if name=="vllm" else "synthetic"), \
                    patch.object(sampling.importlib.metadata,"distribution",return_value=dist), \
                    patch.object(sampling.sys,"executable",str(root/"synthetic-python")):
                fingerprint=sampling.installed_fingerprint(executable)
                self.assertEqual(fingerprint["sources"],pins)
                self.assertEqual(fingerprint["serving_source_contract_sha256"],sampling.SERVING_SOURCE_SHA)
                for name in paths[3:]:
                    file=root/name
                    saved=file.read_bytes()
                    file.write_bytes(saved+b" changed")
                    with self.subTest(name=name),self.assertRaisesRegex(IntegrityError,"source differs"):
                        sampling.installed_fingerprint(executable)
                    file.write_bytes(saved)
    def test_probe_cli_disk_failure_is_unverified_nonzero_exclusive_and_no_launch(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            out=root/"probe.json"
            args=["--snapshot",str(root),"--tokenizer",str(root),"--vllm",str(root/"vllm"),"--output",str(out)]
            with patch.object(sampling_probe,"installed_fingerprint",side_effect=importlib.metadata.PackageNotFoundError("vllm")), \
                    patch.object(sampling_probe.subprocess,"run") as launch:
                self.assertEqual(sampling_probe.main(args),1)
                launch.assert_not_called()
                with self.assertRaises(FileExistsError):
                    sampling_probe.main(args)
            receipt=json.loads(out.read_bytes())
            self.assertFalse(receipt["passed"])
            self.assertEqual(receipt["verification"],"UNVERIFIED")
            self.assertEqual(receipt["generation_requests"],0)
            self.assertFalse(receipt["server_started"])
            with self.assertRaises(IntegrityError):
                sampling.verified_probe(dict(sampling_probe_path=str(out.resolve()),sampling_probe_sha256=sha(out.read_bytes())),"digest")

    def test_probe_refuses_allocated_environment_before_dependency_or_help_calls(self):
        with patch.dict(sampling_probe.os.environ,{"SLURM_JOB_ID":"123"}), \
             patch.object(sampling_probe,"installed_fingerprint") as fingerprint:
            with self.assertRaisesRegex(IntegrityError,"before allocation"):
                sampling_probe.resolve_installed(SimpleNamespace())
            fingerprint.assert_not_called()

    @unittest.skipUnless(os.environ.get("SUN_GN_TOKENIZER"),"Exact tokenizer asset directory not supplied")
    def test_disk_provenance_rebind_and_mutations_with_explicit_mock_fingerprint(self):
        # This tests receipt validation with genuine asset bytes, not a real resolver.
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            tokenizer=Path(os.environ["SUN_GN_TOKENIZER"]).resolve()
            contract=json.loads((sampling.ROOT/"tokenizer_contract.json").read_bytes())
            fingerprint=dict(vllm_executable=str(root/"synthetic-vllm"))
            fixtures=sampling.wire_fixtures(Assets.load())
            resolved=[]
            for fixture in fixtures:
                _,request,params=self.standins(fixture["role"])
                params.seed=fixture["request"]["seed"]
                result=sampling.check_resolved(request,params,fixture["request"],fixture["role"])
                result.update(label=fixture["label"],complete_request=fixture["request"],complete_sampling_params={"synthetic":True})
                resolved.append(result)
            document=dict(format="sun_gn_sampling_probe_v1",passed=True,verification="installed_runtime_cpu_only",
                outside_allocation=True,weights_loaded=False,server_started=False,generation_requests=0,
                model_revision=MODEL_REVISION,sampling_addendum_sha256=sampling.A2_SHA,sampling_policy_sha256=sampling.POLICY_SHA,
                implementation_sha256="synthetic-digest",fingerprint=fingerprint,snapshot=str(root),tokenizer=str(tokenizer),port=18000,
                server_argv=allocation.server_command(root/"synthetic-vllm",root,tokenizer,18000),
                generation_config_sha256=sampling.policy()["generation_config_sha256"],generation_config_fields={"eos_token_id":[151645,151643]},
                tokenizer_assets=contract["assets"],chat_template_sha256="synthetic-template",wire_fixtures=fixtures,resolved=resolved,
                server_sampling_overrides={},watermark_config=None)
            out=root/"synthetic-probe.json"
            def runtime_for(value):
                out.write_text(json.dumps(value),encoding="utf-8")
                return dict(sampling_probe_path=str(out),sampling_probe_sha256=sha(out.read_bytes()),server_chat_template_sha256="synthetic-template")
            with patch.object(sampling,"installed_fingerprint",return_value=fingerprint), \
                    patch.object(sampling,"verify_snapshot_config"):
                receipt=runtime_for(document)
                self.assertEqual(sampling.verified_probe(receipt,"synthetic-digest"),document)
                out.write_text(out.read_text()+" ")
                with self.assertRaises(IntegrityError):
                    sampling.verified_probe(receipt,"synthetic-digest")
                mutations=[dict(document,verification="mock"),dict(document,passed=False),dict(document,outside_allocation=False),
                    dict(document,watermark_config={"unexpected":True}),dict(document,server_sampling_overrides={"top_k":20}),
                    dict(document,server_argv=document["server_argv"]+["--override-generation-config","{}"]),
                    dict(document,tokenizer_assets={}),dict(document,generation_config_fields={"eos_token_id":[151645]}),
                    dict(document,wire_fixtures=list(reversed(fixtures)))]
                altered=deepcopy(document)
                altered["resolved"][0]["effective"]["repetition_penalty"]=1.05
                mutations.append(altered)
                for value in mutations:
                    with self.assertRaises(IntegrityError):
                        sampling.verified_probe(runtime_for(value),"synthetic-digest")
                with patch.object(sampling,"installed_fingerprint",return_value={"changed":True}):
                    with self.assertRaises(IntegrityError):
                        sampling.verified_probe(runtime_for(document),"synthetic-digest")

    def test_cpu_guard_restores_environment_and_refuses_cuda_init(self):
        with patch.dict(sampling_probe.os.environ,{"CUDA_VISIBLE_DEVICES":"synthetic"}):
            with sampling_probe.cpu_only_environment():
                self.assertEqual(sampling_probe.os.environ["CUDA_VISIBLE_DEVICES"],"")
                self.assertEqual(sampling_probe.os.environ["HF_HUB_OFFLINE"],"1")
            self.assertEqual(sampling_probe.os.environ["CUDA_VISIBLE_DEVICES"],"synthetic")
        original=lambda:None
        cuda=SimpleNamespace(is_initialized=lambda:False,_lazy_init=original)
        with sampling_probe.forbid_cuda_initialization(SimpleNamespace(cuda=cuda)):
            with self.assertRaises(IntegrityError):
                cuda._lazy_init()
        self.assertIs(cuda._lazy_init,original)

    def test_server_warning_stops_before_controller_and_is_checked_after_cleanup(self):
        with tempfile.TemporaryDirectory() as path:
            args=helpers.AllocationLifecycleTests().args(path)
            clock=helpers.Clock()
            engine=helpers.FakeEngine(clock,args)
            seen=[]
            def warning(log):
                seen.append(log)
                raise IntegrityError("synthetic override warning")
            receipt=allocation.supervise(args,AllocationContext("stage-one","101","exp",900,1000,"digest"),
                engine,clock,verify_log=warning)
            self.assertEqual(receipt["status"],"failed")
            self.assertEqual([x[0] for x in engine.started],["server"])
            self.assertEqual(len(seen),2)
            self.assertEqual(receipt["sampling_addendum_sha256"],sampling.A2_SHA)
            self.assertEqual(receipt["sampling_policy_sha256"],sampling.POLICY_SHA)
            saved=json.loads((Path(args.ledger)/"stage-one.receipt.json").read_bytes())
            self.assertEqual(saved,receipt)
        with tempfile.TemporaryDirectory() as path:
            log=Path(path)/"server.log"
            log.write_text("synthetic startup\n")
            self.assertEqual(sampling.check_server_log(log),sha(log.read_bytes()))
            log.write_text(sampling.OVERRIDE_WARNING)
            with self.assertRaises(IntegrityError):
                sampling.check_server_log(log)

    def test_release_requires_independent_tokenizer_and_shared_probe_hash(self):
        common=dict(protocol_sha256=PROTOCOL_SHA,protocol_addendum_sha256=ADDENDUM_SHA,contract_sha256=CONTRACT_SHA,
                    benchmark_commit=BENCHMARK,model_revision=MODEL_REVISION,tokenizer_revision=MODEL_REVISION,
                    implementation_sha256="digest",sampling_addendum_sha256=sampling.A2_SHA,sampling_policy_sha256=sampling.POLICY_SHA,
                    sampling_probe_sha256="a"*64,independent_tokenizer_evidence_sha256="b"*64)
        release=dict(common,independent_correctness_gate=True,released_stage="stage-one",independent_tokenizer_gate=True)
        receipt=dict(common,weights_verified=True,existing_access_verified=True,vllm_version="0.31.0",torch_version="synthetic",
                     cuda_version="synthetic",weights_manifest_sha256="synthetic",server_chat_template_sha256="synthetic")
        # This patch exercises the release interlock only, never a real probe claim.
        with patch.object(runtime,"verified_probe",return_value={"fingerprint":{"versions":{"vllm":"0.31.0","torch":"synthetic"}}}) as verify:
            runtime.validate_release(release,receipt,"stage-one","digest")
            verify.assert_called_once()
            for bad in (dict(release,independent_tokenizer_gate=False),dict(release,sampling_probe_sha256="c"*64),
                        dict(release,sampling_policy_sha256="c"*64)):
                with self.assertRaises(IntegrityError):
                    runtime.validate_release(bad,receipt,"stage-one","digest")


if __name__=="__main__":
    unittest.main(verbosity=2)
