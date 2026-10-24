"""Offline lifecycle/receipt tests: injected fake children, never a live server."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import allocation,controller
from .allocation_budget import AllocationContext,validate_prior_allocation,stage_two_time_check
from .core import Assets,PROTOCOL_SHA,CONTRACT_SHA,ADDENDUM_SHA,IntegrityError,run_episode
from .freeze import ROOT,MODEL_REVISION,sha
from .gates import development_gate
from .journal import Journal,CallBoundary,read_journal
from .replay import replay
from .scoring import aggregate
from .test_offline import TokenizerStub,MemoryJournal
from . import test_offline as offline_helpers
from .sampling import A2_SHA,POLICY_SHA

ASSETS=Assets.load()


class Clock:
    def __init__(self):
        self.now=1000.
    def __call__(self):
        return self.now


class FakeEngine:
    def __init__(self,clock,args,fail_at=None):
        self.clock,self.args,self.fail_at=clock,args,fail_at
        self.started=[]
        self.stopped=[]
        self.deadlines=[]
    def start(self,command,log_path,cwd,env):
        kind="server" if "serve" in command else "controller"
        self.started.append((kind,command,dict(env)))
        if self.fail_at==kind:
            raise OSError("synthetic startup failure")
        return kind
    def ready(self,server,port,deadline):
        self.deadlines.append(("ready",deadline))
        self.clock.now+=90
        if self.fail_at=="ready":
            raise TimeoutError("synthetic readiness timeout")
    def wait_controller(self,child,server,deadline):
        self.deadlines.append(("controller",deadline))
        self.clock.now+=100
        if self.fail_at=="controller_wait":
            raise TimeoutError("synthetic controller timeout")
        out=Path(self.args.output)/"controller"
        out.mkdir()
        name="stage_one_complete.json" if self.args.stage=="stage-one" else "evaluation_summary.json"
        (out/name).write_text(json.dumps(dict(completed=True)))
        return 0
    def stop(self,child,deadline):
        self.deadlines.append(("cleanup",deadline))
        self.stopped.append(child)
        self.clock.now+=10
        if self.fail_at=="cleanup" and child=="server":
            raise TimeoutError("synthetic cleanup timeout")


class AllocationBudgetTests(unittest.TestCase):
    def context(self,stage="stage-one",job="101"):
        return AllocationContext(stage,job,"synthetic_experiment",1000.,1100.,"digest")

    def test_allocation_start_propagates_and_reserves_save_shutdown(self):
        c=self.context()
        self.assertEqual(c.hard_deadline_unix,8200)
        self.assertEqual(c.controller_deadline_unix,8080)
        self.assertEqual(c.request_deadline_unix,7960)
        self.assertEqual(c.startup_elapsed(1150),150)
        self.assertEqual(c.request_monotonic_deadline(1150,500),7310)
        loaded=AllocationContext.load(c.document(),"stage-one","digest","synthetic_experiment","101")
        self.assertEqual(loaded,c)
        for key,value in [("wall_seconds",7201),("save_reserve_seconds",0),("job_id","102")]:
            bad=c.document()
            bad[key]=value
            with self.assertRaises(IntegrityError):
                AllocationContext.load(bad,"stage-one","digest","synthetic_experiment","101")

    def test_missing_allocation_start_or_multi_gpu_rejected(self):
        env=dict(SLURM_JOB_ID="101",SLURM_JOB_START_TIME="1000",SLURM_JOB_NUM_NODES="1",SLURM_GPUS_ON_NODE="1")
        self.assertEqual(allocation.allocation_identity(env,"stage-one",dict(experiment_id="exp"),"digest",1100).allocation_start_unix,1000)
        for key,value in [("SLURM_JOB_START_TIME",""),("SLURM_GPUS_ON_NODE","2"),("SLURM_JOB_NUM_NODES","2")]:
            bad=dict(env,**{key:value})
            with self.assertRaises(IntegrityError):
                allocation.allocation_identity(bad,"stage-one",dict(experiment_id="exp"),"digest",1100)

    def test_first_job_gate_uses_measured_startup_saving_shutdown(self):
        episodes,scores,canaries=offline_helpers.GateTests().inputs(10)
        with tempfile.TemporaryDirectory() as path:
            clock=iter([200.,220.])
            boundary=SimpleNamespace(reserve_seconds=120)
            assets=deepcopy(ASSETS)
            assets.manifest["rows"]=[r for r in assets.manifest["rows"] if r["stage"]!="adaptation"]
            # Stop just after the gate so no mock adaptation/checkpoint is needed.
            observed={}
            def gate(*args,**kwargs):
                observed.update(kwargs)
                return dict(passed=False,reasons=["synthetic_stop"])
            with patch.object(controller,"development",return_value=(episodes,scores,canaries,500)), \
                 patch.object(controller,"development_gate",side_effect=gate), \
                 patch.object(controller.time,"monotonic",side_effect=lambda:next(clock)):
                with self.assertRaises(IntegrityError):
                    controller.stage_one(assets,boundary,TokenizerStub(),MemoryJournal(),Path("unused"),Path(path),100,self.context())
            # 100 pre-development + 20 measured saving + 120 final save + 120 shutdown.
            self.assertEqual(observed["startup_save_seconds"],360)
            frozen=json.loads((Path(path)/"gate.json").read_bytes())
            self.assertEqual(frozen["overhead_evidence"]["startup_save_seconds_used"],360)

    def test_stage_two_startup_rechecks_frozen_N_without_fallback(self):
        gate=dict(allowance_seconds=600,actor_conservative_seconds=90)
        c=self.context("stage-two","102")
        self.assertEqual(stage_two_time_check(gate,100,c,1100)["selected_N"],100)
        with self.assertRaises(IntegrityError):
            stage_two_time_check(gate,100,c,1600)


class AllocationLifecycleTests(unittest.TestCase):
    def args(self,root,stage="stage-one"):
        root=Path(root)
        release=root/"release.json"
        runtime=root/"runtime.json"
        release.write_text("{}")
        runtime.write_text("{}")
        ledger=root/"ledger"
        ledger.mkdir()
        return SimpleNamespace(stage=stage,output=root/"output",ledger=ledger,release=release,runtime=runtime,
                               code=root,python=Path("runtime-python"),vllm=Path("runtime-vllm"),
                               snapshot=root/"snapshot",tokenizer=root/"tokenizer",arbench=root/"benchmark",port=18000,prior=None)

    def test_one_shot_claims_and_stage_receipts_prevent_extra_jobs(self):
        with tempfile.TemporaryDirectory() as path:
            ledger=Path(path)/"ledger"
            c=AllocationContext("stage-one","101","exp",1000,1100,"digest")
            allocation.claim_stage(ledger,c,"release","runtime")
            with self.assertRaises(FileExistsError):
                allocation.claim_stage(ledger,c,"release","runtime")
            # A failed/unknown first allocation cannot unlock stage two.
            receipt=dict(stage="stage-one",status="failed",experiment_id="exp",implementation_sha256="digest",
                         protocol_sha256=PROTOCOL_SHA,protocol_addendum_sha256=ADDENDUM_SHA,contract_sha256=CONTRACT_SHA,
                         job_id="101",gpu_count=1,gpu_type="l40s",allocation_elapsed_seconds=7000)
            with self.assertRaises(IntegrityError):
                validate_prior_allocation(receipt,"exp","digest","102")
            receipt["status"]="completed"
            validate_prior_allocation(receipt,"exp","digest","102")
            for change in (dict(allocation_elapsed_seconds=7201),dict(job_id="102"),dict(gpu_count=2)):
                with self.assertRaises(IntegrityError):
                    validate_prior_allocation(dict(receipt,**change),"exp","digest","102")

    def test_lifecycle_counts_startup_and_cleanup_without_real_processes(self):
        with tempfile.TemporaryDirectory() as path:
            args=self.args(path)
            clock=Clock()
            context=AllocationContext("stage-one","101","exp",900,1000,"digest")
            engine=FakeEngine(clock,args)
            def preflight():
                clock.now+=30
            receipt=allocation.supervise(args,context,engine,clock,preflight)
            self.assertEqual(receipt["status"],"completed")
            self.assertEqual(receipt["pre_controller_seconds"],220)
            self.assertEqual(receipt["allocation_elapsed_seconds"],340)
            self.assertEqual([x[0] for x in engine.started],["server","controller"])
            self.assertEqual(engine.stopped,["controller","server"])
            self.assertTrue(receipt["no_retry"])
            saved=json.loads((Path(args.ledger)/"stage-one.receipt.json").read_bytes())
            self.assertEqual(saved,receipt)
            context_doc=json.loads((Path(args.output)/"allocation_context.json").read_bytes())
            self.assertEqual(context_doc["allocation_start_unix"],900)
            self.assertEqual(context_doc["controller_start_unix"],1120)
            server=engine.started[0][1]
            self.assertEqual(server[server.index("--revision")+1],MODEL_REVISION)
            self.assertEqual(server[server.index("--tokenizer-revision")+1],MODEL_REVISION)
            self.assertEqual(engine.started[0][2]["HF_HUB_OFFLINE"],"1")
            command=engine.started[1][1]
            self.assertIn("--allocation-context",command)

    def test_failures_never_restart_server_or_controller(self):
        for failure in ("server","ready","controller","controller_wait","cleanup"):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as path:
                args=self.args(path)
                clock=Clock()
                engine=FakeEngine(clock,args,failure)
                c=AllocationContext("stage-one","101","exp",900,1000,"digest")
                receipt=allocation.supervise(args,c,engine,clock)
                self.assertNotEqual(receipt["status"],"completed")
                self.assertLessEqual(sum(k=="server" for k,_,_ in engine.started),1)
                self.assertLessEqual(sum(k=="controller" for k,_,_ in engine.started),1)
                self.assertTrue(receipt["no_retry"])

    def test_wrapper_static_limits_and_no_submission_path(self):
        source=(ROOT/"gn_allocation.sbatch").read_text()
        for directive in ("#SBATCH --time=02:00:00","#SBATCH --gres=gpu:l40s:1","#SBATCH --nodes=1",
                          "#SBATCH --no-requeue","#SBATCH --signal=B:TERM@120"):
            self.assertIn(directive,source)
        active="\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        for forbidden in ("sbatch ","scontrol requeue","ssh ","pip install","while "):
            self.assertNotIn(forbidden,active)
        self.assertIn("${SLURM_JOB_START_TIME:?",active)
        self.assertNotIn("serve_7b.sbatch",source)

    def test_bash_syntax_only_when_existing_parser_available(self):
        bash=shutil.which("bash")
        if not bash:
            candidate=Path(r"C:\Program Files\Git\bin\bash.exe")
            if candidate.is_file():
                bash=str(candidate)
        if not bash:
            self.skipTest("No existing Bash parser; static directive test still applies")
        run=subprocess.run([bash,"--noprofile","--norc","-n",str(ROOT/"gn_allocation.sbatch")],capture_output=True,text=True,timeout=20)
        if run.returncode and "couldn't create signal pipe, Win32 error 5" in run.stderr:
            self.skipTest("Existing Bash parser blocked before parsing: Win32 signal-pipe error 5; no workaround attempted")
        self.assertEqual(run.returncode,0,run.stderr)

    def test_second_stage_claim_binds_prior_output_and_cannot_repeat(self):
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            ledger=root/"ledger"
            first=AllocationContext("stage-one","101","exp",1000,1100,"digest")
            allocation.claim_stage(ledger,first,"release-one","runtime")
            prior=root/"prior"
            prior.mkdir()
            complete=prior/"stage_one_complete.json"
            complete.write_text('{"completed":true}')
            receipt=dict(stage="stage-one",status="completed",experiment_id="exp",implementation_sha256="digest",
                         protocol_sha256=PROTOCOL_SHA,protocol_addendum_sha256=ADDENDUM_SHA,contract_sha256=CONTRACT_SHA,
                         job_id="101",gpu_count=1,gpu_type="l40s",allocation_elapsed_seconds=7000,
                         controller_output=str(prior.resolve()),controller_complete_sha256=sha(complete.read_bytes()))
            receipt.update(sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA,
                           sampling_probe_sha256=None,runtime_sha256="runtime")
            allocation.write_exclusive(ledger/"stage-one.receipt.json",receipt)
            second=AllocationContext("stage-two","102","exp",10000,10100,"digest")
            with self.assertRaises(IntegrityError):
                allocation.claim_stage(ledger,second,"release-two","runtime",root/"different")
            allocation.claim_stage(ledger,second,"release-two","runtime",prior)
            with self.assertRaises(FileExistsError):
                allocation.claim_stage(ledger,second,"release-two","runtime",prior)

    def test_release_binds_ledger_path_before_any_child_launch(self):
        with tempfile.TemporaryDirectory() as path:
            args=SimpleNamespace(ledger=Path(path)/"ledger")
            allocation.validate_stage_paths(args,dict(allocation_ledger_path=str(args.ledger.resolve())))
            for release in ({},dict(allocation_ledger_path="relative"),dict(allocation_ledger_path=str((Path(path)/"different").resolve()))):
                with self.assertRaises(IntegrityError):
                    allocation.validate_stage_paths(args,release)


class PairedOrderingTests(unittest.TestCase):
    def test_hundred_case_vector_and_CI_ignore_score_or_ID_permutation(self):
        ids=[r["case_id"] for r in ASSETS.manifest["rows"] if r["stage"]=="evaluation"]
        rows=[]
        for i,case in enumerate(ids):
            difference=((i*37)%101-50)/200
            for condition in ("B0","B1","S","E"):
                rows.append(dict(case_id=case,condition=condition,Q=.4+(difference if condition=="E" else 0),
                                 final_EM=False,hit_any=False,invalid_outputs=0,truncated_outputs=0))
        before=aggregate(ids,rows)
        random.Random(42).shuffle(rows)
        after=aggregate(list(reversed(ids)),rows)
        self.assertEqual(before["paired_case_ids"],ids)
        self.assertEqual(before["paired_delta_Q"],after["paired_delta_Q"])
        self.assertEqual(before["CI95"],after["CI95"])
        self.assertEqual(before,after)

    def test_permuted_raw_completion_events_produce_identical_pairs(self):
        ids=[r["case_id"] for r in ASSETS.manifest["rows"] if r["stage"]=="evaluation"][:2]
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            first=root/"first.jsonl"
            journal=Journal(first,dict(mode="synthetic_fixture"))
            script=[]
            called=[]
            def transport(request):
                text=script[len(called)]
                called.append(None)
                return dict(text=text,finish_reason="stop",prompt_tokens=100,completion_tokens=10,response_id=str(len(called)))
            boundary=CallBoundary(transport,TokenizerStub(),journal,actor_limit=208,curator_limit=0,input_limit=100000,
                                  output_limit=100000,deadline=__import__("time").monotonic()+3600,reserve_seconds=120)
            for i,case in enumerate(ids):
                for condition in ("B0","B1","S","E"):
                    script.extend((["malformed"]*(i+1)+["Guess: 0123"]*(24-i)+["malformed"]) if condition=="E" else ["Guess: 0000"]*25+["malformed"])
                    run_episode(ASSETS,boundary,case_id=case,secret="0123",condition=condition,memory=ASSETS.m0,journal=journal)
            journal.close()
            original=read_journal(first)
            second=root/"second.jsonl"
            reordered=Journal(second,original[0]["payload"])
            rest=original[1:]
            events=[r for r in rest if r["event"]!="episode_complete"]+list(reversed([r for r in rest if r["event"]=="episode_complete"]))
            for row in events:
                reordered.append(row["event"],row["payload"])
            reordered.close()
            a,_=replay(first,ASSETS)
            b,_=replay(second,ASSETS)
            self.assertEqual(aggregate(ids,a),aggregate(ids,b))


if __name__=="__main__":
    unittest.main(verbosity=2)
