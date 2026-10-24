"""Run exactly one already released GN stage inside an existing Slurm job.

No submission, installation, SSH, tunnel, retry or resumption code is provided.
Import and offline planning never launch a process. The CLI requires Linux,
Slurm allocation evidence, exact release/runtime bindings and one-shot claims.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

from .allocation_budget import AllocationContext,WALL_SECONDS,SHUTDOWN_RESERVE_SECONDS,validate_prior_allocation
from .core import Assets,CONTRACT_SHA,PROTOCOL_SHA,ADDENDUM_SHA,IntegrityError
from .freeze import ROOT,MODEL_REVISION,sha
from .runtime import PinnedTokenizer,validate_release
from .process_state import owned_group_state
from .sampling import A2_SHA,POLICY_SHA,policy,validate_environment,verify_snapshot_config,check_server_log


class Cancellation:
    """A signal handler records only; exceptions occur at controlled checkpoints."""
    def __init__(self):
        self.requested=False
    def handler(self,signum,frame):
        self.requested=True
    def check(self):
        if self.requested:
            raise TimeoutError("Allocation cancellation requested")


@contextmanager
def record_cancellation(cancellation,deadline,clock=time.time):
    if os.name!="posix" or threading.current_thread() is not threading.main_thread():
        yield
        return
    numbers=(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)
    previous={number:signal.signal(number,cancellation.handler) for number in numbers}
    signal.setitimer(signal.ITIMER_REAL,max(.01,deadline-clock()))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for number,handler in previous.items():
            signal.signal(number,handler)


@contextmanager
def defer_cleanup_cancellation():
    """Disarm the alarm and consume cancellation through cleanup + receipt fsync.

    Only this supervisor's handlers/mask are changed. Child processes/groups are
    still terminated within the original allocation deadline. No permissions,
    scheduler settings or other processes are changed.
    """
    if os.name!="posix" or threading.current_thread() is not threading.main_thread():
        yield False
        return
    signals={signal.SIGALRM,signal.SIGTERM,signal.SIGINT}
    previous_mask=signal.pthread_sigmask(signal.SIG_BLOCK,signals)
    previous_handlers={number:signal.getsignal(number) for number in signals}
    try:
        signal.setitimer(signal.ITIMER_REAL,0)
        for number in signals:
            signal.signal(number,lambda signum,frame:None)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
    try:
        yield True
    finally:
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,signals)
        signal.setitimer(signal.ITIMER_REAL,0)
        # Discard queued cancellation before restoring the caller's handlers.
        # Owned children have already been stopped and terminal receipts fsynced.
        for number in signals:
            signal.signal(number,signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        for number,handler in previous_handlers.items():
            signal.signal(number,handler)


def write_exclusive(path,document):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(document,f,sort_keys=True,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def allocation_identity(env,stage,release,implementation_sha,now_unix):
    job=env.get("SLURM_JOB_ID","")
    start=env.get("SLURM_JOB_START_TIME","")
    experiment=release.get("experiment_id","")
    if (not job.isdigit() or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?",start) or
            not re.fullmatch(r"[A-Za-z0-9_-]{1,80}",experiment) or
            env.get("SLURM_JOB_NUM_NODES")!="1" or env.get("SLURM_GPUS_ON_NODE")!="1"):
        raise IntegrityError("Missing one-node/one-GPU Slurm allocation identity or experiment ID")
    context=AllocationContext(stage,job,experiment,float(start),now_unix,implementation_sha)
    if context.controller_deadline_unix<=now_unix:
        raise IntegrityError("Allocation has no startup/controller window remaining")
    return context


def claim_stage(ledger,context,release_sha,runtime_sha,prior_directory=None,probe_sha=None):
    """O_EXCL claims survive failures/unknown outcomes; there is no reset path."""
    ledger=Path(ledger)
    if context.stage=="stage-one":
        ledger.mkdir(exist_ok=False,parents=True)
        write_exclusive(ledger/"experiment.json",dict(experiment_id=context.experiment_id,
            implementation_sha256=context.implementation_sha256,protocol_sha256=PROTOCOL_SHA,
            contract_sha256=CONTRACT_SHA,max_planned_stages=2,max_allocation_seconds=14400))
    else:
        bound=json.loads((ledger/"experiment.json").read_bytes())
        if bound!=dict(experiment_id=context.experiment_id,implementation_sha256=context.implementation_sha256,
                       protocol_sha256=PROTOCOL_SHA,contract_sha256=CONTRACT_SHA,max_planned_stages=2,max_allocation_seconds=14400):
            raise IntegrityError("Experiment ledger identity mismatch")
        prior=json.loads((ledger/"stage-one.receipt.json").read_bytes())
        validate_prior_allocation(prior,context.experiment_id,context.implementation_sha256,context.job_id)
        if (prior.get("sampling_addendum_sha256")!=A2_SHA or prior.get("sampling_policy_sha256")!=POLICY_SHA or
                prior.get("sampling_probe_sha256")!=probe_sha or prior.get("runtime_sha256")!=runtime_sha):
            raise IntegrityError("Prior allocation sampling provenance differs from stage two")
        if prior_directory is None or prior.get("controller_output")!=str(Path(prior_directory).resolve()):
            raise IntegrityError("Stage two must use the exact completed first-stage output")
        complete=Path(prior_directory)/"stage_one_complete.json"
        if sha(complete.read_bytes())!=prior.get("controller_complete_sha256"):
            raise IntegrityError("Completed first-stage receipt changed")
    write_exclusive(ledger/f"{context.stage}.claim.json",dict(stage=context.stage,job_id=context.job_id,
                    experiment_id=context.experiment_id,allocation_start_unix=context.allocation_start_unix,
                    implementation_sha256=context.implementation_sha256,release_sha256=release_sha,
                    runtime_sha256=runtime_sha,sampling_addendum_sha256=A2_SHA,
                    sampling_policy_sha256=POLICY_SHA,sampling_probe_sha256=probe_sha,status="reserved_no_retry"))


def validate_stage_paths(args,release):
    approved=release.get("allocation_ledger_path")
    if not isinstance(approved,str) or not Path(approved).is_absolute() or Path(approved).resolve()!=Path(args.ledger).resolve():
        raise IntegrityError("Allocation ledger path differs from the one-shot stage release")


def verify_weight_manifest(snapshot,manifest_path,expected_sha):
    raw=Path(manifest_path).read_bytes()
    if sha(raw)!=expected_sha:
        raise IntegrityError("Unverified weight manifest bytes")
    manifest=json.loads(raw)
    if manifest.get("model")!="Qwen/Qwen2.5-7B-Instruct" or manifest.get("revision")!=MODEL_REVISION:
        raise IntegrityError("Weight manifest model/revision mismatch")
    snapshot=Path(snapshot).resolve()
    files=manifest["files"]
    for name,item in files.items():
        if Path(name).name!=name or name in (".","..") or not re.fullmatch(r"[A-Za-z0-9_.-]+",name):
            raise IntegrityError("Unsafe weight inventory name")
        file=snapshot/name
        if not file.is_file() or file.stat().st_size!=item["bytes"]:
            raise IntegrityError("Weight inventory size mismatch")
        with file.open("rb") as stream:
            if hashlib.file_digest(stream,"sha256").hexdigest()!=item["sha256"]:
                raise IntegrityError("Weight inventory hash mismatch")
    required={"config.json","generation_config.json","model.safetensors.index.json"}
    if not required<=files.keys():
        raise IntegrityError("Incomplete weight inventory")
    if files["generation_config.json"]["sha256"]!=policy()["generation_config_sha256"]:
        raise IntegrityError("Weight inventory does not bind the approved generation config")
    shards=set(json.loads((snapshot/"model.safetensors.index.json").read_bytes())["weight_map"].values())
    if not shards or any(Path(n).name!=n or not n.endswith(".safetensors") for n in shards) or not shards<=files.keys():
        raise IntegrityError("Weight shards absent from inventory")


def server_command(vllm,snapshot,tokenizer,port):
    if not 1024<=port<=65535:
        raise IntegrityError("Invalid loopback port")
    return [str(vllm),"serve",str(Path(snapshot).resolve()),"--served-model-name","Qwen/Qwen2.5-7B-Instruct",
            "--revision",MODEL_REVISION,"--tokenizer",str(Path(tokenizer).resolve()),
            "--tokenizer-revision",MODEL_REVISION,"--host","127.0.0.1","--port",str(port),
            "--generation-config","vllm",
            "--dtype","bfloat16","--max-model-len","8192","--max-num-seqs","8",
            "--attention-backend","FLASH_ATTN","--gpu-memory-utilization","0.88","--enable-prefix-caching"]


def controller_command(python,args,context_path,port):
    command=[str(python),"-m","experiments.sun_gn_context.controller",args.stage]
    for name,value in [("arbench",args.arbench),("tokenizer",args.tokenizer),("release",args.release),
                       ("runtime",args.runtime),("output",Path(args.output)/"controller"),("allocation-context",context_path),
                       ("base-url",f"http://127.0.0.1:{port}/v1")]:
        command.extend(["--"+name,str(value)])
    if args.prior:
        command.extend(["--prior",str(args.prior)])
    return command


class ProcessSupervisor:
    """Linux process groups, bounded readiness/termination, no child restart."""
    def __init__(self,clock=time.time,sleep=time.sleep,term_grace_seconds=30,kill_grace_seconds=5):
        self.clock,self.sleep=clock,sleep
        self.term_grace_seconds=term_grace_seconds
        self.kill_grace_seconds=kill_grace_seconds
        self.cancellation=Cancellation()
        self.cleanup_observations=[]

    def start(self,command,log_path,cwd,env):
        log=Path(log_path).open("xb")
        try:
            process=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        finally:
            log.close()
        return process

    def ready(self,server,port,deadline):
        # Health reads are bounded; no generation request is a readiness probe.
        while self.clock()<deadline:
            self.cancellation.check()
            if server.poll() is not None:
                raise IntegrityError("Server exited during startup")
            remaining=deadline-self.clock()
            try:
                with urlopen(f"http://127.0.0.1:{port}/health",timeout=min(2,remaining)) as response:
                    if response.status==200 and server.poll() is None:
                        return
            except (OSError,TimeoutError):
                pass
            self.sleep(min(1,max(0,deadline-self.clock())))
        raise TimeoutError("Server startup deadline")

    def wait_controller(self,controller,server,deadline):
        while self.clock()<deadline:
            self.cancellation.check()
            code=controller.poll()
            if code is not None:
                return code
            if server.poll() is not None:
                raise IntegrityError("Server exited during controller stage")
            self.sleep(min(1,max(0,deadline-self.clock())))
        raise TimeoutError("Controller allocation deadline")

    def stop(self,process,deadline):
        if process is None:
            return
        # Stop the whole group, including server workers; do not infer that a
        # parent exiting means all GPU children have exited.
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            try:
                process.wait(timeout=max(.01,min(self.term_grace_seconds,deadline-self.clock())))
            except (subprocess.TimeoutExpired,TimeoutError,KeyboardInterrupt):
                pass
        finally:
            # An interrupted TERM wait must never skip escalation for this
            # owned group. Cancellation is deferred by supervise during cleanup.
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.wait(timeout=max(.01,min(self.kill_grace_seconds,deadline-self.clock())))
        # Reaping the leader alone does not establish that owned workers exited.
        group_deadline=min(deadline,self.clock()+self.kill_grace_seconds)
        previous_zombies=None
        while True:
            try:
                os.killpg(process.pid,0)
            except ProcessLookupError:
                self.cleanup_observations.append(dict(group_id=process.pid,state="absent"))
                return
            observed=owned_group_state(process.pid)
            if observed["state"]=="zombie_only":
                # Two stable snapshots avoid mistaking a changing membership
                # snapshot for a completed group. Zombies cannot execute work.
                if observed["zombie_pids"]==previous_zombies:
                    self.cleanup_observations.append(dict(group_id=process.pid,**observed))
                    return
                previous_zombies=observed["zombie_pids"]
            else:
                previous_zombies=None
            if self.clock()>=group_deadline:
                self.cleanup_observations.append(dict(group_id=process.pid,**observed))
                raise TimeoutError("Owned process group has live or unconfirmed members")
            self.sleep(min(.05,max(0,group_deadline-self.clock())))


def supervise(args,initial_context,engine,clock=time.time,preflight=lambda:None,prior_elapsed=0,verify_log=None):
    cancellation=Cancellation()
    engine.cancellation=cancellation
    with record_cancellation(cancellation,initial_context.controller_deadline_unix,clock):
        return _supervise(args,initial_context,engine,cancellation,clock,preflight,prior_elapsed,verify_log)


def _supervise(args,initial_context,engine,cancellation,clock,preflight,prior_elapsed,verify_log):
    """Testable lifecycle; fake engine tests never start a real child process."""
    root=Path(args.output)
    root.mkdir(exist_ok=False,parents=True)
    server=controller=None
    status="failed"
    error_type=None
    controller_code=None
    context=initial_context
    runtime_document=json.loads(Path(args.runtime).read_bytes())
    server_argv=server_command(args.vllm,args.snapshot,args.tokenizer,args.port)
    env=dict(os.environ,HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    env.pop("VLLM_ATTENTION_BACKEND",None)
    try:
        cancellation.check()
        preflight() # Existing files/metadata only; inside the allocation clock.
        cancellation.check()
        if clock()>=context.request_deadline_unix:
            raise TimeoutError("Startup consumed the request window")
        server=engine.start(server_argv,root/"server.log",args.code,env)
        cancellation.check()
        # Bound startup to five minutes and the remaining request window.
        engine.ready(server,args.port,min(clock()+300,context.request_deadline_unix))
        if verify_log is not None:
            verify_log(root/"server.log")
        cancellation.check()
        context=AllocationContext(context.stage,context.job_id,context.experiment_id,
                                  context.allocation_start_unix,clock(),context.implementation_sha256)
        if clock()>=context.request_deadline_unix:
            raise TimeoutError("No request window after server startup")
        write_exclusive(root/"allocation_context.json",context.document())
        controller=engine.start(controller_command(args.python,args,root/"allocation_context.json",args.port),
                                root/"controller.log",args.code,env)
        cancellation.check()
        controller_code=engine.wait_controller(controller,server,context.controller_deadline_unix)
        cancellation.check()
        complete=root/"controller"/("stage_one_complete.json" if args.stage=="stage-one" else "evaluation_summary.json")
        if controller_code!=0 or not complete.is_file() or json.loads(complete.read_bytes()).get("completed") is not True:
            raise IntegrityError("Controller did not produce a complete stage")
        status="completed"
    except BaseException as error:
        error_type=type(error).__name__
        status="timed_out" if isinstance(error,TimeoutError) else "failed"
    finally:
        with defer_cleanup_cancellation() as protected:
            cleanup_errors=[]
            for child in (controller,server):
                try:
                    engine.stop(child,context.hard_deadline_unix-2)
                except BaseException as error:
                    cleanup_errors.append(type(error).__name__)
            if cleanup_errors:
                status="cleanup_uncertain"
            if server is not None and verify_log is not None:
                try:
                    verify_log(root/"server.log")
                except BaseException as error:
                    if status=="completed":
                        status="failed"
                        error_type=type(error).__name__
            log_digest=None
            log_error=None
            try:
                if (root/"server.log").is_file():
                    log_digest=sha((root/"server.log").read_bytes())
            except BaseException as error:
                log_error=type(error).__name__
                if status=="completed":
                    status="failed"
                    error_type=log_error
            ended=clock()
            if ended>context.hard_deadline_unix:
                status="budget_exceeded"
            total_elapsed=prior_elapsed+ended-context.allocation_start_unix
            if total_elapsed>14400:
                status="budget_exceeded"
            complete=root/"controller"/("stage_one_complete.json" if args.stage=="stage-one" else "evaluation_summary.json")
            receipt=dict(format="sun_gn_allocation_receipt_v1",stage=context.stage,job_id=context.job_id,
                         experiment_id=context.experiment_id,implementation_sha256=context.implementation_sha256,
                         protocol_sha256=PROTOCOL_SHA,protocol_addendum_sha256=ADDENDUM_SHA,contract_sha256=CONTRACT_SHA,gpu_count=1,gpu_type="l40s",
                         status=status,error_type=error_type,cleanup_errors=cleanup_errors,controller_exit_code=controller_code,
                         cleanup_signal_protection="posix_deferred" if protected else "unsupported_host_or_thread",
                         cancellation_handler="record_only",owned_group_cleanup=list(getattr(engine,"cleanup_observations",[])),
                         allocation_start_unix=context.allocation_start_unix,allocation_end_unix=ended,
                         allocation_elapsed_seconds=ended-context.allocation_start_unix,
                         prior_allocation_elapsed_seconds=prior_elapsed,total_allocation_elapsed_seconds=total_elapsed,
                         pre_controller_seconds=context.controller_start_unix-context.allocation_start_unix,
                         controller_output=str((root/"controller").resolve()),
                         controller_complete_sha256=sha(complete.read_bytes()) if complete.is_file() else None,
                         release_sha256=sha(Path(args.release).read_bytes()),runtime_sha256=sha(Path(args.runtime).read_bytes()),
                         sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA,
                         sampling_probe_sha256=runtime_document.get("sampling_probe_sha256"),
                         server_argv=server_argv,server_log_sha256=log_digest,server_log_error_type=log_error,
                         no_retry=True)
            write_exclusive(root/"allocation_receipt.json",receipt)
            write_exclusive(Path(args.ledger)/f"{context.stage}.receipt.json",receipt)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["stage-one","stage-two"])
    for name in ("code","python","vllm","arbench","tokenizer","snapshot","weights-manifest","release","runtime","ledger","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--prior",type=Path)
    parser.add_argument("--port",type=int)
    args=parser.parse_args(argv)
    if sys.platform!="linux":
        raise IntegrityError("Allocation wrapper requires the reviewed Linux/Slurm environment")
    from .controller import implementation_digest,load_stage_one
    Assets.load()
    release=json.loads(args.release.read_bytes())
    runtime=json.loads(args.runtime.read_bytes())
    digest=implementation_digest()
    sampling_probe=validate_release(release,runtime,args.stage,digest)
    validate_stage_paths(args,release)
    if args.code.resolve()!=ROOT.parents[1] or args.python.resolve()!=Path(sys.executable).resolve():
        raise IntegrityError("Code/runtime executable differs from the reviewed allocation")
    if not args.vllm.is_file() or args.vllm.absolute().parent!=args.python.absolute().parent:
        raise IntegrityError("vLLM must be in the same existing verified runtime")
    validate_environment()
    if args.port is None:
        args.port=sampling_probe["port"]
    if (args.snapshot.resolve()!=Path(sampling_probe["snapshot"]).resolve() or
            args.tokenizer.resolve()!=Path(sampling_probe["tokenizer"]).resolve() or
            server_command(args.vllm,args.snapshot,args.tokenizer,args.port)!=sampling_probe["server_argv"]):
        raise IntegrityError("Allocation launch differs from the CPU sampling resolution probe")
    context=allocation_identity(os.environ,args.stage,release,digest,time.time())
    if args.stage=="stage-two":
        if args.prior is None:
            raise IntegrityError("Second allocation requires the frozen first-stage output")
        load_stage_one(args.prior) # CPU-only full freeze/coverage check before server launch.
    claim_stage(args.ledger,context,sha(args.release.read_bytes()),sha(args.runtime.read_bytes()),args.prior,
                probe_sha=runtime["sampling_probe_sha256"])
    prior_elapsed=json.loads((args.ledger/"stage-one.receipt.json").read_bytes())["allocation_elapsed_seconds"] if args.stage=="stage-two" else 0
    def preflight():
        # CPU-only exact-file checks; no installs, downloads or model loading.
        verify_weight_manifest(args.snapshot,args.weights_manifest,runtime["weights_manifest_sha256"])
        verify_snapshot_config(args.snapshot)
        tokenizer=PinnedTokenizer(args.tokenizer)
        if sha(tokenizer.tokenizer.chat_template)!=runtime["server_chat_template_sha256"]:
            raise IntegrityError("Pinned server/tokenizer chat template mismatch")
        import importlib.metadata
        for package,key in (("vllm","vllm_version"),("torch","torch_version")):
            if importlib.metadata.version(package)!=runtime[key]:
                raise IntegrityError("Installed serving version differs from preflight receipt")
    receipt=supervise(args,context,ProcessSupervisor(),preflight=preflight,prior_elapsed=prior_elapsed,
                       verify_log=check_server_log)
    if receipt["status"]!="completed":
        raise SystemExit(1)


if __name__=="__main__":
    main()
