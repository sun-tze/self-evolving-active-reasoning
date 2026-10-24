"""Single two-stage controller. Live execution requires an independent release.

There is deliberately no scheduler submission, installation, retry, resume,
external API, model switch, or memory-selection command here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
from pathlib import Path
import time

from .core import Assets, CONTRACT_SHA, PROTOCOL_SHA, ADDENDUM_SHA, IntegrityError, run_episode, update_memory, validate_memory
from .freeze import ROOT, MODEL_REVISION, git_bytes, seed, sha
from .gates import development_gate, evaluation_plan
from .journal import Journal, CallBoundary
from .runtime import PinnedTokenizer, LoopbackTransport, validate_release
from .scoring import aggregate, score_episode
from .allocation_budget import AllocationContext,stage_two_time_check
from .sampling import A2_SHA,POLICY_SHA


def implementation_digest():
    hashes = {p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in sorted(ROOT.rglob("*"))
              if p.is_file() and "__pycache__" not in p.parts and (p.name==".gitattributes" or p.suffix in (".py",".md",".json",".jsonl",".txt",".sbatch"))}
    return sha(json.dumps(hashes,sort_keys=True,separators=(",", ":")))


def save_new(path: Path, data):
    with path.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(data,f,sort_keys=True,ensure_ascii=True,indent=2)
        f.write("\n")
        f.flush()
        __import__("os").fsync(f.fileno())


def secret_for(arbench: Path, assets: Assets, row: dict) -> str:
    raw = git_bytes(arbench,row["file"])
    if sha(raw)!=row["file_sha256"]:
        raise IntegrityError("Pinned benchmark data file hash")
    secret = json.loads(raw)[row["row"]]
    if sha(secret)!=row["content_sha256"]:
        raise IntegrityError("Pinned benchmark row content hash")
    return secret


def development(assets, boundary, tokenizer, journal, arbench, directory):
    started = time.monotonic()
    episodes, scores = [], []
    for row in assets.manifest["rows"]:
        if row["stage"]!="development":
            continue
        for condition in row["conditions"]:
            result = run_episode(assets,boundary,case_id=row["case_id"],secret=secret_for(arbench,assets,row),
                                 condition=condition,memory=assets.m0 if condition=="S" else "",journal=journal)
            episodes.append(result)
            scores.append(score_episode(result,assets))
    canaries = []
    for episode in [e for e in episodes if e["condition"]=="B0"][:2]:
        started_canary = time.monotonic()
        response = boundary.call(assets.curator_messages(assets.m0,episode["public_messages"]),
                                 seed(episode["case_id"],0,"canary"),"curator",dict(case_id=episode["case_id"],purpose="canary"))
        record = dict(validation=validate_memory(response["text"],response["finish_reason"],tokenizer.count),
                      response=response,elapsed_seconds=time.monotonic()-started_canary,discarded=True)
        journal.append("canary_discarded",record)
        canaries.append(record)
    return episodes,scores,canaries,time.monotonic()-started


def stage_one(assets,boundary,tokenizer,journal,arbench,directory,startup_save_seconds,allocation_context=None):
    episodes,scores,canaries,elapsed = development(assets,boundary,tokenizer,journal,arbench,directory)
    save_started=time.monotonic()
    save_new(directory/"development.json",dict(episodes=episodes,scores=scores,canaries=canaries))
    measured_save=time.monotonic()-save_started
    # The wrapper's allocation clock starts before CPU preflight/server startup.
    # Include actual development saving and reserved final saving/shutdown too.
    reserves=(allocation_context.save_reserve_seconds+allocation_context.shutdown_reserve_seconds) if allocation_context else 0
    overhead=startup_save_seconds+measured_save+reserves
    gate = development_gate(assets,episodes,scores,canaries,dev_elapsed_seconds=elapsed,
                            startup_save_seconds=overhead)
    gate["overhead_evidence"]=dict(measured_pre_development_seconds=startup_save_seconds,
                                  measured_development_save_seconds=measured_save,
                                  final_save_shutdown_reserved_seconds=reserves,
                                  startup_save_seconds_used=overhead)
    save_new(directory/"gate.json",gate)
    journal.append("gate_frozen",gate)
    if not gate["passed"]:
        raise IntegrityError("Development gate stopped formal experiment")
    memories = directory/"memories"
    memories.mkdir()
    (memories/"M0.txt").write_bytes(assets.m0.encode("utf-8"))
    memory = assets.m0
    # Count maximum observed latency for each future call; never send a request
    # that cannot finish within the current conservative remaining time.
    boundary.reserve_seconds = max(120,gate["actor_conservative_seconds"],gate["curator_conservative_seconds"])
    for row in assets.manifest["rows"]:
        if row["stage"]!="adaptation":
            continue
        episode = run_episode(assets,boundary,case_id=row["case_id"],secret=secret_for(arbench,assets,row),
                              condition="E",memory=memory,journal=journal)
        memory,record = update_memory(assets,boundary,step=row["stage_order"]+1,case_id=row["case_id"],
                                     memory=memory,public_messages=episode["public_messages"],
                                     count=tokenizer.count,journal=journal,directory=memories)
    result = dict(completed=True,N=gate["selected_N"],gate_sha256=sha((directory/"gate.json").read_bytes()),
                  M20_sha256=sha(memory),M0_sha256=sha(assets.m0),counts=boundary.counts,
                  actual_input_tokens=boundary.actual_input,actual_output_tokens=boundary.actual_output,
                  reserved_output_tokens=boundary.reserved_output,
                  protocol_sha256=PROTOCOL_SHA,protocol_addendum_sha256=ADDENDUM_SHA,contract_sha256=CONTRACT_SHA,
                  implementation_sha256=implementation_digest(),model_revision=MODEL_REVISION,
                  sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA,
                  allocation_context_sha256=sha(json.dumps(allocation_context.document(),sort_keys=True,separators=(",", ":"))) if allocation_context else None)
    save_new(directory/"stage_one_complete.json",result)
    journal.append("stage_complete",result)
    return result


def load_stage_one(directory: Path):
    complete = json.loads((directory/"stage_one_complete.json").read_bytes())
    gate = json.loads((directory/"gate.json").read_bytes())
    m20 = (directory/"memories/M20.txt").read_bytes()
    if (complete.get("completed") is not True or complete["N"] not in (50,100) or
            complete["protocol_sha256"]!=PROTOCOL_SHA or complete["protocol_addendum_sha256"]!=ADDENDUM_SHA or complete["contract_sha256"]!=CONTRACT_SHA or
            complete["implementation_sha256"]!=implementation_digest() or complete["model_revision"]!=MODEL_REVISION or
            complete.get("sampling_addendum_sha256")!=A2_SHA or complete.get("sampling_policy_sha256")!=POLICY_SHA or
            complete["gate_sha256"]!=sha((directory/"gate.json").read_bytes()) or
            complete["M20_sha256"]!=sha(m20) or gate["selected_N"]!=complete["N"] or not gate["passed"] or
            complete["counts"]!={"actor":1144,"curator":22}):
        raise IntegrityError("Stage one freeze/coverage mismatch")
    return complete,gate,m20.decode("utf-8")


def stage_two(assets,boundary,journal,arbench,directory,prior_directory):
    prior,gate,m20 = load_stage_one(prior_directory)
    plan = evaluation_plan(assets.manifest,prior["N"])
    scores, failures = [], []
    # Admit in exact FIFO plan order; completion order may differ. At most eight
    # episodes remain in flight. Once a fault is seen, admit no further episodes.
    with ThreadPoolExecutor(max_workers=8) as pool:
        active = {}
        iterator = iter(plan)
        stopped = False
        def admit():
            try:
                row,condition = next(iterator)
            except StopIteration:
                return False
            secret = secret_for(arbench,assets,row)
            journal.append("episode_admitted",dict(case_id=row["case_id"],condition=condition))
            memory = assets.m0 if condition=="S" else m20 if condition=="E" else ""
            future = pool.submit(run_episode,assets,boundary,case_id=row["case_id"],secret=secret,
                                 condition=condition,memory=memory,journal=journal)
            active[future] = (row["case_id"],condition)
            return True
        for _ in range(8):
            if not admit():
                break
        while active:
            finished,_ = wait(active,return_when=FIRST_COMPLETED)
            for future in finished:
                key = active.pop(future)
                try:
                    result = future.result()
                    scores.append(score_episode(result,assets))
                except Exception as error:
                    stopped = True
                    boundary.halted = True
                    failure = dict(case_id=key[0],condition=key[1],error_type=type(error).__name__)
                    journal.append("episode_incomplete",failure)
                    failures.append(failure)
            if not stopped:
                for _ in finished:
                    if not admit():
                        break
    ids = list(dict.fromkeys(row["case_id"] for row,_ in plan))
    summary = aggregate(ids,scores)
    summary.update(infrastructure_failures=failures,completed=not failures,
                   sampling_addendum_sha256=A2_SHA,sampling_policy_sha256=POLICY_SHA,
                   counts=boundary.counts,actual_input_tokens=boundary.actual_input,
                   actual_output_tokens=boundary.actual_output,M20_sha256=sha(m20))
    save_new(directory/"evaluation_summary.json",summary)
    journal.append("stage_complete" if not failures else "stage_incomplete",summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["stage-one","stage-two"])
    for name in ("arbench","tokenizer","release","runtime","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--prior",type=Path)
    parser.add_argument("--allocation-context",type=Path,required=True)
    parser.add_argument("--base-url",required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    assets = Assets.load()
    implementation_sha = implementation_digest()
    release = json.loads(args.release.read_bytes())
    runtime = json.loads(args.runtime.read_bytes())
    sampling_probe=validate_release(release,runtime,args.stage,implementation_sha)
    if (Path(sampling_probe["tokenizer"]).resolve()!=args.tokenizer.resolve() or
            args.base_url!=f'http://127.0.0.1:{sampling_probe["port"]}/v1'):
        raise IntegrityError("Controller tokenizer/endpoint differs from CPU sampling probe")
    allocation=AllocationContext.load(json.loads(args.allocation_context.read_bytes()),args.stage,implementation_sha,
                                      release.get("experiment_id"),__import__("os").environ.get("SLURM_JOB_ID"))
    tokenizer = PinnedTokenizer(args.tokenizer)
    if sha(tokenizer.tokenizer.chat_template)!=runtime["server_chat_template_sha256"]:
        raise IntegrityError("Server chat template differs from pinned tokenizer")
    transport = LoopbackTransport(args.base_url)
    prior = None
    timing = None
    if args.stage=="stage-two":
        if args.prior is None:
            raise IntegrityError("Stage two requires the completed first-stage freeze")
        prior,gate,_ = load_stage_one(args.prior)
        timing=stage_two_time_check(gate,prior["N"],allocation)
    # No directories or requests are created before release/preflight/timing verification.
    args.output.mkdir(exist_ok=False,parents=True)
    journal = Journal(args.output/"journal.jsonl",dict(stage=args.stage,protocol_sha256=PROTOCOL_SHA,
                    contract_sha256=CONTRACT_SHA,implementation_sha256=implementation_sha,
                    runtime=runtime,tokenizer=tokenizer.metadata,allocation_context=allocation.document()))
    if timing:
        journal.append("stage_two_startup_recheck",timing)
    N = prior["N"] if prior else 100
    input_cap = (94000000 if N==100 else 52000000)-(prior["actual_input_tokens"] if prior else 0)
    output_cap = (1494528 if N==100 else 828928)-(prior["reserved_output_tokens"] if prior else 0)
    boundary = CallBoundary(transport,tokenizer,journal,actor_limit=4*N*26 if prior else 1144,
                            curator_limit=0 if prior else 22,input_limit=input_cap,output_limit=output_cap,
                            deadline=allocation.request_monotonic_deadline(),reserve_seconds=120)
    try:
        if prior:
            boundary.reserve_seconds=max(120,gate["actor_conservative_seconds"])
            stage_two(assets,boundary,journal,args.arbench,args.output,args.prior)
        else:
            stage_one(assets,boundary,tokenizer,journal,args.arbench,args.output,allocation.startup_elapsed(),allocation)
    except BaseException as error:
        boundary.halted=True
        journal.append("stage_failed",dict(error_type=type(error).__name__,counts=boundary.counts,
                                          actual_input_tokens=boundary.actual_input,actual_output_tokens=boundary.actual_output))
        raise
    finally:
        journal.close()


if __name__=="__main__":
    main()
