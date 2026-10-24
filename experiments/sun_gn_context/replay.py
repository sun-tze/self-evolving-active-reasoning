"""Recompute the main instrument from raw journals, including incomplete costs.

This is a reproducibility utility, not Claude's independent checker.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .core import Assets, IntegrityError
from .freeze import seed,sha
from .journal import read_journal
from .scoring import aggregate,score_episode


def replay(path: Path, assets: Assets):
    rows=read_journal(path)
    intents,responses={},{}
    episodes=[]
    for row in rows:
        payload=row["payload"]
        if row["event"]=="call_intent":
            call=payload["call_id"]
            if call in intents:
                raise IntegrityError("Duplicate request intent")
            intents[call]=payload
        elif row["event"]=="call_response":
            call=payload["call_id"]
            if call not in intents or call in responses:
                raise IntegrityError("Response without unique intent")
            responses[call]=payload["response"]
        elif row["event"]=="episode_complete":
            episodes.append(payload)
    used_calls=set()
    scores=[]
    for episode in episodes:
        system=None
        messages=[]
        for slot in episode["slots"]:
            call=slot["response"]["call_id"]
            if call not in responses or call in used_calls:
                raise IntegrityError("Complete episode lacks unique returned calls")
            used_calls.add(call)
            intent=intents[call]
            raw=responses[call]
            for key in ("text","finish_reason","prompt_tokens","completion_tokens","response_id"):
                if slot["response"][key]!=raw[key]:
                    raise IntegrityError("Episode differs from raw response")
            request=intent["request"]
            if system is None:
                system=request["messages"][0]
                if system["role"]!="system" or sha(system["content"])!=episode["system_sha256"]:
                    raise IntegrityError("System prompt hash mismatch")
                messages=[system]
            t=slot["slot"]
            messages.append(dict(role="user",content=assets.prompts["final_guess_prompt"] if t==26 else assets.prompts["guess_prompt"].format(turn=t)))
            expected_seed=seed(episode["case_id"],t)
            if (request["messages"]!=messages or request["seed"]!=expected_seed or slot["seed"]!=expected_seed or
                    intent["role"]!="actor" or intent["metadata"].get("case_id")!=episode["case_id"] or
                    intent["metadata"].get("condition")!=episode["condition"] or intent["metadata"].get("slot")!=t):
                raise IntegrityError("Intent transcript/seed/identity mismatch")
            messages.append(dict(role="assistant",content=slot["raw"]))
            if t<26:
                messages.append(dict(role="user",content=slot["delivered_message"]))
        scores.append(score_episode(episode,assets))
    unknown_usage=[call for call,r in responses.items() if any(type(r.get(k)) is not int or r[k]<0 for k in ("prompt_tokens","completion_tokens"))]
    known_input=sum(r["prompt_tokens"] for r in responses.values() if type(r.get("prompt_tokens")) is int and r["prompt_tokens"]>=0)
    known_output=sum(r["completion_tokens"] for r in responses.values() if type(r.get("completion_tokens")) is int and r["completion_tokens"]>=0)
    costs=dict(issued_by_role=dict(Counter(i["role"] for i in intents.values())),
               returned_responses=len(responses),unresolved_intents=sorted(set(intents)-set(responses)),
               input_tokens_all_returned=None if unknown_usage else known_input,
               output_tokens_all_returned=None if unknown_usage else known_output,
               known_input_tokens=known_input,known_output_tokens=known_output,
               unknown_usage_response_ids=unknown_usage,
               finish_reasons=dict(Counter(r.get("finish_reason") for r in responses.values())),
               complete_episodes=len(episodes),journal_sha256=sha(path.read_bytes()))
    return scores,costs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal",type=Path)
    parser.add_argument("--N",type=int,choices=[50,100])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    assets=Assets.load()
    scores,costs=replay(args.journal,assets)
    result=dict(costs=costs,scores=scores)
    if args.N:
        ids=[r["case_id"] for r in assets.manifest["rows"] if r["stage"]=="evaluation"][:args.N]
        result["aggregate"]=aggregate(ids,scores)
    with args.output.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(result,f,indent=2,sort_keys=True)
        f.write("\n")


if __name__=="__main__":
    main()
