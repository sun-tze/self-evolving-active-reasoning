"""CPU-only contract tests; no inference client, weights or scheduler calls."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from .core import Assets, IntegrityError, NotNumberError, feedback, parse_guess, run_episode, update_memory, validate_memory
from .freeze import ROOT, BENCHMARK, canonical_id, condition_order, seed, sha
from .gates import development_gate,evaluation_plan
from .journal import Journal, CallBoundary,read_journal
from .runtime import PinnedTokenizer,LoopbackTransport,validate_release
from .scoring import aggregate,score_episode,SPACE

ASSETS = Assets.load()
FIXTURES = json.loads((ROOT/"frozen/fixtures.json").read_bytes())
TOKENIZER_PATH = os.environ.get("SUN_GN_TOKENIZER")
ARBENCH_PATH = os.environ.get("SUN_GN_ARBENCH")


class TokenizerStub:
    """Injected unit-test dependency only; never used for real/model token claims."""
    def count(self,text):
        return len(text.split())
    def chat_count(self,messages):
        return 100


class MemoryJournal:
    def __init__(self):
        self.rows=[]
    def append(self,event,payload):
        self.rows.append(dict(event=event,payload=deepcopy(payload)))


class ScriptedActor:
    def __init__(self,texts):
        self.texts=list(texts)
        self.requests=[]
    def call(self,messages,seed,role,metadata):
        self.requests.append(deepcopy(dict(messages=messages,seed=seed,role=role,metadata=metadata)))
        text=self.texts[len(self.requests)-1]
        return dict(text=text,finish_reason="stop",prompt_tokens=100,completion_tokens=10,
                    response_id=f"mock-{len(self.requests)}",elapsed_seconds=.001)


def episode(texts=None,condition="E",case="mock"):
    actor=ScriptedActor(texts or (["Guess: 0123"]*25+["no answer"]))
    result=run_episode(ASSETS,actor,case_id=case,secret="0123",condition=condition,
                       memory=ASSETS.m0 if condition in ("S","E") else "",journal=MemoryJournal())
    return result,actor


class ContractTests(unittest.TestCase):
    def test_frozen_bytes(self):
        self.assertEqual(sha(ASSETS.m0),"725f0a736e392ab0a03af5fb60806a19d372afa52ed87401eacb061fce9106af")
        self.assertEqual(len(ASSETS.m0.splitlines()),5)
        self.assertTrue(ASSETS.m0.endswith("\n"))
        self.assertNotIn("\r",ASSETS.m0)
        self.assertEqual(len(SPACE),5040)

    def test_parser_shared_fixtures(self):
        for item in FIXTURES["parser"]:
            with self.subTest(raw=item["raw"]):
                if item["expected"] is None:
                    with self.assertRaises(NotNumberError):
                        parse_guess(item["raw"])
                else:
                    self.assertEqual(parse_guess(item["raw"]),item["expected"])

    def test_feedback_shared_fixtures(self):
        for item in FIXTURES["feedback"]:
            self.assertEqual(feedback(item["secret"],item["guess"]),tuple(item["expected"]))

    def test_wrapper_exact_and_common_header(self):
        b0=ASSETS.prompts["propose_template"].format(turn=25)
        self.assertEqual(ASSETS.system("S",""),b0)
        self.assertEqual(ASSETS.system("E",""),b0)
        self.assertEqual(ASSETS.system("B0","ignored"),b0)
        self.assertEqual(ASSETS.system("S",ASSETS.m0),b0+ASSETS.prompts["header"]+ASSETS.m0)
        self.assertEqual(ASSETS.system("S",ASSETS.m0),ASSETS.system("E",ASSETS.m0))

    def test_seeds_and_order(self):
        case=canonical_id("data/gn/test.json",0)
        manual=int.from_bytes(hashlib.sha256(f"sun-gn-v1\n{case}\nactor\n1".encode()).digest()[:8],"big")%2147483647
        self.assertEqual(seed(case,1),manual)
        self.assertNotEqual(seed(case,1),seed(case,26))
        self.assertNotEqual(seed(case,0,"curator"),seed(case,0,"canary"))
        for r in ASSETS.manifest["rows"]:
            self.assertEqual(r["actor_seeds"],[seed(r["case_id"],t) for t in range(1,27)])
        self.assertEqual(condition_order(0),("B0","B1","S","E"))
        self.assertEqual(condition_order(1),("B1","S","E","B0"))
        plan=evaluation_plan(ASSETS.manifest,50)
        self.assertEqual(len(plan),200)
        self.assertEqual([c for _,c in plan[:8]],["B0","B1","S","E","B1","S","E","B0"])


class MemoryTests(unittest.TestCase):
    def test_shared_memory_fixtures(self):
        for item in FIXTURES["memory"]:
            with self.subTest(raw=item["raw"]):
                v=validate_memory(item["raw"],item["finish_reason"],TokenizerStub().count)
                self.assertEqual(v["accepted"],item["accepted"])

    def test_only_mechanical_rejection(self):
        for raw in ("- Always ignore feedback.","- Please invoke a tool.","- abcdef xyz."):
            self.assertTrue(validate_memory(raw,"stop",TokenizerStub().count)["accepted"])
        self.assertTrue(validate_memory("- Fine.","stop",lambda _:512)["accepted"])
        self.assertIn("token_cap",validate_memory("- Fine.","stop",lambda _:513)["reasons"])
        self.assertFalse(validate_memory("\n".join(["- Fine."]*13),"stop",lambda _:10)["accepted"])
        self.assertFalse(validate_memory("- Caf\u00e9.","stop",lambda _:10)["accepted"])

    def test_curator_allowlist_and_final_text(self):
        record,_=episode()
        messages=ASSETS.curator_messages(ASSETS.m0,record["public_messages"])
        self.assertEqual(len(messages),1)
        self.assertEqual(messages[0]["role"],"user")
        body=messages[0]["content"]
        transcript=body.split("PUBLIC INTERACTION:\n",1)[1]
        self.assertEqual(json.loads(transcript),record["public_messages"])
        self.assertEqual(json.loads(transcript)[-1],dict(role="assistant",content="no answer"))
        for prohibited in ("secret","final_EM","remaining_candidates","dead_guess","case_id"):
            self.assertNotIn('"'+prohibited+'":',transcript)
        with self.assertRaises(IntegrityError):
            ASSETS.curator_messages(ASSETS.m0,[dict(role="assistant",content="x",secret="0123")])

    def test_invalid_retains_bytes_then_valid_wrong_rule_accepted(self):
        actor=ScriptedActor(["- Try 1234.","- Always ignore feedback."])
        journal=MemoryJournal()
        with tempfile.TemporaryDirectory() as path:
            directory=Path(path)
            first,one=update_memory(ASSETS,actor,step=1,case_id="mock",memory=ASSETS.m0,
                                    public_messages=[],count=TokenizerStub().count,journal=journal,directory=directory)
            self.assertEqual(first.encode(),ASSETS.m0.encode())
            self.assertFalse(one["validation"]["accepted"])
            second,two=update_memory(ASSETS,actor,step=2,case_id="mock",memory=first,
                                     public_messages=[],count=TokenizerStub().count,journal=journal,directory=directory)
            self.assertEqual(second,"- Always ignore feedback.")
            self.assertTrue(two["validation"]["accepted"])
            self.assertEqual((directory/"M1.txt").read_bytes(),ASSETS.m0.encode())
            self.assertEqual((directory/"M2.txt").read_bytes(),second.encode())
            self.assertEqual(len(actor.requests),2)


class EpisodeScoringTests(unittest.TestCase):
    def test_shared_trajectories(self):
        texts=[(["malformed"]*25+["Guess: 0123"]),(["Guess: 0123"]*25+["malformed"]),
               (["malformed"]+["Guess: 0123"]*24+["Guess: 4567"])]
        for case,script in zip(FIXTURES["trajectory"],texts):
            record,actor=episode(script)
            scores=score_episode(record,ASSETS)
            with self.subTest(case=case["name"]):
                self.assertEqual(len(actor.requests),26)
                self.assertAlmostEqual(scores["Q"],case["expected_Q"])
                self.assertEqual(scores["final_EM"],case["expected_EM"])
                self.assertEqual(scores["hit_any"],case["expected_hit_any"])

    def test_repeated_digits_are_informative(self):
        record,_=episode(["Guess: 0000"]+["malformed"]*24+["Guess: 0123"])
        scores=score_episode(record,ASSETS)
        self.assertEqual(scores["sizes"],[2016]*25)
        self.assertTrue(scores["diagnostics"][0]["infeasible"])
        self.assertGreater(scores["diagnostics"][0]["expected_entropy"],0)
        self.assertGreater(scores["Q"],0)

    def test_inconsistent_or_incomplete_logs_stop(self):
        record,_=episode()
        record["slots"][1]["delivered_feedback"]=[0,0]
        record["slots"][1]["delivered_message"]=ASSETS.prompts["eval_prompt"].format(same_pos=0,diff_pos=0)
        with self.assertRaises(IntegrityError):
            score_episode(record,ASSETS)
        record,_=episode()
        record["slots"].pop()
        with self.assertRaises(IntegrityError):
            score_episode(record,ASSETS)

    def test_final_never_updates_H(self):
        record,_=episode(["malformed"]*25+["Guess: 0123"])
        score=score_episode(record,ASSETS)
        self.assertEqual(score["remaining_candidates"],5040)
        self.assertEqual(score["Q"],0)
        self.assertTrue(score["final_EM"])

    def test_missing_all_four_common_set_and_bounds(self):
        record,_=episode()
        base=score_episode(record,ASSETS)
        scores=[dict(base,case_id="a",condition=c,Q=.5+(.25 if c=="E" else 0)) for c in ("B0","B1","S","E")]
        scores += [dict(base,case_id="b",condition=c,Q=.5) for c in ("S","E")]
        result=aggregate(["a","b"],scores)
        self.assertEqual(result["paired_n"],1)
        self.assertEqual(result["full_N_bounds"],[-.375,.625])
        self.assertTrue(result["descriptive_only"])
        self.assertFalse(result["positive_primary_supported"])
        self.assertEqual(aggregate(["a"],[])["full_N_bounds"],[-1,1])
        with self.assertRaises(IntegrityError):
            aggregate(["a"],scores[:4]+[scores[0]])


class BoundaryTests(unittest.TestCase):
    def boundary(self,path,transport,tokenizer=None,**kwargs):
        journal=Journal(Path(path)/"journal.jsonl",dict(mode="offline_mock"))
        params=dict(actor_limit=26,curator_limit=1,input_limit=100000,output_limit=100000,
                    deadline=time.monotonic()+3600,reserve_seconds=120)
        params.update(kwargs)
        boundary=CallBoundary(transport,tokenizer or TokenizerStub(),journal,**params)
        return boundary,journal

    def test_durable_intent_response_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as path:
            def transport(request):
                rows=read_journal(Path(path)/"journal.jsonl")
                self.assertEqual(rows[-1]["event"],"call_intent")
                self.assertEqual(set(request),{"model","messages","temperature","top_p","seed","max_tokens"})
                return dict(text="Guess: 0000",finish_reason="stop",prompt_tokens=100,completion_tokens=5,response_id="mock")
            boundary,journal=self.boundary(path,transport)
            response=boundary.call([dict(role="user",content="public")],42,"actor",{})
            self.assertEqual(response["completion_tokens"],5)
            self.assertEqual(read_journal(Path(path)/"journal.jsonl")[-1]["event"],"call_response")
            journal.close()
            p=Path(path)/"journal.jsonl"
            p.write_text(p.read_text().replace("public","tampered"),encoding="utf-8")
            with self.assertRaises(IntegrityError):
                read_journal(p)

    def test_uncertain_call_never_resent(self):
        calls=[]
        with tempfile.TemporaryDirectory() as path:
            def transport(request):
                calls.append(request)
                raise TimeoutError("private connection string must not be logged")
            boundary,journal=self.boundary(path,transport)
            with self.assertRaises(TimeoutError):
                boundary.call([],1,"actor",{})
            with self.assertRaises(IntegrityError):
                boundary.call([],1,"actor",{})
            journal.close()
            self.assertEqual(len(calls),1)
            raw=(Path(path)/"journal.jsonl").read_text()
            self.assertNotIn("private connection",raw)
            self.assertEqual(read_journal(Path(path)/"journal.jsonl")[-1]["event"],"call_uncertain")

    def test_context_and_caps_stop_before_transport(self):
        for kwargs,chat_count in [(dict(actor_limit=0),100),(dict(output_limit=127),100),
                                  (dict(input_limit=99),100),(dict(deadline=time.monotonic()),100),({},8065)]:
            with tempfile.TemporaryDirectory() as path:
                calls=[]
                tokenizer=TokenizerStub()
                tokenizer.chat_count=lambda _:chat_count
                boundary,journal=self.boundary(path,lambda x:calls.append(x),tokenizer,**kwargs)
                with self.assertRaises(IntegrityError):
                    boundary.call([],1,"actor",{})
                self.assertEqual(calls,[])
                journal.close()

    def test_mismatched_service_usage_halts(self):
        with tempfile.TemporaryDirectory() as path:
            boundary,journal=self.boundary(path,lambda _:dict(text="x",finish_reason="stop",prompt_tokens=101,completion_tokens=1,response_id="mock"))
            with self.assertRaises(IntegrityError):
                boundary.call([],1,"actor",{})
            self.assertTrue(boundary.halted)
            journal.close()

    def test_external_api_and_missing_release_refused(self):
        for url in ("https://api.openai.com/v1","http://example.com/v1","http://user:secret@localhost/v1"):
            with self.assertRaises(IntegrityError):
                LoopbackTransport(url)
        with self.assertRaises(IntegrityError):
            validate_release({}, {}, "stage-one", "digest")

    def test_raw_journal_replay_reconstructs_requests_and_costs(self):
        from .replay import replay
        script=["malformed","Guess: 0000"]+["Guess: 0123"]*23+["malformed"]
        with tempfile.TemporaryDirectory() as path:
            responses=[]
            def transport(request):
                text=script[len(responses)]
                responses.append(request)
                return dict(text=text,finish_reason="stop",prompt_tokens=100,completion_tokens=10,response_id=f"mock-{len(responses)}")
            boundary,journal=self.boundary(path,transport)
            record=run_episode(ASSETS,boundary,case_id="synthetic-fixture",secret="0123",condition="E",memory=ASSETS.m0,journal=journal)
            journal.close()
            scored,costs=replay(Path(path)/"journal.jsonl",ASSETS)
            self.assertEqual(scored,[score_episode(record,ASSETS)])
            self.assertEqual(costs["issued_by_role"],dict(actor=26))
            self.assertEqual(costs["input_tokens_all_returned"],2600)
            self.assertEqual(costs["output_tokens_all_returned"],260)
            self.assertEqual(costs["unresolved_intents"],[])

    def test_missing_usage_is_unknown_not_zero(self):
        from .replay import replay
        with tempfile.TemporaryDirectory() as path:
            journal=Journal(Path(path)/"journal.jsonl",dict(mode="synthetic_fixture"))
            journal.append("call_intent",dict(call_id="actor-1",role="actor"))
            journal.append("call_response",dict(call_id="actor-1",response=dict(text="invalid service response")))
            journal.close()
            scores,costs=replay(Path(path)/"journal.jsonl",ASSETS)
            self.assertEqual(scores,[])
            self.assertIsNone(costs["input_tokens_all_returned"])
            self.assertIsNone(costs["output_tokens_all_returned"])
            self.assertEqual(costs["unknown_usage_response_ids"],["actor-1"])


class GateTests(unittest.TestCase):
    def inputs(self,seconds=10):
        episodes,scores=[],[]
        for row in ASSETS.manifest["rows"]:
            if row["stage"]=="development":
                for c in row["conditions"]:
                    episodes.append(dict(case_id=row["case_id"],condition=c,completed=True,slots=[None]*26,elapsed_seconds=seconds))
                    scores.append(dict(case_id=row["case_id"],condition=c,invalid_outputs=0,truncated_outputs=0,Q=.5,final_EM=False))
        canaries=[dict(validation=dict(accepted=True),elapsed_seconds=1)]*2
        return episodes,scores,canaries

    def gate(self,inputs):
        return development_gate(ASSETS,*inputs,dev_elapsed_seconds=500,startup_save_seconds=10)

    def test_full_half_stop_and_stage_one_guard(self):
        full=self.gate(self.inputs(10))
        self.assertTrue(full["passed"])
        self.assertEqual(full["selected_N"],100)
        half=self.gate(self.inputs(60))
        self.assertTrue(half["passed"])
        self.assertEqual(half["selected_N"],50)
        stop=self.gate(self.inputs(110))
        self.assertIn("stage_two_over_90_minutes",stop["reasons"])
        inputs=self.inputs(10)
        inputs[2][0]["elapsed_seconds"]=200
        stop=self.gate(inputs)
        self.assertIn("stage_one_over_120_minutes",stop["reasons"])

    def test_exact_thresholds_and_floor_does_not_switch_metric(self):
        inputs=self.inputs()
        inputs[1][0]["invalid_outputs"]=62
        inputs[1][0]["truncated_outputs"]=31
        self.assertTrue(self.gate(inputs)["passed"])
        inputs[1][0]["invalid_outputs"]=63
        self.assertIn("parse_failure_rate",self.gate(inputs)["reasons"])
        inputs=self.inputs()
        inputs[1][0]["truncated_outputs"]=32
        self.assertIn("truncation_rate",self.gate(inputs)["reasons"])
        self.assertTrue(self.gate(self.inputs())["floor_bound"])
        inputs=self.inputs()
        for row in inputs[1]:
            if row["condition"]=="B0":
                row["Q"]=0
        self.assertIn("insufficient_Q_range",self.gate(inputs)["reasons"])


@unittest.skipUnless(ARBENCH_PATH,"SUN_GN_ARBENCH not supplied")
class PinnedSourceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(ARBENCH_PATH)
        spec=importlib.util.spec_from_file_location("official_gn_utils",root/"arbench/utils/utils_gn.py")
        cls.official=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.official)

    def test_official_feedback_parity_with_repeated_digits(self):
        rng=random.Random(62026)
        pairs=[(s,g) for s in SPACE[:100] for g in ("0000","1111","0011","0123","9876")]
        pairs += [(rng.choice(SPACE),"".join(rng.choice("0123456789") for _ in range(4))) for _ in range(1000)]
        for s,g in pairs:
            self.assertEqual(feedback(s,g),self.official.compare_guess(s,g)[:2])

    def test_parser_parity_except_approved_missing_match_patch(self):
        for fixture in FIXTURES["parser"]:
            official=self.official.extract_and_convert_guess(fixture["raw"])
            if official:
                self.assertEqual(parse_guess(fixture["raw"]),"".join(map(str,official)))
            else:
                with self.assertRaises(NotNumberError):
                    parse_guess(fixture["raw"])

    def test_traditional_loop_request_parity_with_missing_match_patch(self):
        source=(Path(ARBENCH_PATH)/"arbench/reasoner/gn/gn_evaluator.py").read_text(encoding="utf-8")
        function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="_run_traditional_evaluation")
        scripts=["malformed","Guess: 0000","Guess: 0123"]+["Guess: 4567"]*22+["Guess: 0123"]
        # Match the official function in isolation; no OpenAI import or network.
        requests=[]
        def inference(messages,**kwargs):
            requests.append(deepcopy(messages))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=scripts[len(requests)-1]))],
                                   usage=SimpleNamespace(prompt_tokens=100,completion_tokens=10))
        def extract(raw):
            return list(map(int,parse_guess(raw)))
        scope=dict(List=list,tqdm=lambda x:x,json=json,inference=inference,
                   METHOD_DICT={"zero_shot":ASSETS.prompts["propose_template"]},
                   extract_and_convert_guess=extract,NotNumberError=NotNumberError,
                   compare_guess=self.official.compare_guess,POLICY_API_KEY=None,POLICY_BASE_URL=None,
                   **{k:ASSETS.prompts[k] for k in ["guess_prompt","eval_prompt","refine_prompt","final_guess_prompt"]})
        exec(compile(ast.Module(body=[function],type_ignores=[]),"pinned_official_traditional_loop","exec"),scope)
        with tempfile.TemporaryDirectory() as path:
            logs=[]
            scope["_run_traditional_evaluation"](["0123"],logs,str(Path(path)/"official.json"),"mock","zero_shot",25,.7,.7)
        record,actor=episode(scripts,condition="B0")
        self.assertEqual(requests,[r["messages"] for r in actor.requests])
        self.assertEqual(len(requests),26)
        self.assertEqual(score_episode(record,ASSETS)["final_EM"],logs[0]["correctness"])


@unittest.skipUnless(TOKENIZER_PATH,"SUN_GN_TOKENIZER not supplied")
class PinnedTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=PinnedTokenizer(Path(TOKENIZER_PATH))

    def test_real_tokenizer_bounds_and_M0_count(self):
        self.assertEqual(self.tokenizer.count(ASSETS.m0),83)
        for target in (512,513):
            # Repetition is used only to make an exact boundary fixture offline.
            raw="- "+"a "*(target-3)+"a"
            while self.tokenizer.count(raw)<target:
                raw+=" a"
            while self.tokenizer.count(raw)>target:
                raw=raw[:-2]
            self.assertEqual(self.tokenizer.count(raw),target)
            self.assertEqual(validate_memory(raw,"stop",self.tokenizer.count)["accepted"],target==512)
        self.assertLessEqual(self.tokenizer.chat_count([dict(role="system",content=ASSETS.system("B1"))]),8064)

    def test_shared_memory_with_real_tokenizer(self):
        for fixture in FIXTURES["memory"]:
            self.assertEqual(validate_memory(fixture["raw"],fixture["finish_reason"],self.tokenizer.count)["accepted"],fixture["accepted"])


class TwoStageMockTests(unittest.TestCase):
    def test_complete_half_budget_twenty_updates_and_frozen_transfer(self):
        from . import controller
        class CompactJournal:
            def __init__(self):
                self.events=__import__("collections").Counter()
                self.admitted=[]
            def append(self,event,payload):
                self.events[event]+=1
                if event=="episode_admitted":
                    self.admitted.append((payload["case_id"],payload["condition"]))
        tokenizer=TokenizerStub()
        def transport(request):
            curator=request["max_tokens"]==768
            return dict(text="- Preserve public feedback." if curator else "Guess: 0000",finish_reason="stop",
                        prompt_tokens=100,completion_tokens=10,response_id="mock-local")
        def half_gate(assets,episodes,scores,canaries,**kwargs):
            timed=deepcopy(episodes)
            for e in timed:
                e["elapsed_seconds"]=60
            return development_gate(assets,timed,scores,canaries,**kwargs)
        with tempfile.TemporaryDirectory() as path:
            first=Path(path)/"first"
            second=Path(path)/"second"
            first.mkdir()
            second.mkdir()
            j1=CompactJournal()
            b1=CallBoundary(transport,tokenizer,j1,actor_limit=1144,curator_limit=22,input_limit=94000000,
                            output_limit=1494528,deadline=time.monotonic()+7200,reserve_seconds=120)
            with patch.object(controller,"secret_for",return_value="0123"),patch.object(controller,"development_gate",side_effect=half_gate):
                prior=controller.stage_one(ASSETS,b1,tokenizer,j1,Path("unused"),first,10)
            self.assertEqual(prior["N"],50)
            self.assertEqual(prior["counts"],dict(actor=1144,curator=22))
            self.assertEqual(len(list((first/"memories").glob("M*.txt"))),21)
            self.assertEqual((first/"memories/M20.txt").read_text(),"- Preserve public feedback.")
            m20_before=(first/"memories/M20.txt").read_bytes()
            j2=CompactJournal()
            b2=CallBoundary(transport,tokenizer,j2,actor_limit=5200,curator_limit=0,input_limit=52000000-b1.actual_input,
                            output_limit=828928-b1.reserved_output,deadline=time.monotonic()+7200,reserve_seconds=120)
            with patch.object(controller,"secret_for",return_value="0123"):
                summary=controller.stage_two(ASSETS,b2,j2,Path("unused"),second,first)
            self.assertEqual(summary["paired_n"],50)
            self.assertEqual(summary["planned_N"],50)
            self.assertEqual(summary["delta_Q"],0)
            self.assertEqual(summary["counts"],dict(actor=5200,curator=0))
            self.assertEqual(sum(b1.counts.values())+sum(b2.counts.values()),6366)
            self.assertEqual(b1.reserved_output+b2.reserved_output,828928)
            self.assertEqual(j1.events["memory_update"],20)
            self.assertEqual(j2.events["episode_complete"],200)
            self.assertEqual(j2.admitted,[(r["case_id"],c) for r,c in evaluation_plan(ASSETS.manifest,50)])
            self.assertEqual((first/"memories/M20.txt").read_bytes(),m20_before)


if __name__=="__main__":
    unittest.main(verbosity=2)
