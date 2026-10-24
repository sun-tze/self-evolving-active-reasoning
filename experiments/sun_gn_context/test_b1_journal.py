"""B1 regression: genuine fsynced disk journals across both mock stages."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from . import controller
from .core import Assets,IntegrityError
from .freeze import ROOT,sha
from .gates import development_gate,evaluation_plan
from .journal import Journal,CallBoundary,read_journal,normalize_json_wire
from .replay import replay
from .scoring import aggregate
from .test_offline import TokenizerStub

ASSETS=Assets.load()


class B1JournalTests(unittest.TestCase):
    def test_numeric_nested_keys_roundtrip_before_hashing(self):
        payload={"stage_two_estimated_seconds":{100:605.8,50:602.9},"nested":[{2:"two",10:"ten"}]}
        original=deepcopy(payload)
        with tempfile.TemporaryDirectory() as path:
            file=Path(path)/"journal.jsonl"
            journal=Journal(file,dict(mode="synthetic_B1_regression"))
            journal.append("gate_frozen",payload)
            journal.close()
            rows=read_journal(file)
            self.assertEqual(rows[1]["payload"],normalize_json_wire(original))
            self.assertEqual(set(rows[1]["payload"]["stage_two_estimated_seconds"]),{"100","50"})
            self.assertEqual(payload,original) # caller's Python object is not mutated.
            decoded=json.loads(file.read_text().splitlines()[1])
            digest=decoded.pop("sha256")
            self.assertEqual(sha(json.dumps(decoded,sort_keys=True,ensure_ascii=True,separators=(",", ":"))),digest)

    def test_key_alias_collisions_and_nonfinite_values_fail_before_write(self):
        for payload in ({1:"numeric","1":"string"},{"nested":{True:"boolean","true":"string"}},dict(x=float("nan"))):
            with tempfile.TemporaryDirectory() as path:
                file=Path(path)/"journal.jsonl"
                journal=Journal(file,dict(mode="synthetic_B1_collision"))
                before=file.read_bytes()
                with self.assertRaises((IntegrityError,ValueError)):
                    journal.append("ambiguous",payload)
                journal.close()
                self.assertEqual(file.read_bytes(),before)
                self.assertEqual(len(read_journal(file)),1)

    def test_preserved_unchanged_v3_failure_is_still_rejected(self):
        original=ROOT/"b1_v3_numeric_gate.jsonl"
        self.assertEqual(sha(original.read_bytes()),"0edfd8a37cd85e03ac1c4d7ef3d0d10664f0f34d543d46da32ee1048f0b7c033")
        with self.assertRaisesRegex(IntegrityError,"Journal integrity failure"):
            read_journal(original)

    def test_real_disk_stage_one_gate_replay_and_stage_two_prior_consumption(self):
        tokenizer=TokenizerStub()
        def transport(request):
            return dict(text="- Preserve public feedback." if request["max_tokens"]==768 else "Guess: 0000",
                        finish_reason="stop",prompt_tokens=100,completion_tokens=10,response_id="synthetic-local")
        def half_gate(assets,episodes,scores,canaries,**kwargs):
            timed=deepcopy(episodes)
            for episode in timed:
                episode["elapsed_seconds"]=60 # deterministic synthetic latency fixture.
            return development_gate(assets,timed,scores,canaries,**kwargs)
        with tempfile.TemporaryDirectory() as path:
            root=Path(path)
            first=root/"first"
            second=root/"second"
            first.mkdir()
            second.mkdir()
            j1=Journal(first/"journal.jsonl",dict(mode="synthetic_B1_full_stage_one",actual_model_calls=0))
            b1=CallBoundary(transport,tokenizer,j1,actor_limit=1144,curator_limit=22,input_limit=94000000,
                            output_limit=1494528,deadline=time.monotonic()+7200,reserve_seconds=120)
            with patch.object(controller,"secret_for",return_value="0123"),patch.object(controller,"development_gate",side_effect=half_gate):
                completed=controller.stage_one(ASSETS,b1,tokenizer,j1,Path("unused"),first,10)
            j1.close()
            rows=read_journal(first/"journal.jsonl")
            gate=next(row["payload"] for row in rows if row["event"]=="gate_frozen")
            self.assertEqual(set(gate["stage_two_estimated_seconds"]),{"100","50"})
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["selected_N"],50)
            scores1,costs1=replay(first/"journal.jsonl",ASSETS)
            self.assertEqual(len(scores1),44)
            self.assertEqual(costs1["issued_by_role"],dict(actor=1144,curator=22))
            self.assertEqual(costs1["returned_responses"],1166)
            self.assertEqual(costs1["unresolved_intents"],[])
            # Actual stage-two prior loader consumes the on-disk freeze and gate.
            prior,loaded_gate,m20=controller.load_stage_one(first)
            self.assertEqual(prior,completed)
            self.assertEqual(loaded_gate,gate)
            m20_hash=sha(m20)
            j2=Journal(second/"journal.jsonl",dict(mode="synthetic_B1_full_stage_two",actual_model_calls=0))
            b2=CallBoundary(transport,tokenizer,j2,actor_limit=5200,curator_limit=0,input_limit=52000000-b1.actual_input,
                            output_limit=828928-b1.reserved_output,deadline=time.monotonic()+7200,reserve_seconds=120)
            with patch.object(controller,"secret_for",return_value="0123"):
                summary=controller.stage_two(ASSETS,b2,j2,Path("unused"),second,first)
            j2.close()
            rows2=read_journal(second/"journal.jsonl")
            scores2,costs2=replay(second/"journal.jsonl",ASSETS)
            ids=[r["case_id"] for r in ASSETS.manifest["rows"] if r["stage"]=="evaluation"][:50]
            replayed=aggregate(list(reversed(ids)),list(reversed(scores2)))
            for key in ("paired_case_ids","paired_delta_Q","CI95","delta_Q","full_N_bounds"):
                self.assertEqual(replayed[key],summary[key])
            admitted=[(r["payload"]["case_id"],r["payload"]["condition"]) for r in rows2 if r["event"]=="episode_admitted"]
            self.assertEqual(admitted,[(r["case_id"],c) for r,c in evaluation_plan(ASSETS.manifest,50)])
            self.assertEqual(costs2["issued_by_role"],dict(actor=5200))
            self.assertEqual(costs2["complete_episodes"],200)
            self.assertEqual(summary["paired_n"],50)
            self.assertEqual(sha((first/"memories/M20.txt").read_bytes()),m20_hash)
            self.assertEqual(b1.reserved_output+b2.reserved_output,828928)
            self.assertEqual(sum(b1.counts.values())+sum(b2.counts.values()),6366)
            for journal_rows in (rows,rows2):
                for row in journal_rows:
                    stored=row["sha256"]
                    unsigned={k:v for k,v in row.items() if k!="sha256"}
                    self.assertEqual(stored,sha(json.dumps(unsigned,sort_keys=True,ensure_ascii=True,separators=(",", ":"))))
            keep=os.environ.get("SUN_GN_KEEP_B1_DISK")
            if keep:
                destination=Path(keep)
                destination.mkdir(exist_ok=False,parents=True)
                shutil.copytree(first,destination/"stage-one")
                shutil.copytree(second,destination/"stage-two")
                (destination/"REPLAY_STATUS.json").write_text(json.dumps(dict(
                    mode="synthetic_B1_disk_regression",actual_model_calls=0,GPU_jobs=0,
                    stage_one_costs=costs1,stage_two_costs=costs2,
                    gate_event_verified=True,prior_consumed_by_stage_two=True,
                    arrival_order_and_CI_match=True,all_row_hashes_verified=True,
                    mock_boundary_calls=6366,paired_n=summary["paired_n"],
                    implementation_sha256=controller.implementation_digest()),indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    unittest.main(verbosity=2)
