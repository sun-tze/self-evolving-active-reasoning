"""CPU tests, including a local HTTP stub; no real model or GPU calls."""
import copy
import http.server
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common as C
import prepare as P
import runner as R
import analyze as A


def fixture(root):
    indices = list(range(30, 100))
    binding = {"source_lock": C.read_json(HERE / "source_lock.json"), "builder": {"test": "fixture"}}
    manifest = {"format": "c2_real_manifest_v1", "protocol": C.PROTOCOL,
                "protocol_sha256": C.digest(C.PROTOCOL), "binding": binding,
                "binding_sha256": C.digest(binding), "unseen_indices": indices,
                "pilot_indices": C.pilot_indices(indices), "case_bindings": {}}
    rows, targets, costs = [], [], {}
    for i in indices:
        manifest["case_bindings"][str(i)] = {}
        costs[str(i)] = {}
        for role in ["bare", "first"]:
            obs = C.observation(f"Fictional puzzle {i}", [[f"Question {n} {role}", "Yes"] for n in range(25)])
            ident = {"source": role, "position": i, "observation_sha256": C.digest(obs), "dataset_record_sha256": C.digest({"i": i})}
            manifest["case_bindings"][str(i)][role] = ident
            costs[str(i)][role] = {"solver_calls": 26, "tokens": {"in": 1000, "out": 100}, "answerer_calls": 50, "duration_s": 1}
            for cell, (source, writer) in C.CELLS.items():
                if role != source:
                    continue
                body = C.payload(obs, writer, i)
                row = {"format": "c2_real_request_v1", "index": i, "case_id": f"arbench-sp-position-{i:03d}",
                       "cell": cell, "source": role, "writer": writer, "binding_sha256": manifest["binding_sha256"],
                       "source_identity": ident, "observation": obs, "payload": body, "payload_sha256": C.digest(body)}
                row["request_id"] = C.request_id(row)
                rows.append(row)
        targets.append({"index": i, "bottom": "PRIVATE_GOLD_SENTINEL", "target_sha256": C.sha(b"PRIVATE_GOLD_SENTINEL"), "dataset_record_sha256": C.digest({"i": i})})
    C.write_jsonl(root / "requests.jsonl", rows)
    C.write_jsonl(root / "private/targets.jsonl", targets)
    C.write_json(root / "private/historical_costs.json", costs)
    for field, path in [("requests_sha256", "requests.jsonl"), ("targets_sha256", "private/targets.jsonl"), ("historical_costs_sha256", "private/historical_costs.json")]:
        manifest[field] = C.sha((root / path).read_bytes())
    C.write_json(root / "manifest.json", manifest)
    msha = C.sha((root / "manifest.json").read_bytes())
    budget = {"format": "c2_token_budget_v1", "manifest_sha256": msha,
              "model_lock": C.read_json(HERE / "model_lock.json"),
              "request_input_tokens": {r["request_id"]: 100 for r in rows}}
    C.write_json(root / "budget.json", budget)
    return manifest, rows, msha


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest, self.rows, self.msha = fixture(self.root)

    def test_exact_280_four_cells(self):
        self.assertEqual(len(C.validate_requests(self.rows, self.manifest)), 70)
        self.assertEqual(len(self.rows), 280)

    def test_source_field_allowlist_and_verbatim_nonstandard(self):
        history = [[" Q?\n", " Invisible \n"]]*25
        obs = C.observation("Surface", history)
        self.assertEqual(obs["history"][0], {"question": " Q?\n", "reply": " Invisible \n"})
        self.assertEqual(set(obs), {"surface", "history"})

    def test_bad_history_shapes(self):
        for bad in [[], [["Q", "Yes", "gold"]]*25, [{"question": "q", "reply": "a"}]*25, [["q", None]]*25]:
            with self.subTest(bad=str(bad)[:40]), self.assertRaises(ValueError):
                C.observation("s", bad)

    def test_duplicate_and_missing_cells_cases(self):
        for bad in [self.rows[:-1], self.rows[4:], self.rows+[self.rows[0]]]:
            with self.subTest(n=len(bad)), self.assertRaises(ValueError):
                C.validate_requests(bad, self.manifest)

    def test_rehashed_injected_payload_rejected(self):
        for field in ["gold", "bottom", "scoring_targets", "trace", "guess_text"]:
            rows = copy.deepcopy(self.rows)
            rows[0]["observation"][field] = "PRIVATE_SENTINEL"
            rows[0]["payload"] = C.payload(rows[0]["observation"], rows[0]["writer"], rows[0]["index"])
            rows[0]["payload_sha256"] = C.digest(rows[0]["payload"])
            rows[0]["request_id"] = C.request_id(rows[0])
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.validate_requests(rows, self.manifest)

    def test_rehashed_role_model_and_seed_mutation(self):
        for key, value in [("model", "another"), ("seed", 99), ("temperature", True), ("max_tokens", 401)]:
            rows = copy.deepcopy(self.rows)
            rows[0]["payload"][key] = value
            rows[0]["payload_sha256"] = C.digest(rows[0]["payload"])
            rows[0]["request_id"] = C.request_id(rows[0])
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_requests(rows, self.manifest)

    def test_surface_and_dialogue_mutation(self):
        for key in ["surface", "history"]:
            rows = copy.deepcopy(self.rows)
            if key == "surface": rows[0]["observation"][key] += "changed"
            else: rows[0]["observation"][key][0]["reply"] = "No"
            rows[0]["payload"] = C.payload(rows[0]["observation"], rows[0]["writer"], rows[0]["index"])
            rows[0]["payload_sha256"] = C.digest(rows[0]["payload"])
            rows[0]["request_id"] = C.request_id(rows[0])
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_requests(rows, self.manifest)

    def test_bound_source_identity_mutation(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["source_identity"] = dict(rows[0]["source_identity"], position=99)
        rows[0]["request_id"] = C.request_id(rows[0])
        with self.assertRaises(ValueError): C.validate_requests(rows, self.manifest)

    def test_file_and_manifest_tampering(self):
        with self.assertRaises(ValueError): C.load_bundle(self.root, "0"*64)
        (self.root/"requests.jsonl").write_bytes(b"{}\n")
        with self.assertRaises(ValueError): C.load_bundle(self.root, self.msha)

    def test_order_independence_and_seed_pairing(self):
        C.validate_requests(list(reversed(self.rows)), self.manifest)
        for i in self.manifest["unseen_indices"]:
            self.assertEqual(len({r["payload"]["seed"] for r in self.rows if r["index"] == i}), 1)

    def test_pilot_selection_independent_of_scores_order(self):
        indices = self.manifest["unseen_indices"]
        self.assertEqual(C.pilot_indices(indices), C.pilot_indices(list(reversed(indices))))

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
            with self.assertRaises(ValueError): C.decode(raw)

    def test_plan_no_network_and_caps(self):
        with patch.object(R, "http_json", side_effect=AssertionError("network")):
            plan, selected, _ = R.plan(self.root, self.msha, self.root/"budget.json", "pilot", "http://127.0.0.1:18000/v1", 20, 50000, 8000)
        self.assertEqual(len(selected), 20)
        self.assertEqual(plan["totals"]["max_output_tokens"], 8000)
        for calls, inp, out in [(19, 50000, 8000), (20, 1999, 8000), (20, 50000, 7999)]:
            with self.assertRaises(ValueError): R.plan(self.root, self.msha, self.root/"budget.json", "pilot", "http://127.0.0.1:18000/v1", calls, inp, out)

    def test_full_run_requires_expanded_budget(self):
        with self.assertRaises(ValueError): R.plan(self.root, self.msha, self.root/"budget.json", "all", "http://127.0.0.1:18000/v1", 20, 50000, 8000)

    def test_loopback_and_redirect_guard(self):
        for url in ["https://api.openai.com/v1", "http://example.com:80/v1", "http://localhost:80/v1", "http://u:p@127.0.0.1:80/v1", "http://127.0.0.1:80/v1?x=1"]:
            with self.assertRaises(ValueError): R.endpoint(url)

    def test_authorization_required_before_network(self):
        with patch.object(R, "http_json", side_effect=AssertionError("network")), self.assertRaises(ValueError):
            R.execute({}, [], {}, self.root/"run", None, 1200, 120)


class ScoringTests(unittest.TestCase):
    def test_known_character_multiset_cases(self):
        self.assertEqual(C.char_f1("", ""), 0)
        self.assertEqual(C.char_f1("aab", "abb"), 2/3)
        self.assertEqual(C.char_f1("ABC", "abc"), 0)
        self.assertEqual(C.char_f1("! ", " !"), 1)
        self.assertEqual(C.char_f1("é猫", "猫é"), 1)

    def test_factorial_arithmetic_and_negative_effects(self):
        cases = [{"A": .2, "B": .3, "C": .4, "D": .6}, {"A": .8, "B": .6, "C": .5, "D": .4}]
        _, stats = A.paired_stats(cases, draws=1000)
        expected = [-.05, .05, -.05, .05, 0, 0, .1]
        for k, v in zip(A.CONTRASTS, expected): self.assertAlmostEqual(stats[k]["mean"], v)
        self.assertEqual(stats["interaction_(D-C)-(B-A)"]["available"], 2)

    def test_missing_suppresses_only_affected_contrasts(self):
        _, stats = A.paired_stats([{"A": None, "B": .3, "C": .4, "D": .6}], draws=100)
        self.assertIsNone(stats["writer_given_bare_B-A"]["mean"])
        self.assertAlmostEqual(stats["writer_given_first_D-C"]["mean"], .2)
        self.assertIsNone(stats["interaction_(D-C)-(B-A)"]["ci95"])

    def test_bootstrap_resamples_cases_reproducibly(self):
        cases = [{"A": .1, "B": .2, "C": .3, "D": .4}]*5
        self.assertEqual(A.paired_stats(cases), A.paired_stats(cases))
        _, stats = A.paired_stats(cases)
        for endpoint in stats["writer_given_bare_B-A"]["ci95"]: self.assertAlmostEqual(endpoint, .1)

    def test_nonfinite_boolean_invalid_scores(self):
        for val in [True, float("nan"), float("inf"), -1, 2]:
            with self.assertRaises(ValueError): A.paired_stats([dict(A=val,B=0,C=0,D=0)])

    def test_raw_completion_checks_usage_and_truncation(self):
        request = {"payload": {"max_tokens": 400}}
        raw = {"model": C.MODEL, "choices": [{"message": {"content": " full text "}, "finish_reason": "length"}], "usage": None}
        parsed = R.parse_completion(raw, request)
        self.assertEqual(parsed["text"], " full text ")
        self.assertTrue(parsed["truncated"])
        self.assertIsNone(parsed["usage"]["total_tokens"])
        for usage in [{"prompt_tokens": True}, {"completion_tokens": -1}, {"prompt_tokens": 1,"completion_tokens": 2,"total_tokens": 4}, {"completion_tokens": 401}]:
            with self.assertRaises(ValueError): R.parse_completion(dict(raw,usage=usage), request)

    def test_unknown_finish_or_wrong_model_rejected(self):
        raw = {"model": C.MODEL, "choices": [{"message": {"content": "a"}, "finish_reason": "stop"}]}
        with self.assertRaises(ValueError): R.parse_completion(dict(raw,model="other"), {"payload": {"max_tokens": 400}})
        raw["choices"][0]["finish_reason"] = "content_filter"
        with self.assertRaises(ValueError): R.parse_completion(raw, {"payload": {"max_tokens": 400}})


class SourceTests(unittest.TestCase):
    def source(self):
        dataset = [{"surface": f"puzzle-{i}", "bottom": "private target"} for i in range(100)]
        rows = [{"index": i,"harness":"bare","task":"sp","status":"ok","error":None,
                 "history":[["question", "Unknown"]]*25,
                 "trace":[{"messages":[{"content": f"\nPuzzle: puzzle-{i}\n"}]}],
                 "gold":{"prediction":"answer","f1_char":C.char_f1("answer","private target")}}
                 for i in range(100)]
        return dataset, rows

    def test_source_replay_and_allowlist(self):
        dataset, rows = self.source()
        rows[0]["secret"] = "GOLD_DO_NOT_COPY"
        raw = b"".join(C.canonical(r)+b"\n" for r in rows)
        _, identities, audit = P.source_rows(raw, "bare", dataset, C.char_f1)
        self.assertEqual(audit["score_replay_count"], 100)
        self.assertNotIn("GOLD_DO_NOT_COPY", C.canonical(identities).decode())

    def test_source_failures_id_role_surface_score_history_leak(self):
        mutations = [lambda r: r[0].update(index=1), lambda r: r[0].update(index=True),
                     lambda r: r[0].update(harness="wrong"), lambda r: r[0].update(status="failed"),
                     lambda r: r[0]["trace"][0]["messages"][0].update(content="wrong"),
                     lambda r: r[0]["gold"].update(f1_char=.999),
                     lambda r: r[0].update(history=[["private target","Yes"]]*25)]
        for mutate in mutations:
            dataset, rows = self.source(); mutate(rows)
            with self.assertRaises(ValueError): P.source_rows(b"".join(C.canonical(r)+b"\n" for r in rows), "bare", dataset, C.char_f1)

    def test_stream_three_way_crosscheck(self):
        import numpy as np
        stream = [int(i) for i in np.random.default_rng(0).permutation(100)[:30]]
        base = dict(task="sp",split="test",solver_model=C.MODEL,budget=25,solver_mock=None,answerer_mock=False)
        evaluation = dict(base,exclude_stream_seed=0,exclude_stream_limit=30,instances=list(range(100)),excluded_seen=sorted(stream))
        evolution = dict(base,stream_seed=0,limit=30,stream=stream,signals="strict",run_name="sp_strict1_b10g3r2")
        self.assertEqual(len(P.identify_unseen(evaluation,evolution)),70)
        evaluation["excluded_seen"][-1] = 99
        with self.assertRaises(ValueError): P.identify_unseen(evaluation,evolution)


class HTTPIntegrationTests(unittest.TestCase):
    setUp = DesignTests.setUp
    def start_server(self, fail=False, bad_tokens=False):
        captured = []
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def send(self, value, status=200):
                raw=C.canonical(value);self.send_response(status);self.end_headers();self.wfile.write(raw)
            def do_GET(self): self.send({"data":[{"id":C.MODEL}]})
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append((self.path,body))
                if self.path == "/tokenize": self.send({"count": 99 if bad_tokens else 100})
                elif fail: self.send({"error":"test failure"},500)
                else: self.send({"model":C.MODEL,"choices":[{"message":{"content":"stub answer"},"finish_reason":"stop"}],
                                "usage":{"prompt_tokens":100,"completion_tokens":2,"total_tokens":102}})
        server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}/v1", captured

    def run_plan(self,url):
        return R.plan(self.root,self.msha,self.root/"budget.json","pilot",url,20,50000,8000)

    def test_end_to_end_network_allowlist_score_and_resume(self):
        url,captured=self.start_server();doc,selected,counts=self.run_plan(url)
        out=self.root/"run"
        R.execute(doc,selected,counts,out,"CPU_STUB_TEST_ONLY",120,10)
        sent=[b for path,b in captured if path=="/v1/chat/completions"]
        self.assertEqual(sent,[r["payload"] for r in selected])
        self.assertNotIn("PRIVATE_GOLD_SENTINEL",C.canonical(sent).decode())
        report=A.analyze(self.root,self.msha,out,self.root/"report.json")
        self.assertEqual(report["planned_requests"],20)
        self.assertEqual(report["new_inference_cost"]["token_usage"]["total_tokens"]["total_if_complete"],2040)
        R.execute(doc,selected,counts,out,"CPU_STUB_TEST_ONLY",120,10,resume=True)
        self.assertEqual(len([p for p,b in captured if p=="/v1/chat/completions"]),20)

    def test_server_token_mismatch_stops_before_generation(self):
        url,captured=self.start_server(bad_tokens=True);doc,selected,counts=self.run_plan(url)
        with self.assertRaises(ValueError):R.execute(doc,selected,counts,self.root/"run","CPU_STUB_TEST_ONLY",120,10)
        self.assertFalse(any(p=="/v1/chat/completions" for p,b in captured))

    def test_network_failure_no_retry_and_null_analysis(self):
        url,captured=self.start_server(fail=True);doc,selected,counts=self.run_plan(url);out=self.root/"run"
        with self.assertRaises(Exception):R.execute(doc,selected,counts,out,"CPU_STUB_TEST_ONLY",120,10)
        self.assertEqual(len([p for p,b in captured if p=="/v1/chat/completions"]),1)
        with self.assertRaises(ValueError):R.execute(doc,selected,counts,out,"CPU_STUB_TEST_ONLY",120,10,resume=True)
        report=A.analyze(self.root,self.msha,out,self.root/"report.json")
        self.assertTrue(all(s["mean"] is None for s in report["contrasts"].values()))
        self.assertIsNone(report["new_inference_cost"]["token_usage"]["total_tokens"]["total_if_complete"])

    def test_ambiguous_journal_does_not_resend(self):
        url,captured=self.start_server();doc,selected,counts=self.run_plan(url);out=self.root/"run"
        with patch.object(R,"http_json",side_effect=RuntimeError("test interruption before inference")):
            with self.assertRaises(RuntimeError):R.execute(doc,selected,counts,out,"CPU_STUB_TEST_ONLY",120,10)
        r=selected[0]
        R.append_event(out/"journal.jsonl",{"event":"started","request_id":r["request_id"],"payload_sha256":r["payload_sha256"]})
        with self.assertRaises(ValueError):R.execute(doc,selected,counts,out,"CPU_STUB_TEST_ONLY",120,10,resume=True)
        self.assertFalse(captured)

    def test_duplicate_foreign_or_unpaired_journal_rejected(self):
        _,selected,_=self.run_plan("http://127.0.0.1:18000/v1");r=selected[0]
        e={"event":"started","request_id":r["request_id"],"payload_sha256":r["payload_sha256"]}
        for events in [[e,e],[dict(e,request_id="foreign")],[dict(e,event="result",status="completed")]]:
            with self.assertRaises(ValueError):R.journal_state(events,selected)

    def test_target_tampering_is_rejected(self):
        (self.root/"private/targets.jsonl").write_bytes(b"{}\n")
        with self.assertRaises(ValueError):A.analyze(self.root,self.msha,self.root/"run",self.root/"report.json")

    def test_unique_service_marker_rejects_old_server(self):
        url,captured=self.start_server();doc,selected,counts=self.run_plan(url)
        doc["server_marker"]="c2-pilot-new-job-unique-marker"
        with self.assertRaises(ValueError):R.execute(doc,selected,counts,self.root/"run","CPU_STUB_TEST_ONLY",120,10)
        self.assertFalse(captured)

    def test_absolute_deadline_stops_dribbling_body(self):
        class Slow(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                self.send_response(200);self.end_headers()
                for c in b'{"slow":true}':
                    try:self.wfile.write(bytes([c]));self.wfile.flush()
                    except OSError:break
                    time.sleep(.08)
        server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Slow)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        start=time.monotonic()
        with self.assertRaises(TimeoutError):R.http_json(f"http://127.0.0.1:{server.server_port}/slow",timeout=.15)
        self.assertLess(time.monotonic()-start,.75)

    def test_redirect_response_not_followed(self):
        class Redirect(http.server.BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                self.send_response(302);self.send_header("Location","http://example.com/");self.end_headers()
        server=http.server.ThreadingHTTPServer(("127.0.0.1",0),Redirect)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        with self.assertRaises(ValueError):R.http_json(f"http://127.0.0.1:{server.server_port}/redirect",timeout=1)


if __name__ == "__main__": unittest.main()
