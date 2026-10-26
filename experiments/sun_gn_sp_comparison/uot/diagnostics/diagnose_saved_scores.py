"""Offline diagnostics of recorded UoT depth-1 decisions; never model inference.

Extracts only pinned function definitions and literal prompt assignments. The
original evaluator consumes saved responses through an injected transport and
writes its incidental log only to StringIO. No evaluator imports are executed.
This is a diagnostic of saved proposals, not a full scientific run audit, a
counterfactual trajectory evaluation, or evidence of self-evolution.

Root must review before one bounded invocation. Only --output is written; it
must not exist. Raw prompts, secret numbers and answer lists are not exported.
"""
import argparse
import ast
import hashlib
import inspect
import io
import itertools
import json
import math
import random
import re
import time
import typing
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


PINS = {
    "gn_evaluator.py": "aa6643d004ed7b510452ce9fe581b371112616a6702dfee3a26132759cb221a9",
    "prompt.py": "5dafd196d5542fad163898fbfecca411fb13b1b30b636ebc0577d38a7bbcc713",
    "utils_gn.py": "f57cc0a77615859ea44fad07a0e5b849aa95e6d428963d72d3c49405c20274e7",
}
FUNCTIONS = {
    "_gn_extract_multiple_guesses", "_gn_generate_all_answers",
    "_gn_calculate_remaining", "_gn_calculate_remaining_with_model",
    "_gn_estimate_uncertainty_reduction", "_gn_simulate_rounds",
    "_gn_analyze_candidates_with_simulation", "_run_uot_evaluation",
}
MODEL = "Qwen/Qwen2.5-7B-Instruct"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def key(number):
    return tuple(number)


class MissingRecordedResponse(Exception):
    pass


def load_functions(reference, namespace):
    for name, wanted in PINS.items():
        data = (reference / name).read_bytes()
        require(digest(data) == wanted, "Pinned source mismatch: " + name)
        tree = ast.parse(data.decode("utf-8"), filename=name)
        nodes = []
        if name == "prompt.py":
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    ast.literal_eval(node.value)
                    nodes.append(node)
        else:
            names = FUNCTIONS if name == "gn_evaluator.py" else {"compare_guess"}
            nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in names]
            require({n.name for n in nodes} == names, "Missing pinned definitions")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)


class SavedReplay:
    def __init__(self, reference, intents, responses, episode):
        self.intents, self.responses, self.episode = intents, responses, episode
        self.cursor = 0
        self.last_intent = self.last_response = None
        self.decisions, self.branches = [], []
        self.active_decision = None
        self.current_candidate = None
        self.missing_call_id = None
        self.ns = {n: getattr(typing, n) for n in
                   ("Any", "Dict", "Generator", "List", "Union", "Optional", "Tuple")}
        self.ns.update(
            Counter=Counter, random=random.Random(episode["seed"]), re=re,
            math=math, itertools=itertools, json=json, inference=self.inference,
            tqdm=lambda xs: xs, print=lambda *a, **kw: None,
            POLICY_API_KEY=None, POLICY_BASE_URL=None, open=self.log_sink)
        load_functions(reference, self.ns)
        self.original_branch = self.ns["_gn_calculate_remaining_with_model"]
        self.original_simulate = self.ns["_gn_simulate_rounds"]
        self.original_analyze = self.ns["_gn_analyze_candidates_with_simulation"]
        self.ns["_gn_calculate_remaining_with_model"] = self.branch
        self.ns["_gn_simulate_rounds"] = self.simulate
        self.ns["_gn_analyze_candidates_with_simulation"] = self.analyze

    @staticmethod
    def log_sink(path, mode="r", *args, **kwargs):
        require(path == "__memory_only_log__" and mode == "w",
                "Unexpected upstream filesystem operation")
        return io.StringIO()

    def inference(self, messages, **kwargs):
        if self.cursor == len(self.intents):
            self.missing_call_id = "NO_RECORDED_INTENT"
            raise MissingRecordedResponse("No further recorded intent")
        intent = self.intents[self.cursor]
        self.cursor += 1
        self.last_intent, self.last_response = intent, None
        request, metadata = intent["request"], intent["metadata"]
        require(metadata["call_index"] == self.cursor, "Episode call index differs")
        require(metadata["upstream_caller"] == inspect.currentframe().f_back.f_code.co_name,
                "Upstream caller differs")
        require(request["messages"] == messages, "Saved request prompt differs")
        require(request["seed"] == self.episode["seed"], "Saved request seed differs")
        require(request["model"] == MODEL and request["max_tokens"] == 512,
                "Saved model or output cap differs")
        require(metadata.get("memory_injected") is False and
                metadata.get("method") == "uot" and metadata.get("task") == "GN" and
                metadata.get("phase") == "native_algorithm", "Unexpected metadata")
        for field in ("temperature", "top_p"):
            require(request[field] == kwargs[field] == metadata[field],
                    "Saved sampling parameter differs: " + field)
        expected_role = ("uot_proposal" if metadata["upstream_caller"] ==
                         "_run_uot_evaluation" else "uot_simulation")
        require(intent["role"] == expected_role, "Saved role differs")
        response = self.responses.get(intent["call_id"])
        if response is None:
            self.missing_call_id = intent["call_id"]
            raise MissingRecordedResponse("Recorded intent has no response")
        self.last_response = response
        require(isinstance(response["text"], str), "Invalid saved response text")
        require(type(response["completion_tokens"]) is int and
                0 <= response["completion_tokens"] <= 512, "Invalid token accounting")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response["text"]))],
            usage=SimpleNamespace(prompt_tokens=response["prompt_tokens"],
                                  completion_tokens=response["completion_tokens"]))

    def branch(self, model, answer_set, guess, same_pos, diff_pos, temperature, top_p):
        record = {
            "episode_index": self.episode["index"],
            "decision_index": self.active_decision["decision_index"],
            "candidate_position": self.current_candidate["candidate_position"],
            "feedback": [same_pos, diff_pos],
            "response_present": False,
        }
        self.branches.append(record)
        try:
            result = self.original_branch(model, answer_set, guess, same_pos, diff_pos,
                                          temperature, top_p)
        except MissingRecordedResponse:
            record["call_id"] = self.missing_call_id
            record["native_verified_count"] = None
            record["status"] = "MISSING_RESPONSE_EXCLUDED_NOT_EMPTY"
            raise
        intent, response = self.last_intent, self.last_response
        prompt = intent["request"]["messages"][0]["content"]
        header = re.search(r"\(showing (\d+) out of (\d+) total\):\n(.*?)\n\nBased on",
                           prompt, re.S)
        require(header is not None, "Cannot recover displayed sample")
        shown, total = int(header[1]), int(header[2])
        sample = [tuple(map(int, n)) for n in re.findall(r"^([0-9]{4})$", header[3], re.M)]
        state = {key(n) for n in answer_set}
        require(shown == len(sample) == min(50, len(answer_set)) and
                total == len(answer_set) and len(set(sample)) == shown and
                set(sample) <= state, "Displayed sample is inconsistent")
        exact_bucket = {key(n) for n in answer_set
                        if self.ns["compare_guess"](n, guess)[:2] == (same_pos, diff_pos)}
        extracted = [tuple(map(int, s)) for s in re.findall(r"\d{4}", response["text"])]
        verified = [key(n) for n in result[0]]
        independently_verified = [n for n in extracted
                                  if self.ns["compare_guess"](n, guess)[:2] == (same_pos, diff_pos)]
        require(verified == independently_verified, "Native verified-list reconstruction differs")
        record.update({
            "call_id": intent["call_id"], "status": "RECORDED_RESPONSE",
            "response_present": True, "state_count": len(answer_set),
            "exact_bucket_count": len(exact_bucket),
            "native_verified_count": len(verified),
            "native_count_minus_exact": len(verified) - len(exact_bucket),
            "displayed_sample_count": shown,
            "state_larger_than_50_input_sample": total > shown,
            "exact_valid_in_displayed_sample": len(set(sample) & exact_bucket),
            "finish_reason": response["finish_reason"],
            "finish_length": response["finish_reason"] == "length",
            "completion_tokens": response["completion_tokens"],
            "at_512_output_cap": response["completion_tokens"] == 512,
            "raw_extracted_occurrences": len(extracted),
            "raw_duplicate_occurrences": len(extracted) - len(set(extracted)),
            "verified_duplicate_occurrences": len(verified) - len(set(verified)),
            "verified_nonmember_occurrences": sum(n not in state for n in verified),
            "verified_not_shown_occurrences": sum(n not in set(sample) for n in verified),
            "verified_repeated_digit_occurrences": sum(len(set(n)) != 4 for n in verified),
            "unique_correct_survivors_returned": len(set(verified) & exact_bucket),
            "exact_survivors_not_returned": len(exact_bucket - set(verified)),
        })
        return result

    def simulate(self, answer_set, guess, depth=3, model=None, temperature=0.1, top_p=0.9):
        require(depth == 1, "Diagnostic supports only pinned depth 1")
        position = len(self.active_decision["candidates"]) + 1
        _, partition = self.ns["_gn_estimate_uncertainty_reduction"](answer_set, guess)
        exact_score = 0.0
        for count in partition.values():
            exact_score += (count / len(answer_set)) * (len(answer_set) - count)
        candidate = {
            "candidate_position": position,
            "candidate_sha256": digest(bytes(guess)),
            "exact_expected_candidate_count_reduction": exact_score,
            "branch_order_and_counts": [[k, n] for k, n in partition.items()],
            "native_expected_candidate_count_reduction": None,
            "complete": False,
        }
        self.active_decision["candidates"].append(candidate)
        self.current_candidate = candidate
        result = self.original_simulate(answer_set, guess, depth, model, temperature, top_p)
        candidate["native_expected_candidate_count_reduction"] = result[0]
        candidate["complete"] = True
        return result

    def analyze(self, answer_set, candidates, depth=3, model=None, temperature=0.1, top_p=0.9):
        decision = {
            "episode_index": self.episode["index"],
            "decision_index": len(self.decisions) + 1,
            "state_count": len(answer_set),
            "state_sha256": digest(json.dumps(answer_set, separators=(",", ":")).encode()),
            "proposal_call_id": self.last_intent["call_id"],
            "proposals_count": len(candidates), "candidates": [],
            "complete": False, "included_in_rank_comparison": False,
        }
        self.decisions.append(decision)
        self.active_decision = decision
        try:
            result = self.original_analyze(answer_set, candidates, depth, model, temperature, top_p)
        except MissingRecordedResponse:
            decision["excluded_reason"] = "incomplete simulation responses; no imputation"
            raise
        rows = decision["candidates"]
        require(len(rows) == len(candidates) and all(r["complete"] for r in rows),
                "Incomplete native decision")
        native_order = sorted(range(len(rows)), key=lambda i:
                              rows[i]["native_expected_candidate_count_reduction"], reverse=True)
        exact_order = sorted(range(len(rows)), key=lambda i:
                             rows[i]["exact_expected_candidate_count_reduction"], reverse=True)
        expected = [(candidates[i], rows[i]["native_expected_candidate_count_reduction"])
                    for i in native_order]
        require([(r[0], r[1]) for r in result[0]] == expected, "Native ranking replay differs")
        native_pick, exact_pick = native_order[0], exact_order[0]
        decision.update({
            "complete": True, "included_in_rank_comparison": True,
            "native_rank_positions": [i + 1 for i in native_order],
            "exact_count_rank_positions": [i + 1 for i in exact_order],
            "rank_changed": native_order != exact_order,
            "selected_position_changed": native_pick != exact_pick,
            "selected_guess_changed": candidates[native_pick] != candidates[exact_pick],
            "native_selected_exact_count_reduction": rows[native_pick]["exact_expected_candidate_count_reduction"],
            "exact_best_count_reduction": rows[exact_pick]["exact_expected_candidate_count_reduction"],
            "one_step_expected_count_reduction_gap":
                rows[exact_pick]["exact_expected_candidate_count_reduction"] -
                rows[native_pick]["exact_expected_candidate_count_reduction"],
            "comparison_scope": "same recorded proposals/state; not a counterfactual trajectory",
        })
        return result

    def run(self, secret):
        logs = []
        status = "complete"
        try:
            self.ns["_run_uot_evaluation"]([secret], logs, "__memory_only_log__",
                                           "caller_frozen_model", 25, 0.7, 0.7, 1)
        except MissingRecordedResponse:
            status = "recorded_prefix_only"
        require(self.cursor == len(self.intents), "Saved intents left unconsumed")
        return logs, status


def diagnose(evidence, reference):
    runtime_path = evidence / "allocation/runtime-journal.jsonl"
    controller = evidence / "allocation/controller"
    journal = read_jsonl(runtime_path)
    require(journal[0]["event"] == "run_start" and
            journal[0]["payload"]["job_id"] == "26955564", "Unexpected run identity")
    require([r["index"] for r in journal] == list(range(len(journal))),
            "Runtime event indices are not contiguous")
    for previous, current in zip(journal, journal[1:]):
        require(current["previous_sha256"] == previous["sha256"],
                "Runtime previous-hash linkage differs")
    cases_path = controller / "private-cases.json"
    cases = read_json(cases_path)
    require(digest(cases_path.read_bytes()) ==
            journal[0]["payload"]["bindings"]["private-cases.json"], "Private-case binding differs")
    intents = [r["payload"] for r in journal if r["event"] == "call_intent"]
    response_rows = [r["payload"] for r in journal if r["event"] == "call_response"]
    responses = {r["call_id"]: r["response"] for r in response_rows}
    ids = [r["call_id"] for r in intents]
    require(len(set(ids)) == len(ids) and len(responses) == len(response_rows),
            "Duplicate call id")
    require(set(responses) <= set(ids), "Orphan response")
    require(ids == ["gn-compare-" + str(i) for i in range(1, len(ids) + 1)],
            "Noncontiguous runtime call sequence")
    controller_rows = read_jsonl(controller / "journal.jsonl")
    starts = [r["payload"] for r in controller_rows if r["kind"] == "episode_started"]
    saved = {r["payload"]["index"]: r["payload"] for r in controller_rows
             if r["kind"] == "episode_saved"}
    decisions, branches, episode_reports = [], [], []
    require(starts and starts[0]["call_start_exclusive"] == 0, "Missing initial episode")
    for i, episode in enumerate(starts):
        begin = episode["call_start_exclusive"]
        end = starts[i + 1]["call_start_exclusive"] if i + 1 < len(starts) else len(intents)
        require(0 <= begin < end <= len(intents), "Invalid episode boundaries")
        replay = SavedReplay(reference, intents[begin:end], responses, episode)
        logs, status = replay.run(cases[episode["case_id"]])
        report = {"episode_index": episode["index"], "replay_status": status,
                  "saved_episode_present": episode["index"] in saved,
                  "intents_replayed": replay.cursor,
                  "responses_replayed": sum(r["call_id"] in responses for r in intents[begin:end]),
                  "missing_call_id": replay.missing_call_id}
        if episode["index"] in saved:
            save = saved[episode["index"]]
            source_path = controller / save["file"]
            require(source_path.resolve().is_relative_to(controller.resolve()), "Invalid episode path")
            require(digest(source_path.read_bytes()) == save["sha256"], "Saved episode hash differs")
            source = read_json(source_path)
            require(status == "complete" and len(logs) == 1, "Saved complete episode did not replay")
            require(logs[0] == source["official_result"], "Native saved episode result differs")
            require(source["accounting"]["simulate_depth"] == 1 and
                    source["seed"] == episode["seed"] and source["case_id"] == episode["case_id"],
                    "Saved episode identity/depth differs")
            require(save["call_end_inclusive"] == end and save["call_start_exclusive"] == begin,
                    "Saved episode call coverage differs")
            report["official_result_reproduced_exactly"] = True
        else:
            require(status == "recorded_prefix_only", "Unrecorded complete episode needs investigation")
        decisions.extend(replay.decisions)
        branches.extend(replay.branches)
        episode_reports.append(report)
    complete = [d for d in decisions if d["included_in_rank_comparison"]]
    observed = [b for b in branches if b["response_present"]]
    return {
        "status": "OFFLINE_RECORDED_DECISION_DIAGNOSTIC_COMPLETE",
        "not_scientific_full_run_audit": True,
        "scope": "Recorded native depth-1 proposals only; no new trajectories, models, or GPU",
        "objective": "sum branch_probability * (state_count - survivor_count)",
        "exact_replacement": "true partition cardinality; original probabilities and stable tie order",
        "entropy_is_not_substituted": True,
        "source_pins": PINS,
        "evidence_sha256": {
            "runtime-journal.jsonl": digest(runtime_path.read_bytes()),
            "controller/journal.jsonl": digest((controller / "journal.jsonl").read_bytes()),
            "private-cases.json": digest(cases_path.read_bytes()),
        },
        "coverage": {"saved_complete_episodes": len(saved), "started_episodes": len(starts),
                     "call_intents": len(intents), "call_responses": len(responses),
                     "missing_response_call_ids": [i for i in ids if i not in responses],
                     "complete_decisions": len(complete), "incomplete_decisions_excluded": len(decisions) - len(complete),
                     "recorded_simulation_responses": len(observed)},
        "summary": {
            "decisions_rank_changed": sum(d["rank_changed"] for d in complete),
            "decisions_selected_guess_changed": sum(d["selected_guess_changed"] for d in complete),
            "decisions_selected_position_changed": sum(d["selected_position_changed"] for d in complete),
            "branches_state_exceeds_displayed_50_sample": sum(b["state_larger_than_50_input_sample"] for b in observed),
            "branches_finish_length": sum(b["finish_length"] for b in observed),
            "branches_at_512_output_cap": sum(b["at_512_output_cap"] for b in observed),
            "branches_with_verified_duplicates": sum(b["verified_duplicate_occurrences"] > 0 for b in observed),
            "branches_with_verified_nonmembers": sum(b["verified_nonmember_occurrences"] > 0 for b in observed),
            "branches_native_count_differs_from_exact": sum(b["native_count_minus_exact"] != 0 for b in observed),
        },
        "limitations": [
            "Missing response is excluded, never interpreted as an empty list.",
            "Max-50 input sampling and 512-output-token truncation are separate observed constraints.",
            "This does not estimate counterfactual final success or question savings.",
            "Episode four has only an observed prefix; a partial run is not a 20-case result.",
            "Not an independent replacement for the authorized full-result audit or scheduler accounting.",
        ],
        "episodes": episode_reports, "decisions": decisions, "branches": branches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite an existing diagnostic")
    require(not args.output.resolve().is_relative_to(args.evidence.resolve()) and
            not args.output.resolve().is_relative_to(args.reference.resolve()),
            "Output cannot be inside evidence or reference")
    started = time.monotonic()
    try:
        report = diagnose(args.evidence, args.reference)
        exit_code = 0
    except Exception as error:
        report = {"status": "DIAGNOSTIC_STOPPED_NOT_PASS", "error_type": type(error).__name__,
                  "error": str(error), "new_model_calls": 0,
                  "instruction": "Do not impute missing fields or publish success; inspect this discrepancy."}
        exit_code = 1
    report["elapsed_seconds"] = time.monotonic() - started
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"],
                      "coverage": report.get("coverage"), "summary": report.get("summary")}))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
