"""Offline checks for the GN representation baseline.

Constructed aggregates in the success-bar tests are not measurements.
Mock-policy episodes are not model results and are not a scientific lift.
"""
from __future__ import annotations

from collections import Counter
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from arm0_kimi.client import (
    DEFAULT_QWEN_BASE_URL,
    KimiActor,
    MockActor,
    QwenActor,
    describe_qwen,
)
from arm0_kimi.domain import CANDIDATES, DeterministicBelief, feedback, make_cases
from arm0_kimi.official_cases import (
    EXPECTED_TEST_SHA256,
    EXPECTED_TRAIN_SHA256,
    LOCKED_N20_SECRETS,
    PACKAGE_FIXTURE,
    SMOKE_N2_SECRETS,
    load_official_cases,
    sha256_file,
    verify_official_file,
)
from arm0_kimi.prompts import ARMS, prompt_for
from arm0_kimi.readout import representation_baseline_readout
from arm0_kimi.runner import RunConfig, inspect_episode, run_episode, run_matrix
from arm0_kimi.smoke import main

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = PACKAGE_FIXTURE


def compare_guess(list1, list2):
    """Pinned AR-Bench 9971322 ``arbench.utils.utils_gn.compare_guess``.

    Copied so the offline test does not import the submodule.
    """
    if isinstance(list1, str):
        list1 = [int(digit) for digit in list1]
    if isinstance(list2, str):
        list2 = [int(digit) for digit in list2]
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")
    same_position_count = sum(a == b for a, b in zip(list1, list2))
    same_elements = sum((Counter(list1) & Counter(list2)).values())
    different = same_elements - same_position_count
    return same_position_count, different, same_position_count * 2 + different


def extract_and_convert_guess(text):
    """Pinned AR-Bench last-``\\d{4}`` extractor. Accepts repeated digits."""
    import re
    matches = re.findall(r"\d{4}", text)
    if not matches:
        return []
    return [int(char) for char in matches[-1]]


def _block(em, repeat, consistency, episodes=18):
    return {
        "episodes": episodes,
        "EM": em,
        "repeat_guess": repeat,
        "feedback_consistency": consistency,
    }


def _arms(ledger_em, cand_em, chrono_em, ledger_rep, cand_rep, chrono_rep,
          ledger_con, cand_con, chrono_con, episodes=18):
    return {
        "chrono": _block(chrono_em, chrono_rep, chrono_con, episodes),
        "ledger": _block(ledger_em, ledger_rep, ledger_con, episodes),
        "ledger_belief": _block(cand_em, cand_rep, cand_con, episodes),
    }


def _per_seed(by_arm, seeds=(0, 1, 2)):
    return {str(seed): by_arm for seed in seeds}


class _Resp:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _completion(text="Guess: 0123"):
    return {
        "id": "offline-test",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
    }


class OfficialCaseTests(unittest.TestCase):
    def test_fixture_matches_pinned_arbench_test_split(self):
        self.assertEqual(sha256_file(FIXTURE), EXPECTED_TEST_SHA256)
        self.assertEqual(FIXTURE.stat().st_size, 800)
        report = verify_official_file(FIXTURE)
        self.assertTrue(report["sha256_ok"])
        self.assertEqual(report["n_secrets"], 100)
        self.assertFalse(report["is_train_split"])
        self.assertTrue(report["locked_n20_ok"])
        self.assertEqual(report["arbench_commit"], "9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1")
        secrets = json.loads(FIXTURE.read_text())
        self.assertEqual(len(set(secrets)), 100)
        self.assertEqual(sum(item.startswith("0") for item in secrets), 9)

    def test_loader_prefix_case_ids_and_no_seed_argument(self):
        smoke = load_official_cases(2)
        matrix = load_official_cases(20)
        full = load_official_cases(100)
        self.assertEqual([case["secret"] for case in smoke], list(SMOKE_N2_SECRETS))
        self.assertEqual([case["case_id"] for case in smoke], ["arbench-test-0000", "arbench-test-0001"])
        self.assertEqual([case["secret"] for case in matrix], list(LOCKED_N20_SECRETS))
        self.assertEqual(matrix[0]["case_id"], "arbench-test-0000")
        self.assertEqual(matrix[19]["case_id"], "arbench-test-0019")
        self.assertEqual(matrix[0]["sha256"], EXPECTED_TEST_SHA256)
        self.assertEqual(matrix[0]["source"], "arbench_gn_test")
        self.assertEqual(len(full), 100)
        self.assertEqual([case["secret"] for case in full[:20]], list(LOCKED_N20_SECRETS))
        with self.assertRaises(TypeError):
            load_official_cases(2, seed=0)  # type: ignore[call-arg]

    def test_train_digest_sha_mismatch_drift_and_missing_file(self):
        with patch("arm0_kimi.official_cases.sha256_file", return_value=EXPECTED_TRAIN_SHA256):
            with self.assertRaisesRegex(ValueError, "train.json"):
                load_official_cases(2, json_path=FIXTURE)
        with patch("arm0_kimi.official_cases.sha256_file", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_official_cases(2, json_path=FIXTURE)
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "test.json"
            with self.assertRaisesRegex(FileNotFoundError, "synthetic"):
                load_official_cases(2, json_path=missing)
            drifted = json.loads(FIXTURE.read_text())
            drifted[0] = "0123"
            path = Path(tmp) / "drift.json"
            path.write_text(json.dumps(drifted))
            digest = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "8362, 1058"):
                load_official_cases(2, json_path=path, expected_sha256=digest)
            drifted[0] = "8362"
            drifted[2] = "0123"
            path.write_text(json.dumps(drifted))
            digest = sha256_file(path)
            with self.assertRaisesRegex(ValueError, "locked protocol"):
                load_official_cases(20, json_path=path, expected_sha256=digest)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_official_cases(101)
        short = ROOT / "arm0_kimi" / "fixtures" / "arbench_gn_test.json"
        self.assertTrue(short.is_file())

    def test_default_path_is_checksummed_without_network(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AR0_GN_TEST_JSON", None)
            path = verify_official_file()["path"]
            self.assertEqual(sha256_file(Path(path)), EXPECTED_TEST_SHA256)


class QwenPayloadTests(unittest.TestCase):
    def test_request_body_knobs_and_empty_key(self):
        captured = []

        def transport(request):
            captured.append(request)
            return _completion("UNIQUE_PROSE\nGuess: 0123")

        env = {key: value for key, value in os.environ.items()
               if key not in {"VLLM_API_KEY", "MOONSHOT_API_KEY", "KIMI_API_KEY"}}
        with patch.dict(os.environ, env, clear=True):
            actor = QwenActor(transport=transport)
            response = actor.chat(
                [{"role": "user", "content": "ping"}],
                seed=2, arm="ledger", case_id="arbench-test-0000", turn=1,
            )
        self.assertEqual(response.text, "UNIQUE_PROSE\nGuess: 0123")
        request = captured[0]
        payload = json.loads(request.data.decode())
        self.assertEqual(set(payload), {"model", "messages", "seed", "temperature", "max_tokens"})
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 800)
        self.assertEqual(payload["seed"], 2)
        self.assertEqual(payload["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(request.get_header("Authorization"), "Bearer EMPTY")
        self.assertEqual(request.full_url, DEFAULT_QWEN_BASE_URL + "/chat/completions")
        self.assertNotIn("EMPTY", json.dumps(response.usage))

    def test_default_transport_emits_the_same_body_once(self):
        captured = []

        class Opener:
            def open(self, request, timeout=None):
                captured.append((request, timeout))
                return _Resp(_completion())

        with patch("arm0_kimi.client.build_opener", return_value=Opener()) as built:
            QwenActor(base_url="http://127.0.0.1:19999/v1", temperature=0.7, max_tokens=800).chat(
                [{"role": "user", "content": "ping"}],
                seed=0, arm="chrono", case_id="c", turn=1,
            )
        handler = built.call_args.args[0]
        self.assertEqual(handler.proxies, {})
        request, timeout = captured[0]
        self.assertEqual(timeout, 600.0)
        payload = json.loads(request.data.decode())
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 800)
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("top_p", payload)
        self.assertEqual(len(captured), 1)

    def test_transport_failure_does_not_retry_or_echo_the_key(self):
        calls = {"n": 0}

        class Opener:
            def open(self, request, timeout=None):
                calls["n"] += 1
                raise HTTPError(request.full_url, 503, "down", {}, io.BytesIO(b"dummy-secret-value"))

        with patch("arm0_kimi.client.build_opener", return_value=Opener()):
            actor = QwenActor(api_key="dummy-secret-value")
            with self.assertRaises(RuntimeError) as caught:
                actor.chat([{"role": "user", "content": "ping"}], seed=1, arm="ledger", case_id="c", turn=1)
        self.assertEqual(calls["n"], 1)
        self.assertNotIn("dummy-secret-value", str(caught.exception))

        class Down:
            def open(self, request, timeout=None):
                calls["n"] += 1
                raise URLError("timed out")

        with patch("arm0_kimi.client.build_opener", return_value=Down()):
            actor = QwenActor()
            with self.assertRaisesRegex(RuntimeError, "retries=0"):
                actor.chat([{"role": "user", "content": "ping"}], seed=1, arm="ledger", case_id="c", turn=1)
        self.assertEqual(calls["n"], 2)

    def test_refuses_moonshot_and_does_not_need_a_kimi_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Moonshot/Kimi"):
                QwenActor(base_url="https://api.moonshot.cn/v1")
            actor = QwenActor(transport=lambda request: _completion())
            actor.chat([{"role": "user", "content": "ping"}], seed=0, arm="chrono", case_id="c", turn=1)
        described = describe_qwen()
        self.assertTrue(described["wired"])
        self.assertFalse(described["sends_reasoning_effort"])
        self.assertFalse(described["sends_top_p"])
        self.assertFalse(described["requires_moonshot_key"])
        self.assertFalse(described["http_on_import"])

    def test_kimi_support_remains_and_is_not_called_live(self):
        captured = []

        def fake_urlopen(request, timeout=None):
            captured.append((request, timeout))
            return _Resp(_completion("Guess: 0123"))

        with patch.dict(os.environ, {"MOONSHOT_API_KEY": "kimi-test-key"}, clear=False):
            with patch("arm0_kimi.client.urlopen", fake_urlopen):
                actor = KimiActor("kimi-k3.1-pro")
                actor.chat(
                    [{"role": "user", "content": "ping"}],
                    seed=1, arm="chrono", case_id="c", turn=4,
                )
        payload = json.loads(captured[0][0].data.decode())
        self.assertEqual(payload["temperature"], 1.0)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(payload["seed"], 1)
        self.assertNotIn("top_p", payload)
        self.assertIn("api.moonshot.ai", captured[0][0].full_url)
        self.assertEqual(captured[0][0].get_header("Authorization"), "Bearer kimi-test-key")

        attempts = {"n": 0}

        def failing(request, timeout=None):
            attempts["n"] += 1
            raise URLError("timed out")

        with patch.dict(os.environ, {"KIMI_API_KEY": "kimi-test-key"}, clear=False):
            with patch("arm0_kimi.client.urlopen", failing), patch("arm0_kimi.client.time.sleep"):
                actor = KimiActor("kimi-k3.1-pro")
                with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
                    actor.chat([{"role": "user", "content": "ping"}], seed=0, arm="chrono", case_id="c", turn=1)
        self.assertEqual(attempts["n"], 3)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MOONSHOT_API_KEY"):
                KimiActor("kimi-k3.1-pro")


class SemanticsTests(unittest.TestCase):
    def test_feedback_matches_pinned_compare_guess_and_final_em_rule(self):
        secrets = json.loads(FIXTURE.read_text())
        self.assertEqual(len(CANDIDATES), 5040)
        for secret in secrets:
            for guess in list(LOCKED_N20_SECRETS) + secrets[:20]:
                observed = feedback(guess, secret)
                official = compare_guess(secret, guess)
                self.assertEqual(observed, official[:2])
                self.assertEqual(official[2], 2 * observed[0] + observed[1])
                self.assertNotEqual(observed, (3, 1))
                self.assertEqual(observed[0] == 4, guess == secret)
        from arm0_kimi.domain import extract_guess
        self.assertEqual(extract_and_convert_guess("before 1122 after Guess: 1123"), [1, 1, 2, 3])
        self.assertIsNone(extract_guess("Guess: 1123"))
        self.assertEqual(extract_guess("1234 then Guess: 5678"), "5678")

    def test_budget_final_em_and_no_singleton_substitution(self):
        case = load_official_cases(1)[0]
        self.assertEqual(case["secret"], "8362")
        config = RunConfig(arm="ledger", n=1, seeds=(0,), max_turns=25, case_source="official")
        record = run_episode("ledger", case, 0, MockActor(0), config)
        self.assertEqual(record["n_queries"], 25)
        self.assertEqual(len(record["calls"]), 26)
        self.assertEqual([call["final"] for call in record["calls"]], [False] * 25 + [True])
        self.assertEqual([call["turn"] for call in record["calls"]], list(range(1, 27)))
        self.assertTrue(record["primary_EM"])
        self.assertEqual(record["final_guess"], extract_field(record["final_raw_response"]))
        self.assertIn(record["final_guess"], record["final_raw_response"])
        # Mock policy solves. That is not a representation result.
        self.assertEqual(record["study_label"], "representation_baseline")
        self.assertFalse(record["context_evolution"])

        class Scripted:
            def __init__(self):
                self.usage_calls = []

            def chat(self, messages, *, seed, arm, case_id, turn, belief=None, final=False):
                from arm0_kimi.client import ChatResponse
                self.usage_calls.append(final)
                if final:
                    text = "I choose otherwise.\nGuess: 9876"
                elif turn == 1:
                    text = "UNIQUE_PROSE\nGuess: 8362"
                else:
                    text = "UNIQUE_PROSE\nGuess: 0123"
                usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                return ChatResponse(text=text, usage=usage, finish_reason="stop")

        short = RunConfig(arm="ledger_belief", n=1, seeds=(0,), max_turns=3, case_source="official")
        scripted = Scripted()
        solved_midway = run_episode("ledger_belief", case, 1, scripted, short)
        self.assertEqual(solved_midway["n_queries"], 3)
        self.assertEqual(len(solved_midway["calls"]), 4)
        self.assertEqual(scripted.usage_calls, [False, False, False, True])
        self.assertTrue(solved_midway["determined_before_final"])
        self.assertEqual(solved_midway["final_guess"], "9876")
        self.assertFalse(solved_midway["primary_EM"])
        self.assertTrue(solved_midway["determined_but_wrong_final"])
        self.assertNotEqual(solved_midway["final_guess"], case["secret"])
        same, _, _ = compare_guess(case["secret"], solved_midway["final_guess"])
        self.assertNotEqual(same, 4)

    def test_host_rendering_hides_raw_prose_and_does_not_label_the_secret(self):
        case = load_official_cases(1)[0]
        secret = case["secret"]

        class Recording(MockActor):
            def __init__(self):
                super().__init__(0)
                self.seen = []

            def chat(self, messages, *, seed, arm, case_id, turn, belief=None, final=False):
                self.seen.append({
                    "arm": arm,
                    "final": final,
                    "messages": messages,
                    "panel": None if belief is None else belief.panel(),
                })
                from arm0_kimi.client import ChatResponse
                text = "UNIQUE_PROSE line\nGuess: 0123"
                usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                self.usage.add(usage, arm=arm, case_id=case_id, turn=turn)
                return ChatResponse(text=text, usage=usage, finish_reason="stop")

        self.assertNotIn("DETERMINISTIC_BELIEF", prompt_for("ledger_belief", 25, [], final=False))
        for arm in ARMS:
            actor = Recording()
            config = RunConfig(arm=arm, n=1, seeds=(0,), max_turns=1, case_source="official")
            record = run_episode(arm, case, 0, actor, config)
            first, second = actor.seen
            blob = json.dumps(first["messages"])
            self.assertNotIn(secret, blob)
            self.assertNotIn("UNIQUE_PROSE", blob)
            user = second["messages"][1]["content"]
            system = second["messages"][0]["content"]
            self.assertNotIn(secret, system)
            if arm == "chrono":
                self.assertNotIn("UNIQUE_PROSE", user)
                self.assertIn("CHRONOLOGICAL HISTORY", user)
                self.assertNotIn("DETERMINISTIC_BELIEF", user)
            else:
                self.assertIn("UNIQUE_PROSE line", user)
                self.assertIn("LOSSLESS_NUMBERED_LEDGER", user)
                self.assertLess(user.index("[01] Q"), user.index("[01] A"))
                answer = feedback("0123", secret)
                self.assertIn(f"{answer[0]}A{answer[1]}B", user)
            if arm == "ledger_belief":
                self.assertIn("DETERMINISTIC_BELIEF", user)
                self.assertTrue(user.endswith(second["panel"]))
                panel = user.split("DETERMINISTIC_BELIEF\n", 1)[1]
                obj = json.loads(panel)
                self.assertEqual(obj["schema"], "gn_belief_v1")
                for key in ("secret", "answer", "gold", "target", "true_number"):
                    self.assertNotIn(key, obj)
                self.assertNotIn("the secret is", user.lower())
                self.assertEqual("DETERMINISTIC_BELIEF\n" + panel, second["panel"])
            else:
                self.assertNotIn("DETERMINISTIC_BELIEF", user)
            self.assertEqual(record["invalid_query_count"], 0)
            self.assertEqual(len(record["calls"]), 2)

    def test_invalid_guess_is_kept_and_does_not_prune(self):
        case = load_official_cases(1)[0]

        class Blank:
            def chat(self, messages, *, seed, arm, case_id, turn, belief=None, final=False):
                from arm0_kimi.client import ChatResponse
                return ChatResponse(text="I decline", usage={}, finish_reason="stop")

        config = RunConfig(arm="ledger", n=1, seeds=(0,), max_turns=1)
        record = run_episode("ledger", case, 0, Blank(), config)
        self.assertIsNone(record["final_guess"])
        self.assertFalse(record["primary_EM"])
        self.assertEqual(record["invalid_query_count"], 1)
        self.assertEqual(record["n_alive_end"], 5040)
        self.assertIsNone(record["repeat_guess"])
        self.assertIn("NO_VALID_GUESS", record["calls"][1]["messages"][1]["content"])
        self.assertEqual(record["public_feedback"][0]["feedback"], None)


def extract_field(text):
    from arm0_kimi.domain import extract_guess
    return extract_guess(text)


class SuccessBarTests(unittest.TestCase):
    def test_gate_requires_beating_ledger_and_a_consistency_move(self):
        # These aggregates are constructed to exercise the gate. They are not a run.
        passing = _arms(0.40, 0.60, 0.20, 0.30, 0.30, 0.40, 0.50, 0.70, 0.45)
        readout = representation_baseline_readout(passing, _per_seed(passing))
        self.assertTrue(readout["passes"])
        self.assertTrue(readout["beats_ledger"])
        self.assertEqual(readout["fail_closed_reasons"], [])
        self.assertFalse(readout["scientific_lift_claimed"])
        self.assertFalse(readout["context_evolution"])
        self.assertFalse(readout["diagnostics_are_success"])
        self.assertEqual(readout["control"], "ledger")
        self.assertEqual(readout["candidate"], "ledger_belief")
        self.assertEqual(readout["study_label"], "representation_baseline")

        diagnostics_only = _arms(0.40, 0.40, 0.20, 0.30, 0.10, 0.40, 0.50, 0.90, 0.45)
        refused = representation_baseline_readout(diagnostics_only, _per_seed(diagnostics_only))
        self.assertFalse(refused["passes"])
        self.assertIn("candidate_does_not_beat_ledger_control", refused["fail_closed_reasons"])
        self.assertNotIn("em_lift_without_consistency_move", refused["fail_closed_reasons"])

        chrono_only = _arms(0.70, 0.50, 0.20, 0.20, 0.10, 0.40, 0.40, 0.80, 0.30)
        chrono_readout = representation_baseline_readout(chrono_only, _per_seed(chrono_only))
        self.assertFalse(chrono_readout["passes"])
        self.assertIn("beats_chrono_alone_is_not_success", chrono_readout["fail_closed_reasons"])
        self.assertIn("lift_vs_chrono_loses_to_ledger", chrono_readout["fail_closed_reasons"])

        flat_repeat = _arms(0.40, 0.40, 0.20, 0.10, 0.50, 0.20, 0.60, 0.60, 0.40)
        flat = representation_baseline_readout(flat_repeat, _per_seed(flat_repeat))
        self.assertIn("em_flat_while_repeat_rises", flat["fail_closed_reasons"])
        self.assertFalse(flat["passes"])

        no_move = _arms(0.40, 0.55, 0.20, 0.20, 0.20, 0.20, 0.50, 0.50, 0.50)
        lifted = representation_baseline_readout(no_move, _per_seed(no_move))
        self.assertIn("em_lift_without_consistency_move", lifted["fail_closed_reasons"])
        self.assertFalse(lifted["passes"])

        incomplete = representation_baseline_readout(passing, _per_seed(passing, seeds=(0, 1)))
        self.assertIn("multi_seed_coverage_incomplete", incomplete["fail_closed_reasons"])
        self.assertFalse(incomplete["passes"])

    def test_mock_matrix_does_not_count_as_a_lift(self):
        with tempfile.TemporaryDirectory() as tmp:
            configs = [
                RunConfig(arm=arm, n=1, seeds=(0,), max_turns=1, backend="mock", case_source="synthetic")
                for arm in ARMS
            ]
            result = run_matrix(configs, Path(tmp))
        self.assertFalse(result["scientific_lift_claimed"])
        self.assertFalse(result["readout"]["passes"])
        self.assertIn("multi_seed_coverage_incomplete", result["readout"]["fail_closed_reasons"])
        self.assertTrue(result["mock_is_not_a_representation_result"])


class MatrixResumeTests(unittest.TestCase):
    def test_official_cases_are_fixed_across_arms_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("arm0_kimi.runner.make_cases", side_effect=AssertionError("synthetic fallback")):
                result = run_matrix([
                    RunConfig(
                        arm=arm, n=20, seeds=(0, 1, 2), max_turns=1,
                        backend="mock", case_source="official",
                    )
                    for arm in ARMS
                ], Path(tmp))
            self.assertEqual(result["case_source"], "official")
            self.assertEqual(result["official_sha256"], EXPECTED_TEST_SHA256)
            self.assertTrue(result["decoding_seeds_only"])
            self.assertEqual(result["case_ids"], [f"arbench-test-{index:04d}" for index in range(20)])
            self.assertEqual(result["executed_episodes"], 180)
            self.assertEqual(result["resumed_episodes"], 0)
            secrets = {}
            for arm in ARMS:
                for seed in (0, 1, 2):
                    rows = []
                    for case_id in result["case_ids"]:
                        path = Path(tmp) / "episodes" / f"{arm}__s{seed}__{case_id}.json"
                        record = json.loads(path.read_text())
                        rows.append(record["secret"])
                        self.assertEqual(record["case_source"], "official")
                        self.assertEqual(len(record["calls"]), 2)
                        self.assertEqual(record["protocol"]["official_sha256"], EXPECTED_TEST_SHA256)
                        self.assertIsNone(record["protocol"]["reasoning_effort"])
                        self.assertFalse(record["protocol"]["top_p_sent"])
                    secrets[(arm, seed)] = rows
            expected = list(LOCKED_N20_SECRETS)
            for key, rows in secrets.items():
                self.assertEqual(rows, expected, key)
            self.assertEqual(result["usage"]["calls"], 180 * 2)
            for arm in ARMS:
                self.assertEqual(result["by_arm"][arm]["episodes"], 60)

    def test_resume_skips_completed_episodes_and_does_not_double_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            configs = [
                RunConfig(arm="chrono", n=2, seeds=(0, 1), max_turns=1, case_source="official")
            ]
            first = run_matrix(configs, out)
            episode = next((out / "episodes").glob("*.json"))
            before = episode.read_bytes()
            writes = []
            real = __import__("arm0_kimi.runner", fromlist=["write_json"]).write_json

            def spy(path, value):
                writes.append(Path(path).name)
                real(path, value)

            with patch("arm0_kimi.runner.MockActor") as actor:
                actor.side_effect = AssertionError("completed episode was executed again")
                with patch("arm0_kimi.runner.write_json", spy):
                    second = run_matrix(configs, out)
            self.assertEqual(episode.read_bytes(), before)
            self.assertTrue(all(not name.startswith("chrono__") for name in writes))
            self.assertEqual(first["executed_episodes"], 4)
            self.assertEqual(second["executed_episodes"], 0)
            self.assertEqual(second["resumed_episodes"], 4)
            self.assertEqual(second["by_arm"]["chrono"]["episodes"], 4)
            self.assertEqual(second["usage"]["calls"], first["usage"]["calls"])
            self.assertEqual(len(list((out / "episodes").glob("*.json"))), 4)

    def test_crash_rerun_and_protocol_conflict(self):
        state = {"n": 0, "fail": True}

        class Flaky(MockActor):
            def chat(self, messages, *, seed, arm, case_id, turn, belief=None, final=False):
                state["n"] += 1
                if state["fail"] and state["n"] == 3:
                    raise RuntimeError("injected failure")
                return super().chat(
                    messages, seed=seed, arm=arm, case_id=case_id, turn=turn,
                    belief=belief, final=final,
                )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            configs = [RunConfig(arm="chrono", n=2, seeds=(0,), max_turns=1)]
            with patch("arm0_kimi.runner.MockActor", Flaky):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    run_matrix(configs, out)
                self.assertEqual(len(list((out / "episodes").glob("*.json"))), 1)
                state["fail"] = False
                recovered = run_matrix(configs, out)
            self.assertEqual(state["n"], 5)
            self.assertEqual(recovered["executed_episodes"], 1)
            self.assertEqual(recovered["resumed_episodes"], 1)
            self.assertEqual(recovered["by_arm"]["chrono"]["episodes"], 2)
            self.assertEqual(recovered["usage"]["calls"], 4)
            self.assertEqual(len(list((out / "episodes").glob("*.json"))), 2)

            path = next((out / "episodes").glob("*.json"))
            before = path.read_bytes()
            secret = json.loads(before)["secret"]
            shifted = [RunConfig(arm="chrono", n=2, seeds=(0,), max_turns=2)]
            with self.assertRaisesRegex(RuntimeError, "fresh --out") as caught:
                run_matrix(shifted, out)
            self.assertNotIn(secret, str(caught.exception))
            self.assertEqual(path.read_bytes(), before)

            path.write_text("{not json")
            rerun = run_matrix(configs, out)
            self.assertEqual(rerun["by_arm"]["chrono"]["episodes"], 2)
            self.assertEqual(len(list((out / "episodes").glob("*.json"))), 2)
            reloaded = json.loads(path.read_text())
            self.assertEqual(reloaded["status"], "complete")

            reloaded["status"] = "incomplete"
            path.write_text(json.dumps(reloaded))
            again = run_matrix(configs, out)
            self.assertEqual(again["by_arm"]["chrono"]["episodes"], 2)
            self.assertEqual(again["usage"]["calls"], 4)

            lied = json.loads(path.read_text())
            lied["primary_EM"] = not lied["primary_EM"]
            path.write_text(json.dumps(lied))
            fixed = run_matrix(configs, out)
            corrected = json.loads(path.read_text())
            self.assertEqual(corrected["primary_EM"], corrected["final_guess"] == corrected["secret"])
            self.assertEqual(fixed["by_arm"]["chrono"]["episodes"], 2)
            self.assertEqual(len(list((out / "episodes").glob("*.json"))), 2)

    def test_synthetic_default_differs_by_seed_and_official_cli_does_not_fall_back(self):
        self.assertNotEqual(
            [case["secret"] for case in make_cases(2, 0)],
            [case["secret"] for case in make_cases(2, 1)],
        )
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--n", "1", "--max-turns", "1", "--seeds", "0,1", "--arm", "chrono", "--out", tmp])
            self.assertEqual(code, 0)
            ids = sorted(path.name for path in (Path(tmp) / "episodes").glob("*.json"))
            self.assertTrue(any("synthetic-0-" in name for name in ids))
            self.assertTrue(any("synthetic-1-" in name for name in ids))
        with tempfile.TemporaryDirectory() as tmp:
            with patch("arm0_kimi.runner.make_cases", side_effect=AssertionError("synthetic fallback")):
                code = main([
                    "--backend", "mock", "--cases", "official", "--arm", "all",
                    "--n", "2", "--seeds", "0,1,2", "--max-turns", "1", "--out", tmp,
                ])
            self.assertEqual(code, 0)
            summary = json.loads((Path(tmp) / "run_summary.json").read_text())
            self.assertEqual(summary["case_ids"], ["arbench-test-0000", "arbench-test-0001"])
            self.assertEqual(summary["executed_episodes"], 18)
            self.assertFalse(summary["readout"]["passes"])
            self.assertFalse(summary["scientific_lift_claimed"])
            secrets = {
                json.loads(path.read_text())["secret"]
                for path in (Path(tmp) / "episodes").glob("*.json")
            }
            self.assertEqual(secrets, set(SMOKE_N2_SECRETS))

    def test_vllm_cli_sends_qwen_knobs_for_each_official_arm(self):
        captured = []

        def transport(self, request):
            captured.append(request)
            body = json.loads(request.data.decode())
            # Echo a stable guess. The first turn has no prose to preserve yet.
            return _completion("UNIQUE_PROSE\nGuess: 0123")

        env = {key: value for key, value in os.environ.items()
               if key not in {"VLLM_API_KEY", "MOONSHOT_API_KEY", "KIMI_API_KEY"}}
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, env, clear=True):
                with patch.object(QwenActor, "_default_transport", transport):
                    with patch("arm0_kimi.runner.make_cases", side_effect=AssertionError("synthetic fallback")):
                        code = main([
                            "--backend", "vllm", "--cases", "official", "--arm", "all",
                            "--n", "1", "--seed", "0", "--max-turns", "1", "--out", tmp,
                        ])
            self.assertEqual(code, 0)
            summary = json.loads((Path(tmp) / "run_summary.json").read_text())
        self.assertEqual(summary["model"], "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(summary["temperature_sent"], 0.7)
        self.assertEqual(summary["max_tokens_sent"], 800)
        self.assertIsNone(summary["reasoning_effort_sent"])
        self.assertFalse(summary["top_p_sent"])
        self.assertEqual(summary["case_ids"], ["arbench-test-0000"])
        self.assertEqual(len(captured), 6)
        classified = {"chrono": [], "ledger": [], "ledger_belief": []}
        for request in captured:
            payload = json.loads(request.data.decode())
            self.assertEqual(set(payload), {"model", "messages", "seed", "temperature", "max_tokens"})
            self.assertEqual(payload["temperature"], 0.7)
            self.assertEqual(payload["max_tokens"], 800)
            self.assertEqual(payload["seed"], 0)
            self.assertEqual(payload["model"], "Qwen/Qwen2.5-7B-Instruct")
            self.assertEqual(request.get_header("Authorization"), "Bearer EMPTY")
            self.assertTrue(request.full_url.startswith("http://127.0.0.1:18000/v1/chat/completions"))
            user = payload["messages"][1]["content"]
            self.assertNotIn("8362", payload["messages"][0]["content"])
            if "DETERMINISTIC_BELIEF" in user:
                classified["ledger_belief"].append(user)
            elif "CHRONOLOGICAL HISTORY" in user or user.startswith("No public guesses"):
                classified["chrono"].append(user)
            else:
                classified["ledger"].append(user)
        self.assertEqual({key: len(value) for key, value in classified.items()},
                         {"chrono": 2, "ledger": 2, "ledger_belief": 2})
        self.assertTrue(all("UNIQUE_PROSE" not in text for text in classified["chrono"]))
        self.assertTrue(any("UNIQUE_PROSE" in text for text in classified["ledger"]))
        self.assertTrue(any("UNIQUE_PROSE" in text for text in classified["ledger_belief"]))
        self.assertTrue(all("8362" not in text.split("DETERMINISTIC_BELIEF", 1)[0]
                            or "0123" in text
                            for text in classified["ledger_belief"]))
        # Turn 1, before any guess, does not contain the hidden secret.
        self.assertTrue(any("8362" not in text and "QUERY" in text for text in classified["ledger_belief"]))

    def test_cli_guards(self):
        self.assertEqual(main(["--n", "1"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--backend", "vllm", "--base-url", "https://api.moonshot.ai/v1",
                                   "--n", "1", "--max-turns", "1", "--out", tmp]), 2)
            self.assertEqual(list(Path(tmp).glob("episodes/*.json")), [])
            self.assertEqual(main(["--cases", "official", "--gn-json", str(Path(tmp) / "missing.json"),
                                   "--n", "1", "--max-turns", "1", "--out", tmp]), 2)
        config = RunConfig(arm="chrono", n=1, seeds=(0,), backend="vllm")
        self.assertEqual(config.base_url, "http://127.0.0.1:18000/v1")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 800)
        self.assertEqual(config.model, "Qwen/Qwen2.5-7B-Instruct")
        with self.assertRaises(ValueError):
            RunConfig(arm="ledger_belief_reflect", n=1, seeds=(0,))
        self.assertEqual(ARMS, ("chrono", "ledger", "ledger_belief"))
        sources = "\n".join(path.read_text() for path in (ROOT / "arm0_kimi").glob("*.py"))
        self.assertNotIn("import subprocess", sources)
        self.assertNotIn("os.system", sources)
        self.assertNotIn("sbatch", sources)


class InspectUnitTests(unittest.TestCase):
    def test_missing_file_is_a_rerun(self):
        config = RunConfig(arm="ledger", n=1, seeds=(0,))
        kind, record, why = inspect_episode(Path("/tmp/does-not-exist-arm0.json"), config,
                                            {"case_id": "synthetic-0-0000", "secret": "0123"}, 0)
        self.assertEqual((kind, record, why), ("rerun", None, "missing"))


if __name__ == "__main__":
    unittest.main()
