import json
from pathlib import Path
import unittest
from unittest.mock import patch

from arm0_kimi.client import KimiActor
from arm0_kimi.domain import CANDIDATES, DeterministicBelief, Feedback, Turn, extract_guess, feedback, make_cases
from arm0_kimi.prompts import prompt_for, render_ledger
from arm0_kimi.runner import RunConfig, aggregate, run_episode
from arm0_kimi.smoke import main


class CoreTests(unittest.TestCase):
    def test_feedback_matches_ab_rule_and_parser(self):
        self.assertEqual(feedback("0123", "0132"), (2, 2))
        self.assertEqual(feedback("0123", "3012"), (0, 4))
        self.assertEqual(extract_guess("Try 1234 then Guess: 0123"), "0123")
        self.assertIsNone(extract_guess("Guess: 1123"))

    def test_all_feedback_types_and_prune_are_exact(self):
        for guess in CANDIDATES[:12] + CANDIDATES[-12:]:
            buckets = {}
            for secret in CANDIDATES:
                buckets.setdefault(feedback(guess, secret), set()).add(secret)
            for code, expected in buckets.items():
                belief = DeterministicBelief()
                belief.observe(guess, Feedback(*code))
                self.assertEqual(set(belief.feasible), expected)
                self.assertTrue(expected)

    def test_ledger_preserves_complete_raw_text_and_order(self):
        turns = [Turn(1, "Reason\nseveral lines\nGuess: 0123", "0123", Feedback(1, 2)),
                 Turn(2, "Guess: 4567", "4567", Feedback(0, 1))]
        rendered = render_ledger(turns)
        self.assertIn("Reason\nseveral lines\nGuess: 0123", rendered)
        self.assertLess(rendered.index("[01] Q"), rendered.index("[01] A"))
        self.assertLess(rendered.index("[01] A"), rendered.index("[02] Q"))

    def test_belief_fields_and_unasked_exclusions(self):
        belief = DeterministicBelief()
        belief.observe("0123", Feedback(0, 0))
        obj = belief.object()
        self.assertEqual(obj["exclusions"]["asked"], ["0123"])
        self.assertEqual(obj["exclusions"]["digits_not_in_any_feasible_secret"], list("0123"))
        self.assertTrue(obj["consistency"]["history_nonempty"])
        self.assertEqual(obj["pinned_positions"], {})

    def test_mock_episode_solves_without_final_singleton_substitution(self):
        case = make_cases(1, 0)[0]
        config = RunConfig(arm="ledger_belief", n=1, seeds=(0,), max_turns=6)
        from arm0_kimi.client import MockActor
        record = run_episode("ledger_belief", case, 0, MockActor(0), config)
        self.assertEqual(record["valid_query_count"], 6)
        self.assertTrue(record["determined_before_final"])
        self.assertTrue(record["primary_EM"])
        self.assertEqual(record["feedback_consistency"], 1.0)

    def test_missing_live_key_fails_before_network(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires MOONSHOT_API_KEY"):
                KimiActor("kimi-k3.1-pro")

    def test_cli_mock_mock_and_live_key_guard(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            code = main(["--n", "1", "--max-turns", "1", "--out", tmp])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads((Path(tmp) / "run_summary.json").read_text())["by_arm"]["ledger_belief"]["episodes"], 1)
        with patch.dict("os.environ", {}, clear=True):
            with tempfile.TemporaryDirectory() as live:
                self.assertEqual(main(["--backend", "kimi", "--n", "1", "--out", live]), 2)


if __name__ == "__main__":
    unittest.main()
