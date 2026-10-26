import unittest
from constraint_readout import feedback, resolve_public, public_state


class ConstraintReadoutTests(unittest.TestCase):
    def test_repeated_guess_uses_multiset_intersection(self):
        self.assertEqual(feedback("0123", "0011"), (1, 1))

    def test_no_evidence_abstains(self):
        self.assertEqual(resolve_public([]), {"status": "ambiguous", "candidate_count": 5040, "answer": None})

    def test_leading_zero_is_preserved(self):
        result = resolve_public([dict(turn=1, guess="0123", same=4, different=0)])
        self.assertEqual(result, dict(status="unique", candidate_count=1, answer="0123"))

    def test_conflicting_records_fail_closed(self):
        result = resolve_public([dict(turn=1, guess="0123", same=4, different=0), dict(turn=2, guess="0123", same=0, different=0)])
        self.assertEqual(result["status"], "inconsistent")
        self.assertIsNone(result["answer"])

    def test_rejects_label_fields(self):
        with self.assertRaises(ValueError):
            resolve_public([dict(turn=1, guess="0123", same=4, different=0, secret="0123")])

    def test_rejects_invalid_public_records(self):
        for row in [dict(turn=1, guess="0123", same=3, different=2), dict(turn=1, guess="１２３４", same=1, different=0)]:
            with self.assertRaises(ValueError):
                resolve_public([row])

    def test_state_limit_does_not_select_subset(self):
        self.assertEqual(public_state([])['candidate_count'], 5040)
        self.assertIsNone(public_state([])['candidates'])
        state = public_state([dict(turn=1, guess='0123', same=4, different=0)])
        self.assertEqual(state['candidates'], ['0123'])


if __name__ == "__main__":
    unittest.main()
