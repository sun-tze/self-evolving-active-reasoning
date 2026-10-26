"""Portable synthetic checks; no experimental outcome fixtures."""
from copy import deepcopy
from collections import Counter
import math
import random
import statistics
import unittest

from paired_statistics import paired_statistics, percentile
from question_policy import PublicMonitor, validate_note
from verify_results import ARMS, anonymous_conditions, compare, SCHEMA


class PublicChecks(unittest.TestCase):
    def test_monitor_multiset_repeated_digits_and_malformed(self):
        monitor = PublicMonitor()
        monitor.observe(None, None)
        self.assertEqual(len(monitor.candidates), 5040)
        monitor.observe('0000', [1, 0])
        self.assertEqual(len(monitor.candidates), 4*9*8*7)
        remaining = []
        for candidate in monitor.candidates:
            same = sum(a == b for a, b in zip(candidate, '0123'))
            common = sum((Counter(candidate) & Counter('0123')).values())
            if [same, common-same] == [4, 0]: remaining.append(candidate)
        monitor.observe('0123', [4, 0])
        self.assertEqual(monitor.candidates, remaining)
        self.assertEqual(remaining, ['0123'])

    def test_numeric_strategies_allowed_truncated_note_rejected(self):
        self.assertTrue(validate_note('- Consider opening with 0123.', 'stop')['accepted'])
        self.assertFalse(validate_note('- Consider opening with 0123.', 'length')['accepted'])

    def test_paired_bootstrap_matches_manual_case_block_draws(self):
        values = dict(B0=[10, 20, 25], S=[12, 19, 25], E0=[4, 20, 25], E1=[7, 17, 25], E2=[10, 14, 25])
        conditions = {arm:[dict(case_id=str(i), capped25=v, penalized26=v) for i, v in enumerate(costs)] for arm, costs in values.items()}
        result = paired_statistics(conditions, replicates=100)
        differences = [-3, -3, 0]
        rng = random.Random(62026)
        draws = [sum(differences[rng.randrange(3)] for _ in range(3))/3 for _ in range(100)]
        self.assertEqual(result['primary']['mean_difference'], statistics.mean(differences))
        self.assertEqual(result['primary']['CI95'], [percentile(draws, .025), percentile(draws, .975)])
        conditions['E1'].reverse()
        with self.assertRaises(ValueError): paired_statistics(conditions, replicates=1)

    def test_schema_derives_unresolved_penalty_and_rejects_private_fields(self):
        arm = dict(first_unique_turn=None, queries=25, invalid=25, truncated=0, valid_queries=0,
                   information_bits=0, input_tokens=100, output_tokens=25)
        document = dict(schema=SCHEMA, N=100,
                        cases=[dict(case_index=i, arms={name:deepcopy(arm) for name in ARMS}) for i in range(100)], expected={})
        rows = anonymous_conditions(document)
        self.assertEqual(rows['B0'][0]['capped25'], 25)
        self.assertEqual(rows['B0'][0]['penalized26'], 26)
        document['cases'][0]['arms']['B0']['secret'] = '0123'
        with self.assertRaises(ValueError): anonymous_conditions(document)

    def test_result_comparison_rejects_changed_estimate(self):
        compare({'x':[-.5, 1]}, {'x':[-.5, 1]})
        with self.assertRaises(ValueError): compare({'x':[-.4, 1]}, {'x':[-.5, 1]})


if __name__ == '__main__':
    unittest.main()
