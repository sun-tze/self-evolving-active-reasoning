"""Synthetic mechanism checks only; no model/backend or research outcomes."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import portable_search as portable


class PortableSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((Path(__file__).parent/'protocol.json').read_bytes())

    def test_fixed_controller_skip_cache_and_freeze(self):
        reps = {str(rep):dict(train=[f'train-{rep}-{i}' for i in range(18)],
            selection=[f'selection-{rep}-{i}' for i in range(48)]) for rep in range(2)}
        cohorts = {str(c):[f'eval-{c}-{i}' for i in range(100)] for c in range(2)}
        seeds = {portable.seed('selected-'+run, rd, 'curator'):(run, rd)
                 for run in portable.RUNS for rd in range(6)}
        calls, episodes, train = [], [], {}

        def episode(p, call, feedback, cid, note, tag, meta):
            phase = meta['phase']
            if phase == 'evaluation':
                self.assertEqual(len(calls), 24)
                self.assertEqual(note, '' if meta['arm'] == 'B0' else f'- {meta["arm"]} gamma.')
            cost = (10 if meta['round'] == -1 else {0:9, 4:9, 5:8}[meta['round']]) if phase == 'selection' else 1
            # Distinct public traces expose crossed or incorrectly ordered batches.
            guess = f'{int(meta["run"][1])*100+meta["position"]:04d}' if phase == 'train' else '0123'
            row = dict(case_id=cid, metadata=deepcopy(meta), seed_tag=tag,
                public_history=[dict(turn=1, guess=guess, feedback=[0, 0], before=5040, after=1)],
                resolved=True, first_unique_turn=cost, capped25=cost, slots=[{} for _ in range(cost)])
            episodes.append(row)
            if phase == 'train':
                train[(meta['run'], meta['position'])] = row
            return row

        def call(req):
            self.assertEqual(set(req), {'model', 'messages', 'temperature', 'top_p', 'max_tokens', 'seed'})
            self.assertEqual((req['temperature'], req['top_p'], req['max_tokens']), (.7, .7, 512))
            run, rd = seeds[req['seed']]
            self.assertNotIn((run, rd), calls)
            calls.append((run, rd))
            content = req['messages'][0]['content']
            summaries = json.loads(content.split('\nPUBLIC TRAINING EPISODES:\n', 1)[1])
            expected = [{key:train[(run, i)][key] for key in ('public_history', 'resolved', 'capped25')}
                        for i in range(3*rd, 3*rd+3)] if run[0] == 'E' else []
            self.assertEqual(summaries, expected)
            text = [f'- {run} alpha.', 'KEEP', 'invalid', f'- {run} alpha.',
                    f'- {run} beta.', f'- {run} gamma.'][rd]
            return dict(text=text, finish_reason='stop')

        with patch.object(portable, 'run_episode', episode):
            search = portable.run_search(self.protocol, reps, call, lambda *_:self.fail('No real environment'))
            learning = list(episodes)
            evaluations = portable.evaluate(self.protocol, search, cohorts, call, lambda *_:None)
            bad = deepcopy(search)
            bad['policies']['E0'] = '- Mutated after freezing.'
            before = len(episodes)
            with self.assertRaisesRegex(ValueError, 'Frozen policies changed'):
                portable.evaluate(self.protocol, bad, cohorts, call, lambda *_:None)
            self.assertEqual(len(episodes), before)
        self.assertEqual(len(calls), 24)
        self.assertEqual(len(learning), 36+96+4*3*48)
        baseline = [r for r in learning if r['metadata']['phase'] == 'selection' and r['metadata']['round'] == -1]
        self.assertEqual(Counter(r['metadata']['run'] for r in baseline), {'B0':48, 'B1':48})
        self.assertEqual(Counter(r['metadata']['run'] for r in learning if r['metadata']['phase'] == 'train'), {'E0':18, 'E1':18})
        for run in portable.RUNS:
            rows = [r for r in search['decisions'] if r['run'] == run]
            self.assertEqual([r['incumbent_round'] for r in rows], [-1, 0, 0, 0, 0, 0])
            self.assertEqual([r['decision']['reason'] for r in rows], ['mean_gain_and_resolution_pass',
                'keep_requested', 'mechanically_invalid', 'byte_identical', 'selection_criteria_not_met', 'mean_gain_and_resolution_pass'])
            self.assertEqual([r['decision']['promoted'] for r in rows], [True, False, False, False, False, True])
            self.assertEqual(rows[-1]['decision']['metrics']['mean_paired_difference'], -1)
        self.assertEqual(Counter((r['metadata']['cohort'], r['metadata']['index'], r['metadata']['arm']) for r in evaluations),
            Counter((c, i, arm) for c in range(2) for i in range(100) for arm in portable.ARMS))
        self.assertEqual(len(evaluations), 1000)

    def test_parser_seed_malformed_slot_and_early_stop(self):
        self.assertEqual(portable.parse_as_original('0123456789'), '4567')
        self.assertEqual(portable.parse_as_original('Guess: ０１２３'), '0123')
        expected_seed = int.from_bytes(hashlib.sha256(b'sun-gn-v1\nfixture\ntag\n2').digest()[:8], 'big') % 2147483647
        self.assertEqual(portable.seed('fixture', 2, 'tag'), expected_seed)
        calls, feedback_calls = [], []
        def call(req):
            calls.append(deepcopy(req))
            return dict(text='no number' if len(calls) == 1 else 'Guess: 01234', finish_reason='stop')
        def feedback(cid, guess):
            feedback_calls.append((cid, guess))
            return (4, 0)
        row = portable.run_episode(self.protocol, call, feedback, 'fixture', '', 'tag', {})
        self.assertEqual((row['first_unique_turn'], row['capped25'], len(row['slots'])), (2, 2, 2))
        self.assertEqual(feedback_calls, [('fixture', '0123')])
        self.assertEqual(row['slots'][0]['before'], row['slots'][0]['after'])
        self.assertEqual(calls[0]['messages'][0]['content'], self.protocol['actor_prompts']['propose_template'].format(turn=25))
        self.assertEqual(calls[1]['seed'], expected_seed)
        self.assertEqual(calls[1]['messages'][-2]['content'], self.protocol['actor_prompts']['refine_prompt'])
        self.assertEqual(set(calls[0]), {'model', 'messages', 'temperature', 'top_p', 'max_tokens', 'seed'})

    def test_unresolved_consumes_exactly_twenty_five(self):
        calls = []
        def call(req):
            calls.append(req)
            return dict(text='invalid', finish_reason='length')
        row = portable.run_episode(self.protocol, call, lambda *_:self.fail('Malformed guess reached environment'),
                                   'fixture', '- A note.', 'tag', {})
        self.assertEqual(len(calls), 25)
        self.assertIsNone(row['first_unique_turn'])
        self.assertFalse(row['resolved'])
        self.assertEqual(row['capped25'], 25)
        self.assertEqual(calls[0]['messages'][0]['content'],
            self.protocol['actor_prompts']['propose_template'].format(turn=25)+self.protocol['actor_prompts']['header']+'- A note.')

    def test_call_failure_is_not_retried(self):
        calls = []
        def call(req):
            calls.append(req)
            raise RuntimeError('Synthetic transport failure')
        with self.assertRaisesRegex(RuntimeError, 'Synthetic transport failure'):
            portable.run_episode(self.protocol, call, lambda *_:None, 'fixture', '', 'tag', {})
        self.assertEqual(len(calls), 1)

    def test_new_public_environment_is_disjoint_deterministic_and_exact(self):
        from example_environment import make_environment, number_feedback
        from question_policy import UNIVERSE
        import random
        reps, cohorts, feedback = make_environment()
        again_reps, again_cohorts, again_feedback = make_environment()
        self.assertEqual((reps, cohorts), (again_reps, again_cohorts))
        ids = portable.check_cases(reps, cohorts)
        self.assertEqual(len(ids), 332)
        self.assertTrue(all(cid.startswith('public-replication-20261026-') for cid in ids))
        values = random.Random(20261026).sample(UNIVERSE, 332)
        self.assertEqual(len(set(values)), 332)
        self.assertEqual(len(UNIVERSE), 5040)
        self.assertIn('0123', UNIVERSE)
        for cid, value in zip(ids, values):
            self.assertEqual(feedback(cid, value), (4, 0))
            self.assertEqual(feedback(cid, '0000'), again_feedback(cid, '0000'))
        self.assertEqual(number_feedback('0123', '0000'), (1, 0))
        self.assertEqual(number_feedback('1023', '0000'), (1, 0))
        self.assertEqual(number_feedback('0123', '3210'), (0, 4))
        with self.assertRaises(ValueError):
            number_feedback('0123', '01234')


if __name__ == '__main__':
    unittest.main()
