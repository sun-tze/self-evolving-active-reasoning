"""Tests for gn_state.py: `python -I -B test_gn_state.py -v` (standard library only).

The feedback check loads AR-Bench `utils_gn.py` (submodule at 9971322f, or
ARBENCH_ROOT) by path and pins its hash. Extract tests build synthetic
run_compare controllers; no private record, model or network is used.
"""
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gn_state as G  # noqa: E402

REPO = HERE.parents[3]
UTILS_GN = Path(os.environ.get('ARBENCH_ROOT', REPO / 'AR-Bench')) / 'arbench' / 'utils' / 'utils_gn.py'
UTILS_GN_SHA256 = 'f57cc0a77615859ea44fad07a0e5b849aa95e6d428963d72d3c49405c20274e7'


def random_guess(rng):
    if rng.random() < 0.2:
        return ''.join(rng.choice('0123456789') for _ in range(4))
    return rng.choice(G.UNIVERSE)


def history(secret, guesses):
    return [['', 0, 0] if g == '' else [g, *G.feedback(secret, g)] for g in guesses]


class Feedback(unittest.TestCase):
    def test_matches_arbench_compare_guess(self):
        if not UTILS_GN.exists():
            raise unittest.SkipTest('AR-Bench utils_gn.py not found; set ARBENCH_ROOT')
        self.assertEqual(hashlib.sha256(UTILS_GN.read_bytes()).hexdigest(), UTILS_GN_SHA256)
        spec = importlib.util.spec_from_file_location('pinned_utils_gn', UTILS_GN)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rng = random.Random(1)
        for _ in range(20000):
            secret, guess = rng.choice(G.UNIVERSE), random_guess(rng)
            expected = module.compare_guess(secret, guess)[:2]
            self.assertEqual(G.feedback(secret, guess), expected)
            self.assertEqual(G.row(guess)[G.INDEX[secret]], G.code(*expected))

    def test_every_first_guess_splits_the_universe_alike(self):
        def squares(guess):
            counts = {}
            for c in G.row(guess):
                counts[c] = counts.get(c, 0) + 1
            return sorted(counts.values())
        rng = random.Random(2)
        for guess in rng.sample(G.UNIVERSE, 5):
            self.assertEqual(squares(guess), squares(G.UNIVERSE[0]))


class Histories(unittest.TestCase):
    def test_rows_match_naive_recomputation(self):
        rng = random.Random(3)
        for _ in range(30):
            secret, guesses = rng.choice(G.UNIVERSE), []
            for _ in range(rng.randint(1, 9)):
                roll = rng.random()
                if roll < 0.1:
                    guesses.append('')
                elif roll < 0.25 and [g for g in guesses if g]:
                    guesses.append(rng.choice([g for g in guesses if g]))
                elif roll < 0.4:
                    guesses.append(secret)
                else:
                    guesses.append(random_guess(rng))
            logged = history(secret, guesses)
            rows = G.history_rows(logged, secret)
            seen = []
            for entry, row in zip(logged, rows):
                if entry[0] == '':
                    self.assertFalse(row['parsed'])
                    continue
                state = [u for u in G.UNIVERSE
                         if all(G.feedback(u, g) == (b, c) for g, b, c in seen)]
                seen.append(entry)
                after = [u for u in state if G.feedback(u, entry[0]) == tuple(entry[1:])]
                self.assertEqual((row['before'], row['after']), (len(state), len(after)))
                self.assertEqual(row['consistent'], entry[0] in state)
                self.assertEqual(row['repeat'], entry[0] in [g for g, _, _ in seen[:-1]])
                self.assertEqual(row['informed'], len(seen) > 1)
                self.assertEqual(row['hit'], entry[0] == secret)

    def test_summary_of_a_handmade_history(self):
        summary = G.summarize(G.history_rows(history('0123', ['4567', '4567', '', '1111', '0123']), '0123'))
        self.assertEqual(summary['logged_guesses'], 5)
        self.assertEqual(summary['parse_failures'], 1)
        self.assertEqual(summary['informed_guesses'], 3)
        self.assertEqual(summary['consistent_informed'], 1)
        self.assertEqual((summary['repeats'], summary['non_distinct']), (1, 1))
        self.assertEqual((summary['identified_turn'], summary['first_hit_turn']), (5, 5))
        self.assertTrue(summary['last_guess_correct'])
        self.assertFalse(summary['identified_but_last_wrong'])
        self.assertAlmostEqual(summary['bits_after_questions'], G.math.log2(5040))

    def test_readout_failure_is_counted(self):
        rng = random.Random(4)
        guesses = rng.sample([u for u in G.UNIVERSE if u != '5678'], 25) + ['0000']
        summary = G.summarize(G.history_rows(history('5678', guesses), '5678'))
        self.assertIsNotNone(summary['identified_turn'])
        self.assertFalse(summary['last_guess_correct'])
        self.assertTrue(summary['identified_but_last_wrong'])

    def test_rejects_bad_records(self):
        with self.assertRaises(ValueError):
            G.history_rows([['1234', 1, 0]], '1234')
        with self.assertRaises(ValueError):
            G.history_rows([['1234', 0, 0], ['5678', 0, 0], ['9012', 0, 0]])
        for entry in (['12345', 0, 0], ['1234', 3, 2], ['1234', '0', 0], ['1234', 0]):
            with self.assertRaises(ValueError):
                G.history_rows([entry])


class Reference(unittest.TestCase):
    def test_policies_on_a_subset(self):
        secrets = G.UNIVERSE[::97]
        first = G.reference(secrets)
        self.assertEqual(first, G.reference(secrets))
        by = {r['policy']: r for r in first}
        for name in ('consistent_first', 'consistent_greedy_expected_size', 'mixture_p1'):
            self.assertEqual(by[name]['hit_within_26_guesses'], len(secrets))
            self.assertEqual(by[name]['measured_consistency_rate'], 1.0)
            self.assertLessEqual(by[name]['guesses_to_hit_max'], 9)
        self.assertLess(by['mixture_p0']['measured_consistency_rate'], 0.1)
        for r in first:
            self.assertEqual(r['identified_after_25_questions'], len(secrets))


def save(directory, relative, value, hashes):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True) + '\n'
    path.write_text(data, encoding='utf-8', newline='\n')
    hashes[relative] = hashlib.sha256(data.encode('utf-8')).hexdigest()
    return {'path': relative, 'sha256': hashes[relative]}


def build_controller(directory, cases, evaluations, updates=()):
    """evaluations: (split, checkpoint, arm, case_id, guesses or None for a failed slot)."""
    directory.mkdir()
    hashes, index, rows, update_rows = {}, 0, [], []
    save(directory, 'private-cases.json', cases, hashes)

    def episode(case_id, guesses):
        nonlocal index
        index += 1
        if guesses is None:
            value = {'status': 'execution_failure', 'case_id': case_id, 'official_result': None, 'metrics': None}
        else:
            logged = history(cases[case_id], guesses)
            value = {'status': 'complete', 'case_id': case_id,
                     'official_result': {'guess_list': logged, 'true_number': cases[case_id]},
                     'metrics': {'official_success': bool(logged) and logged[-1][1] == 4}}
        return value['status'], save(directory, f'episodes/{index:05d}.json', value, hashes)

    for split, checkpoint, arm, case_id, guesses in evaluations:
        status, ref = episode(case_id, guesses)
        rows.append({'split': split, 'checkpoint': checkpoint, 'arm': arm, 'case_id': case_id,
                     'status': status, 'episode': ref})
    for arm, step, case_id, guesses in updates:
        _, ref = episode(case_id, guesses)
        update_rows.append({'arm': arm, 'step': step, 'case_id': case_id, 'initial_episode': ref})
    result = {'evaluations': rows, 'updates': update_rows, 'artifact_sha256': dict(hashes)}
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    return directory


class Extract(unittest.TestCase):
    CASES = {'case-alpha': '0123', 'case-bravo': '4567', 'case-charlie': '8901'}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = random.Random(5)
        self.lucky = rng.sample(G.UNIVERSE, 6) + ['0123']
        self.repeating = ['1357'] * 25 + ['2468']

    def controller(self, name, evaluations, updates=()):
        return build_controller(self.root / name, self.CASES, evaluations, updates)

    def test_aggregates_are_anonymous_and_match_direct_computation(self):
        c = self.controller('run', [
            ('final', 6, 'ZeroShot', 'case-alpha', self.lucky),
            ('final', 6, 'ZeroShot', 'case-bravo', self.repeating),
            ('final', 6, 'ZeroShot', 'case-charlie', None),
            ('dev', 0, 'DC_Cumulative', 'case-bravo', ['4567'])],
            updates=[('DC_Cumulative', 1, 'case-charlie', ['0123', '8901'])])
        value, case_ids = G.extract([c])
        G.leak_check(value, case_ids)
        text = json.dumps(value)
        for private in [*self.CASES, *self.CASES.values(), '1357', '2468']:
            self.assertNotIn(private, text)
        groups = {(g['phase'], g['checkpoint'], g['arm']): g for g in value['groups']}
        self.assertEqual(list(groups), [('train', 1, 'DC_Cumulative'), ('dev', 0, 'DC_Cumulative'),
                                        ('final', 6, 'ZeroShot')])
        final = groups[('final', 6, 'ZeroShot')]
        direct = G.aggregate([G.summarize(G.history_rows(history(s, g), s))
                              for s, g in (('0123', self.lucky), ('4567', self.repeating))])
        for key, expected in direct.items():
            self.assertEqual(final[key], expected, key)
        self.assertEqual((final['missing'], final['official_success_episodes']), (1, 1))
        self.assertEqual(final['last_guess_disagrees_with_official'], 0)
        self.assertEqual(final['repeat_rate'], 24 / 33)

    def test_accepts_list_form_native_target(self):
        c = self.controller('run', [('final', 6, 'UoT', 'case-alpha', self.lucky)])
        path = c / 'episodes' / '00001.json'
        episode = json.loads(path.read_text(encoding='utf-8'))
        episode['official_result']['true_number'] = [0, 1, 2, 3]
        result = json.loads((c / 'result.json').read_text(encoding='utf-8'))
        ref = result['evaluations'][0]['episode']
        ref['sha256'] = save(c, ref['path'], episode, result['artifact_sha256'])['sha256']
        (c / 'result.json').write_text(json.dumps(result), encoding='utf-8', newline='\n')
        value, _ = G.extract([c])
        self.assertEqual(value['groups'][0]['last_guess_correct_episodes'], 1)

    def test_rejects_tampered_episode(self):
        c = self.controller('run', [('final', 6, 'ZeroShot', 'case-alpha', self.lucky)])
        path = c / 'episodes' / '00001.json'
        path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8', newline='\n')
        with self.assertRaisesRegex(ValueError, 'binding'):
            G.extract([c])

    def test_rejects_feedback_that_does_not_match_the_case(self):
        c = self.controller('run', [('final', 6, 'ZeroShot', 'case-alpha', self.lucky)])
        cases = dict(self.CASES, **{'case-alpha': '3210'})
        data = json.dumps(cases, indent=2, sort_keys=True) + '\n'
        (c / 'private-cases.json').write_text(data, encoding='utf-8', newline='\n')
        result = json.loads((c / 'result.json').read_text(encoding='utf-8'))
        result['artifact_sha256']['private-cases.json'] = hashlib.sha256(data.encode()).hexdigest()
        (c / 'result.json').write_text(json.dumps(result), encoding='utf-8', newline='\n')
        with self.assertRaises(ValueError):
            G.extract([c])

    def test_one_complete_record_per_slot_across_sources(self):
        first = self.controller('first', [('final', 6, 'ZeroShot', 'case-alpha', None)])
        second = self.controller('second', [('final', 6, 'ZeroShot', 'case-alpha', self.lucky)])
        value, _ = G.extract([first, second])
        self.assertEqual((value['groups'][0]['episodes'], value['groups'][0]['missing']), (1, 0))
        with self.assertRaisesRegex(ValueError, 'Two complete records'):
            G.extract([second, second])

    def test_leak_check_rejects_private_strings(self):
        c = self.controller('run', [('final', 6, 'ZeroShot', 'case-alpha', self.lucky)])
        value, case_ids = G.extract([c])
        for arm in ('case-alpha', 'arm_0123', 'two words'):
            bad = json.loads(json.dumps(value))
            bad['groups'][0]['arm'] = arm
            with self.assertRaises(ValueError):
                G.leak_check(bad, case_ids | {'case-alpha'})


if __name__ == '__main__':
    unittest.main()
