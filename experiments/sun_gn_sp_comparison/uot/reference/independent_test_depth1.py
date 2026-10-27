"""Independent, offline acceptance of Claude Code's depth1_batch prototype.

Run with the Python standard library only. Reference functions are extracted from
hash-pinned source ASTs; their imports/CLI and live transport are never executed.
No network, model, GPU, subprocess, or real dispatch is used. This checks successful
simulation request/reducer equivalence, not live serving or failure-trace equality.
Written independently; implementation and reference files remain unmodified.
"""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
import unittest


ROOT = Path(__file__).resolve().parent
REFERENCE_PINS = {
    'gn_evaluator.py': 'aa6643d004ed7b510452ce9fe581b371112616a6702dfee3a26132759cb221a9',
    'utils_gn.py': 'f57cc0a77615859ea44fad07a0e5b849aa95e6d428963d72d3c49405c20274e7',
    'prompt.py': '5dafd196d5542fad163898fbfecca411fb13b1b30b636ebc0577d38a7bbcc713',
}
PROTOTYPE_PIN = 'e1ea6f02174393c042455ab85bbf4dac9aa3efcc27862b758b4ca9a6243486e0'
FUNCTIONS = {
    '_gn_calculate_remaining_with_model',
    '_gn_estimate_uncertainty_reduction',
    '_gn_simulate_rounds',
    '_gn_analyze_candidates_with_simulation',
}


def pinned_bytes(path, expected):
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise AssertionError('Source pin mismatch: ' + str(path) + ': ' + actual)
    return data


def load_prototype():
    path = ROOT / 'depth1_batch.py'
    pinned_bytes(path, PROTOTYPE_PIN)
    spec = importlib.util.spec_from_file_location('independent_depth1_subject', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass resolves its defining module
    spec.loader.exec_module(module)
    return module


def extract_functions(filename, names, namespace):
    data = pinned_bytes(ROOT / 'reference' / filename, REFERENCE_PINS[filename])
    tree = ast.parse(data.decode('utf-8'), filename=filename)
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError('Missing reference function in ' + filename)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)


def reference_namespace(rng, inference):
    ns = dict(Counter=Counter, List=List, Tuple=Tuple, Dict=Dict,
              Optional=Optional, random=rng, inference=inference, re=re,
              math=math, POLICY_API_KEY=None, POLICY_BASE_URL=None,
              print=lambda *args, **kwargs: None)
    extract_functions('utils_gn.py', {'compare_guess'}, ns)
    extract_functions('gn_evaluator.py', FUNCTIONS, ns)
    return ns


class TracedRandom(random.Random):
    """Record populations and results, without changing Random's implementation."""
    def __init__(self, seed):
        super().__init__(seed)
        self.trace = []

    def sample(self, population, k, **kwargs):
        result = super().sample(population, k, **kwargs)
        normalize = lambda p: tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in p)
        self.trace.append(('sample', normalize(population), k, normalize(result)))
        return result

    def choice(self, population):
        result = super().choice(population)
        self.trace.append(('choice', tuple(population), result))
        return result


def tape_payload(index):
    # Includes no match, duplicates, Unicode decimal digits, chunks of longer
    # runs, repeated digits, ordinary valid guesses, and arbitrary surrounding text.
    texts = ('No candidate.', '0123 1023 4567', '9999 9999 99999999',
             '٠١٢٣ ٩٩٩٩ x0123y', '[0, 1, 2, 3] 123 12345', '')
    return texts[index % len(texts)], ('length' if index % 2 else 'stop'), 17 + index, index % 13


def response_for(request, payload):
    text, finish_reason, prompt_tokens, completion_tokens = payload
    return dict(logical_id=request['logical_id'], request_sha256=request['request_sha256'],
                text=text, finish_reason=finish_reason,
                usage=dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


class Depth1Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for filename, pin in REFERENCE_PINS.items():
            pinned_bytes(ROOT / 'reference' / filename, pin)
        cls.subject = load_prototype()

    def test_reference_and_implementation_pins(self):
        manifest = json.loads((ROOT / 'reference-hashes.json').read_text(encoding='utf-8-sig'))
        for filename, pin in REFERENCE_PINS.items():
            self.assertEqual(manifest[filename], pin)
            pinned_bytes(ROOT / 'reference' / filename, pin)
        pinned_bytes(ROOT / 'depth1_batch.py', PROTOTYPE_PIN)

    def compare_native(self, answers, candidates, *, seed=41, prelude=False, payload=tape_payload):
        native_rng, planned_rng = TracedRandom(seed), TracedRandom(seed)
        if prelude:
            # The planner receives a used episode RNG, not a fresh seed. Model
            # seed remains the episode seed even after these candidate-fill draws.
            for rng in (native_rng, planned_rng):
                rng.choice(range(10))
                rng.sample(range(10), 4)
        native_calls = []

        def inference(messages, **kwargs):
            index = len(native_calls)
            self.assertEqual(kwargs['model'], 'offline-reference-model')
            self.assertIsNone(kwargs['api_key'])
            self.assertIsNone(kwargs['base_url'])
            native_calls.append(dict(messages=deepcopy(messages),
                                     temperature=kwargs['temperature'], top_p=kwargs['top_p']))
            text, finish_reason, prompt_tokens, completion_tokens = payload(index)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                                   usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                                                         completion_tokens=completion_tokens))

        native_candidates = deepcopy(candidates)
        planned_candidates = deepcopy(candidates)
        ns = reference_namespace(native_rng, inference)
        native = ns['_gn_analyze_candidates_with_simulation'](
            deepcopy(answers), native_candidates, depth=1,
            model='offline-reference-model', temperature=0.1, top_p=0.9)
        plan = self.subject.plan_depth1(deepcopy(answers), planned_candidates,
                                        rng=planned_rng, seed=seed, first_call_index=2, depth=1)
        requests = plan.requests()
        self.assertEqual(len(requests), len(native_calls))
        self.assertEqual([dict(messages=r['messages'], temperature=r['temperature'], top_p=r['top_p'])
                          for r in requests], native_calls)
        for index, request in enumerate(requests):
            self.assertEqual(request['serial_call_index'], index + 2)
            self.assertEqual(request['seed'], seed)
            self.assertEqual(request['max_tokens'], 512)
            self.assertEqual(request['role'], 'uot_simulation')
            self.assertEqual(request['temperature'], 0.1)
            self.assertEqual(request['top_p'], 0.9)
        self.assertEqual(planned_rng.trace, native_rng.trace)
        self.assertEqual(planned_rng.getstate(), native_rng.getstate())
        responses = [response_for(request, payload(index)) for index, request in enumerate(requests)]
        actual = self.subject.reduce_depth1(plan, list(reversed(responses)))
        self.assertEqual(actual, native)
        # Equality of duplicate guess values alone cannot prove stable sorting.
        original_positions = lambda scores, original: [next(i for i, c in enumerate(original) if c is row[0])
                                                       for row in scores]
        self.assertEqual(original_positions(actual[0], planned_candidates),
                         original_positions(native[0], native_candidates))
        return plan, actual, requests, planned_rng

    def test_hand_calculated_order_parser_and_duplicate_identity(self):
        answers = [[0, 1, 2, 3], [1, 0, 2, 3], [4, 5, 6, 7]]
        candidates = [[0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1]]
        hostile = '9999 9999 99999999 \u0669\u0669\u0669\u0669 x9999y'

        def payload(index):
            return (hostile if index == 2 else ''), 'length', index + 1, 1

        plan, actual, requests, rng = self.compare_native(answers, candidates, payload=payload)
        self.assertEqual([r['partition_key'] for r in requests],
                         ['4_0', '2_2', '0_0', '4_0', '2_2', '0_0', '1_0', '0_0'])
        self.assertEqual(len(requests), 8)
        self.assertEqual(len(rng.trace), 8)
        self.assertEqual(self.subject.parse_remaining(hostile, (0, 1, 2, 3), 0, 0),
                         [[9, 9, 9, 9]] * 6)
        self.assertEqual([row[1] for row in actual[0]], [3.0, 3.0, 1.0])
        self.assertIs(actual[0][0][0], plan.candidates[1])
        self.assertIs(actual[0][1][0], plan.candidates[2])
        self.assertIs(actual[0][2][0], plan.candidates[0])
        self.assertEqual(actual[0][2][2], {'4_0': 3, '2_2': 3, '0_0': -3})
        self.assertEqual(actual[1:], (36, 8))

    def test_all_ties_preserve_proposal_order(self):
        answers = [[0, 1, 2, 3], [1, 0, 2, 3], [4, 5, 6, 7]]
        candidates = [[0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1]]
        plan, actual, _, _ = self.compare_native(answers, candidates,
                                                payload=lambda i: ('', 'stop', 2, 0))
        self.assertEqual([row[1] for row in actual[0]], [3.0, 3.0, 3.0])
        for index, row in enumerate(actual[0]):
            self.assertIs(row[0], plan.candidates[index])

    def test_empty_and_singleton_make_no_dispatch_or_rng_draw(self):
        for answers in ([], [[0, 1, 2, 3]]):
            with self.subTest(size=len(answers)):
                candidates = [[0, 1, 2, 3], [0, 1, 2, 3], [9, 9, 9, 9]]
                plan, actual, requests, rng = self.compare_native(answers, candidates)
                self.assertEqual(requests, [])
                self.assertEqual(rng.trace, [])
                self.assertEqual(actual[1:], (0, 0))
                self.assertEqual([(row[1], row[2]) for row in actual[0]], [(0.0, {})] * 3)
                fresh = random.Random(41)
                before = fresh.getstate()

                def forbidden_dispatch(requests):
                    self.fail('Empty/singleton simulation must never dispatch')

                self.subject.run_depth1(answers, candidates, rng=fresh, seed=41,
                                        first_call_index=2, depth=1, dispatch=forbidden_dispatch)
                self.assertEqual(fresh.getstate(), before)

    def test_50_and_51_sample_boundary_after_candidate_rng_draws(self):
        for size in (50, 51):
            with self.subTest(size=size):
                answers = [list(p) for p in itertools.islice(itertools.permutations(range(10), 4), size)]
                _, _, requests, rng = self.compare_native(
                    answers, [[0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1]], prelude=True)
                samples = rng.trace[2:]
                self.assertEqual(len(samples), len(requests))
                self.assertTrue(all(draw[0] == 'sample' and draw[2] == 50 for draw in samples))
                for request in requests:
                    text = request['messages'][0]['content']
                    self.assertIn(f'showing 50 out of {size} total', text)
                    self.assertEqual('... (and 1 more possible answers)' in text, size == 51)

    def test_full_5040_answer_space(self):
        answers = [list(p) for p in itertools.permutations(range(10), 4)]
        self.assertEqual(len(answers), 5040)
        _, _, requests, _ = self.compare_native(
            answers, [[0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1]], prelude=True)
        self.assertTrue(requests)
        self.assertTrue(all('showing 50 out of 5040 total' in r['messages'][0]['content']
                            and '... (and 4990 more possible answers)' in r['messages'][0]['content']
                            for r in requests))

    def test_bad_responses_fail_without_redispatch(self):
        def mutate(kind, responses):
            if kind == 'missing':
                responses.pop()
            elif kind == 'duplicate':
                responses.append(deepcopy(responses[0]))
            elif kind == 'unknown_id':
                responses[0]['logical_id'] = 'not-a-request'
            elif kind == 'wrong_hash':
                responses[0]['request_sha256'] = '0' * 64
            elif kind == 'swapped_hash':
                responses[0]['request_sha256'] = responses[1]['request_sha256']
            elif kind == 'negative_usage':
                responses[0]['usage']['prompt_tokens'] = -1
            elif kind == 'boolean_usage':
                responses[0]['usage']['completion_tokens'] = True
            elif kind == 'cap_exceeded':
                responses[0]['usage']['completion_tokens'] = 513
            elif kind == 'invalid_finish':
                responses[0]['finish_reason'] = 'error'
            elif kind == 'invalid_text':
                responses[0]['text'] = None
            return responses

        kinds = ('missing', 'duplicate', 'unknown_id', 'wrong_hash', 'swapped_hash',
                 'negative_usage', 'boolean_usage', 'cap_exceeded', 'invalid_finish', 'invalid_text')
        for kind in kinds:
            with self.subTest(kind=kind):
                calls = []

                def dispatch(requests):
                    calls.append(len(requests))
                    return mutate(kind, [response_for(r, ('', 'stop', 4, 0)) for r in requests])

                with self.assertRaises(self.subject.BatchFailure):
                    self.subject.run_depth1([[0, 1, 2, 3], [4, 5, 6, 7]],
                        [[0, 1, 2, 3], [1, 0, 2, 3], [9, 9, 9, 9]],
                        rng=random.Random(1), seed=1, first_call_index=2, depth=1, dispatch=dispatch)
                self.assertEqual(len(calls), 1)

    def test_dispatch_exception_is_not_retried(self):
        calls = []

        def dispatch(requests):
            calls.append(len(requests))
            raise RuntimeError('synthetic offline failure')

        with self.assertRaises(self.subject.BatchFailure):
            self.subject.run_depth1([[0, 1, 2, 3], [4, 5, 6, 7]],
                [[0, 1, 2, 3], [1, 0, 2, 3], [9, 9, 9, 9]],
                rng=random.Random(1), seed=1, first_call_index=2, depth=1, dispatch=dispatch)
        self.assertEqual(len(calls), 1)

    def test_request_copy_isolation_and_changed_candidate_rejection(self):
        candidates = [[0, 1, 2, 3], [1, 0, 2, 3], [9, 9, 9, 9]]
        plan = self.subject.plan_depth1([[0, 1, 2, 3], [4, 5, 6, 7]], candidates,
                                       rng=random.Random(1), seed=1, first_call_index=2, depth=1)
        first = plan.requests()
        first[0]['messages'][0]['content'] = 'mutated caller copy'
        self.assertNotEqual(first[0]['messages'], plan.requests()[0]['messages'])
        responses = [response_for(r, ('', 'stop', 1, 0)) for r in plan.requests()]
        candidates[0][0] = 8
        with self.assertRaises(ValueError):
            self.subject.reduce_depth1(plan, responses)


if __name__ == '__main__':
    unittest.main(verbosity=2)
