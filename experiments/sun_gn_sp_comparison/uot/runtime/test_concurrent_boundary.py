"""Offline tests for concurrent_boundary.py with fake transports; no model or network.

The native-UoT tests run the pinned arbench_adapter and the pinned AR-Bench
sources (submodule AR-Bench at 9971322f, or ARBENCH_ROOT) on synthetic secrets
with a deterministic fake model. Run: python -I -B test_concurrent_boundary.py -v
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
UOT = HERE.parent
REPO = UOT.parents[2]
PINS = {
    UOT.parent / 'source/baseline-adapters/arbench_adapter.py':
        'd63887f85329b871e9f7d169747c02a290be4e31e3292ac474469a0b3d8288d6',
    UOT / 'reference/uot_binding.py':
        'f92b9c665754be7fafb55ff892bae87df9f2e2f0e820978b3797259fcfc5b072',
}
VENDOR = Path(os.environ.get('ARBENCH_ROOT', REPO / 'AR-Bench'))


def load(path, name, pin=None):
    if pin is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest() != pin:
        raise unittest.SkipTest('Pinned source differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cb = load(HERE / 'concurrent_boundary.py', '_concurrent_boundary_under_test')


class Tokenizer:
    @staticmethod
    def chat_count(messages):
        return sum(len(m['content']) for m in messages) // 4 + 4 * len(messages)


def feedback(a, b):
    same = sum(x == y for x, y in zip(a, b))
    return same, len(set(a) & set(b)) - same


def fake_model(request):
    """Deterministic stand-in: proposals pick listed answers, simulations list survivors."""
    prompt = request['messages'][0]['content']
    rng = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
    if prompt.startswith('I am playing a number guessing game.'):
        guess = re.search(r'When I guess: (\d{4})', prompt)[1]
        match = re.search(r'I receive feedback: (\d) digits.*?, (\d) digits', prompt)
        same, diff = int(match[1]), int(match[2])
        shown = re.search(r'total\):\n(.*?)\n\nBased on', prompt, re.S)[1]
        valid = [n for n in re.findall(r'^(\d{4})$', shown, re.M) if feedback(n, guess) == (same, diff)]
        listed = valid[:rng.randint(0, 40)]
        tokens = min(512, 5 * len(listed) + 3)
        return '\n'.join(listed), ('length' if tokens == 512 else 'stop'), tokens
    listed = re.findall(r'^(\d{4})$', prompt, re.M)
    pool = listed or [''.join(rng.sample('0123456789', 4)) for _ in range(3)]
    picks = [rng.choice(pool) for _ in range(3)]
    return 'My next guesses: ' + ', '.join(picks), 'stop', 20


class FakeTransport:
    """Shared stats across the per-thread instances made by factory()."""

    def __init__(self, latency=0.0, respond=fake_model):
        self.latency, self.respond = latency, respond
        self.lock = threading.Lock()
        self.in_flight = self.max_in_flight = self.sent = 0
        self.instances = 0
        self.on_send = None

    def factory(self):
        with self.lock:
            self.instances += 1

        def transport(request):
            with self.lock:
                self.sent += 1
                self.in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                if self.on_send is not None:
                    self.on_send(request)
                time.sleep(self.latency)
                text, finish, completion = self.respond(request)
                return {'text': text, 'finish_reason': finish,
                        'prompt_tokens': Tokenizer.chat_count(request['messages']),
                        'completion_tokens': completion,
                        'response_id': hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]}
            finally:
                with self.lock:
                    self.in_flight -= 1
        return transport


def synthetic_classify(metadata):
    if set(metadata) != {'call_index', 'temperature', 'top_p'}:
        raise ValueError('synthetic metadata only')
    return 'actor', 512, 0.7, 0.7


def synthetic_episode(calls):
    def run(case, chat):
        try:
            for i in range(1, calls + 1):
                chat(messages=[{'role': 'user', 'content': f"case {case['index']} step {i}"}],
                     max_tokens=512, seed=case['index'],
                     metadata={'call_index': i, 'temperature': 0.7, 'top_p': 0.7})
            return {'status': 'complete'}
        except Exception as exc:
            return {'status': 'execution_failure', 'error': type(exc).__name__}
    return run


class Base(unittest.TestCase):
    def setUp(self):
        # Cleanups run last-in first-out: journals close before the directory goes.
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def boundary(self, transport, name='run', *, classify=synthetic_classify, max_in_flight=4,
                 limits=None, deadline=None, clock=time.time):
        journal = cb.HashChainJournal(self.dir / (name + '.jsonl'))
        self.addCleanup(journal.close)
        return cb.ConcurrentBoundary(
            run_id=name, transport_factory=transport.factory, tokenizer=Tokenizer, journal=journal,
            classify=classify, max_in_flight=max_in_flight, clock=clock,
            limits=limits or dict(max_calls=100000, max_input_reserved=10 ** 9,
                                  max_output_reserved=10 ** 9),
            deadline=deadline if deadline is not None else time.time() + 3600)

    def run_synthetic(self, transport, *, cases=8, calls=10, workers=4, name='run', **kwargs):
        boundary = self.boundary(transport, name, max_in_flight=workers, **kwargs)
        outcomes = cb.run_cases([{'index': i} for i in range(1, cases + 1)], boundary=boundary,
                                run_episode=synthetic_episode(calls),
                                max_parallel_episodes=workers, wait_seconds=60)
        boundary.journal.close()
        return boundary, outcomes


class ConcurrencyTests(Base):
    def test_overlap_and_wall_clock(self):
        serial_transport, parallel_transport = FakeTransport(0.03), FakeTransport(0.03)
        started = time.monotonic()
        serial, _ = self.run_synthetic(serial_transport, workers=1, name='serial')
        serial_seconds = time.monotonic() - started
        started = time.monotonic()
        parallel, outcomes = self.run_synthetic(parallel_transport, workers=4, name='parallel')
        parallel_seconds = time.monotonic() - started
        self.assertEqual([o['status'] for o in outcomes], ['complete'] * 8)
        self.assertEqual(serial_transport.max_in_flight, 1)
        self.assertEqual(parallel_transport.max_in_flight, 4)
        self.assertEqual(parallel.max_observed_in_flight, 4)
        self.assertGreater(serial_seconds / parallel_seconds, 2.5)
        self.assertLessEqual(parallel_transport.instances, 4)
        self.assertEqual(cb.verify_journal(self.dir / 'parallel.jsonl')['calls'], 80)
        print(f'\n  80 calls x 30 ms: serial {serial_seconds:.2f}s, 4 in flight '
              f'{parallel_seconds:.2f}s', file=sys.stderr)

    def test_lock_is_not_held_during_transport(self):
        transport = FakeTransport(0.001)
        boundary = self.boundary(transport)
        free = []

        def probe(request):
            acquired = boundary._lock.acquire(blocking=False)
            free.append(acquired)
            if acquired:
                boundary._lock.release()
        transport.on_send = probe
        cb.run_cases([{'index': 1}], boundary=boundary, run_episode=synthetic_episode(3),
                     max_parallel_episodes=1, wait_seconds=10)
        self.assertEqual(free, [True, True, True])

    def test_admission_is_atomic_under_contention(self):
        transport = FakeTransport(0.005)
        boundary, outcomes = self.run_synthetic(
            transport, cases=8, calls=10, workers=8,
            limits=dict(max_calls=37, max_input_reserved=10 ** 9, max_output_reserved=10 ** 9))
        self.assertEqual(boundary.calls, 37)
        self.assertEqual(transport.sent, 37)
        self.assertEqual(boundary.reserved_output, 37 * 512)
        self.assertTrue(boundary.halted)
        self.assertEqual(boundary.halt_reason, 'admission')
        self.assertTrue(all(o['status'] in ('execution_failure', 'not_started') for o in outcomes))
        audit = cb.verify_journal(self.dir / 'run.jsonl')
        self.assertEqual((audit['calls'], audit['open_calls'], audit['halted']), (37, [], True))

    def test_deadline_refuses_before_transport(self):
        transport = FakeTransport()
        boundary, outcomes = self.run_synthetic(transport, cases=2, calls=3, workers=2,
                                                deadline=time.time() + 119)
        self.assertEqual(transport.sent, 0)
        self.assertEqual(boundary.calls, 0)
        self.assertTrue(boundary.halted)
        self.assertEqual({o['status'] for o in outcomes}, {'execution_failure', 'not_started'})


class FailureTests(Base):
    def test_transport_failure_halts_drains_and_counts_uncertain(self):
        transport = FakeTransport(0.02)
        respond = transport.respond

        def failing(request):
            if request['messages'][0]['content'] == 'case 3 step 4':
                raise TimeoutError('injected timeout after send')
            return respond(request)
        transport.respond = failing
        boundary, outcomes = self.run_synthetic(transport, cases=8, calls=10, workers=4)
        statuses = {o['index']: o['status'] for o in outcomes}
        self.assertEqual(statuses[3], 'execution_failure')
        self.assertTrue(all(s in ('execution_failure', 'complete', 'not_started')
                            for s in statuses.values()))
        self.assertIn('not_started', statuses.values())
        accounting = boundary.accounting()
        self.assertEqual(boundary.halt_reason, 'transport')
        self.assertEqual(accounting['uncertain_calls'], 1)
        self.assertEqual(accounting['open_calls'], 0)
        self.assertEqual(accounting['uncertain_reserved_output'], 512)
        self.assertEqual(accounting['admitted_calls'],
                         accounting['completed_calls'] + accounting['uncertain_calls'])
        self.assertEqual(accounting['conservative_output'], accounting['actual_output'] + 512)
        records = cb.read_journal(self.dir / 'run.jsonl')
        halt = next(r['index'] for r in records if r['event'] == 'boundary_halt')
        self.assertFalse([r for r in records[halt:] if r['event'] == 'call_intent'])
        cb.verify_journal(self.dir / 'run.jsonl')

    def test_integrity_failure_is_uncertain_and_halts(self):
        transport = FakeTransport(0.001)
        respond = transport.respond
        transport.respond = lambda r: ('x', 'stop', 600) if r['messages'][0]['content'] == \
            'case 1 step 2' else respond(r)
        boundary, outcomes = self.run_synthetic(transport, cases=1, calls=5, workers=1)
        self.assertEqual(outcomes[0]['status'], 'execution_failure')
        self.assertEqual(boundary.halt_reason, 'integrity')
        self.assertEqual((boundary.completed, boundary.uncertain_calls), (1, 1))
        cb.verify_journal(self.dir / 'run.jsonl')

    def test_wait_deadline_reports_abandoned_and_open_cost(self):
        transport = FakeTransport()
        release = threading.Event()
        respond = transport.respond

        def hang(request):
            if request['messages'][0]['content'] == 'case 2 step 1':
                release.wait(10)
            return respond(request)
        transport.respond = hang
        boundary = self.boundary(transport, max_in_flight=2)
        outcomes = cb.run_cases([{'index': i} for i in range(1, 5)], boundary=boundary,
                                run_episode=synthetic_episode(2), max_parallel_episodes=2,
                                wait_seconds=0.5)
        statuses = [o['status'] for o in outcomes]
        self.assertEqual(statuses[1], 'abandoned_running')
        self.assertEqual(boundary.halt_reason, 'controller_wait_deadline')
        accounting = boundary.accounting()
        self.assertEqual(accounting['open_calls'], 1)
        self.assertGreaterEqual(accounting['uncertain_reserved_output'], 512)
        returned = cb.read_journal(self.dir / 'run.jsonl')[-1]
        self.assertEqual(returned['event'], 'run_returned')
        self.assertEqual(returned['payload']['accounting']['open_calls'], 1)
        release.set()
        time.sleep(0.2)

    def test_journal_tamper_is_detected(self):
        self.run_synthetic(FakeTransport(), cases=2, calls=2, workers=2)
        path = self.dir / 'run.jsonl'
        lines = path.read_text(encoding='utf-8').splitlines()
        record = json.loads(lines[1])
        record['payload']['reserved_output_tokens'] = 1
        lines[1] = json.dumps(record, sort_keys=True, separators=(',', ':'))
        path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            cb.read_journal(path)


class NativeUoTTests(Base):
    CASES = [{'index': i, 'case_id': f'synthetic-{i}', 'secret': s, 'seed': 1000 + i}
             for i, s in enumerate(('0123', '4567', '8902', '3579', '6014', '2895'), start=1)]

    @classmethod
    def setUpClass(cls):
        for name in ('arbench/reasoner/gn/gn_evaluator.py', 'arbench/reasoner/gn/prompt.py',
                     'arbench/utils/utils_gn.py'):
            if not (VENDOR / name).exists():
                raise unittest.SkipTest('Pinned AR-Bench sources not found; set ARBENCH_ROOT')
        cls.adapter = load(*next((p, '_arbench_adapter_pinned', h) for p, h in PINS.items()
                                 if p.name == 'arbench_adapter.py'))
        cls.adapter.VENDOR = VENDOR
        cls.binding = load(*next((p, '_uot_binding_pinned', h) for p, h in PINS.items()
                                 if p.name == 'uot_binding.py'))

    def native(self, case, chat):
        return cb_native_uot(self.adapter)(case, chat)

    def run_native(self, name, workers, latency=0.0):
        # Same run id for both runs, so runtime call ids inside the results match.
        transport = FakeTransport(latency)
        journal = cb.HashChainJournal(self.dir / (name + '.jsonl'))
        self.addCleanup(journal.close)
        boundary = cb.ConcurrentBoundary(
            run_id='native', transport_factory=transport.factory, tokenizer=Tokenizer,
            journal=journal, classify=self.binding.classify_uot_metadata,
            max_in_flight=workers, deadline=time.time() + 3600,
            limits=dict(max_calls=21500, max_input_reserved=10 ** 9, max_output_reserved=21500 * 512))
        store = cb.EpisodeStore(self.dir / (name + '-episodes'), journal)
        outcomes = cb.run_cases(self.CASES, boundary=boundary, run_episode=self.native,
                                max_parallel_episodes=workers, wait_seconds=120, store=store)
        journal.close()
        return boundary, transport, outcomes

    def test_runner_matches_pinned_serial_invocation(self):
        tree = ast.parse((UOT / 'reference/uot_binding.py').read_text(encoding='utf-8'))
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                    getattr(n.func, 'attr', None) == 'run_gn_episode')
        pinned = {k.arg: ast.unparse(k.value) for k in call.keywords}
        ours = next(n for n in ast.walk(ast.parse(
            (HERE / 'native_uot.py').read_text(encoding='utf-8')))
            if isinstance(n, ast.Call) and getattr(n.func, 'attr', None) == 'run_gn_episode')
        mine = {k.arg: ast.unparse(k.value) for k in ours.keywords}
        differing = {k for k in pinned if pinned[k] != mine.get(k)}
        # Only how the per-case transport, secret, seed and case id are supplied may differ.
        self.assertEqual(differing, {'chat', 'secret', 'seed', 'case_id'})
        self.assertEqual(set(pinned), set(mine))
        native = load(HERE / 'native_uot.py', '_native_uot_constants')
        for name in ('MAX_TURNS', 'SIMULATE_DEPTH', 'EPISODE_CALL_BOUND'):
            self.assertEqual(getattr(native, name), getattr(self.binding, name))

    def test_serial_and_concurrent_runs_are_identical_per_case(self):
        # A small fake latency lets calls overlap; with zero latency the native
        # algorithm's own CPU work under the GIL dominates and overlap is rare.
        _, serial_transport, serial = self.run_native('serial', 1, latency=0.005)
        boundary, parallel_transport, parallel = self.run_native('parallel', 4, latency=0.005)
        self.assertEqual([o['status'] for o in serial], ['complete'] * len(self.CASES))
        self.assertEqual(serial_transport.max_in_flight, 1)
        self.assertGreaterEqual(parallel_transport.max_in_flight, 2)
        for a, b in zip(serial, parallel):
            self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        audits = [cb.verify_journal(self.dir / (n + '.jsonl')) for n in ('serial', 'parallel')]
        self.assertEqual(audits[0]['cases'], audits[1]['cases'])

        def per_case(name):
            rows = {}
            for r in cb.read_journal(self.dir / (name + '.jsonl')):
                if r['event'] == 'call_intent':
                    rows.setdefault(r['payload']['case_index'], []).append(r['payload']['request'])
            return rows
        self.assertEqual(per_case('serial'), per_case('parallel'))
        saved = [r for r in cb.read_journal(self.dir / 'parallel.jsonl') if r['event'] == 'episode_saved']
        self.assertEqual(sorted(r['payload']['index'] for r in saved), list(range(1, 7)))
        print(f"\n  native UoT: {boundary.calls} calls over {len(self.CASES)} synthetic cases, "
              f"identical serial/4-way results; max {parallel_transport.max_in_flight} in flight",
              file=sys.stderr)

    def test_native_failure_mid_run_is_recorded_not_imputed(self):
        transport = FakeTransport(0.002)
        respond, count = transport.respond, [0]

        def fail_once(request):
            with transport.lock:
                count[0] += 1
                fail = count[0] == 150
            if fail:
                raise ConnectionError('injected connection reset')
            return respond(request)
        transport.respond = fail_once
        boundary = self.boundary(transport, 'native', classify=self.binding.classify_uot_metadata,
                                 max_in_flight=3)
        outcomes = cb.run_cases(self.CASES, boundary=boundary, run_episode=self.native,
                                max_parallel_episodes=3, wait_seconds=120)
        boundary.journal.close()
        statuses = [o['status'] for o in outcomes]
        self.assertIn('execution_failure', statuses)
        self.assertNotIn('controller_error', statuses)
        for outcome in outcomes:
            if outcome['status'] == 'execution_failure':
                self.assertIsNone(outcome['result']['official_result'])
        self.assertEqual(boundary.accounting()['uncertain_calls'], 1)
        cb.verify_journal(self.dir / 'native.jsonl')


def cb_native_uot(adapter):
    native = load(HERE / 'native_uot.py', '_native_uot_under_test')
    return native.native_uot_runner(adapter)


if __name__ == '__main__':
    unittest.main()
