"""Bounded throughput probe: replay recorded native UoT requests, serially and K-way.

Sends the cancelled run's own request bodies (model, messages, sampling, seed,
512-token cap) through ConcurrentBoundary and reports calls per second, latency,
finish reasons, and how often the generated text equals the recorded text. Two
disjoint slices of the recorded order are used, so prefix-cache reuse cannot cross
phases: "parallel" = the first N calls split round-robin over K streams, "serial" =
the next M calls on one stream. Token counts are the recorded ones; the server's
prompt_tokens must equal them. No trajectory, case result or score is produced.

Against a real server this needs an approved execution scope and an already
running vLLM on 127.0.0.1. --fake-latency replays recorded responses offline.
Only --output and the two phase journals are written; none may exist.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
UOT_BINDING_SHA256 = 'f92b9c665754be7fafb55ff892bae87df9f2e2f0e820978b3797259fcfc5b072'


def load(path, name, pin=None):
    if pin is not None and hashlib.sha256(Path(path).read_bytes()).hexdigest() != pin:
        raise ValueError('Pinned source differs: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cb = load(HERE / 'concurrent_boundary.py', '_probe_concurrent_boundary')


def key(value):
    return hashlib.sha256(cb.canonical(value)).hexdigest()


def load_recorded(path):
    """(intent, response) pairs in call order, and calls per recorded episode."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    if [r['index'] for r in rows] != list(range(len(rows))) or any(
            b['previous_sha256'] != a['sha256'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Recorded journal index or hash linkage differs')
    responses = {r['payload']['call_id']: r['payload']['response']
                 for r in rows if r['event'] == 'call_response'}
    intents = [r['payload'] for r in rows if r['event'] == 'call_intent']
    episodes = []
    for intent in intents:
        if intent['metadata']['call_index'] == 1:
            episodes.append(0)
        episodes[-1] += 1
    pairs = [dict(intent, response=responses[intent['call_id']])
             for intent in intents if intent['call_id'] in responses]
    return pairs, episodes


class RecordedTokenizer:
    def __init__(self, pairs):
        self.counts = {key(p['request']['messages']): p['input_tokens_counted'] for p in pairs}

    def chat_count(self, messages):
        return self.counts[key(messages)]


class Expectation:
    """Wraps a transport factory: every request must equal a recorded request body."""

    def __init__(self, pairs, inner_factory):
        self.pairs = {key(p['request']): p for p in pairs}
        self.inner_factory = inner_factory
        self.lock = threading.Lock()
        self.compared = self.identical = 0

    def factory(self):
        inner = self.inner_factory()

        def transport(request):
            pair = self.pairs.get(key(request))
            if pair is None:
                raise ValueError('Request differs from every recorded request body')
            raw = inner(request)
            if isinstance(raw, dict) and isinstance(raw.get('text'), str):
                with self.lock:
                    self.compared += 1
                    self.identical += raw['text'] == pair['response']['text']
            return raw
        return transport


def fake_factory(pairs, latency):
    """Offline stand-in: the recorded response after latency scaled by its output length."""
    recorded = {key(p['request']): p['response'] for p in pairs}

    def factory():
        def transport(request):
            raw = dict(recorded[key(request)])
            time.sleep(latency * max(raw['completion_tokens'], 1) / 512)
            return raw
        return transport
    return factory


def stream_runner(case, chat):
    try:
        for index, pair in enumerate(case['pairs'], start=1):
            chat(messages=pair['request']['messages'], max_tokens=pair['request']['max_tokens'],
                 seed=pair['request']['seed'], metadata=dict(pair['metadata'], call_index=index))
        return {'status': 'complete'}
    except Exception as exc:
        return {'status': 'execution_failure',
                'error': {'type': type(exc).__name__, 'message': str(exc)}}


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1) + 0.5))]
    return {'n': len(values), 'min': values[0], 'median': pick(0.5), 'p90': pick(0.9),
            'max': values[-1]}


def run_phase(name, pairs, streams, *, inner_factory, classify, journal_path, seconds):
    expectation = Expectation(pairs, inner_factory)
    journal = cb.HashChainJournal(journal_path)
    boundary = cb.ConcurrentBoundary(
        run_id=name, transport_factory=expectation.factory, tokenizer=RecordedTokenizer(pairs),
        journal=journal, classify=classify, max_in_flight=streams,
        deadline=time.time() + seconds,
        limits=dict(max_calls=len(pairs), max_output_reserved=512 * len(pairs),
                    max_input_reserved=sum(p['input_tokens_counted'] for p in pairs)))
    cases = [{'index': s + 1, 'pairs': pairs[s::streams]} for s in range(streams)]
    started = time.monotonic()
    outcomes = cb.run_cases(cases, boundary=boundary, run_episode=stream_runner,
                            max_parallel_episodes=streams, wait_seconds=seconds)
    wall = time.monotonic() - started
    journal.close()
    audit = cb.verify_journal(journal_path)
    records = cb.read_journal(journal_path)
    responses = [r['payload'] for r in records if r['event'] == 'call_response']
    accounting = boundary.accounting()
    return {
        'streams': streams, 'planned_calls': len(pairs), 'wall_seconds': wall,
        'stream_statuses': Counter(o['status'] for o in outcomes),
        'halt_reason': boundary.halt_reason, 'accounting': accounting,
        'journal_calls': audit['calls'],
        'completed_calls_per_second': accounting['completed_calls'] / wall,
        'output_tokens_per_second': accounting['actual_output'] / wall,
        'latency_seconds': quantiles([r['elapsed_seconds'] for r in responses]),
        'finish_reasons': Counter(r['response'].get('finish_reason') for r in responses
                                  if isinstance(r['response'], dict)),
        'texts_compared': expectation.compared,
        'texts_identical_to_recorded': expectation.identical,
    }


def feasibility(parallel, recorded_episode_calls, *, cases, allocation_seconds,
                startup_seconds, safety, deadline_margin=240):
    """Projected wall time of a full native run at the measured K-way rate.

    Two call scenarios: the recorded mean per complete case, and 1.5x the recorded
    maximum. Makespan is the larger of total calls / measured rate and
    ceil(cases / K) waves of the longest case at the measured median latency.
    """
    complete = recorded_episode_calls[:-1]  # the last recorded episode is a prefix
    rate = parallel['completed_calls_per_second']
    latency = parallel['latency_seconds']['median']
    waves = math.ceil(cases / parallel['streams'])
    budget = allocation_seconds - deadline_margin
    scenarios = {}
    for name, per_case in (('recorded_mean', sum(complete) / len(complete)),
                           ('pessimistic_1.5x_recorded_max', 1.5 * max(complete))):
        total = cases * per_case
        projected = startup_seconds + max(total / rate, waves * per_case * latency)
        scenarios[name] = {'calls_per_case': per_case, 'total_calls': total,
                           'projected_seconds': projected,
                           'fits_with_safety': projected * safety <= budget}
    return {'recorded_complete_case_calls': complete, 'cases': cases,
            'allocation_seconds': allocation_seconds, 'request_budget_seconds': budget,
            'safety_factor': safety, 'scenarios': scenarios,
            'gate_pass': scenarios['pessimistic_1.5x_recorded_max']['fits_with_safety'],
            'caveat': 'three complete recorded cases; a hard case can need up to 1075 calls'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recorded', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parallel-calls', type=int, default=144)
    parser.add_argument('--serial-calls', type=int, default=18)
    parser.add_argument('--streams', type=int, default=8)
    parser.add_argument('--phase-seconds', type=float, default=420)
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--fake-latency', type=float, default=None)
    parser.add_argument('--cases', type=int, default=20)
    parser.add_argument('--allocation-seconds', type=float, required=True)
    parser.add_argument('--startup-seconds', type=float, default=140)
    parser.add_argument('--safety', type=float, default=1.25)
    args = parser.parse_args()
    journals = [args.output.with_suffix('.' + n + '.jsonl') for n in ('parallel', 'serial')]
    if any(p.exists() for p in [args.output] + journals):
        raise SystemExit('Refusing to overwrite an existing output or journal')
    if (args.base_url is None) == (args.fake_latency is None):
        raise SystemExit('Choose exactly one of --base-url (real server) or --fake-latency')
    pairs, episodes = load_recorded(args.recorded)
    if args.parallel_calls + args.serial_calls > len(pairs):
        raise SystemExit('Not enough recorded calls')
    parallel_pairs = pairs[:args.parallel_calls]
    serial_pairs = pairs[args.parallel_calls:args.parallel_calls + args.serial_calls]
    if args.base_url is not None:
        transport = load(HERE / 'loopback_transport.py', '_probe_loopback')
        inner = lambda: transport.LoopbackChatTransport(args.base_url, timeout=120)
    else:
        inner = fake_factory(pairs, args.fake_latency)
    binding = load(HERE.parent / 'reference/uot_binding.py', '_probe_uot_binding',
                   UOT_BINDING_SHA256)
    phases = {}
    for (name, chosen, streams), journal in zip(
            (('parallel', parallel_pairs, args.streams), ('serial', serial_pairs, 1)), journals):
        phases[name] = run_phase(name, chosen, streams, inner_factory=inner,
                                 classify=binding.classify_uot_metadata,
                                 journal_path=journal, seconds=args.phase_seconds)
        if phases[name]['halt_reason'] is not None:
            break  # a failing server is not measured further
    complete = len(phases) == 2 and all(p['halt_reason'] is None for p in phases.values())
    report = {
        'status': 'THROUGHPUT_PROBE_COMPLETE' if complete else 'THROUGHPUT_PROBE_HALTED',
        'mode': 'real_server' if args.base_url else 'offline_fake_latency',
        'workload': {'recorded_pairs': len(pairs), 'parallel_slice': [0, args.parallel_calls],
                     'serial_slice': [args.parallel_calls, args.parallel_calls + args.serial_calls]},
        'phases': phases,
        'parallel_over_serial_rate': (phases['parallel']['completed_calls_per_second'] /
                                      phases['serial']['completed_calls_per_second']
                                      if complete else None),
        # A halted probe never yields a gate; it counts as not passing.
        'feasibility': feasibility(phases['parallel'], episodes, cases=args.cases,
                                   allocation_seconds=args.allocation_seconds,
                                   startup_seconds=args.startup_seconds, safety=args.safety)
                       if complete else {'gate_pass': False, 'reason': 'probe halted'},
        'not_claimed': ['scientific results', 'serial-equivalent text under batching',
                        'speedup on other hardware or server settings'],
    }
    with args.output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, indent=2, default=dict)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'mode', 'parallel_over_serial_rate')}))


if __name__ == '__main__':
    main()
