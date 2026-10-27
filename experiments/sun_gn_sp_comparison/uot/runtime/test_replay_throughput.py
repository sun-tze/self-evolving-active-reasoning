"""Offline tests for loopback_transport.py and replay_throughput.py.

Uses a synthetic recorded journal and, for HTTP, a local server on 127.0.0.1 in
this process. No model, GPU or external network. Run with python -I -B.
"""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load(HERE / 'replay_throughput.py', '_replay_under_test')
loopback = load(HERE / 'loopback_transport.py', '_loopback_under_test')
MODEL = loopback.MODEL


def synthetic_recorded(path, episode_calls=(10, 12, 7), drop_last_response=True):
    """Old-format runtime journal: run_start, then intent/response pairs."""
    rows, previous = [], None

    def append(event, payload):
        nonlocal previous
        record = {'event': event, 'index': len(rows), 'payload': payload,
                  'previous_sha256': previous, 'time_ns': len(rows)}
        record['sha256'] = previous = hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()).hexdigest()
        rows.append(record)
    append('run_start', {'job_id': 'synthetic'})
    number = 0
    for episode, calls in enumerate(episode_calls, start=1):
        for call_index in range(1, calls + 1):
            number += 1
            proposal = call_index % 5 == 1
            temperature, top_p = (0.7, 0.7) if proposal else (0.1, 0.9)
            content = f'episode {episode} call {call_index} ' + 'x' * (40 * call_index)
            metadata = {'task': 'GN', 'method': 'uot', 'call_index': call_index,
                        'temperature': temperature, 'top_p': top_p, 'json_format': False,
                        'memory_injected': False, 'phase': 'native_algorithm',
                        'upstream_caller': '_run_uot_evaluation' if proposal else
                        '_gn_calculate_remaining_with_model'}
            counted = len(content) // 4 + 10
            append('call_intent', {
                'call_id': f'gn-compare-{number}', 'role': 'uot_proposal' if proposal else
                'uot_simulation', 'metadata': metadata, 'input_tokens_counted': counted,
                'reserved_output_tokens': 512,
                'request': {'model': MODEL, 'messages': [{'role': 'user', 'content': content}],
                            'temperature': temperature, 'top_p': top_p, 'max_tokens': 512,
                            'seed': 1000 + episode}})
            last = episode == len(episode_calls) and call_index == calls
            if not (last and drop_last_response):
                tokens = 512 if number % 3 else 37
                append('call_response', {'call_id': f'gn-compare-{number}', 'elapsed_seconds': 9.6,
                                         'response': {'text': f'reply {number}',
                                                      'finish_reason': 'length' if tokens == 512 else 'stop',
                                                      'prompt_tokens': counted,
                                                      'completion_tokens': tokens,
                                                      'response_id': f'r{number}'}})
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


class FakeServer:
    """OpenAI-style chat endpoint on 127.0.0.1 answering from a recorded journal."""

    def __init__(self, recorded, delay=0.05, mode='ok'):
        pairs, _ = probe.load_recorded(recorded)
        answers = {probe.key(p['request']): p['response'] for p in pairs}
        state = {'in_flight': 0, 'max_in_flight': 0, 'lock': threading.Lock()}
        self.state = state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with state['lock']:
                    state['in_flight'] += 1
                    state['max_in_flight'] = max(state['max_in_flight'], state['in_flight'])
                try:
                    time.sleep(delay)
                    if mode == 'redirect':
                        self.send_response(302)
                        self.send_header('Location', 'http://127.0.0.1:1/elsewhere')
                        self.end_headers()
                        return
                    if mode == 'error':
                        self.send_response(500)
                        self.end_headers()
                        return
                    raw = answers[probe.key(request)]
                    body = json.dumps({'id': raw['response_id'], 'model': MODEL, 'choices': [
                        {'message': {'role': 'assistant', 'content': raw['text']},
                         'finish_reason': raw['finish_reason']}],
                        'usage': {'prompt_tokens': raw['prompt_tokens'],
                                  'completion_tokens': raw['completion_tokens']}}).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with state['lock']:
                        state['in_flight'] -= 1

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/v1'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.recorded = self.dir / 'recorded.jsonl'
        synthetic_recorded(self.recorded)

    def server(self, **kwargs):
        server = FakeServer(self.recorded, **kwargs)
        self.addCleanup(server.close)
        return server


class LoopbackTransportTests(Base):
    def request(self):
        pairs, _ = probe.load_recorded(self.recorded)
        return dict(pairs[0]['request'])

    def test_round_trip(self):
        server = self.server(delay=0)
        raw = loopback.LoopbackChatTransport(server.url, timeout=5)(self.request())
        self.assertEqual(set(raw), {'text', 'finish_reason', 'prompt_tokens',
                                    'completion_tokens', 'response_id'})
        self.assertEqual(raw['text'], 'reply 1')

    def test_only_loopback_urls(self):
        for url in ('http://localhost:1/v1', 'https://127.0.0.1:1/v1', 'http://10.0.0.1:1/v1',
                    'http://127.0.0.1/v1', 'http://u:p@127.0.0.1:1/v1'):
            with self.assertRaises(ValueError):
                loopback.LoopbackChatTransport(url)

    def test_errors_redirects_and_timeouts_raise(self):
        for kwargs, timeout in (({'mode': 'error', 'delay': 0}, 5),
                                ({'mode': 'redirect', 'delay': 0}, 5),
                                ({'delay': 1.0}, 0.2)):
            server = self.server(**kwargs)
            with self.assertRaises(Exception):
                loopback.LoopbackChatTransport(server.url, timeout=timeout)(self.request())

    def test_unexpected_request_keys_are_refused(self):
        request = dict(self.request(), stream=True)
        with self.assertRaises(ValueError):
            loopback.LoopbackChatTransport('http://127.0.0.1:9/v1')(request)


class ProbeTests(Base):
    def run_probe(self, *extra):
        output = self.dir / 'probe.json'
        result = subprocess.run(
            [sys.executable, '-I', '-B', str(HERE / 'replay_throughput.py'), '--recorded',
             str(self.recorded), '--output', str(output), '--parallel-calls', '16',
             '--serial-calls', '8', '--streams', '4', '--phase-seconds', '200',
             '--allocation-seconds', '3600', *extra],
            capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(output.read_text(encoding='utf-8'))

    def test_recorded_episodes_and_pairs(self):
        pairs, episodes = probe.load_recorded(self.recorded)
        self.assertEqual(episodes, [10, 12, 7])
        self.assertEqual(len(pairs), 28)

    def test_offline_fake_latency(self):
        report = self.run_probe('--fake-latency', '0.05')
        self.assertEqual(report['status'], 'THROUGHPUT_PROBE_COMPLETE')
        parallel, serial = report['phases']['parallel'], report['phases']['serial']
        self.assertEqual((parallel['planned_calls'], serial['planned_calls']), (16, 8))
        self.assertEqual(parallel['accounting']['completed_calls'], 16)
        self.assertEqual(parallel['texts_identical_to_recorded'], 16)
        self.assertEqual(parallel['accounting']['max_observed_in_flight'], 4)
        self.assertEqual(report['workload']['serial_slice'], [16, 24])
        feasibility = report['feasibility']
        self.assertEqual(feasibility['recorded_complete_case_calls'], [10, 12])
        self.assertEqual(feasibility['scenarios']['pessimistic_1.5x_recorded_max']
                         ['calls_per_case'], 18.0)

    def test_real_transport_over_local_http(self):
        server = self.server(delay=0.05)
        report = self.run_probe('--base-url', server.url)
        parallel, serial = report['phases']['parallel'], report['phases']['serial']
        self.assertEqual(report['status'], 'THROUGHPUT_PROBE_COMPLETE')
        self.assertEqual(server.state['max_in_flight'], 4)
        self.assertGreater(report['parallel_over_serial_rate'], 2.0)
        self.assertEqual(serial['texts_identical_to_recorded'], 8)
        print(f"\n  local HTTP, 50 ms/call: {report['parallel_over_serial_rate']:.2f}x "
              f"calls/s at 4 streams", file=sys.stderr)

    def test_server_error_halts_and_counts_uncertain(self):
        server = self.server(mode='error', delay=0)
        report = self.run_probe('--base-url', server.url)
        self.assertEqual(report['status'], 'THROUGHPUT_PROBE_HALTED')
        parallel = report['phases']['parallel']
        self.assertEqual(parallel['halt_reason'], 'transport')
        self.assertGreaterEqual(parallel['accounting']['uncertain_calls'], 1)
        self.assertEqual(parallel['accounting']['open_calls'], 0)
        self.assertNotIn('serial', report['phases'])
        self.assertEqual(report['feasibility'], {'gate_pass': False, 'reason': 'probe halted'})

    def test_feasibility_arithmetic(self):
        parallel = {'streams': 8, 'completed_calls_per_second': 0.5,
                    'latency_seconds': {'median': 12.0}}
        result = probe.feasibility(parallel, [100, 200, 150, 40], cases=20,
                                   allocation_seconds=10000, startup_seconds=140, safety=1.0)
        mean = result['scenarios']['recorded_mean']
        self.assertEqual(mean['total_calls'], 3000)
        self.assertEqual(mean['projected_seconds'], 140 + max(3000 / 0.5, 3 * 150 * 12.0))
        worst = result['scenarios']['pessimistic_1.5x_recorded_max']
        self.assertEqual(worst['projected_seconds'], 140 + max(6000 / 0.5, 3 * 300 * 12.0))
        self.assertFalse(result['gate_pass'])


if __name__ == '__main__':
    unittest.main()
