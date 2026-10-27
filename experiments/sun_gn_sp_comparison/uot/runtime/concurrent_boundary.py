"""Bounded concurrent model boundary for independent GN episodes.

Episodes run the unmodified native algorithm (arbench_adapter.run_gn_episode),
each with its own namespace and seeded random.Random, so calls from different
episodes may overlap while each episode keeps its serial call order. No response
reaches another episode. This replaces the old QwenBoundary, which held its lock
across the transport call and therefore serialized everything.

Admission is atomic under a short lock: schema/role/cap/sampling checks, context,
call, input and output reservations, the request deadline, a run/case/call ID and
a durable intent record. The transport runs outside the lock, at most
max_in_flight at once, with one transport object per worker thread. The first
failure of any kind halts admission permanently. Calls already sent finish or time
out and are recorded. A call whose transport raised, whose response failed the
integrity check, or that is still open when the caller stops waiting is counted at
its full reservation as uncertain cost.

This module creates no process, socket, model or GPU work by itself; the caller
supplies the transport factory, tokenizer, deadline and limits. Nothing here is
authorization to run.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
CONTEXT = 8192
JOURNAL_FORMAT = 'gn_concurrent_journal_v1'
RAW_KEYS = frozenset({'text', 'finish_reason', 'prompt_tokens', 'completion_tokens', 'response_id'})


class BoundaryHalted(RuntimeError):
    """Admission refused because an earlier failure halted the boundary."""


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


class HashChainJournal:
    """Append-only JSONL; each record carries the previous record's hash; fsync per append."""

    def __init__(self, path, clock_ns=time.time_ns):
        self.path = Path(path)
        self._handle = self.path.open('x', encoding='utf-8', newline='\n')
        self._lock = threading.Lock()
        self._clock_ns = clock_ns
        self._index, self._previous = 0, None

    def append(self, event, payload):
        with self._lock:
            record = {'format': JOURNAL_FORMAT, 'index': self._index, 'event': event,
                      'payload': payload, 'previous_sha256': self._previous,
                      'time_ns': self._clock_ns()}
            record['sha256'] = hashlib.sha256(canonical(record)).hexdigest()
            self._handle.write(canonical(record).decode('utf-8') + '\n')
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._index, self._previous = self._index + 1, record['sha256']
            return record['sha256']

    def close(self):
        with self._lock:
            self._handle.close()


def read_journal(path):
    """Records in order; raises if a hash, index or link differs."""
    records, previous = [], None
    for index, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines()):
        record = json.loads(line)
        claimed = record.pop('sha256')
        if (record.get('format') != JOURNAL_FORMAT or record.get('index') != index or
                record.get('previous_sha256') != previous or
                hashlib.sha256(canonical(record)).hexdigest() != claimed):
            raise ValueError('Journal hash chain differs at record ' + str(index))
        record['sha256'] = previous = claimed
        records.append(record)
    return records


class ConcurrentBoundary:
    def __init__(self, *, run_id, transport_factory, tokenizer, journal, classify, limits,
                 deadline, max_in_flight, admission_margin=120, clock=time.time,
                 monotonic=time.monotonic):
        if not isinstance(run_id, str) or not run_id or '/' in run_id:
            raise ValueError('Explicit run id without "/" required')
        if (not isinstance(limits, dict) or set(limits) != {'max_calls', 'max_input_reserved',
                                                             'max_output_reserved'} or
                any(type(v) is not int or v <= 0 for v in limits.values())):
            raise ValueError('Finite call, input and output reservations required')
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('Explicit finite request deadline required')
        if type(max_in_flight) is not int or not 1 <= max_in_flight <= 64:
            raise ValueError('Bounded positive in-flight limit required')
        self.run_id, self.transport_factory, self.tokenizer = run_id, transport_factory, tokenizer
        self.journal, self.classify, self.limits = journal, classify, dict(limits)
        self.deadline, self.margin, self.clock, self.monotonic = (
            deadline, admission_margin, clock, monotonic)
        self.max_in_flight = max_in_flight
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._local = threading.local()
        self._open = {}         # call_id -> (counted input, reserved output)
        self._last_index = {}   # case_index -> last admitted call_index
        self.halted, self.halt_reason = False, None
        self.calls = self.reserved_input = self.reserved_output = 0
        self.actual_input = self.actual_output = self.completed = 0
        self.uncertain_calls = self.uncertain_input = self.uncertain_output = 0
        self.max_observed_in_flight = 0

    def chat_for(self, case_index):
        if type(case_index) is not int or case_index < 1:
            raise ValueError('Positive integer case index required')

        def chat(*, messages, max_tokens, seed, metadata):
            return self.call(case_index=case_index, messages=messages, max_tokens=max_tokens,
                             seed=seed, metadata=metadata)
        return chat

    def accounting(self):
        with self._lock:
            return self._accounting()

    def _accounting(self):
        open_input = sum(i for i, _ in self._open.values())
        open_output = sum(o for _, o in self._open.values())
        return dict(admitted_calls=self.calls, completed_calls=self.completed,
                    uncertain_calls=self.uncertain_calls, open_calls=len(self._open),
                    reserved_input=self.reserved_input, reserved_output=self.reserved_output,
                    actual_input=self.actual_input, actual_output=self.actual_output,
                    uncertain_reserved_input=self.uncertain_input + open_input,
                    uncertain_reserved_output=self.uncertain_output + open_output,
                    conservative_output=self.actual_output + self.uncertain_output + open_output,
                    max_observed_in_flight=self.max_observed_in_flight)

    def _halt(self, reason, call_id=None, error=None):
        # Caller holds the lock. Only the first failure is the halt reason.
        if not self.halted:
            self.halted, self.halt_reason = True, reason
            self.journal.append('boundary_halt', dict(
                reason=reason, call_id=call_id,
                error_type=type(error).__name__ if error is not None else None,
                message=str(error) if error is not None else None,
                accounting=self._accounting()))

    def halt(self, reason):
        with self._lock:
            self._halt(reason)

    def _transport(self):
        transport = getattr(self._local, 'transport', None)
        if transport is None:
            transport = self._local.transport = self.transport_factory()
        return transport

    def _admit(self, case_index, messages, max_tokens, seed, metadata):
        if (not isinstance(messages, list) or not messages or
                any(not isinstance(m, dict) or set(m) != {'role', 'content'} or
                    m['role'] not in ('system', 'user', 'assistant') or
                    not isinstance(m['content'], str) for m in messages)):
            raise ValueError('Unsupported public message schema')
        if type(seed) is not int or type(max_tokens) is not int:
            raise TypeError('Integer seed and requested cap required')
        role, cap, temperature, top_p = self.classify(metadata)
        if max_tokens != cap or cap <= 0:
            raise ValueError('Requested cap differs from the frozen role cap')
        if metadata.get('temperature') != temperature or metadata.get('top_p') != top_p:
            raise ValueError('Requested sampling differs from the frozen role')
        if metadata.get('json_format', False) or metadata.get('use_json_mode', False):
            raise ValueError('Structured output requires separate serving review')
        counted = self.tokenizer.chat_count(messages)
        if type(counted) is not int or counted <= 0:
            raise ValueError('Pinned tokenizer did not return a positive count')
        request = dict(model=MODEL, messages=deepcopy(messages), temperature=temperature,
                       top_p=top_p, max_tokens=cap, seed=seed)
        with self._lock:
            if self.halted:
                raise BoundaryHalted('Boundary halted: ' + str(self.halt_reason))
            call_index = metadata.get('call_index')
            if call_index != self._last_index.get(case_index, 0) + 1:
                raise ValueError('Case call index is not contiguous')
            if (counted + cap > CONTEXT or self.calls >= self.limits['max_calls'] or
                    self.reserved_input + counted > self.limits['max_input_reserved'] or
                    self.reserved_output + cap > self.limits['max_output_reserved'] or
                    self.clock() + self.margin >= self.deadline):
                raise RuntimeError('Context, request, token reservation or deadline admission failed')
            self.calls += 1
            self.reserved_input += counted
            self.reserved_output += cap
            self._last_index[case_index] = call_index
            call_id = f'{self.run_id}/case-{case_index:03d}/call-{call_index:04d}'
            self._open[call_id] = (counted, cap)
            self.max_observed_in_flight = max(self.max_observed_in_flight, len(self._open))
            self.journal.append('call_intent', dict(
                call_id=call_id, case_index=case_index, call_index=call_index, role=role,
                request=request, metadata=deepcopy(metadata), input_tokens_counted=counted,
                reserved_output_tokens=cap))
        return call_id, request, counted, cap

    def _uncertain(self, call_id, counted, cap):
        # Caller holds the lock.
        if self._open.pop(call_id, None) is not None:
            self.uncertain_calls += 1
            self.uncertain_input += counted
            self.uncertain_output += cap

    def call(self, *, case_index, messages, max_tokens, seed, metadata):
        self._slots.acquire()
        try:
            try:
                call_id, request, counted, cap = self._admit(case_index, messages, max_tokens,
                                                             seed, metadata)
            except BaseException as exc:
                with self._lock:
                    self._halt('admission', error=exc)
                raise
            started = self.monotonic()
            try:
                raw = self._transport()(deepcopy(request))
            except BaseException as exc:
                with self._lock:
                    self._uncertain(call_id, counted, cap)
                    self.journal.append('call_transport_error', dict(
                        call_id=call_id, error_type=type(exc).__name__, message=str(exc),
                        elapsed_seconds=self.monotonic() - started,
                        cost='uncertain; counted at full reservation'))
                    self._halt('transport', call_id, exc)
                raise
            with self._lock:
                self.journal.append('call_response', dict(
                    call_id=call_id, response=raw, elapsed_seconds=self.monotonic() - started))
                if (not isinstance(raw, dict) or set(raw) != RAW_KEYS or
                        not isinstance(raw['text'], str) or
                        raw['finish_reason'] not in ('stop', 'length') or
                        not isinstance(raw['response_id'], str) or not raw['response_id'] or
                        type(raw['prompt_tokens']) is not int or raw['prompt_tokens'] != counted or
                        type(raw['completion_tokens']) is not int or
                        not 0 <= raw['completion_tokens'] <= cap):
                    error = RuntimeError('Response/tokenizer identity integrity failure')
                    self._uncertain(call_id, counted, cap)
                    self._halt('integrity', call_id, error)
                    raise error
                self._open.pop(call_id)
                self.completed += 1
                self.actual_input += raw['prompt_tokens']
                self.actual_output += raw['completion_tokens']
            return {'text': raw['text'], 'finish_reason': raw['finish_reason'],
                    'usage': {'prompt_tokens': raw['prompt_tokens'],
                              'completion_tokens': raw['completion_tokens']},
                    'response_id': raw['response_id'], 'model': MODEL, 'runtime_call_id': call_id}
        finally:
            self._slots.release()


class EpisodeStore:
    """Exclusive per-case result files, bound into the journal as each case ends."""

    def __init__(self, directory, journal):
        self.directory, self.journal = Path(directory), journal
        self.directory.mkdir(parents=True, exist_ok=False)

    def save(self, outcome):
        name = f"{outcome['index']:05d}.json"
        data = canonical(outcome)
        with (self.directory / name).open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.journal.append('episode_saved', dict(index=outcome['index'], status=outcome['status'],
                                                  file=name, sha256=hashlib.sha256(data).hexdigest()))


def run_cases(cases, *, boundary, run_episode, max_parallel_episodes, wait_seconds, store=None):
    """Fork/join over independent cases; outcomes in the original case order.

    cases: dicts with a positive int 'index' plus whatever run_episode needs.
    run_episode(case, chat) returns the adapter's result dict. Cases start in
    order; none starts after a halt. After wait_seconds the boundary is halted,
    cases not yet started are cancelled and running cases are reported as
    abandoned; their open calls stay in the uncertain accounting.
    """
    if type(max_parallel_episodes) is not int or max_parallel_episodes < 1:
        raise ValueError('Positive parallel episode limit required')
    if len({c['index'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case index')

    def task(case):
        if boundary.halted:
            return {'index': case['index'], 'status': 'not_started'}
        result = run_episode(case, boundary.chat_for(case['index']))
        outcome = {'index': case['index'], 'status': result['status'], 'result': result}
        if store is not None:
            store.save(outcome)
        return outcome

    executor = ThreadPoolExecutor(max_workers=max_parallel_episodes,
                                  thread_name_prefix='gn-episode')
    futures = [executor.submit(task, case) for case in cases]
    done, _ = wait(futures, timeout=wait_seconds)
    if len(done) < len(futures):
        boundary.halt('controller_wait_deadline')
    executor.shutdown(wait=False, cancel_futures=True)
    outcomes = []
    for case, future in zip(cases, futures):
        if future in done:
            try:
                outcomes.append(future.result())
            except BaseException as exc:
                outcomes.append({'index': case['index'], 'status': 'controller_error',
                                 'error': {'type': type(exc).__name__, 'message': str(exc)}})
        elif future.cancelled():
            outcomes.append({'index': case['index'], 'status': 'not_started'})
        else:
            outcomes.append({'index': case['index'], 'status': 'abandoned_running'})
    boundary.journal.append('run_returned', dict(
        statuses=[[o['index'], o['status']] for o in outcomes], halted=boundary.halted,
        halt_reason=boundary.halt_reason, accounting=boundary.accounting()))
    return outcomes


def verify_journal(path):
    """Structural audit of a closed run journal; returns per-case call counts."""
    records = read_journal(path)
    intents, terminal, halted_at, per_case = {}, {}, None, {}
    for record in records:
        event, payload = record['event'], record['payload']
        if event == 'call_intent':
            if halted_at is not None:
                raise ValueError('Call admitted after the boundary halted')
            if payload['call_id'] in intents:
                raise ValueError('Duplicate call id')
            expected = per_case.get(payload['case_index'], 0) + 1
            if payload['call_index'] != expected:
                raise ValueError('Per-case call index is not contiguous')
            per_case[payload['case_index']] = expected
            intents[payload['call_id']] = record['index']
        elif event in ('call_response', 'call_transport_error'):
            if payload['call_id'] not in intents or payload['call_id'] in terminal:
                raise ValueError('Orphan or repeated terminal call record')
            terminal[payload['call_id']] = event
        elif event == 'boundary_halt':
            halted_at = record['index'] if halted_at is None else halted_at
    returned = [r for r in records if r['event'] == 'run_returned']
    if len(returned) != 1 or records[-1]['event'] != 'run_returned':
        raise ValueError('Journal must end with exactly one run_returned record')
    open_calls = sorted(set(intents) - set(terminal))
    if returned[0]['payload']['accounting']['open_calls'] != len(open_calls):
        raise ValueError('Open-call accounting differs from the journal')
    return dict(cases=per_case, calls=len(intents), open_calls=open_calls,
                halted=halted_at is not None)
