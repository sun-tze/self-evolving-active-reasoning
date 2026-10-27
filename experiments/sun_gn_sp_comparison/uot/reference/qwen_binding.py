"""Thin GN binding to the already-used tokenizer and loopback transport.

No weights, credential, provider, process, socket or cluster operation is
created by importing this module. Callers own the existing allocation and
must supply admission; this file grants no new execution scope.
"""
from copy import deepcopy
import math
import threading
import time

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
CONTEXT = 8192


class QwenBoundary:
    """Adapt old LoopbackTransport's flat result to the new chat contract.

    Unlike the old readout Boundary, sampling comes from the declared role,
    and every requested output cap is reserved, including DC's doubled cap.
    Native prompts, parser, scorer and stopping rules are untouched.
    """
    def __init__(self, *, transport, tokenizer, journal, protocol, plan,
                 deadline, max_input_reserved, clock=time.time):
        if type(max_input_reserved) is not int or max_input_reserved <= 0:
            raise ValueError('Explicit finite input-token reservation required')
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('Explicit finite request deadline required')
        self.transport, self.tokenizer, self.journal = transport, tokenizer, journal
        self.protocol, self.plan = deepcopy(protocol), deepcopy(plan)
        self.deadline, self.max_input, self.clock = deadline, max_input_reserved, clock
        self.lock = threading.RLock()
        self.halted = False
        self.calls = self.reserved_input = self.reserved_output = 0
        self.actual_input = self.actual_output = 0

    def _role(self, metadata):
        role, method = metadata.get('role', 'actor'), metadata.get('method')
        if role == 'actor':
            return ('actor', self.protocol['actor']['max_tokens'],
                    self.protocol['actor']['temperature'], self.protocol['actor']['top_p'])
        if method == 'DC' and role == 'curator':
            return ('DC_curator', 2 * self.protocol['actor']['max_tokens'],
                    self.protocol['dc']['temperature'], 1.0)
        if method == 'ACE' and role in ('reflector', 'curator'):
            return ('ACE_' + role, self.protocol['ace'][role + '_max_tokens'], 0.0, 1.0)
        raise ValueError('Unrecognized method/role at Qwen boundary')

    def call(self, *, messages, max_tokens, seed, metadata):
        with self.lock:
            if self.halted:
                raise RuntimeError('Qwen shared boundary already halted')
            call_id = None
            try:
                if (not isinstance(messages, list) or not messages or
                    any(not isinstance(m, dict) or set(m) != {'role', 'content'} or
                        m['role'] not in ('system', 'user', 'assistant') or
                        not isinstance(m['content'], str) for m in messages)):
                    raise ValueError('Unsupported public message schema')
                if type(seed) is not int or type(max_tokens) is not int:
                    raise TypeError('Integer seed and requested cap required')
                role, cap, temperature, top_p = self._role(metadata)
                if max_tokens != cap or cap <= 0:
                    raise ValueError('Requested cap differs from the frozen role cap')
                if (metadata.get('temperature', temperature) != temperature or
                    metadata.get('top_p', top_p) != top_p):
                    raise ValueError('Requested sampling differs from the frozen role')
                # The carried server proof covers the six-key unstructured wire.
                # Do not silently add an unreviewed JSON/grammar server mode.
                if metadata.get('json_format', False) or metadata.get('use_json_mode', False):
                    raise ValueError('Structured output requires separate serving review')
                counted = self.tokenizer.chat_count(messages)
                if type(counted) is not int or counted <= 0:
                    raise ValueError('Pinned tokenizer did not return a positive count')
                if (counted + cap > CONTEXT or self.calls >= self.plan['max_calls'] or
                    self.reserved_input + counted > self.max_input or
                    self.reserved_output + cap > self.plan['max_output_tokens'] or
                    self.clock() + 120 >= self.deadline):
                    raise RuntimeError('Context, request, token reservation or deadline admission failed')
                self.calls += 1
                self.reserved_input += counted
                self.reserved_output += cap
                call_id = 'gn-compare-' + str(self.calls)
                request = dict(model=MODEL, messages=deepcopy(messages), temperature=temperature,
                               top_p=top_p, max_tokens=cap, seed=seed)
                self.journal.append('call_intent', dict(call_id=call_id, role=role,
                    request=request, metadata=deepcopy(metadata), input_tokens_counted=counted,
                    reserved_output_tokens=cap))
                started = time.monotonic()
                raw = self.transport(request)
                self.journal.append('call_response', dict(call_id=call_id, response=raw,
                    elapsed_seconds=time.monotonic() - started))
                if (not isinstance(raw, dict) or set(raw) != {
                        'text', 'finish_reason', 'prompt_tokens', 'completion_tokens', 'response_id'} or
                    not isinstance(raw['text'], str) or raw['finish_reason'] not in ('stop', 'length') or
                    not isinstance(raw['response_id'], str) or not raw['response_id'] or
                    type(raw['prompt_tokens']) is not int or raw['prompt_tokens'] != counted or
                    type(raw['completion_tokens']) is not int or not 0 <= raw['completion_tokens'] <= cap):
                    raise RuntimeError('Response/tokenizer identity integrity failure')
                self.actual_input += raw['prompt_tokens']
                self.actual_output += raw['completion_tokens']
                return {'text': raw['text'], 'finish_reason': raw['finish_reason'],
                        'usage': {'prompt_tokens': raw['prompt_tokens'],
                                  'completion_tokens': raw['completion_tokens']},
                        'response_id': raw['response_id'], 'model': MODEL,
                        'runtime_call_id': call_id}
            except BaseException as exc:
                self.halted = True
                self.journal.append('boundary_halt', dict(call_id=call_id,
                    error_type=type(exc).__name__, calls=self.calls,
                    reserved_input=self.reserved_input, reserved_output=self.reserved_output))
                raise


def run_with_existing_server(protocol, cases, *, transport, tokenizer, journal,
                             request_deadline, max_input_reserved, admission,
                             output_dir, run_comparison, validate_protocol):
    """Bind existing runtime objects; caller retains server lifecycle/cleanup.

    Supply the real run_compare functions explicitly. This avoids another
    import/config framework and is straightforward to exercise with fake IO.
    The admission callback remains the caller's evidence/scope gate.
    """
    frozen, plan = validate_protocol(protocol, cases)
    boundary = QwenBoundary(transport=transport, tokenizer=tokenizer, journal=journal,
        protocol=frozen, plan=plan, deadline=request_deadline,
        max_input_reserved=max_input_reserved)
    result = run_comparison(frozen, cases, chat=boundary.call,
        count_tokens=tokenizer.count, admission=admission, output_dir=output_dir)
    journal.append('comparison_returned', dict(status=result['status'], calls=boundary.calls,
        halted=boundary.halted, reserved_input=boundary.reserved_input,
        reserved_output=boundary.reserved_output, actual_input=boundary.actual_input,
        actual_output=boundary.actual_output))
    return result
