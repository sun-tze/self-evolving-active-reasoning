"""Depth-1 UoT simulation batching: ordered request plan plus exact serial reducer.

Local prototype only: stdlib, no network, model, process, file I/O or activation.
For simulate_depth == 1 it reproduces the work of the pinned
gn_evaluator._gn_analyze_candidates_with_simulation (REFERENCE_SHA256):

* plan_depth1 draws from the caller's episode random.Random exactly as the
  serial original does (one rng.sample per candidate/feedback branch, candidate
  order then partition insertion order) and builds every request up front. At
  depth 1 no response affects control flow or RNG, so branches are independent
  given those pre-drawn samples.
* The caller-supplied dispatch(requests) is invoked once and never retried.
  Responses may arrive in any order but must echo logical_id and request_sha256.
* reduce_depth1 restores serial order, then applies the original parser (its
  weaknesses kept), path reductions, float accumulation order, usage sums and
  stable descending sort. Any missing, duplicate, unknown or invalid response
  raises BatchFailure; partial scores are never returned.

Claimed: identical request bytes, RNG progression and reduction GIVEN IDENTICAL
RESPONSES. Not claimed: that a batched server generates the same text as serial
calls (batch composition can change logits even with an identical seed), nor any
speedup. Requests use a new schema (PLAN_FORMAT) without legacy upstream_caller /
call_index metadata, so uot_binding.classify_uot_metadata rejects them by design.
NOT integrated with arbench_adapter (serial inference closure, call counter,
events), QwenBoundary (holds its lock across transport), uot_binding or the
controller pins/admission; that needs a new, separately reviewed adapter and
serving boundary.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import random
import re

PLAN_FORMAT = 'gn_uot_depth1_sim_request_v1'
REFERENCE_SHA256 = {
    'gn_evaluator.py': 'aa6643d004ed7b510452ce9fe581b371112616a6702dfee3a26132759cb221a9',
    'utils_gn.py': 'f57cc0a77615859ea44fad07a0e5b849aa95e6d428963d72d3c49405c20274e7',
}
ROLE = 'uot_simulation'
MAX_TOKENS = 512               # frozen per-call cap (uot_binding ROLES, CallLimits)
TEMPERATURE, TOP_P = 0.1, 0.9  # hard-coded by _run_uot_evaluation for simulations
CANDIDATES = 3                 # _run_uot_evaluation always passes exactly three
SAMPLE_LIMIT = 50
EPISODE_CALL_BOUND = 1075      # uot_binding: 25 * (1 + 3 * 14)
RESPONSE_KEYS = frozenset({'logical_id', 'request_sha256', 'text', 'finish_reason', 'usage'})
USAGE_KEYS = frozenset({'prompt_tokens', 'completion_tokens'})


class BatchFailure(RuntimeError):
    """Dispatch raised, or returned incomplete/invalid responses. Never retried."""


def canonical_bytes(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _feedback(a, b):
    """utils_gn.compare_guess (same_pos, diff_pos) for equal-length digit sequences."""
    same = sum(x == y for x, y in zip(a, b))
    return same, sum((Counter(a) & Counter(b)).values()) - same


def _digits(value, what):
    if (type(value) is not list or len(value) != 4 or
            any(type(d) is not int or not 0 <= d <= 9 for d in value)):
        raise ValueError(what + ' must be a list of four integer digits')
    return tuple(value)


def feedback_partition(answers, guess):
    """_gn_estimate_uncertainty_reduction's partition; insertion order preserved."""
    partition = {}
    for candidate in answers:
        s, d = _feedback(candidate, guess)
        key = f"{s}_{d}"
        partition[key] = partition.get(key, 0) + 1
    return partition


def simulation_prompt(total, guess, same_pos, diff_pos, sample_answers):
    """Byte-identical to the prompt built in _gn_calculate_remaining_with_model."""
    guess_str = "".join([str(d) for d in guess])
    feedback_str = f"{same_pos} digits are present in the answer and in the correct positions, {diff_pos} digits are present in the answer but in the different positions"
    sample_size = min(SAMPLE_LIMIT, total)
    answer_set_str = "\n".join(["".join(map(str, ans)) for ans in sample_answers])
    if total > sample_size:
        answer_set_str += f"\n... (and {total - sample_size} more possible answers)"
    return f"""I am playing a number guessing game. The secret is a 4-digit number where all digits are different (0-9).
When I guess: {guess_str}
I receive feedback: {feedback_str}

Here is my current set of possible answers (showing {sample_size} out of {total} total):
{answer_set_str}

Based on this feedback and my current set of possible answers, please determine which numbers from my answer set are still valid.
A number is valid if it would give exactly the same feedback ({same_pos} correct digits in correct positions, {diff_pos} correct digits in wrong positions) when compared to my guess {guess_str}.

Format your answer as a list of 4-digit numbers, one per line. Only include numbers that satisfy the feedback criteria."""


def parse_remaining(text, guess, same_pos, diff_pos):
    """Original simulation parser, weaknesses deliberately kept: every
    non-overlapping \\d{4} (any Unicode decimal digits, also chunks of longer
    digit runs) whose feedback matches is kept, including duplicates,
    repeated-digit numbers and numbers outside the answer set. Truncated text is
    parsed as-is; empty text keeps nothing (scored as maximal reduction)."""
    verified = []
    for number_str in re.findall(r"\d{4}", text):
        candidate = [int(d) for d in number_str]
        s, d = _feedback(candidate, guess)
        if s == same_pos and d == diff_pos:
            verified.append(candidate)
    return verified


@dataclass(frozen=True)
class Depth1Plan:
    seed: int
    first_call_index: int
    answer_set_size: int
    candidates: tuple         # caller's guess objects, returned by reference as the original does
    frozen_candidates: tuple  # value snapshot used for prompts and parsing
    branches: tuple           # per candidate: ((partition_key, count, request_position), ...)
    ids: tuple                # logical IDs, serial order
    wire: tuple               # canonical request bytes (without request_sha256), serial order
    shas: tuple               # sha256 of each wire entry
    digest: str               # sha256 binding the whole ordered plan

    def requests(self):
        """Fresh request dicts in serial order; mutating them cannot alter the plan."""
        return [dict(json.loads(w), request_sha256=s) for w, s in zip(self.wire, self.shas)]


def plan_depth1(answer_set, candidates, *, rng, seed, first_call_index, depth):
    """Build every depth-1 simulation request, drawing rng exactly as the original.

    rng is the episode's random.Random (the adapter's namespace `random`). seed is
    the explicit episode seed the adapter sends with every call. first_call_index
    is the call_index the serial adapter would give the first simulation of this
    turn; it is recorded as serial_call_index, not as adapter-issued metadata.
    """
    if type(depth) is not int or depth != 1:
        raise ValueError('Only simulate_depth == 1 is supported')
    if not isinstance(rng, random.Random) or isinstance(rng, random.SystemRandom):
        raise TypeError('The seeded episode random.Random instance is required')
    if type(seed) is not int or type(first_call_index) is not int or first_call_index < 1:
        raise ValueError('Explicit integer seed and positive serial call index required')
    if type(answer_set) is not list or type(candidates) is not list or len(candidates) != CANDIDATES:
        raise ValueError('answer_set list and exactly three candidate guesses required')
    answers = tuple(_digits(a, 'answer') for a in answer_set)
    frozen = tuple(_digits(c, 'candidate') for c in candidates)
    total = len(answers)
    branches, ids, wire, shas = [], [], [], []
    for ci, guess in enumerate(frozen):
        rows = []
        # The original returns (0.0, {}) with no call or draw when len(answer_set) <= 1.
        partition = feedback_partition(answers, guess) if total > 1 else {}
        for pi, (key, count) in enumerate(partition.items()):
            same_pos, diff_pos = map(int, key.split('_'))
            sample = rng.sample(answers, min(SAMPLE_LIMIT, total))
            index = first_call_index + len(wire)
            body = {
                'format': PLAN_FORMAT, 'role': ROLE,
                'logical_id': f'sim-{index}-c{ci}-p{pi}', 'serial_call_index': index,
                'candidate_index': ci, 'partition_key': key,
                'messages': [{'role': 'user', 'content': simulation_prompt(
                    total, guess, same_pos, diff_pos, sample)}],
                'max_tokens': MAX_TOKENS, 'temperature': TEMPERATURE, 'top_p': TOP_P,
                'seed': seed,
            }
            rows.append((key, count, len(wire)))
            ids.append(body['logical_id'])
            wire.append(canonical_bytes(body))
            shas.append(hashlib.sha256(wire[-1]).hexdigest())
        branches.append(tuple(rows))
    if first_call_index + len(wire) - 1 > EPISODE_CALL_BOUND:
        raise ValueError('Plan exceeds the native per-episode call bound; nothing dispatched')
    digest = hashlib.sha256((PLAN_FORMAT + '\n' + '\n'.join(shas)).encode('ascii')).hexdigest()
    return Depth1Plan(seed, first_call_index, total, tuple(candidates), frozen,
                      tuple(branches), tuple(ids), tuple(wire), tuple(shas), digest)


def order_responses(plan, responses):
    """Validate one complete batch; return copies in original serial order."""
    if type(responses) not in (list, tuple):
        raise BatchFailure('Batch must return a list of responses')
    position = {rid: i for i, rid in enumerate(plan.ids)}
    ordered = [None] * len(plan.ids)
    for r in responses:
        if type(r) is not dict or set(r) != RESPONSE_KEYS or type(r['logical_id']) is not str:
            raise BatchFailure('Malformed batch response')
        i = position.get(r['logical_id'])
        if i is None:
            raise BatchFailure('Unknown logical_id ' + r['logical_id'])
        if ordered[i] is not None:
            raise BatchFailure('Duplicate response for ' + r['logical_id'])
        usage = r['usage']
        if (type(r['request_sha256']) is not str or r['request_sha256'] != plan.shas[i] or
                type(r['text']) is not str or r['finish_reason'] not in ('stop', 'length') or
                type(usage) is not dict or set(usage) != USAGE_KEYS or
                any(type(v) is not int or v < 0 for v in usage.values()) or
                usage['completion_tokens'] > MAX_TOKENS):
            raise BatchFailure('Invalid response for ' + r['logical_id'])
        ordered[i] = deepcopy(r)
    missing = sum(r is None for r in ordered)
    if missing:
        raise BatchFailure(f'Incomplete batch: {missing} of {len(ordered)} responses missing')
    return ordered


def reduce_depth1(plan, responses):
    """(candidate_scores, total_in, total_out) exactly as the original computes them."""
    if tuple(map(tuple, plan.candidates)) != plan.frozen_candidates:
        raise ValueError('Candidate guesses changed after planning')
    ordered = order_responses(plan, responses)
    total = plan.answer_set_size
    candidate_scores, total_in, total_out = [], 0, 0
    for cand, guess, rows in zip(plan.candidates, plan.frozen_candidates, plan.branches):
        total_uncertainty, path_reductions, ui, uo = 0.0, {}, 0, 0
        for key, count, position in rows:
            same_pos, diff_pos = map(int, key.split('_'))
            prob = count / total
            response = ordered[position]
            new_answer_set = parse_remaining(response['text'], guess, same_pos, diff_pos)
            ui += response['usage']['prompt_tokens']
            uo += response['usage']['completion_tokens']
            # Both original depth-1 exits (<= 1 remaining, depth == 1) do exactly this.
            path_reductions[key] = total - len(new_answer_set)
            total_uncertainty += prob * (total - len(new_answer_set))
        total_in += ui
        total_out += uo
        candidate_scores.append((cand, total_uncertainty, path_reductions))
    candidate_scores.sort(key=lambda x: x[1], reverse=True)  # stable: ties keep proposal order
    return candidate_scores, total_in, total_out


def run_depth1(answer_set, candidates, *, rng, seed, first_call_index, depth, dispatch):
    """Plan, dispatch once, reduce. Returns the original triple or raises; never partial.

    dispatch(requests) -> list of response dicts with exactly RESPONSE_KEYS. A future
    serving boundary implementing it owns admission, journaling and token checks.
    """
    plan = plan_depth1(answer_set, candidates, rng=rng, seed=seed,
                       first_call_index=first_call_index, depth=depth)
    responses = []
    if plan.ids:  # the original makes no call when len(answer_set) <= 1
        try:
            responses = list(dispatch(plan.requests()))
        except Exception as exc:
            raise BatchFailure('Batch dispatch failed; not retried') from exc
    return reduce_depth1(plan, responses)


def offline_fake_dispatch(responder, arrival_order=None):
    """Offline test double; no I/O. responder(request) -> (text, finish_reason,
    prompt_tokens, completion_tokens). arrival_order(n) -> index permutation used
    to return responses out of order. dispatch.batches records each batch size."""
    def dispatch(requests):
        dispatch.batches.append(len(requests))
        out = []
        for request in requests:
            text, finish_reason, prompt_tokens, completion_tokens = responder(request)
            out.append({'logical_id': request['logical_id'],
                        'request_sha256': request['request_sha256'],
                        'text': text, 'finish_reason': finish_reason,
                        'usage': {'prompt_tokens': prompt_tokens,
                                  'completion_tokens': completion_tokens}})
        return [out[i] for i in arrival_order(len(out))] if arrival_order else out
    dispatch.batches = []
    return dispatch
