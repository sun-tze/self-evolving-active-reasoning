"""Exact public-feedback state for Guess Number histories, and model-free references.

Standard library only. `reference` needs no model, network, benchmark checkout or
private file. `extract` reads separately supplied private run evidence (one or
more run_compare controller directories) and writes group aggregates that contain
no case identities, secrets or guesses.

Feedback follows AR-Bench `compare_guess`: digits in the right position, and
shared digits (multiset intersection) in other positions. The state after a
history is the set of four-distinct-digit secrets that would have produced every
logged feedback. It is computed offline; no actor in these runs was shown it.
"""
import argparse
import functools
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter
from itertools import permutations
from pathlib import Path

UNIVERSE = tuple(''.join(p) for p in permutations('0123456789', 4))
INDEX = {u: i for i, u in enumerate(UNIVERSE)}
QUESTION_TURNS = 25
GUESS_LIMIT = QUESTION_TURNS + 1
MIX_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9, 1.0)
SEED = 20261027
PHASES = ('train', 'dev', 'final')


def feedback(secret, guess):
    bulls = sum(a == b for a, b in zip(secret, guess))
    return bulls, sum((Counter(secret) & Counter(guess)).values()) - bulls


def code(bulls, cows):
    return 5 * bulls + bulls + cows


@functools.lru_cache(maxsize=None)
def row(guess):
    """Feedback codes of `guess` against every universe secret (distinct digits)."""
    digits = set(guess)
    out = bytearray(len(UNIVERSE))
    for i, s in enumerate(UNIVERSE):
        bulls = (s[0] == guess[0]) + (s[1] == guess[1]) + (s[2] == guess[2]) + (s[3] == guess[3])
        out[i] = 5 * bulls + len(digits.intersection(s))
    return bytes(out)


# ---------------------------------------------------------------- histories

def history_rows(guess_list, secret=None):
    """One row per logged guess; a blank guess is a parse failure with no feedback."""
    state, asked, informed, rows = list(range(len(UNIVERSE))), set(), False, []
    for turn, entry in enumerate(guess_list, 1):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError('Unexpected guess_list entry')
        guess, bulls, cows = entry
        if guess == '':
            rows.append({'turn': turn, 'parsed': False})
            continue
        if not (isinstance(guess, str) and re.fullmatch(r'[0-9]{4}', guess)
                and type(bulls) is int and type(cows) is int and 0 <= bulls <= bulls + cows <= 4):
            raise ValueError('Unexpected guess_list entry')
        if secret is not None and feedback(secret, guess) != (bulls, cows):
            raise ValueError('Logged feedback differs from recomputed feedback')
        before = len(state)
        consistent = INDEX.get(guess) in set(state)
        codes = row(guess)
        state = [i for i in state if codes[i] == code(bulls, cows)]
        if not state:
            raise ValueError('Logged feedback is inconsistent with every secret')
        rows.append({'turn': turn, 'parsed': True, 'informed': informed,
                     'distinct_digits': len(set(guess)) == 4, 'repeat': guess in asked,
                     'consistent': consistent, 'before': before, 'after': len(state),
                     'hit': bulls == 4})
        asked.add(guess)
        informed = True
    return rows


def summarize(rows):
    parsed = [r for r in rows if r['parsed']]
    informed = [r for r in parsed if r['informed']]
    questions = [r for r in parsed if r['turn'] <= QUESTION_TURNS]
    identified = next((r['turn'] for r in parsed if r['after'] == 1), None)
    last_correct = bool(rows) and rows[-1]['parsed'] and rows[-1]['hit']
    remaining = questions[-1]['after'] if questions else len(UNIVERSE)
    return {'logged_guesses': len(rows), 'parse_failures': len(rows) - len(parsed),
            'informed_guesses': len(informed),
            'consistent_informed': sum(r['consistent'] for r in informed),
            'repeats': sum(r['repeat'] for r in parsed),
            'non_distinct': sum(not r['distinct_digits'] for r in parsed),
            'bits_after_questions': math.log2(len(UNIVERSE) / remaining),
            'identified_turn': identified,
            'first_hit_turn': next((r['turn'] for r in parsed if r['hit']), None),
            'last_guess_correct': last_correct,
            'identified_but_last_wrong': identified is not None and not last_correct}


def aggregate(summaries):
    def total(key):
        return sum(s[key] for s in summaries)
    def ratio(a, b):
        return a / b if b else None
    parsed = total('logged_guesses') - total('parse_failures')
    turns = [s['identified_turn'] for s in summaries if s['identified_turn'] is not None]
    rates = [s['consistent_informed'] / s['informed_guesses'] for s in summaries if s['informed_guesses']]
    bits = [s['bits_after_questions'] for s in summaries]
    return {'episodes': len(summaries), 'logged_guesses': total('logged_guesses'),
            'parse_failure_rate': ratio(total('parse_failures'), total('logged_guesses')),
            'consistency_rate_pooled': ratio(total('consistent_informed'), total('informed_guesses')),
            'consistency_rate_episode_median': statistics.median(rates) if rates else None,
            'repeat_rate': ratio(total('repeats'), parsed),
            'non_distinct_rate': ratio(total('non_distinct'), parsed),
            'bits_after_questions_mean': statistics.fmean(bits) if bits else None,
            'bits_after_questions_median': statistics.median(bits) if bits else None,
            'identified_episodes': len(turns),
            'identified_turn_median': statistics.median(turns) if turns else None,
            'hit_any_episodes': sum(s['first_hit_turn'] is not None for s in summaries),
            'last_guess_correct_episodes': sum(s['last_guess_correct'] for s in summaries),
            'identified_but_last_wrong_episodes': sum(s['identified_but_last_wrong'] for s in summaries)}


# ------------------------------------------------------- reference policies

def _consistent_first(state, asked, rng):
    return UNIVERSE[state[0]]


def _consistent_random(state, asked, rng):
    return UNIVERSE[rng.choice(state)]


@functools.lru_cache(maxsize=None)
def _greedy(state):
    if len(state) == len(UNIVERSE):
        return UNIVERSE[0]   # every first guess splits the universe identically up to relabeling
    best, choice = None, None
    for g in state:
        codes = row(UNIVERSE[g])
        counts = Counter(codes[i] for i in state)
        score = sum(c * c for c in counts.values())
        if best is None or score < best:
            best, choice = score, g
    return UNIVERSE[choice]


def _consistent_greedy(state, asked, rng):
    """Consistent guess with the smallest expected remaining state (sum of squared bucket sizes)."""
    return _greedy(tuple(state))


def _mixture(p):
    def choose(state, asked, rng):
        if rng.random() < p:
            return UNIVERSE[rng.choice(state)]
        while True:
            guess = UNIVERSE[rng.randrange(len(UNIVERSE))]
            if guess not in asked:
                return guess
    return choose


def play(choose, secret, rng):
    """Guess until the secret is hit or GUESS_LIMIT guesses (25 questions + final) are used.

    Returns (hit turn or None, informed guesses, consistent informed guesses,
    state size after the 25 questions; 1 once the secret has been hit)."""
    state, asked, informed, consistent = list(range(len(UNIVERSE))), set(), 0, 0
    after_questions = None
    for turn in range(1, GUESS_LIMIT + 1):
        guess = choose(state, asked, rng)
        if guess in asked:
            raise AssertionError('Reference policies never repeat a guess')
        if turn > 1:
            informed += 1
            consistent += INDEX[guess] in set(state)
        if guess == secret:
            return turn, informed, consistent, 1 if turn <= QUESTION_TURNS else after_questions
        asked.add(guess)
        codes = row(guess)
        target = codes[INDEX[secret]]
        state = [i for i in state if codes[i] == target]
        if turn == QUESTION_TURNS:
            after_questions = len(state)
    return None, informed, consistent, after_questions


def policies():
    named = [('consistent_first', None, _consistent_first),
             ('consistent_greedy_expected_size', None, _consistent_greedy)]
    named += [(f'mixture_p{p:g}', p, _mixture(p)) for p in MIX_GRID]
    return named


def rng_for(name, index):
    digest = hashlib.sha256(f'{SEED}:{name}:{index}'.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def reference(secrets=UNIVERSE):
    results = []
    for name, p, choose in policies():
        turns, informed, consistent, sizes = Counter(), 0, 0, []
        for secret in secrets:
            turn, i, c, size = play(choose, secret, rng_for(name, INDEX[secret]))
            turns[turn] += 1
            informed += i
            consistent += c
            sizes.append(size)
        solved = sorted(t for t in turns.elements() if t is not None)
        results.append({
            'policy': name, 'consistent_choice_probability': p, 'secrets': len(secrets),
            'hit_within_26_guesses': len(solved), 'hit_within_25_guesses': sum(t <= 25 for t in solved),
            'hit_rate_within_26': len(solved) / len(secrets),
            'identified_after_25_questions': sum(s == 1 for s in sizes),
            'bits_after_25_questions_mean': statistics.fmean(math.log2(len(UNIVERSE) / s) for s in sizes),
            'measured_consistency_rate': consistent / informed if informed else None,
            'guesses_to_hit_mean': statistics.fmean(solved) if solved else None,
            'guesses_to_hit_median': statistics.median(solved) if solved else None,
            'guesses_to_hit_max': solved[-1] if solved else None,
            'distribution': {str(t): n for t, n in sorted(turns.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))}})
    return results


# ----------------------------------------------------------- private extract

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_text(value):
    """Native logs store the target as a digit string (prompting methods) or a list of ints (UoT)."""
    return value if isinstance(value, str) else ''.join(str(d) for d in value)


def _load_bound(controller, ref, artifacts):
    path = controller / ref['path']
    digest = sha256(path)
    if digest != ref['sha256'] or artifacts.get(ref['path']) != digest:
        raise ValueError('Episode bytes differ from the run result binding')
    return json.loads(path.read_text(encoding='utf-8'))


def controller_rows(controller):
    """Yield (phase, checkpoint, arm, case_id, status, episode-or-None) from one run_compare controller."""
    controller = Path(controller)
    result = json.loads((controller / 'result.json').read_text(encoding='utf-8'))
    artifacts = result['artifact_sha256']
    if artifacts.get('private-cases.json') != sha256(controller / 'private-cases.json'):
        raise ValueError('Case file differs from the run result binding')
    cases = json.loads((controller / 'private-cases.json').read_text(encoding='utf-8'))
    for item in result['evaluations']:
        episode = _load_bound(controller, item['episode'], artifacts) if item['status'] == 'complete' else None
        yield item['split'], item['checkpoint'], item['arm'], item['case_id'], item['status'], episode, cases
    for item in result['updates']:
        episode = _load_bound(controller, item['initial_episode'], artifacts)
        yield 'train', item['step'], item['arm'], item['case_id'], episode['status'], episode, cases


def extract(controllers):
    """Return (anonymous aggregates, case ids seen) for the leak check."""
    slots, sources, case_ids = {}, [], set()
    for controller in controllers:
        sources.append(sha256(Path(controller) / 'result.json'))
        for phase, checkpoint, arm, case_id, status, episode, cases in controller_rows(controller):
            case_ids.add(case_id)
            key = (phase, checkpoint, arm, case_id)
            complete = status == 'complete' and episode is not None and episode.get('official_result') is not None
            if key in slots and (slots[key] is not None) and complete:
                raise ValueError('Two complete records for one evaluation slot; supply one source per slot')
            if not complete:
                slots.setdefault(key, None)
                continue
            if episode.get('case_id', case_id) != case_id:
                raise ValueError('Episode case differs from its evaluation slot')
            secret = cases[case_id]
            native = episode['official_result']
            if target_text(native.get('true_number', secret)) != secret:
                raise ValueError('Native target differs from the bound case')
            summary = summarize(history_rows(native['guess_list'], secret))
            official = (episode.get('metrics') or {}).get('official_success')
            summary['official_success'] = official
            summary['last_guess_matches_official'] = official is None or official == summary['last_guess_correct']
            slots[key] = summary
    groups = {}
    for (phase, checkpoint, arm, _), summary in slots.items():
        group = groups.setdefault((phase, checkpoint, arm), {'complete': [], 'missing': 0})
        if summary is None:
            group['missing'] += 1
        else:
            group['complete'].append(summary)
    rows = []
    for (phase, checkpoint, arm), group in sorted(groups.items(), key=lambda kv: (PHASES.index(kv[0][0]), kv[0][1], kv[0][2])):
        summaries = group['complete']
        rows.append({'phase': phase, 'checkpoint': checkpoint, 'arm': arm, 'missing': group['missing'],
                     'official_success_episodes': sum(bool(s['official_success']) for s in summaries),
                     'last_guess_disagrees_with_official': sum(not s['last_guess_matches_official'] for s in summaries),
                     **aggregate(summaries)})
    return {'schema_version': 1, 'source_result_sha256': sources, 'groups': rows,
            'definitions': DEFINITIONS}, case_ids


DEFINITIONS = {
    'state': 'Four-distinct-digit secrets consistent with every logged (guess, feedback) pair; computed offline, never shown to an actor.',
    'consistency_rate': 'Share of parsed guesses made after at least one earlier parsed guess that were members of the current state.',
    'repeat_rate': 'Share of parsed guesses equal to an earlier parsed guess in the same episode.',
    'bits_after_questions': 'log2(5040 / state size) after the first 25 logged turns (all turns when fewer were logged).',
    'identified': 'The state became a single secret at some logged turn; a measure of information acquired, not of actor belief.',
    'identified_but_last_wrong': 'Identified, but the last logged guess was not the secret (a readout failure for native final-guess scoring).',
    'missing': 'Slots without a complete episode record; never counted as failures.'}


def leak_check(value, case_ids):
    """Structural check: outside the fixed definitions and hex digests, every string
    is an identifier-like name with no four-digit run and no case identity."""
    def strings(node):
        if isinstance(node, dict):
            for key, item in node.items():
                yield key
                if node is not value['definitions']:
                    yield from strings(item)
        elif isinstance(node, list):
            for item in node:
                yield from strings(item)
        elif isinstance(node, str):
            yield node
    for text in strings(value):
        if re.fullmatch(r'[0-9a-f]{64}', text):
            continue
        if (not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', text) or re.search(r'[0-9]{4}', text)
                or any(case_id in text for case_id in case_ids)):
            raise ValueError('A non-aggregate string would be exported')
    if value['definitions'] != DEFINITIONS:
        raise ValueError('Definitions changed')


def write_exclusive(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    ref = commands.add_parser('reference', help='Exhaustive model-free reference policies over all 5040 secrets')
    ref.add_argument('--output', type=Path, required=True)
    ext = commands.add_parser('extract', help='Anonymous state aggregates from private run_compare controllers')
    ext.add_argument('--controller', type=Path, action='append', required=True)
    ext.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'reference':
        write_exclusive(args.output, {'schema_version': 1, 'universe': len(UNIVERSE),
                                      'question_turns': QUESTION_TURNS, 'guess_limit': GUESS_LIMIT,
                                      'seed_rule': 'random.Random(int(sha256(f"20261027:{policy}:{secret_index}")[:16], 16))',
                                      'policies': reference()})
    else:
        value, case_ids = extract(args.controller)
        leak_check(value, case_ids)
        write_exclusive(args.output, value)
