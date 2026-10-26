"""Portable sequential extraction of the fixed selection mechanism.

call(request) -> {text: str, finish_reason: 'stop'|'length'} performs ONE
generation with the supplied model/messages/temperature/top_p/max_tokens/seed.
The adapter must pin protocol['revision'] and enforce context/resource limits.
feedback(case_id, guess) -> (same_position, different_position) owns the hidden
environment; no labels or case identifiers are passed in model requests.

repetitions = {'0': {'train': [18 opaque IDs], 'selection': [48 opaque IDs]},
               '1': {'train': [18 opaque IDs], 'selection': [48 opaque IDs]}}
cohorts = {'0': [100 opaque IDs], '1': [100 opaque IDs]}
The caller supplies new, disjoint cases and their fixed order. IDs affect seeds.
run_search returns all decisions and four frozen notes plus B0; evaluate checks
their canonical hash before all 1000 evaluations. run_experiment combines both.
Returned records may contain caller-owned transcripts/notes: they are not an
anonymous publication format. No files, network, models or installation here.

The sequential schedule preserves learning barriers, not serving concurrency.
This is mechanism reproduction on caller data, not an exact private-run replay.
"""
from copy import deepcopy
import hashlib
import json
import re

from question_policy import PublicMonitor, curator_messages
from selection_policy import select_policy, validate_note


RUNS = ('E0', 'R0', 'E1', 'R1')
ARMS = ('B0',) + RUNS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seed(case_id, slot, purpose='actor'):
    return int.from_bytes(hashlib.sha256(f'sun-gn-v1\n{case_id}\n{purpose}\n{slot}'.encode()).digest()[:8], 'big') % 2147483647


def parse_as_original(raw):
    matches = re.findall(r'\d{4}', raw)
    return ''.join(str(int(char)) for char in matches[-1]) if matches else None


def policy_hash(policies):
    return hashlib.sha256(json.dumps(policies, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_protocol(p):
    expected = dict(N=200, trajectories=2, updates_per_trajectory=6, batch_size=3,
        cohorts=2, cohort_N=100, selection_N=48, actor_tokens=128, curator_tokens=512,
        note_bytes=800, initial_note='', temperature=.7, top_p=.7, evaluation_arms=list(ARMS))
    require(all(p[key] == value for key, value in expected.items()), 'Changed fixed design')


def check_cases(repetitions, cohorts=None):
    require(set(repetitions) == {'0', '1'}, 'Require two learning blocks')
    groups = []
    for rep in ('0', '1'):
        require(set(repetitions[rep]) == {'train', 'selection'}, 'Learning case fields')
        groups.extend(((repetitions[rep]['train'], 18), (repetitions[rep]['selection'], 48)))
    if cohorts is not None:
        require(set(cohorts) == {'0', '1'}, 'Require two evaluation cohorts')
        groups.extend((cohorts[str(c)], 100) for c in range(2))
    ids = []
    for values, n in groups:
        require(type(values) is list and len(values) == n
                and all(type(value) is str and value for value in values), 'Case list/schema')
        ids.extend(values)
    require(len(ids) == len(set(ids)), 'Repeated case identifier across splits')
    return ids


def request(p, call, messages, cap, call_seed):
    response = call(dict(model=p['model'], messages=deepcopy(messages), temperature=p['temperature'],
        top_p=p['top_p'], max_tokens=cap, seed=call_seed))
    require(type(response) is dict and type(response.get('text')) is str
            and response.get('finish_reason') in ('stop', 'length'), 'Invalid model response')
    return deepcopy(response)  # Errors propagate; never retry, repair or replace.


def run_episode(p, call, feedback, case_id, note, tag, metadata):
    prompts = p['actor_prompts']
    system = prompts['propose_template'].format(turn=25)
    if note:
        system += prompts['header'] + note
    messages = [dict(role='system', content=system)]
    monitor, slots = PublicMonitor(), []
    for turn in range(1, 26):
        messages.append(dict(role='user', content=prompts['guess_prompt'].format(turn=turn)))
        response = request(p, call, messages, 128, seed(case_id, turn, tag))
        guess = parse_as_original(response['text'])
        messages.append(dict(role='assistant', content=response['text']))
        counts = None if guess is None else list(feedback(case_id, guess))
        text = prompts['refine_prompt'] if guess is None else prompts['eval_prompt'].format(
            same_pos=counts[0], diff_pos=counts[1])
        messages.append(dict(role='user', content=text))
        row = monitor.observe(guess, counts)
        slots.append(dict(**row, response=response, delivered_message=text))
        if monitor.resolved:
            break
    return dict(case_id=case_id, metadata=deepcopy(metadata), seed_tag=tag, slots=slots,
        public_history=monitor.records, resolved=monitor.resolved,
        capped25=len(slots) if monitor.resolved else 25,
        first_unique_turn=len(slots) if monitor.resolved else None)


def selection_vector(records):
    return [dict(selection_index=row['metadata']['index'], first_unique_turn=row['first_unique_turn'],
        capped25=row['capped25'], resolved=row['resolved'], queries=len(row['slots']))
        for row in sorted(records, key=lambda row:row['metadata']['index'])]


def run_search(p, repetitions, call, feedback):
    check_protocol(p)
    case_ids = check_cases(repetitions)
    notes = {run:'' for run in RUNS}
    cached, lineage, episodes, decisions = {}, {}, [], []

    def episode(cid, note, tag, metadata):
        result = run_episode(p, call, feedback, cid, note, tag, metadata)
        episodes.append(result)
        return result

    def bank(run, rd, note):
        rep = int(run[1])
        return selection_vector([episode(cid, note, f'selected-bank-{rep}',
            dict(phase='selection', run=run, repetition=rep, mode=run[0], round=rd, index=index))
            for index, cid in enumerate(repetitions[str(rep)]['selection'])])

    # Exactly one physical empty baseline per block, shared by E and R.
    for rep in range(2):
        vector = bank('B'+str(rep), -1, '')
        for mode in ('E', 'R'):
            cached[mode+str(rep)] = vector
            lineage[mode+str(rep)] = -1
    for rd in range(6):
        train = {}
        for run in ('E0', 'E1'):
            rep = int(run[1])
            train[run] = [episode(repetitions[str(rep)]['train'][position], notes[run],
                f'selected-train-{rep}', dict(phase='train', run=run, repetition=rep,
                    mode='E', round=rd, position=position)) for position in range(3*rd, 3*rd+3)]
        proposals = {}
        for run in RUNS:
            summaries = [{key:row[key] for key in ('public_history', 'resolved', 'capped25')}
                         for row in train.get(run, [])]
            response = request(p, call, curator_messages(notes[run], summaries, p['curator_prompt']),
                               512, seed('selected-'+run, rd, 'curator'))
            valid = validate_note(response['text'], response['finish_reason'])
            eligible = valid['accepted'] and not valid['keep'] and valid['normalized'] != notes[run]
            proposals[run] = dict(response=response, validation=valid, eligible=eligible)
        # Complete every eligible challenger before any round decision.
        candidates = {run:bank(run, rd, proposals[run]['validation']['normalized'])
                      for run in RUNS if proposals[run]['eligible']}
        for run in RUNS:
            proposal = proposals[run]
            decision = select_policy(incumbent_note=notes[run], challenger_raw=proposal['response']['text'],
                finish_reason=proposal['response']['finish_reason'],
                incumbent_episodes=cached[run] if proposal['eligible'] else None,
                challenger_episodes=candidates.get(run))
            decisions.append(dict(run=run, round=rd, incumbent_round=lineage[run],
                response=proposal['response'], decision=decision))
            if decision['promoted']:
                cached[run], lineage[run] = candidates[run], rd
            notes[run] = decision['selected_note']
    policies = dict(B0='', **notes)
    return dict(policies=policies, policies_sha256=policy_hash(policies), decisions=decisions,
        episodes=episodes, curator_calls=24, learning_case_ids=case_ids)


def evaluate(p, search, cohorts, call, feedback):
    check_protocol(p)
    policies = deepcopy(search['policies'])
    require(set(policies) == set(ARMS) and policies['B0'] == ''
            and policy_hash(policies) == search['policies_sha256'], 'Frozen policies changed')
    require(len(search['decisions']) == 24 and search['curator_calls'] == 24, 'Incomplete learning')
    require(set(cohorts) == {'0', '1'} and all(type(cohorts[c]) is list and len(cohorts[c]) == 100
            for c in ('0', '1')), 'Require both complete cohorts')
    ids = cohorts['0'] + cohorts['1']
    require(all(type(cid) is str and cid for cid in ids) and len(set(ids)) == 200
            and not set(ids) & set(search['learning_case_ids']), 'Evaluation cases overlap')
    results = []
    for cohort in range(2):
        for index, cid in enumerate(cohorts[str(cohort)]):
            ordinal = cohort*100+index
            order = ARMS[ordinal % 5:] + ARMS[:ordinal % 5]
            for arm in order:
                results.append(run_episode(p, call, feedback, cid, policies[arm],
                    f'selected-evaluation-cohort-{cohort}', dict(phase='evaluation', arm=arm,
                        cohort=cohort, index=index, ordinal=ordinal)))
    return results


def run_experiment(p, repetitions, cohorts, call, feedback):
    check_cases(repetitions, cohorts)
    search = run_search(p, repetitions, call, feedback)
    return dict(search=search, evaluation=evaluate(p, search, cohorts, call, feedback))
