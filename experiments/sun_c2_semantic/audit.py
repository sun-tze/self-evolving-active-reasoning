"""Validate frozen inputs/reviews and analyze without inference or score mutation."""
import argparse
import itertools
import statistics
from pathlib import Path
from experiments.sun_c2_real.common import read_json, require
from .prepare import HERE, ROOT, ANALYSIS_SHA, POSITIONS, make_packets, sha, write

AXES = ('semantic_correctness', 'referee_consistency', 'unsupported_severity', 'evidence_faithfulness')


def verify_inputs(directory=HERE):
    frozen = directory / 'frozen'
    freeze = read_json(frozen / 'FREEZE.json')
    for path, digest in freeze['sha256'].items():
        require(sha(frozen / path) == digest, 'frozen file changed: ' + path)
    original_path = ROOT / 'experiments/sun_c2_real/reports/pilot_25941208_analysis.json'
    require(sha(original_path) == ANALYSIS_SHA == freeze['analysis_sha256'], 'historical analysis changed')
    originals = {r['request_id']: r for r in read_json(original_path)['responses']}
    source = read_json(frozen / 'source_selected.json')
    require(len(source) == 20 and {(r['index'], r['cell']) for r in source} ==
            {(i, c) for i in POSITIONS for c in 'ABCD'}, 'incomplete/duplicate case pairing')
    require(len({r['request_id'] for r in source}) == 20, 'duplicate request identity')
    for r in source:
        old = originals[r['request_id']]
        require((r['index'], r['cell'], r['answer'], r['character_f1']) ==
                (old['index'], old['cell'], old['response']['text'], old['score']), 'source answer/score mismatch')
    packets, mappings = make_packets(source, sha(directory / 'rubric_v1.json'))
    require(mappings == read_json(frozen / 'unblind_map.json'), 'unblinding map changed')
    for reviewer in ('r1', 'r2'):
        require(packets[reviewer] == read_json(frozen / f'packet_{reviewer}.json'), 'packet reconstruction mismatch')
    return source, packets, mappings


def validate_review(review, packet, packet_sha):
    require(review['reviewer_id'] == packet['reviewer_id'], 'reviewer mismatch')
    require(review['rubric_sha256'] == packet['rubric_sha256'], 'rubric mismatch')
    require(review['packet_sha256'] == packet_sha, 'packet hash mismatch')
    require(isinstance(review['independence_statement'], str) and bool(review['independence_statement'].strip()), 'missing independence statement')
    items = {x['item_id']: x for x in packet['items']}
    ratings = review['ratings']
    require(len(ratings) == len(items) and {r['item_id'] for r in ratings} == set(items), 'incomplete/duplicate ratings')
    for rating in ratings:
        item = items[rating['item_id']]
        for axis in AXES:
            require(type(rating[axis]) is int and 0 <= rating[axis] <= 2, 'invalid ordinal ' + axis)
        require(rating['evidence_faithfulness'] == min(rating['referee_consistency'], 2-rating['unsupported_severity']), 'faithfulness formula mismatch')
        require(isinstance(rating['rationale'], str) and bool(rating['rationale'].strip()), 'missing rationale')
        require(isinstance(rating['issues'], list), 'issues must be a list')
        for issue in rating['issues']:
            require(issue['kind'] in ('contradicted', 'unsupported', 'ambiguous'), 'invalid issue kind')
            require(issue['role'] in ('central', 'peripheral'), 'invalid issue role')
            require(bool(issue['answer_quote']) and issue['answer_quote'] in item['answer'], 'quote not in answer')
            valid_refs = {'SURFACE'} | {h['ref'] for h in item['history']}
            require(isinstance(issue['evidence_refs'], list) and set(issue['evidence_refs']) <= valid_refs, 'invalid evidence reference')
            require(issue['kind'] != 'contradicted' or bool(issue['evidence_refs']), 'contradiction needs evidence')
            require(isinstance(issue['reason'], str) and bool(issue['reason'].strip()), 'issue needs reason')
            if issue['kind'] in ('unsupported', 'contradicted'):
                minimum = 2 if issue['role'] == 'central' else 1
                require(rating['unsupported_severity'] >= minimum, 'issue contradicts unsupported score')
            if issue['kind'] == 'contradicted':
                maximum = 0 if issue['role'] == 'central' else 1
                require(rating['referee_consistency'] <= maximum, 'issue contradicts consistency score')
        if rating['unsupported_severity']:
            role = 'central' if rating['unsupported_severity'] == 2 else 'peripheral'
            require(any(x['kind'] in ('unsupported', 'contradicted') and x['role'] == role for x in rating['issues']), 'unsupported severity lacks matching issue')
        if rating['referee_consistency'] < 2:
            role = 'central' if rating['referee_consistency'] == 0 else 'peripheral'
            require(any(x['kind'] == 'contradicted' and x['role'] == role for x in rating['issues']), 'contradiction severity lacks matching issue')
    return {r['item_id']: r for r in ratings}


def agreement(left, right):
    n = len(left)
    observed = sum((a-b)**2 for a, b in zip(left, right)) / n
    expected = sum((a-b)**2 for a in left for b in right) / (n*n)
    return {'n': n, 'exact_count': sum(a == b for a, b in zip(left, right)),
            'exact_fraction': sum(a == b for a, b in zip(left, right))/n,
            'quadratic_weighted_kappa': None if expected == 0 else 1-observed/expected}


def contrasts(scores):
    a, b, c, d = (scores[x] for x in 'ABCD')
    return {'writer_marginal': ((b-a)+(d-c))/2, 'dialogue_marginal': ((c-a)+(d-b))/2,
            'interaction': (d-c)-(b-a), 'writer_bare_B-A': b-a, 'writer_first_D-C': d-c,
            'dialogue_plain_C-A': c-a, 'dialogue_evidence_D-B': d-b}


def rank_pairs(f1, metric):
    def sign(v):
        return 0 if abs(v) <= 1e-12 else (1 if v > 0 else -1)
    pairs = []
    for a, b in itertools.combinations('ABCD', 2):
        fd, md = f1[a]-f1[b], metric[a]-metric[b]
        fs, ms = sign(fd), sign(md)
        kind = ('both_tie' if fs == ms == 0 else 'f1_tie' if fs == 0 else
                'metric_tie' if ms == 0 else 'reversal' if fs != ms else 'same')
        pairs.append({'pair': a+b, 'f1_difference': fd, 'metric_difference': md, 'classification': kind})
    return pairs


def analyze(directory=HERE):
    source, packets, mappings = verify_inputs(directory)
    reviews, by_case = {}, {}
    review_sha = {}
    for who in ('r1', 'r2'):
        path = directory / 'reviews' / f'{who}.json'
        reviews[who] = validate_review(read_json(path), packets[who], sha(directory / 'frozen' / f'packet_{who}.json'))
        review_sha[who] = sha(path)
        by_case[who] = {(mappings[who][k]['index'], mappings[who][k]['cell']): v for k, v in reviews[who].items()}
    rows, disagreements = [], []
    for source_row in sorted(source, key=lambda r: (r['index'], r['cell'])):
        key = (source_row['index'], source_row['cell'])
        r1, r2 = (by_case[who][key] for who in ('r1', 'r2'))
        row = {'index': key[0], 'cell': key[1], 'character_f1': source_row['character_f1'],
               'semantic_correctness': (r1['semantic_correctness']+r2['semantic_correctness'])/2,
               'evidence_faithfulness': (r1['evidence_faithfulness']+r2['evidence_faithfulness'])/2,
               'reviewer_item_ids': {who: by_case[who][key]['item_id'] for who in ('r1', 'r2')},
               'raw_scores': {who: {a: by_case[who][key][a] for a in AXES} for who in ('r1', 'r2')}}
        rows.append(row)
        different = {a: [r1[a], r2[a]] for a in AXES if r1[a] != r2[a]}
        if different:
            disagreements.append({'index': key[0], 'cell': key[1], 'differences': different,
                                  'rationales': {'r1': r1['rationale'], 'r2': r2['rationale']}})
    metrics = ('character_f1', 'semantic_correctness', 'evidence_faithfulness')
    means = {c: {m: statistics.mean(r[m] for r in rows if r['cell'] == c) for m in metrics} for c in 'ABCD'}
    per_case = []
    for index in POSITIONS:
        cells = {r['cell']: r for r in rows if r['index'] == index}
        scores = {m: {c: cells[c][m] for c in 'ABCD'} for m in metrics}
        per_case.append({'index': index, 'scores': scores,
                         'contrasts_normalized': {m: contrasts({c: v/(1 if m == 'character_f1' else 2) for c, v in scores[m].items()}) for m in metrics},
                         'ranking_pairs': {m: rank_pairs(scores['character_f1'], scores[m]) for m in metrics[1:]}})
    reversals = {}
    for m in metrics[1:]:
        pairs = [p for case in per_case for p in case['ranking_pairs'][m]]
        reversals[m] = {'within_case_counts': {kind: sum(p['classification'] == kind for p in pairs) for kind in ('reversal', 'same', 'metric_tie', 'f1_tie', 'both_tie')},
                        'total_pairs': len(pairs), 'condition_mean_pairs': rank_pairs({c: means[c]['character_f1'] for c in 'ABCD'}, {c: means[c][m] for c in 'ABCD'})}
    indicators = {who: {c: {'fully_semantically_correct': sum(by_case[who][(i,c)]['semantic_correctness'] == 2 for i in POSITIONS),
                            'central_contradiction': sum(by_case[who][(i,c)]['referee_consistency'] == 0 for i in POSITIONS),
                            'central_unsupported': sum(by_case[who][(i,c)]['unsupported_severity'] == 2 for i in POSITIONS), 'denominator': 5} for c in 'ABCD'} for who in ('r1', 'r2')}
    return {'format': 'c2_semantic_audit_v1', 'n_cases': 5, 'n_answers': 20, 'n_reviewers': 2,
            'score_scale': 'Semantic/faithfulness means 0..2; contrasts divide these by 2; historical F1 unchanged 0..1.',
            'freeze_sha256': sha(directory / 'frozen/FREEZE.json'), 'raw_review_sha256': review_sha,
            'original_analysis_sha256': ANALYSIS_SHA, 'new_inference_calls': 0,
            'cell_means': means, 'rows': rows, 'per_case': per_case,
            'contrasts_normalized': {m: {name: statistics.mean(case['contrasts_normalized'][m][name] for case in per_case) for name in per_case[0]['contrasts_normalized'][m]} for m in metrics},
            'agreement': {axis: agreement([by_case['r1'][(r['index'],r['cell'])][axis] for r in rows], [by_case['r2'][(r['index'],r['cell'])][axis] for r in rows]) for axis in AXES},
            'disagreements': disagreements, 'ranking_reversals': reversals, 'answer_level_indicators': indicators,
            'limitations': ['Post-hoc pilot rubric; frozen before these ratings and remaining 65 outcomes.', 'Independent same-model executions, not independent human experts.', 'Procedural blinding; four answers share case context.', 'No adjudication replaces original ratings; primary is the mean of two.', 'Five cases, descriptive only; no new inference or significance claims.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite results. Choose a new output path.')
    result = analyze()
    write(args.output, result)
    print('Validated two blind reviews; saved descriptive audit of existing 20 answers.')


if __name__ == '__main__':
    main()
