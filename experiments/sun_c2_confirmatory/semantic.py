"""Deterministic review packets and paired semantic analysis; no model calls."""
import argparse
import statistics
import tempfile
from pathlib import Path
from .study import C, HERE, RUBRIC, file_sha, verify, targets
from .run import plan
import analyze as A
from experiments.sun_c2_semantic.audit import AXES, validate_review, agreement, rank_pairs


def verified_analysis(bundle, analysis_path, freeze_sha, run_dir):
    run_dir = Path(run_dir)
    record = C.read_json(run_dir/'run.json')
    saved = record['plan']
    expected, _, _ = plan(bundle, freeze_sha, saved['endpoint'], saved['server_marker'])
    C.require(saved == expected, 'run does not match the frozen execution plan')
    analysis = C.read_json(analysis_path)
    # Recompute from the append-only raw journal, including response parsing and F1.
    with tempfile.TemporaryDirectory() as temporary:
        rebuilt = A.analyze(bundle, expected['manifest_sha256'], run_dir, Path(temporary)/'analysis.json')
    C.require(analysis == rebuilt, 'analysis differs from the original run journal')
    return analysis


def packets(bundle, analysis_path, freeze_sha, run_dir):
    frozen, manifest, selected = verify(bundle, freeze_sha)
    analysis = verified_analysis(bundle, analysis_path, freeze_sha, run_dir)
    C.require(analysis['scope'] == 'confirmatory' and analysis['provenance']['manifest_sha256'] == frozen['manifest_sha256'], 'analysis belongs to another population')
    answers = analysis['responses']
    C.require(len(answers) == 260 and {r['request_id'] for r in answers} == {r['request_id'] for r in selected}, 'response identity/coverage mismatch')
    responses = {r['request_id']: r for r in answers}
    sources = {r['request_id']: r for r in selected}
    indices = sorted({r['index'] for r in selected})
    refs = targets(bundle, manifest, indices)
    anchors = C.read_json(HERE/'anchors.json')['cases']
    result = {}
    for name, assignment in C.read_json(HERE/'assignments.json').items():
        items = []
        for entry in assignment:
            r = sources[entry['request_id']]
            response = responses[r['request_id']]
            C.require(response['index'] == r['index'] and response['cell'] == r['cell'], 'response pairing mismatch')
            if response['status'] != 'completed':
                continue
            text = response['response']['text']
            C.require(type(text) is str, 'invalid answer text')
            items.append({'item_id': entry['item_id'], 'surface': r['observation']['surface'],
                          'history': [{'ref':f'H{n:02d}',**h} for n,h in enumerate(r['observation']['history'],1)],
                          'reference_solution': refs[r['index']]['bottom'], 'semantic_anchors': anchors[str(r['index'])],
                          'answer': text})
        result[name] = {'reviewer_id': name, 'rubric_sha256':file_sha(RUBRIC),
                        'review_protocol_sha256':file_sha(HERE/'review_protocol.json'), 'items':items}
    return result, analysis, selected


def build_packets(bundle, analysis_path, freeze_sha, run_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    expected, _, _ = packets(bundle, analysis_path, freeze_sha, run_dir)
    for name, packet in expected.items():
        C.write_json(out/f'{name}.json', packet)
    C.write_json(out/'binding.json', {'freeze_sha256':freeze_sha, 'analysis_sha256':file_sha(analysis_path),
                 'packets':{name:file_sha(out/f'{name}.json') for name in expected}})


def summarize(rows, indices, flags):
    metrics = ['semantic_correctness', 'evidence_faithfulness']
    complete = all(all(r.get(m) is not None for m in metrics) for r in rows)
    contrasts, means = {}, {}
    for metric in metrics:
        cases = [{r['cell']: None if r[metric] is None else r[metric]/2 for r in rows if r['index']==i} for i in indices]
        _, contrasts[metric] = A.paired_stats(cases)
        complete_cases = [s for s in cases if all(v is not None for v in s.values())]
        contrasts[metric]['complete_case_descriptive'] = A.paired_stats(complete_cases)[1] if complete_cases else None
        for cell in 'ABCD':
            values = [r[metric] for r in rows if r['cell']==cell and r[metric] is not None]
            means.setdefault(cell,{})[metric] = {'mean_0_to_2':statistics.mean(values) if values else None,
                                                 'fully_scored_2':sum(v == 2 for v in values),
                                                 'available':len(values), 'expected':len(indices)}
    primary = contrasts['semantic_correctness']['writer_marginal']
    return {'cell_means':means, 'paired_contrasts_normalized':contrasts,
            'primary_status': 'incomplete_withheld' if not complete else
               ('positive_effect_supported' if primary['ci95'] is not None and primary['ci95'][0]>0 else 'positive_effect_not_established'),
            'reference_flag_sensitivity':{group:{'cases':sum((i in flags)==flag for i in indices),
                'semantic_mean_0_to_2':statistics.mean(values) if (values:=[r['semantic_correctness'] for r in rows if (r['index'] in flags)==flag and r['semantic_correctness'] is not None]) else None}
                 for group,flag in [('flagged',True),('unflagged',False)]}}


def aggregate(bundle, analysis_path, freeze_sha, run_dir, packet_dir, review_dir, out):
    expected, analysis, selected = packets(bundle, analysis_path, freeze_sha, run_dir)
    packet_dir, review_dir = Path(packet_dir), Path(review_dir)
    binding = C.read_json(packet_dir/'binding.json')
    C.require(binding['freeze_sha256'] == freeze_sha and binding['analysis_sha256'] == file_sha(analysis_path), 'packet source changed')
    by_request = {r['request_id']:{} for r in selected}
    assignments = C.read_json(HERE/'assignments.json')
    annotation_hashes = {}
    for name, packet in expected.items():
        path = packet_dir/f'{name}.json'
        C.require(C.read_json(path) == packet and file_sha(path) == binding['packets'][name], 'review packet changed')
        review = review_dir/f'{name}.json'
        if not review.exists():
            continue
        rating = validate_review(C.read_json(review), packet, file_sha(path))
        annotation_hashes[name] = file_sha(review)
        mapping = {r['item_id']:r['request_id'] for r in assignments[name]}
        for item, value in rating.items():
            by_request[mapping[item]][name[:2]] = {axis:value[axis] for axis in AXES}
    f1 = {r['request_id']:r for r in analysis['responses']}
    rows, disagreements = [], []
    for r in sorted(selected,key=lambda r:(r['index'],r['cell'])):
        ratings = by_request[r['request_id']]
        row = {'index':r['index'],'cell':r['cell'],'request_id':r['request_id'],
               'character_f1':f1[r['request_id']]['score'], 'inference_status':f1[r['request_id']]['status'],
               'raw_scores':ratings}
        for metric in ['semantic_correctness','evidence_faithfulness']:
            row[metric] = statistics.mean(ratings[w][metric] for w in ('r1','r2')) if set(ratings)=={'r1','r2'} else None
        if len(ratings)==2:
            diff = {axis:[ratings['r1'][axis],ratings['r2'][axis]] for axis in AXES if ratings['r1'][axis]!=ratings['r2'][axis]}
            if diff:
                disagreements.append({'index':r['index'],'cell':r['cell'],'differences':diff})
        rows.append(row)
    indices = sorted({r['index'] for r in selected})
    flags = {int(i) for i,anchor in C.read_json(HERE/'anchors.json')['cases'].items() if anchor.get('reference_flags')}
    result = summarize(rows, indices, flags)
    paired = [r for r in rows if len(r['raw_scores'])==2]
    result.update({'format':'c2_confirmatory_semantic_v1', 'cases':65,'answers':260,
        'annotation_count':sum(len(r['raw_scores']) for r in rows), 'expected_annotations':520,
        'missing_annotations':520-sum(len(r['raw_scores']) for r in rows), 'rows':rows,
        'disagreements':disagreements,'freeze_sha256':freeze_sha,'analysis_sha256':file_sha(analysis_path),
        'annotation_sha256':annotation_hashes,
        'agreement':{axis:agreement([r['raw_scores']['r1'][axis] for r in paired], [r['raw_scores']['r2'][axis] for r in paired]) if paired else None for axis in AXES},
        'f1_contrasts':analysis['contrasts'], 'inference_cost':analysis['new_inference_cost'],
        'failures':[{'index':r['index'],'cell':r['cell'],'status':r['inference_status']} for r in rows if r['inference_status']!='completed'],
        'truncated_completions':analysis['truncated_completions'],
        'reference_flagged_indices':sorted(flags), 'ranking_pairs':[]})
    for i in indices:
        cells = {r['cell']:r for r in rows if r['index']==i}
        for metric in ['semantic_correctness','evidence_faithfulness']:
            if all(cells[c][metric] is not None and cells[c]['character_f1'] is not None for c in 'ABCD'):
                result['ranking_pairs'].append({'index':i,'metric':metric,
                    'pairs':rank_pairs({c:cells[c]['character_f1'] for c in 'ABCD'}, {c:cells[c][metric] for c in 'ABCD'})})
    result['condition_mean_ranking_pairs'] = {}
    for metric in ('semantic_correctness','evidence_faithfulness'):
        if all(result['cell_means'][c][metric]['available']==65 and analysis['cell_means'][c]['available']==65 for c in 'ABCD'):
            result['condition_mean_ranking_pairs'][metric] = rank_pairs(
                {c:analysis['cell_means'][c]['mean'] for c in 'ABCD'},
                {c:result['cell_means'][c][metric]['mean_0_to_2'] for c in 'ABCD'})
    result['ordinal_counts_by_reviewer_and_cell'] = {
        who:{cell:{axis:{str(value):sum(r['raw_scores'].get(who,{}).get(axis)==value for r in rows if r['cell']==cell)
                       for value in (0,1,2)} for axis in AXES} for cell in 'ABCD'} for who in ('r1','r2')}
    C.require(not Path(out).exists(), 'refusing to overwrite semantic result')
    C.write_json(out,result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['packets','analyze'])
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--packets',type=Path)
    p.add_argument('--reviews',type=Path)
    a=p.parse_args()
    if a.action=='packets':
        build_packets(a.bundle,a.analysis,a.freeze_sha256,a.run,a.out)
    else:
        C.require(a.packets is not None and a.reviews is not None, 'packet and annotation directories required')
        aggregate(a.bundle,a.analysis,a.freeze_sha256,a.run,a.packets,a.reviews,a.out)
