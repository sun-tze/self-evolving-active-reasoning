"""Offline extraction of the existing pilot; never invokes a model or changes scores."""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from experiments.sun_c2_real.common import load_bundle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST_SHA = '1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548'
ANALYSIS_SHA = 'cf049d5853ec73473f324b88a1f5164e52e39310ad999d3c06c4f3a73f920664'
POSITIONS = [7, 29, 40, 58, 70]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def make_packets(records, rubric_sha):
    packets, mappings = {}, {}
    for reviewer, seed in [('r1', 62101), ('r2', 62102)]:
        ordered = sorted(records, key=lambda r: (r['index'], r['cell']))
        random.Random(seed).shuffle(ordered)
        items, mapping = [], {}
        for n, r in enumerate(ordered):
            opaque = hashlib.sha256(f'c2-semantic-v1:{reviewer}:{n}'.encode()).hexdigest()[:12]
            item_id = 'item-' + opaque
            items.append({'item_id': item_id, 'surface': r['observation']['surface'],
                          'history': [{'ref': f'H{k:02d}', **h} for k, h in enumerate(r['observation']['history'], 1)],
                          'reference_solution': r['reference_solution'], 'semantic_anchors': r['semantic_anchors'],
                          'answer': r['answer']})
            mapping[item_id] = {'index': r['index'], 'cell': r['cell'], 'request_id': r['request_id']}
        packets[reviewer] = {'reviewer_id': reviewer, 'rubric_sha256': rubric_sha, 'items': items}
        mappings[reviewer] = mapping
    return packets, mappings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE / 'frozen')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite frozen artifacts; use a new output directory.')
    manifest, requests = load_bundle(args.bundle, MANIFEST_SHA)
    assert sorted(manifest['pilot_indices']) == POSITIONS
    analysis_path = ROOT / 'experiments/sun_c2_real/reports/pilot_25941208_analysis.json'
    assert sha(analysis_path) == ANALYSIS_SHA
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    targets = {}
    # No remaining-case model outputs exist or are loaded. Only selected targets are retained.
    for line in (args.bundle / 'private/targets.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['index'] in POSITIONS:
            assert hashlib.sha256(row['bottom'].encode()).hexdigest() == row['target_sha256']
            targets[row['index']] = row
    assert set(targets) == set(POSITIONS)
    anchors = json.loads((HERE / 'anchors_v1.json').read_text(encoding='utf-8'))
    by_id = {r['request_id']: r for r in requests}
    records = []
    for response in analysis['responses']:
        r = by_id[response['request_id']]
        assert r['index'] == response['index'] and r['cell'] == response['cell']
        assert response['status'] == 'completed' and response['index'] in POSITIONS
        records.append({'index': r['index'], 'cell': r['cell'], 'request_id': r['request_id'],
                        'observation': r['observation'], 'source_identity': r['source_identity'],
                        'payload_sha256': r['payload_sha256'], 'target_sha256': targets[r['index']]['target_sha256'],
                        'reference_solution': targets[r['index']]['bottom'],
                        'semantic_anchors': anchors[str(r['index'])],
                        'answer': response['response']['text'], 'character_f1': response['score']})
    assert len(records) == 20 and {(r['index'], r['cell']) for r in records} == {(i, c) for i in POSITIONS for c in 'ABCD'}
    rubric_sha = sha(HERE / 'rubric_v1.json')
    packets, mappings = make_packets(records, rubric_sha)
    write(args.output / 'source_selected.json', records)
    write(args.output / 'unblind_map.json', mappings)
    for reviewer, packet in packets.items():
        write(args.output / f'packet_{reviewer}.json', packet)
    inventory = {p.name: sha(p) for p in sorted(args.output.glob('*.json'))}
    for name in ['rubric_v1.json', 'anchors_v1.json', 'confirmatory_v1.json', 'prepare.py']:
        inventory['../' + name] = sha(HERE / name)
    write(args.output / 'FREEZE.json', {
        'version': 'semantic-audit-v1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen_before_independent_ratings; pilot_outcomes_already_known_to_designer',
        'manifest_sha256': MANIFEST_SHA, 'analysis_sha256': ANALYSIS_SHA,
        'source_requests_sha256': sha(args.bundle / 'requests.jsonl'),
        'selected_positions': POSITIONS, 'new_inference_calls': 0, 'sha256': inventory,
        'scope': 'Only existing 20 responses. Remaining 65 cases have no generated outcomes and were not evaluated.',
        'limitations': 'Post-hoc exploratory pilot rubric; not a preregistration before pilot results. Procedural blinding, shared filesystem, same-model judges.'})
    print('Frozen 20 selected answers and two independently shuffled blind packets; no inference.')


if __name__ == '__main__':
    main()
