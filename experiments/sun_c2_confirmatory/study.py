"""Freeze and verify the remaining-case design without model execution."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REAL = HERE.parent / 'sun_c2_real'
sys.path.insert(0, str(REAL))
import common as C
import runner as R

# Keep the original manifest identity explicit; no new preparation or selection.
MANIFEST_SHA = '1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548'
RUBRIC = HERE.parent / 'sun_c2_semantic/rubric_v1.json'
FILES = [HERE / n for n in ('anchors.json', 'protocol.json', 'review_protocol.json',
         'study.py', 'run.py', 'guard.py', 'submit_once.py', 'semantic.py', 'request_set.json', 'assignments.json')]
FILES += [REAL / n for n in ('common.py', 'runner.py', 'analyze.py', 'budget.py',
          'runtime_preflight.py', 'model_lock.json', 'weights_lock.json')]
FILES += [RUBRIC, HERE.parent / 'sun_c2_semantic/audit.py', HERE.parent / 'sun_c2_semantic/prepare.py',
          ROOT / 'engaging/c2_confirmatory.sbatch']
FREEZE_NAME = 'freeze_v2.json'
PARENT_SHA = '732f6a7f71df04757016b265a71ea3549286305baffc4f7325e4af5e000dca9b'
AMENDED_FILES = {'experiments/sun_c2_confirmatory/' + n for n in ('study.py', 'submit_once.py')}
ACTIVE_FILES = FILES + [HERE/'freeze.json', HERE/'cli_preflight.py']


def file_sha(path):
    return C.sha(Path(path).read_bytes())


def targets(bundle, manifest, indices):
    path = Path(bundle) / 'private/targets.jsonl'
    C.require(file_sha(path) == manifest['targets_sha256'], 'target-file hash mismatch')
    rows = C.read_jsonl(path)
    C.require(len(rows) == 70 and len({r['index'] for r in rows}) == 70, 'duplicate/incomplete targets')
    selected = {r['index']: r for r in rows if r['index'] in indices}
    C.require(set(selected) == set(indices), 'missing reference')
    for i, row in selected.items():
        C.require(C.sha(row['bottom'].encode()) == row['target_sha256'], 'reference text hash mismatch')
        for binding in manifest['case_bindings'][str(i)].values():
            C.require(binding['dataset_record_sha256'] == row['dataset_record_sha256'], 'reference identity mismatch')
    return selected


def assignments(selected):
    result = {}
    for stream, seed in [('r1', 62101), ('r2', 62102)]:
        rng = random.Random(seed)
        indices = sorted({r['index'] for r in selected})
        rng.shuffle(indices)
        for offset in range(0, len(indices), 5):
            name = f'{stream}_b{offset//5+1:02d}'
            rows = sorted([r for r in selected if r['index'] in indices[offset:offset+5]], key=lambda r: r['request_id'])
            rng.shuffle(rows)
            result[name] = [{'item_id': f'item-{rng.getrandbits(128):032x}', 'request_id': r['request_id']} for r in rows]
    return result


def freeze(bundle):
    C.require(not (HERE / 'freeze.json').exists(), 'existing freeze is immutable; create a new version for amendments')
    manifest, rows = C.load_bundle(bundle, MANIFEST_SHA)
    indices = C.scope_indices(manifest, 'confirmatory')
    C.require(sorted(manifest['pilot_indices']) == [7, 29, 40, 58, 70], 'pilot identity changed')
    selected = R.schedule(rows, indices)
    reference = targets(bundle, manifest, indices)
    anchors = C.read_json(HERE / 'anchors.json')['cases']
    C.require(set(anchors) == {str(i) for i in indices}, 'anchor coverage mismatch')
    budget = C.read_json(Path(bundle) / 'budget.json')
    limits = C.read_json(HERE / 'protocol.json')['gpu_limits']
    doc, checked, counts = R.plan(bundle, MANIFEST_SHA, Path(bundle)/'budget.json', 'confirmatory',
                                 'http://127.0.0.1:18000/v1', limits['completion_calls'],
                                 limits['input_tokens'], limits['output_tokens_cap'])
    C.require(checked == selected and doc['totals'] == {'calls': 260, 'input_tokens': 194338, 'max_output_tokens': 104000}, 'unexpected budget')
    C.write_json(HERE / 'request_set.json', {
        'manifest_sha256': MANIFEST_SHA, 'original_requests_sha256': manifest['requests_sha256'],
        'excluded_pilot_positions': sorted(manifest['pilot_indices']), 'indices': indices,
        'requests': [{'request_id': r['request_id'], 'index': r['index'], 'cell': r['cell'],
                      'payload_sha256': r['payload_sha256'], 'input_tokens': counts[r['request_id']]} for r in selected],
        'totals': doc['totals'], 'max_input_per_request': max(counts[r['request_id']] for r in selected),
        'input_by_cell': {c: sum(counts[r['request_id']] for r in selected if r['cell'] == c) for c in 'ABCD'}})
    C.write_json(HERE / 'assignments.json', assignments(selected))
    output = {'format': 'c2_confirmatory_freeze_v1', 'recorded_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': MANIFEST_SHA, 'budget_sha256': file_sha(Path(bundle)/'budget.json'),
              'targets_sha256': manifest['targets_sha256'], 'rubric_sha256': file_sha(RUBRIC),
              'files': {str(p.relative_to(ROOT)).replace('\\','/'): file_sha(p) for p in FILES},
              'reference_bindings': {str(i): {'target_sha256': reference[i]['target_sha256'],
                                     'dataset_record_sha256': reference[i]['dataset_record_sha256']} for i in indices},
              'new_model_outputs_generated_or_inspected': False, 'gpu_execution_authorized': False}
    C.write_json(HERE / 'freeze.json', output)
    return output


def check_amendment(frozen):
    """Bind the execution-only amendment to the immutable original design."""
    C.require(file_sha(HERE/'freeze.json') == PARENT_SHA, 'original freeze changed')
    original = C.read_json(HERE/'freeze.json')
    C.require(frozen.get('parent_freeze_sha256') == PARENT_SHA, 'wrong parent freeze')
    C.require(frozen.get('format') == 'c2_confirmatory_freeze_v2', 'wrong amended freeze')
    preserved = set(original) - {'files', 'format', 'recorded_utc'}
    C.require(all(frozen.get(k) == original[k] for k in preserved), 'amendment changed design metadata')
    C.require(all(frozen['files'].get(p) == h for p, h in original['files'].items()
                  if p not in AMENDED_FILES), 'amendment changed protected file')


def amend_preflight():
    """Explicit one-time versioning; never rebuild requests, anchors or assignments."""
    path = HERE/FREEZE_NAME
    C.require(not path.exists(), 'amended freeze already exists')
    original = C.read_json(HERE/'freeze.json')
    frozen = dict(original, format='c2_confirmatory_freeze_v2',
                  recorded_utc=datetime.now(timezone.utc).isoformat(),
                  parent_freeze_sha256=PARENT_SHA,
                  amendment='Replace CLI-help execution with bounded static installed-CLI verification.',
                  files={p.relative_to(ROOT).as_posix(): file_sha(p) for p in ACTIVE_FILES})
    check_amendment(frozen)
    C.write_json(path, frozen)
    return frozen


def verify(bundle, expected_freeze_sha=None):
    path = HERE / FREEZE_NAME
    if expected_freeze_sha is not None:
        C.require(file_sha(path) == expected_freeze_sha, 'freeze identity mismatch')
    frozen = C.read_json(path)
    check_amendment(frozen)
    C.require(frozen['manifest_sha256'] == MANIFEST_SHA, 'wrong freeze')
    C.require(set(frozen['files']) == {p.relative_to(ROOT).as_posix() for p in ACTIVE_FILES}, 'incomplete code freeze')
    for rel, digest in frozen['files'].items():
        candidate = (ROOT / rel).resolve()
        C.require(candidate.is_relative_to(ROOT), 'invalid frozen path')
        C.require(file_sha(candidate) == digest, 'frozen file changed: ' + rel)
    manifest, rows = C.load_bundle(bundle, MANIFEST_SHA)
    C.require(file_sha(Path(bundle)/'budget.json') == frozen['budget_sha256'], 'budget changed')
    selected = R.schedule(rows, C.scope_indices(manifest, 'confirmatory'))
    saved = C.read_json(HERE / 'request_set.json')
    C.require(saved['original_requests_sha256'] == manifest['requests_sha256'], 'original request file mismatch')
    C.require(saved['indices'] == C.scope_indices(manifest, 'confirmatory'), 'selection changed')
    C.require([(r['request_id'],r['index'],r['cell'],r['payload_sha256']) for r in selected] ==
              [(r['request_id'],r['index'],r['cell'],r['payload_sha256']) for r in saved['requests']], 'request selection changed')
    C.require(C.read_json(HERE/'assignments.json') == assignments(selected), 'review assignment changed')
    return frozen, manifest, selected


def authorization(path, expected_freeze_sha):
    record = C.read_json(path)
    C.require(record.get('approved') is True, 'explicit authorization required')
    C.require(record.get('freeze_sha256') == expected_freeze_sha == file_sha(HERE/FREEZE_NAME), 'authorization belongs to another freeze')
    C.require(record.get('limits') == C.read_json(HERE/'protocol.json')['gpu_limits'], 'authorization limits differ')
    C.require(isinstance(record.get('authorization_id'), str) and bool(record['authorization_id'].strip()), 'missing authorization reference')
    C.require(re.fullmatch('[0-9a-f]{40}', record.get('code_commit','')) is not None, 'missing pinned code commit')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--amend-preflight', action='store_true')
    args = parser.parse_args()
    if args.amend_preflight:
        C.require(not args.freeze, 'choose only one freeze action')
        amend_preflight()
        verify(args.bundle)
    elif args.freeze:
        freeze(args.bundle)
    else:
        verify(args.bundle)
    print('Confirmed 65 cases, 260 unchanged payloads, and frozen evaluation inputs. No inference.')
