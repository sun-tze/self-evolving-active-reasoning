"""Additional target provenance verification, without changing the frozen packets."""
import argparse
import hashlib
from pathlib import Path
from experiments.sun_c2_real.common import load_bundle, read_json, read_jsonl, require
from .prepare import HERE, MANIFEST_SHA, POSITIONS, sha, write


def verified_targets(path, manifest, positions=POSITIONS):
    require(sha(path) == manifest['targets_sha256'], 'targets file differs from pinned manifest')
    selected = [r for r in read_jsonl(path) if r['index'] in positions]
    require(len(selected) == len(positions) and {r['index'] for r in selected} == set(positions), 'target coverage mismatch')
    for target in selected:
        require(hashlib.sha256(target['bottom'].encode()).hexdigest() == target['target_sha256'], 'target text hash mismatch')
        for binding in manifest['case_bindings'][str(target['index'])].values():
            require(binding['dataset_record_sha256'] == target['dataset_record_sha256'], 'dataset record mismatch')
    return {r['index']: r for r in selected}


def verify(bundle, frozen=HERE / 'frozen'):
    manifest, _ = load_bundle(bundle, MANIFEST_SHA)
    targets = verified_targets(bundle / 'private/targets.jsonl', manifest)
    source = read_json(frozen / 'source_selected.json')
    for row in source:
        target = targets[row['index']]
        require(row['reference_solution'] == target['bottom'] and row['target_sha256'] == target['target_sha256'], 'frozen reference mismatch')
        require(row['source_identity']['dataset_record_sha256'] == target['dataset_record_sha256'], 'frozen identity mismatch')
    return {'status': 'passed', 'manifest_sha256': MANIFEST_SHA,
            'targets_file_sha256': manifest['targets_sha256'], 'source_selected_sha256': sha(frozen / 'source_selected.json'),
            'selected_reference_count': len(targets), 'selected_answer_count': len(source),
            'checked': ['full targets file hash against pinned manifest', 'selected target text hashes',
                        'selected dataset record identity against both historical source bindings',
                        'all frozen reference texts and target identities match original bundle'],
            'remaining_case_outcomes_read': False,
            'note': 'Supplemental check after independent code review. Original frozen packets/rubric unchanged. Full original targets file remains in private bundle; this committed report records the performed verification.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'refusing to overwrite provenance report')
    write(args.output, verify(args.bundle))
    print('Pinned target-file hash and all five frozen references verified.')


if __name__ == '__main__':
    main()
