"""Confirmatory runner: planning by default; execution requires pinned approval."""
import argparse
import os
import subprocess
from pathlib import Path
from .study import C, R, HERE, ROOT, MANIFEST_SHA, file_sha, verify, authorization
from .guard import inference


def plan(bundle, freeze_sha, endpoint, marker=None):
    frozen, _, selected = verify(bundle, freeze_sha)
    limits = C.read_json(HERE/'protocol.json')['gpu_limits']
    doc, scheduled, counts = R.plan(bundle, MANIFEST_SHA, Path(bundle)/'budget.json', 'confirmatory',
                                    endpoint, limits['completion_calls'], limits['input_tokens'],
                                    limits['output_tokens_cap'], marker)
    C.require(scheduled == selected, 'execution schedule changed')
    C.require(doc['totals'] == C.read_json(HERE/'request_set.json')['totals'], 'execution budget changed')
    doc['evaluation_freeze_sha256'] = freeze_sha
    return doc, selected, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--base-url', default='http://127.0.0.1:18000/v1')
    p.add_argument('--server-marker')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--authorization-file', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    approved = None
    if a.authorization_file is not None:
        approved = authorization(a.authorization_file, a.freeze_sha256)
        C.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == approved['code_commit'], 'code commit differs from authorization')
    doc, selected, counts = plan(a.bundle, a.freeze_sha256, a.base_url, a.server_marker)
    if not a.execute:
        print(C.canonical(doc).decode())
        return
    C.require(a.authorization_file is not None and a.out is not None and a.server_marker, 'execution arguments missing')
    C.require(os.environ.get('SLURM_JOB_ID'), 'execution requires Slurm allocation')
    limits = approved['limits']
    inference(approved)
    R.execute(doc, selected, counts, a.out, approved['authorization_id'],
              limits['runner_deadline_seconds'], limits['request_timeout_seconds'], resume=False)


if __name__ == '__main__':
    main()
