"""Private, single-submission controller. No approval means no scheduler call."""
import argparse
import os
from pathlib import Path
import subprocess
from .study import C, HERE, ROOT, REAL, MANIFEST_SHA, verify, authorization
from . import guard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--authorization-file', type=Path, required=True)
    p.add_argument('--private-root', type=Path, required=True)
    p.add_argument('--runtime-python', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    approved = authorization(a.authorization_file, a.freeze_sha256)
    C.require(a.runtime_python.resolve() == (Path.home()/'code/ar_harness/conda_env/bin/python').resolve(), 'runtime path differs from launcher')
    C.require(a.cache.resolve() == (Path.home()/'orcd/pool/ar_harness/hf_cache').resolve(), 'cache path differs from launcher')
    verify(a.bundle, a.freeze_sha256)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    C.require(commit == approved['code_commit'], 'code commit differs from authorization')
    C.require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'clean checkout required')
    out = a.private_root.resolve()
    C.require(not out.is_relative_to(ROOT), 'operational records must be outside public checkout')
    out.mkdir(parents=True, exist_ok=True)
    reservation = guard.ledger(approved) / 'SUBMISSION_ATTEMPTED.json'
    C.require(not reservation.exists(), 'submission already attempted; no automatic retry')
    def checked(command, logname, timeout):
        with (out/logname).open('wb') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    checked(['bash','-n',str(ROOT/'engaging/c2_confirmatory.sbatch')], 'syntax.log', 30)
    checked([str(a.runtime_python), str(REAL/'runtime_preflight.py'), '--bundle',str(a.bundle),
             '--manifest-sha256',MANIFEST_SHA,'--cache',str(a.cache),'--out',str(out/'runtime_preflight.json')],
             'runtime_preflight.log', 600)
    # Confirm CLI availability without starting a service or allocating a GPU.
    checked([str(a.runtime_python.parent/'vllm'),'serve','--help=all'], 'vllm_cli.log', 120)
    C.require('--attention-backend' in (out/'vllm_cli.log').read_text(errors='replace'), 'required vLLM CLI option unavailable')
    args = ['--no-requeue','--account=mit_general','--partition=mit_normal_gpu','--gres=gpu:l40s:1',
            '--cpus-per-task=8','--mem=32G','--time=01:00:00',
            '--job-name=c2-confirm65-'+C.sha(approved['authorization_id'].encode())[:8],
            '--output='+str(out/'slurm_%j.out'),'--error='+str(out/'slurm_%j.err'),
            str(ROOT/'engaging/c2_confirmatory.sbatch'),str(ROOT),str(a.bundle.resolve()),
            a.freeze_sha256,str(a.authorization_file.resolve())]
    checked(['sbatch','--test-only',*args], 'slurm_test_only.log', 120)
    # O_EXCL is the submission guard, including an unknown scheduler response.
    guard.claim(reservation, {'approval_sha256':C.digest(approved), 'code_commit':commit,
                'freeze_sha256':a.freeze_sha256,'command':['sbatch','--parsable',*args]})
    env = {k:v for k,v in os.environ.items() if not k.startswith('SBATCH_')}
    completed = subprocess.run(['sbatch','--parsable',*args], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=120)
    C.write_json(out/'submission_receipt.json', {'returncode':completed.returncode,
                 'stdout':completed.stdout,'stderr':completed.stderr})
    C.require(completed.returncode == 0, 'submission failed; no retry authorized')
    job = completed.stdout.strip().split(';')[0]
    C.require(job.isdigit(), 'ambiguous submission receipt; inspect scheduler, do not resubmit')
    guard.receipt(approved, job)
    print('Submitted single authorized job '+job)


if __name__ == '__main__':
    main()
