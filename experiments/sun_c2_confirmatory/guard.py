"""Canonical per-authorization ledger shared by submission and allocated jobs."""
import argparse
import os
from pathlib import Path
import time
from .study import C, authorization

LEDGER_ROOT = Path.home()/'orcd/pool/ar_harness/c2/authorizations'


def ledger(record):
    return LEDGER_ROOT/C.sha(record['authorization_id'].encode())


def claim(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(C.canonical(record)+b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def allocated_job(record, wait_seconds=0):
    job = os.environ.get('SLURM_JOB_ID')
    C.require(job is not None and job.isdigit(), 'Slurm allocation required')
    directory = ledger(record)
    reservation = C.read_json(directory/'SUBMISSION_ATTEMPTED.json')
    C.require(reservation['approval_sha256'] == C.digest(record), 'submission approval mismatch')
    receipt = directory/'job.json'
    deadline = time.monotonic()+wait_seconds
    while not receipt.exists() and time.monotonic()<deadline:
        time.sleep(.2)
    C.require(receipt.exists(), 'missing scheduler receipt; no execution authorized')
    C.require(C.read_json(receipt)['job_id'] == job, 'authorization belongs to another Slurm job')
    return directory, job


def receipt(record, job):
    directory = ledger(record)
    temporary = directory/'job.pending.json'
    claim(temporary, {'job_id':job})
    # Publish a complete file atomically; a racing launch cannot read partial JSON.
    os.link(temporary, directory/'job.json')
    temporary.unlink()


def launch(record):
    directory, job = allocated_job(record, wait_seconds=30)
    claim(directory/'LAUNCH_CLAIM.json', {'job_id':job})


def inference(record):
    directory, job = allocated_job(record)
    C.require(C.read_json(directory/'LAUNCH_CLAIM.json')['job_id'] == job, 'missing launch claim')
    claim(directory/'INFERENCE_CLAIM.json', {'job_id':job})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--authorization-file',type=Path,required=True)
    parser.add_argument('--freeze-sha256',required=True)
    args=parser.parse_args()
    launch(authorization(args.authorization_file,args.freeze_sha256))
