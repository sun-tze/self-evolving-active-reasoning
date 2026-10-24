#!/bin/bash
# Invoke only after parent release, in an existing permitted Slurm environment.
# This file never submits, requeues or retries a job.
#SBATCH --job-name=sun-gn-context
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Must run in the released allocation}"
: "${SLURM_JOB_START_TIME:?Allocation start epoch is required; do not default to current time}"
: "${GN_RUNTIME_PYTHON:?Existing verified runtime Python required}"
: "${GN_CODE_ROOT:?Exact reviewed implementation checkout required}"
cd "$GN_CODE_ROOT"
exec "$GN_RUNTIME_PYTHON" -m experiments.sun_gn_context.allocation "$@"
