#!/bin/bash
# Confirmatory evaluation; one authorized allocation, no automatic retries.
# Pass CODE BUNDLE FREEZE_SHA256 AUTHORIZATION_FILE as arguments.
#SBATCH --job-name=c2-confirm65
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
: "${1:?CODE required}" "${2:?BUNDLE required}" "${3:?FREEZE_SHA256 required}"
: "${4:?approved AUTHORIZATION_FILE required}"
CODE=$(realpath "$1")
BUNDLE=$(realpath "$2")
FREEZE_SHA256=$3
AUTHORIZATION_FILE=$(realpath "$4")
MANIFEST_SHA256=1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548
ENV_CODE=$HOME/code/ar_harness
POOL=$HOME/orcd/pool/ar_harness
export HF_HOME=$POOL/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_USE_FLASHINFER_SAMPLER=0
# vLLM 0.31.0 no longer accepts VLLM_ATTENTION_BACKEND; use the serve CLI.
unset VLLM_ATTENTION_BACKEND
export CUDA_HOME=$ENV_CODE/conda_env/lib/python3.12/site-packages/nvidia/cu13
export PATH=$ENV_CODE/conda_env/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}
PY=$ENV_CODE/conda_env/bin/python
VLLM=$ENV_CODE/conda_env/bin/vllm
RUN_ROOT=$POOL/c2/runs/confirmatory_${SLURM_JOB_ID:?Run through Slurm}
mkdir "$RUN_ROOT"
PORT=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
SERVER_MARKER=c2-confirm65-${SLURM_JOB_ID}-$("$PY" -c 'import secrets; print(secrets.token_hex(12))')
cd "$CODE"
"$PY" -m experiments.sun_c2_confirmatory.guard --authorization-file "$AUTHORIZATION_FILE" \
  --freeze-sha256 "$FREEZE_SHA256"
# Fail on bundle, token budget, or publication-code problems before loading GPU weights.
"$PY" -m experiments.sun_c2_confirmatory.run --bundle "$BUNDLE" --freeze-sha256 "$FREEZE_SHA256" \
  --authorization-file "$AUTHORIZATION_FILE" --base-url "http://127.0.0.1:$PORT/v1" \
  --server-marker "$SERVER_MARKER" > "$RUN_ROOT/plan.json"
REVISION=$("$PY" -c 'import json; print(json.load(open("experiments/sun_c2_real/model_lock.json"))["revision"])')
git rev-parse HEAD > "$RUN_ROOT/code_commit.txt"
git status --porcelain > "$RUN_ROOT/code_status.txt"
test ! -s "$RUN_ROOT/code_status.txt" || { echo 'Code checkout must be clean'; exit 1; }
printf '%s\n' "$FREEZE_SHA256" > "$RUN_ROOT/evaluation_freeze_sha256.txt"
"$PY" -m pip freeze > "$RUN_ROOT/environment.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$RUN_ROOT/gpu.csv"
PID=
cleanup() {
  if [ -n "$PID" ]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
"$VLLM" serve Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen/Qwen2.5-7B-Instruct "$SERVER_MARKER" \
  --revision "$REVISION" --tokenizer-revision "$REVISION" --host 127.0.0.1 \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 48 \
  --attention-backend FLASH_ATTN \
  --gpu-memory-utilization 0.88 --enable-prefix-caching \
  --port "$PORT" > "$RUN_ROOT/vllm.log" 2>&1 &
PID=$!
ready=false
for i in $(seq 1 60); do
  kill -0 "$PID" 2>/dev/null || { echo 'vLLM exited'; exit 1; }
  if curl --noproxy '*' --max-time 2 -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    kill -0 "$PID" 2>/dev/null || { echo 'vLLM exited during readiness check'; exit 1; }
    ready=true
    break
  fi
  sleep 5
done
$ready || { echo 'vLLM readiness timeout'; exit 1; }
status=0
"$PY" -m experiments.sun_c2_confirmatory.run --bundle "$BUNDLE" --freeze-sha256 "$FREEZE_SHA256" \
  --authorization-file "$AUTHORIZATION_FILE" --base-url "http://127.0.0.1:$PORT/v1" \
  --server-marker "$SERVER_MARKER" \
  --execute --out "$RUN_ROOT/inference" || status=$?
# Retain an incomplete report when a call fails; do not retry.
if [ -f "$RUN_ROOT/inference/run.json" ]; then
  "$PY" experiments/sun_c2_real/analyze.py --bundle "$BUNDLE" --manifest-sha256 "$MANIFEST_SHA256" \
    --run "$RUN_ROOT/inference" --out "$RUN_ROOT/analysis.json" || status=$?
fi
exit "$status"
