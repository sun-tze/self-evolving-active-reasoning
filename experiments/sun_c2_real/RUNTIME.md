# Engaging runtime reproduction

The deployed environment was created using the project's existing Miniforge /
Python 3.12 layout. It serves the same Qwen2.5-7B model with the existing BF16,
16,384 context, 48 sequence, 0.88 GPU memory, prefix-cache, and Flash Attention
settings. No new serving architecture was introduced.

The account initially had no serving environment or model cache. Both have now
been prepared without allocating a GPU. The final CPU preflight is in
`reports/runtime_preflight.json`, with the complete installed package inventory.
`requirements-runtime.txt` freezes that inventory for a new Linux Python 3.12
environment. Keep the CPU staging environment separate: its Transformers 5.19
is outside vLLM 0.31.0's supported range.

## Installed versions and source build

- Python 3.12 (Miniforge module `26.7.2-0`).
- vLLM 0.31.0, torch 2.13.0 (compiled for CUDA 13.0).
- Transformers 5.17.0, tokenizers 0.23.3.
- CUDA toolkit metadata 13.0.3.0; resolved nvcc 13.4.92; GCC 12.2.0.
- llguidance 1.7.6, built locally with Rust 1.99.0 and `CARGO_BUILD_JOBS=2`.

The llguidance binary wheels require glibc 2.31, while the login environment has
2.28. A binary-only attempt correctly failed. Only this dependency was rebuilt
from its official PyPI source; the remaining Python packages used binary wheels.
No system glibc changes, relabeled third-party wheels, CPU-specific native flags,
or replacement GPU libraries were used. The resulting wheel is a local Linux
wheel, not a claim of manylinux portability. Its import was verified on the login
node; target compute-node compatibility remains part of the authorized pilot.

Source SHA-256:
`db7febbe412ed2015501904646750071d7e00e6df7f85c4b956ad4f206fd2df7`

Built wheel SHA-256:
`a0adcc3ef3836cc097447e904ba4add2fb1147998ab08e27efe79cd7fd12ef43`

The source URL, build tool version, and wheel record are preserved in
`reports/runtime_build_provenance.json`. The matching cached wheel remains on
the cluster. Different build-tool versions may produce different wheel hashes.

## Recreate on a fresh account only

The current account is already set up; these are reproduction instructions, not
additional work the user needs to perform. From this branch's clone, with a
fresh target environment, this one shell block prepares software and the pinned
public model cache. It does not submit Slurm work or load a model:

```bash
set -euo pipefail
module load miniforge/26.7.2-0
conda create --override-channels -c conda-forge -p "$HOME/code/ar_harness/conda_env" python=3.12 pip rust=1.99.0 -y
export PATH="$HOME/code/ar_harness/conda_env/bin:$PATH"
CARGO_BUILD_JOBS=2 python -m pip install --only-binary=:all: --no-binary=llguidance -r experiments/sun_c2_real/requirements-runtime.txt
python -m pip check
export HF_HOME="$HOME/orcd/pool/ar_harness/hf_cache"
HF_HUB_DISABLE_XET=1 python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28", allow_patterns=["*.safetensors","*.json","*.txt"], max_workers=2))'
```

Then run the single `runtime_preflight.py` command in `README.md` against the
prepared bundle. It reads and hashes model files without loading their tensors.
All four weight shards must match the official revision's LFS hashes; the seven
small files are pinned using official Git blob identities and SHA-256.

The actual CPU preflight passed, including exact agreement across all 280 token
counts despite the supported Transformers version difference. `pip check` found
no broken requirements, and Python.h / ninja / nvcc / CUDA headers / g++ were
present. These checks do **not** test the target GPU driver, CUDA/JIT compilation,
engine startup, or served `/tokenize`. The installed torch CUDA and nvcc minor
versions are recorded explicitly; compatibility must be observed inside the
approved L40S allocation, using the existing attention/compiler settings. A
startup failure stops the bounded job; do not modify the frozen data, retry
inference automatically, or increase its budget to work around a failure.
