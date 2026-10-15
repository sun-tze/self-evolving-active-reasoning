# C2 real-data pilot delivery — 2026-10-15

**Current update:** the first authorized attempt, Job **25934600**, was cancelled
by the assistant after a nonfatal warning; it produced no writer results. The
complete log reached application startup during cancellation cleanup. The
attention-selection fix at **`58f7e7b`**, branch `sun/c2-vllm-attention-fix`, is
deployed on Engaging and passed **95/95 CPU tests** and independent code review.
The fixed manifest and 20-call budget are unchanged. No second GPU job has
been submitted; new single-job authorization is required. See the current
[review and failure reclassification](REVIEW.md) and
[validation evidence](reports/attention_fix_validation.json).

The original delivery record below describes the state before Job 25934600.

## Scope and current result

Fork branch: `sun-tze/self-evolving-active-reasoning: sun/c2-real-data-adapter`.
Base: `e16ccf743e2d27f37b1d4316e3cc2109ce48a32d`.
Initial implementation publication: `9c18cb762f0de527c5ba2bda50166dbc039dfe64`.
No upstream changes, merged PRs, paid API calls, GPU allocations, or real writer
inference were performed. The original synthetic implementation remains byte-identical.

The adapter, loopback-only inference runner, offline scoring, paired factorial
analysis, cost reporting, and independent code review are complete. All 280
real requests were generated and independently compared to their raw source
projections. The authenticated MIT OnDemand shell was used to deploy a separate
clone and reproduce the CPU work without asking the user to run commands.

## Evidence

| Check | Result | Evidence |
|---|---|---|
| Existing synthetic tests on Engaging | 54/54 passed | `reports/tests_engaging.json` |
| New adapter/runner/analysis tests on Engaging | 40/40 passed | same |
| Windows tests | 53 legacy + 40 new passed; 1 symlink-privilege skip | `reports/tests_windows.json` |
| Exact unseen split and source identity | 70 unseen, exact 30 excluded, 100 unique positional IDs per history source | `reports/audit.json` |
| Raw projection comparison | 280/280 passed | `reports/second_path_audit.json` |
| Scorer differential cases | 1,005/1,005 passed; all 200 historical scores replayed | second-path and primary audits |
| Cross-platform regeneration | Identical manifest SHA-256 on Windows and Engaging | below |
| Independent review | Three P2 issues fixed and independently rechecked; no unresolved code blockers | `REVIEW.md` |
| Installed runtime and cache CPU preflight | Passed; all 11 model files match external hashes; 280/280 frozen input-token counts match | `reports/runtime_preflight.json` |
| New Slurm scheduling check | `sbatch --test-only` succeeded, L40S/account/partition accepted | `reports/engaging_cpu_validation.json` |

Manifest SHA-256 (external integrity anchor):

```text
1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548
```

The dataset's own `index` has only 99 unique values. Historical episode `index`
means array position, so using the raw dataset label as a join key would mispair
cases. The adapter verifies all 200 historical prompt surfaces and uses positional
identity. Outgoing fields are exclusively surface, recorded questions/replies,
and standardized system/writer instructions. Targets, historical predictions,
trace/scoring information, and source labels never enter the wire payload.

## Deployed locations

- Code: `~/code/ar_harness_c2_real` (new clone; original checkout preserved).
- CPU environment: `~/orcd/pool/ar_harness/c2/cpu_env`.
- Verified source inputs: `~/orcd/pool/ar_harness/c2/inputs`.
- Immutable prepared bundle: `~/orcd/pool/ar_harness/c2/pilot-v1`.
- CPU logs/reports: `~/orcd/pool/ar_harness/c2/`.
- Existing-layout inference environment: `~/code/ar_harness/conda_env`.
- Pinned Qwen cache: `~/orcd/pool/ar_harness/hf_cache`.

The last two locations were absent initially and are now populated. CPU runtime
preflight passed: dependency consistency, Python/compiler/CUDA header presence,
all 11 model-file hashes, and all 280 runtime-tokenizer counts. See
`reports/runtime_preflight.json` and [RUNTIME.md](RUNTIME.md). No GPU startup has
been attempted; `--test-only` predicts scheduling and does not validate CUDA or serving.

## Concrete pilot budget and interpretation

| Scope | Cases | Calls | Exact input tokens | Maximum output tokens |
|---|---:|---:|---:|---:|
| Proposed pilot | 5 | 20 | 15,210 | 8,000 |
| All prepared requests | 70 | 280 | 209,548 | 112,000 |

Pilot positions: **7, 29, 40, 58, 70**, chosen by a deterministic hash before new
outcomes. One L40S, 8 CPUs, 32 GiB RAM, hard 30-minute Slurm cap: **at most 0.5
L40S GPU-hours**, including startup. No dollar cost is asserted without an
institutional rate. There are no automatic inference retries. A failed or
ambiguous request stops the run and is retained for review.

A/B use the bare dialogue; C/D use the first-selected dialogue. A/C use the plain
writer, B/D the evidence-constrained writer. The report separates writer effects
(B-A, D-C), dialogue effects (C-A, D-B), marginal effects, and interaction
(D-C)-(B-A), with paired case-bootstrap descriptive intervals. Missing components
suppress dependent estimates. This pilot tests feasibility and output behavior;
five cases do not establish a general performance claim. If pilot feedback changes
the method, reserve the remaining 65 cases for confirmation.

Historical questioner costs include the original finalizers and remain separate
from new writer costs; referee token cost is unknown. Character F1 is lexical
overlap, not a semantic correctness measure. Hash checks establish consistency
with pinned artifacts, not signed historical authenticity or semantic removal of
answer-relevant facts from the recorded dialogue.

## Reproduction and launch boundary

The complete CPU reproduction command is `stage.py`; exact options and analysis
commands are in [README.md](README.md). It downloads pinned public inputs and
small tokenizer files, creates all 280 requests, scores audit fixtures, counts
inputs, and writes a zero-network execution plan. CPU environment dependencies
must not be installed over the vLLM environment.

From the deployed clone, rerun CPU tests with:

```bash
~/orcd/pool/ar_harness/c2/cpu_env/bin/python experiments/sun_c2_real/checks.py --out ~/orcd/pool/ar_harness/c2/tests_rerun.json
```

Scheduling-only check (already executed, submits nothing):

```bash
sbatch --test-only engaging/c2_pilot.sbatch "$PWD" "$HOME/orcd/pool/ar_harness/c2/pilot-v1" 1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548 NOT_AUTHORIZED experiments/sun_c2_real/REVIEW.md
```

**Do not remove `--test-only` until the human explicitly approves the concrete
20-call budget.** An authorization reference records that approval; placeholders
are not approval. The assistant can submit the reviewed job through the existing
OnDemand session after that approval. The user need not manually execute commands.
The job checks served-model identity and all pilot token counts before its first
completion, records its environment/commit/journal, and writes offline analysis.

Remaining work: obtain explicit pilot authorization; validate driver/CUDA/JIT and served tokenization on the allocated
L40S; execute and analyze the 20 calls; inspect failures and outputs before any
larger experiment. No C2 real-model effect estimate exists yet.
