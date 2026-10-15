# Independent code review — 2026-10-15

## Completed real pilot: independent results verification

Following a new explicit user authorization, **Job 25941208** executed the
reviewed `58f7e7b` code and original manifest. The read-only independent reviewer
verified all 20 started/result pairs, request/payload/target hashes, scheduled
order, five-case four-condition pairing, budget limits and Slurm accounting.
Without calling the project's analysis function, the reviewer recomputed all
20 character F1 scores, all case contrasts, seven overall contrasts and the
10,000-draw paired-bootstrap intervals. All results match.

**Conclusion: results verification passed; no pairing, scoring, statistics or
budget inconsistency found.** Slurm completed with exit 0:0 after 178 seconds;
20/20 requests succeeded, using 15,210 input and 1,817 output tokens. Maximum
single-response length was 168 tokens; no response was truncated. Logs confirm
20 successful tokenization requests and 20 successful completion requests.

All contrast intervals include zero. A positive interaction estimate does not
establish a reliable gain, semantic success, or absence of other effects.
Case 29 illustrates a lower-scoring answer retaining the correct fire-resistant
suit mechanism while a higher-scoring answer invents a safe room; case 70's
high-scoring D answer proposes a hidden passage rather than the target's VR
pause mechanism. The reviewer also checked the histories and found examples
where generated answers repeat denied claims. Character F1 and the evidence
instruction do not establish semantic correctness or faithfulness.

The model generation-default warning is nonfatal. Engine-process/semaphore
cleanup warnings occur after all completion responses, during shutdown; there
is no evidence of corruption of the returned records. They remain in the logs.

Verified SHA-256 values:

- `analysis.json`: `cf049d5853ec73473f324b88a1f5164e52e39310ad999d3c06c4f3a73f920664`
- `inference/journal.jsonl`: `1d4a58a09e4d3d20feb865308e5312063f085666b8e17408754d88d953fe011a`
- `vllm.log`: `49664aca962183b190637720cb6faaf8fee5e9dc9acbf96f2692b69cd2e17d5c`

See [the pilot report](Research_Pilot_Report_25941208.txt) and
[machine-readable evidence](reports/pilot_25941208_result.json). No additional
inference was performed for review. The earlier review stages below are retained
as historical records of what was known before this completed pilot.

## Follow-up: vLLM 0.31 attention selection

Reviewed execution commit: `58f7e7bb68bf4caf1b36826073685f3ceafb5fdc`,
branch `sun/c2-vllm-attention-fix`, compared with
`6d685a617ca4a14710c9c727f5e0dd53f4f50294`. The later documentation commit
does not replace this execution pin. The user explicitly requested another
read-only review by the independent reviewer.

**Code review: approved for another bounded pilot, with no blocking findings.
No second GPU job has been submitted. New human authorization is required.**
This is a CPU/code-level readiness conclusion, not proof that the explicitly
selected attention backend has started on an L40S.

Exactly two runtime/test files differ from the previous execution commit:

| File | Change | SHA-256 of canonical Git LF bytes |
|---|---|---|
| `engaging/c2_pilot.sbatch` | Unset the obsolete environment variable and pass `--attention-backend FLASH_ATTN` on the actual serve command | `9bbc341fb49fc368ec3434019666449bb77d377523a190e5e96be8f3b60d4daf` |
| `experiments/sun_c2_real/tests/test_runtime.py` | Add a launcher compatibility regression | `a7a541d42ec17d94a5de13315c78796425a2b16663ca8f5079c3ae1103506cc7` |

The independent reviewer parsed the continued serve command, checked the unique
backend flag, and reconstructed the new launcher from the old one using only
the intended substitutions. The new regression passes at this commit and fails
against the old launcher. This static test does not itself execute vLLM.

Deployment and validation:

- Engaging checkout `~/code/ar_harness_c2_real` is clean at the execution pin.
  The original checkout and the first pilot's detached worktree are preserved.
- `bash -n` passed on Engaging and independently under local Git Bash.
- Engaging CPU tests: **95/95 passed**, comprising 54 legacy and 41 real-data tests.
- Windows CPU tests: **94 passed, one unchanged symlink-privilege skip**;
  the corresponding test passes on Engaging. No failures or errors.
- Installed vLLM **0.31.0** source confirms the serve parser entry point,
  `--attention-backend` registration, and `FLASH_ATTN` backend choice. File
  hashes and the registration excerpt are retained. Full CLI help exceeded a
  120-second CPU limit; a corrected direct parser probe exceeded a 300-second
  limit while importing dependencies (exit 124, observed I/O wait). It did not
  reach argument parsing, so **complete installed-parser execution is not
  marked passed**. An earlier probe also used an obsolete module path; that was
  a validation-helper error, not a failure of the project code. All probe logs
  are retained. No further environment changes were made to chase this check.
- No dependency changes, model generation, paid API calls, or GPU submissions
  were made during this repair validation.

The manifest remains
`1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548`.
Both the local and Engaging dry plans select positions **7, 29, 40, 58, 70**,
with **20 calls, 15,210 input tokens, at most 8,000 output tokens**.
The runner hash remains
`38cc557744d83c4adba3112ec17a9335f92303e0c82c5cd4f53f763c20c14519`.
The model/tokenizer revision, sampling inputs, payloads, scorer, pairing,
loopback restriction, unique server identity, per-call limits, no-retry policy,
1 L40S allocation and 30-minute Slurm cap are unchanged.

### Correction to the first pilot's failure interpretation

The complete retained logs for **Job 25934600** do not demonstrate a fatal vLLM
startup failure. The unknown `VLLM_ATTENTION_BACKEND` warning was nonfatal;
weights loaded and CUDA graph capture completed. The later warning about
model-supplied generation defaults was also nonfatal. The log records
`Application startup complete.` during cancellation cleanup.

The assistant cancelled the job after interpreting the environment-variable
warning too conservatively. Slurm records `CANCELLED`; the batch was eventually
killed during cancellation. This is an operator cancellation, not evidence of
an OOM, CUDA exception or fatal startup traceback. No writer calls or scored
outputs were produced. Application startup alone does not establish that the
health, server identity, tokenizer or inference checks passed.

The complete old `vllm.log` SHA-256 is
`e173413640ca21e624833f9c5783875a94993a81a6b90d5269bf17ef748b8fbe`;
the same bytes were verified on Engaging and in the local evidence archive.
The prior run, submit receipt, cancellation record and Slurm logs are retained.

For the next authorized run, known nonfatal warnings are retained as evidence
and do not by themselves establish failure. Stop on process exit, readiness
timeout, a fatal CUDA/OOM/traceback, failed identity/tokenizer/budget checks, or
a failed/ambiguous inference request. Do not retry or submit a replacement job.
The existing `generation_config=auto` behavior is preserved: explicit request
temperature/top-p override corresponding model defaults; other pinned model
defaults remain in force. Changing those defaults would exceed this repair's
two-file scope.

Version-specific reference: [vLLM 0.31 serve CLI](https://docs.vllm.ai/en/v0.31.0/cli/serve/).
Machine-readable follow-up evidence is in `reports/attention_fix_validation.json`.
Full CPU and CLI logs remain under `~/orcd/pool/ar_harness/c2/attention_fix_58f7e7b`
and in the local `.c2-local/attention-fix-58f7e7b` archive.

## Original adapter review (before Job 25934600)

The user explicitly authorized a separate read-only review agent. The reviewer
inspected the adapter, source/config bindings, runner, scoring/statistics, costs,
Slurm script, tests, original C2 protocol, and historical Experiment 10 sources.
The implementing agent applied fixes; the reviewer independently rechecked them.

**Final conclusion: approved at the code level for a bounded pilot. No unresolved
blocking findings. Cluster compatibility checks and explicit human GPU
authorization are still required.** This is not a claim that a real pilot ran.

| Finding | Resolution and independent evidence |
|---|---|
| P2: socket inactivity timeout did not bound total HTTP time | Replaced urllib I/O with a local HTTP connection and absolute timer that shuts down socket I/O. Body-dribble regression passes; independent slow-header probe aborted after 188 ms for a 150 ms deadline. |
| P2: model-cache inventory could accept a nonempty truncated shard | Added external `weights_lock.json`: all four official LFS SHA-256 values and seven small-file Git blob/SHA-256 values independently cross-checked at the fixed revision. Five corruption/completeness regressions pass. |
| P2: port collision could accept an old service's health response | Select a free port, check launched child before/after health, add an unpredictable per-job served-model alias, and require the marker in `/models` before tokenization or generation. Old-service rejection regression passes. |

Reviewer verification:

- 40/40 new CPU tests pass, including local HTTP integration tests.
- All 12 legacy synthetic files are byte-identical to `e16ccf7`.
- Independently loaded and hashed raw inputs without using the adapter's
  validation helper: both sources contain 100 unique positional IDs; the exact
  30 evolution-seen positions agree across archived configs and NumPy RNG.
- All 280 projections and four-cell pairings match the raw records.
- No complete hidden story from any of the 100 targets appears in outgoing
  payloads. All 100 surfaces and all 100 targets are unique as exact strings.
- The dataset has only 99 unique `index` values, validating the positional join.
- Scorer, contrasts, missingness behavior, and separation of historical and new
  costs are consistent with the protocol and its stated limitations.

Reviewed runtime file hashes:

| File | SHA-256 |
|---|---|
| `runner.py` | `38cc557744d83c4adba3112ec17a9335f92303e0c82c5cd4f53f763c20c14519` |
| `../../engaging/c2_pilot.sbatch` | `b2e71c64327a6d7c6fdd12d933b7d23eca994775473fd065e18ba719c1b1d97d` |

The reviewer did not authenticate to Engaging, run Slurm validation, load model
weights, or perform real inference. The implementing agent subsequently completed
Engaging CPU validation and `sbatch --test-only`; see `DELIVERY.md`. Hash consistency is not signed historical authenticity.
Structural field isolation does not establish semantic decontamination of the
recorded dialogue. The review also corrected readiness documentation to account
for the approximately seven-minute worst-case polling period.

Additional reviewed SHA-256 values:

- `runtime_preflight.py`: `e6f0aafbe6f6cb0f21b58b7d5ac4eb9122b6b9e01f9f172fa682dbd5b6c9af7a`
- `weights_lock.json`: `5e3d9ed5ee5aac087019211d83eadc76dc26e3f9366e2f1637ec10977b8a3f75`

The reviewer also verified vLLM 0.31.0 tokenizer API compatibility and identified
the Transformers version boundary and llguidance glibc wheel mismatch during
deployment. CPU staging and GPU-serving dependencies remain isolated. This
review does not claim a GPU driver/JIT/startup test.
