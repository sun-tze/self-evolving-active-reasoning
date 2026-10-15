# Confirmatory preparation validation

Date: 2026-10-15. Scope: CPU validation and independent review before remaining-case generation. No GPU submission or new Qwen inference occurred.

## Frozen identity

- Freeze SHA256: `732f6a7f71df04757016b265a71ea3549286305baffc4f7325e4af5e000dca9b`.
- Recorded freeze time: `2026-10-15T17:25:58.361429+00:00`.
- Original manifest SHA256: `1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548`.
- Original target file SHA256: `ee86128b43acedcdddbf53bb5c11671a992debeb5ea7a8e4fcec248a02a2f161`.
- Original semantic rubric and historical pilot responses/scores are unchanged.
- All executable files in the freeze use LF line endings. The freeze verifier passed against the original local bundle.

## Data and budget

The selected set contains exactly 65 original unseen cases, with four distinct cells per case. Positions 7, 29, 40, 58 and 70 are excluded. All 260 selected request identities and canonical payload hashes match the original 280-request bundle. Reference-file bytes, reference text hashes and case bindings were checked. References and evaluation anchors are not part of the model payload allowlist.

All 280 original token counts were reproduced with the pinned tokenizer on CPU; the resulting budget equals the original budget. The 260 selected requests total 194,338 input tokens and at most 104,000 output tokens. The largest input is 1,216 tokens. The exact selected request inventory and counts are in `request_set.json`.

The 65 reference anchors were independently checked against the original reference-only data. Seven corrections narrowed anchors to mechanisms supported by the references or clarified sufficient alternatives; these were applied before the freeze. Eight reference-quality flags are retained without exclusions. No remaining-case generated answers were available to the curator or reviewer.

## CPU tests

All 54 distinct relevant tests passed: 41 existing real-adapter/runtime tests and 13 new confirmatory tests. There were 55 test executions because one affected submission-claim regression was rerun after a review fix. No failed tests remain. Previously passed synthetic-writer and pilot-semantic suites were not rerun because their implementations did not change.

Commands for the affected suites:

```sh
python -m unittest discover -s experiments/sun_c2_real/tests -v
python -m unittest experiments.sun_c2_confirmatory.test_confirmatory -v
bash -n engaging/c2_confirmatory.sbatch
```

The new suite was executed in focused batches as changes were completed. It covers population exclusion and exact payload preservation, each budget cap, two complete independent assignments, reference tampering, journal-to-analysis reconstruction and tampering rejection, blind packet fields, paired effects and incomplete-result withholding, approval/freeze/commit/limit binding, absence of scheduler calls without approval, canonical duplicate-submission rejection, job-bound launch/inference claims, runtime-path binding, and full synthetic 260-answer aggregation with 520 ratings, one retained disagreement and ranking reversals. The synthetic answers are test fixtures, not new experimental outputs.

The launcher passed Bash syntax validation. This validates syntax only, not scheduling eligibility or GPU startup.

## Independent review

A read-only reviewer inspected the reference anchors, population/payload integrity, frozen provenance, blind-review construction, statistical implementation and submission controls. Two material implementation findings were repaired before freezing:

1. The submission guard initially depended on the chosen output directory. It now uses a canonical authorization-ID ledger, atomic scheduler receipt publication, job-ID binding and separate one-time launch/inference claims.
2. Preflight accepted runtime/cache paths independently of the launcher. It now rejects paths that differ from the launcher's fixed environment and cache.

The reviewer approved the repaired implementation and final numerical summaries with no unresolved blocking code findings. Selected final SHA256 values:

| File | SHA256 |
|---|---|
| `guard.py` | `83e930fb961aa80eb7bb2a68cfec323465c06e547cc01bb772075c4c9801bb29` |
| `submit_once.py` | `650329829d9b8a42e8e1c8358226a5f8f4bfcf07792d3709b31a0760a94bba68` |
| `semantic.py` | `312586f708cabd1aab7651890c0405b9f17b524c12702609abdf793d4da64aa2` |
| `engaging/c2_confirmatory.sbatch` | `c7803af62c3b11e524c8e884c1e811a1a19d31c4386751a846f0004415b3682c` |

The review did not itself rerun tests, change files, inspect new outcomes or authorize execution.

## Operational checks still required

Fresh Engaging runtime/cache validation, vLLM CLI inspection and Slurm `--test-only` remain pending because remote browser access timed out during preparation. The submission controller requires them to pass before submission, using the exact environment and cache that the launcher will execute. Any failed preflight prevents allocation; any true startup/tokenizer/inference failure stops the single job without retries. Successful historical pilot startup is supporting evidence, not a substitute for these checks.

The proposed allocation is one L40S, eight CPUs, 32 GiB host memory and a 60-minute wall limit under account `mit_general` / partition `mit_normal_gpu`. Approval must bind the published commit, freeze hash and exact resource limits. This validation record grants no GPU authorization.
