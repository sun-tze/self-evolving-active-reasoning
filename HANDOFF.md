# C2 evaluation checkpoint

Current checkpoint: the [CPU preflight amendment](experiments/sun_c2_confirmatory/AMENDMENT.md) replaces CLI-help execution with a bounded static verifier. All 23 focused tests and the installed Engaging check passed; independent review found no blocking issues. Active freeze: `freeze_v2.json`, SHA256 `ac354acd199bee67d952c311fd01474148e74fd5285fe9d39b3b7ee0939d6fcb`. The original freeze and experimental design are preserved. Obtain renewed authorization for the new published commit and V2 hash before running the controller. No GPU job has been submitted.

Execution update (2026-10-15): the authorized attempt at `72e97fd` stopped before GPU submission when the vLLM CLI preflight exceeded its fixed 120-second timeout. Runtime, model integrity and token-count checks passed. No job, new output or semantic rating was produced. No retry was attempted. See [execution status](experiments/sun_c2_confirmatory/RUN_STATUS.md). Frozen code and evaluation inputs remain unchanged; do not resume automatically.

## Current study

The remaining 65-case confirmatory evaluation is prepared on `sun/c2-confirmatory-65`. Read [the study documentation](experiments/sun_c2_confirmatory/README.md), [protocol](experiments/sun_c2_confirmatory/protocol.json), [freeze inventory](experiments/sun_c2_confirmatory/freeze.json) and [validation record](experiments/sun_c2_confirmatory/VALIDATION.md).

The study uses 260 original requests, excludes all five pilot cases, and retains the original Qwen model and sampling settings. All 65 outcome-blind anchors, two review streams and analysis rules are frozen before new generation. Code, CPU tests and independent review are complete. No remaining-case model outputs have been generated or inspected.

GPU execution is not authorized by this checkpoint. The proposed envelope is one L40S job for at most 60 minutes, 260 completions, 194,338 input tokens and 104,000 output tokens. No paid API, retries, requeues or additional jobs are included. Fresh MIT runtime/cache validation and Slurm `--test-only` must pass before the single authorized submission. The controller enforces those prerequisites.

## Continuation

1. Use the published branch and its exact commit. Do not force-push a divergent local checkout.
2. Verify `freeze_v2.json` and the original bundle using the commands in the study documentation. The bundle remains at `~/orcd/pool/ar_harness/c2/pilot-v1` on Engaging. Preserve the original `freeze.json` as the immutable parent.
3. Obtain explicit authorization for the published commit, freeze hash and full resource envelope. Keep the authorization record private.
4. Use the single-submission controller; preserve its canonical approval ledger and scheduler receipt. Inspect an uncertain submission rather than submitting again.
5. Preserve raw run records, generate blind packets, obtain two fresh-context reviews per answer, and use the frozen analysis. Do not supply condition labels or other reviewers' scores to reviewers.
6. Publish English research results, paired contrasts and intervals, numerical ratings, necessary examples, failures and limitations. Keep operational records and review working files private. Do not modify upstream or merge a PR.

## Completed exploratory evidence

The successful five-case pilot and its 20 original answers remain in `experiments/sun_c2_real/reports/pilot_25941208_analysis.json` (SHA256 `cf049d5853ec73473f324b88a1f5164e52e39310ad999d3c06c4f3a73f920664`). Pilot positions are 7, 29, 40, 58 and 70. The original manifest SHA256 is `1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548`.

The completed exploratory semantic audit is in `experiments/sun_c2_semantic/REPORT.md`; numerical results are in `results/audit_v1.json`. Its unchanged rubric, frozen packets and original reviews remain available there. Reproduce those saved results without new inference or ratings:

```sh
python -m experiments.sun_c2_semantic.audit --output audit_reproduction.json
```

The rubric was developed after pilot outcomes were known. The prior audit's original freeze timing and subsequent repository packaging are documented in its existing provenance records; commit dates alone do not establish a pre-pilot registration. The confirmatory study excludes those cases and does not change their responses or scores.

Historical full logs are retained in the private MIT pool under `c2/runs/pilot_25941208`. Existing passed test records should be reused unless a code change or new failure warrants another execution. The current study's focused validation is documented separately.
