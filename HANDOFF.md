# C2 Semantic Audit handoff

Checkpoint date: 2026-10-15. Branch: `sun/c2-semantic-audit` in `sun-tze/self-evolving-active-reasoning`.

## Scope and authorization

Audit the existing 20 answers from successful Job 25941208 only. Two independent blind reviewers are explicitly authorized. No new Qwen inference, GPU jobs, paid APIs, upstream changes or PR merge. User has about 35% Work quota remaining: prioritize research value, preserve checkpoints, avoid repeated passed tests and cosmetic work.

## Completed at this initial checkpoint

- Original real pilot and its analysis are committed at base `1e0798dbe71a6f040f84560d94ce8626eeae5b0d`; execution commit was `58f7e7bb68bf4caf1b36826073685f3ceafb5fdc`.
- New offline extraction code, reproducible rubric, selected-case causal anchors and confirmatory protocol prepared under `experiments/sun_c2_semantic/`.
- Frozen selected data and two blinded, shuffled packets will be committed alongside their SHA256 inventory. Original scores and responses remain unchanged.
- No independent semantic ratings existed before this checkpoint; do not describe planned ratings as completed.

## Pending

1. Two fresh-context blind agents rate their assigned packet independently and write raw JSON reviews using the rubric schema. Each reads only `rubric_v1.json` and its `frozen/packet_r1.json` or `packet_r2.json`. They must not inspect source_selected, maps, historical report, F1 or each other's ratings.
2. Validate complete 20-item coverage, hashes, integer ranges, faithfulness formula, exact issue quotes and evidence references. Preserve raw ratings; primary aggregate is mean of both. Record every disagreement.
3. Compute A/B/C/D semantic means, normalized paired contrasts, strict ranking reversals against F1, agreement. Add minimal meaningful CPU tests, independent code/method check, concise research report.
4. Update this file and push final checkpoint. Do not rerun the earlier completed GPU pilot or old CPU suites merely for reporting.

## Reproducible continuation from another account

Clone the fork and checkout `sun/c2-semantic-audit`. Python 3.11+ standard library is sufficient. All selected surfaces, dialogues, reference targets and 20 verbatim responses needed for review are in `experiments/sun_c2_semantic/frozen/source_selected.json`; no MIT login or local `.c2-local` directory is required for evaluation/analysis after extraction.

The extraction command, only if the original private bundle is available, is:

```text
python -m experiments.sun_c2_semantic.prepare --bundle .c2-local/pilot-final --output <NEW_DIRECTORY>
```

It refuses to overwrite frozen output. Default authoritative extraction is `experiments/sun_c2_semantic/frozen/`. `FREEZE.json` contains file hashes and timing. Reviewers must not read `unblind_map.json` until both reviews are committed. The hash inventory is not adversarial isolation; blinding is procedural.

## Historical evidence locations

- Published immutable pilot outputs/F1: `experiments/sun_c2_real/reports/pilot_25941208_analysis.json` (SHA256 `cf049d5853ec73473f324b88a1f5164e52e39310ad999d3c06c4f3a73f920664`).
- Original pilot report: `experiments/sun_c2_real/Research_Pilot_Report_25941208.txt`; provenance/runtime result: `experiments/sun_c2_real/reports/pilot_25941208_result.json`.
- Fixed manifest SHA256: `1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548`. Selected positions 7,29,40,58,70; 5 x 4 answers.
- Local complete run logs/archive: `.c2-local/pilot2-58f7e7b/evidence/` and `pilot_25941208_evidence.tar.gz` (archive SHA256 `8bf08541c5a8ac16e79747afdfde4fbd7d034f63144f1a7956584e343c7b8a39`). These large run artifacts are not necessary to continue the semantic audit. Engaging preserved run directory is `/home/sunyz/code/ar_harness/runs/pilot_25941208`; consult historical delivery documents for staging details.
- Prior test results already saved under `experiments/sun_c2_real/reports/`: Engaging 95/95; Windows 94 pass and one platform skip. New audit tests are pending.

## Research limitations and next decision

The pilot outcomes were already known to the rubric designer. This is exploratory post-hoc rubric design, frozen before independent semantic ratings and before the remaining 65 outcomes. Same-model independent agents are not independent human experts. Only 5 cases: no generalization or significance claim. Primary scoring retains disagreements; no silent consensus. The remaining 65-case experiment is a proposal requiring new authorization, future outcome-blind reference anchors and budget checks.

Repository packaging note: the freeze time and file hashes are recorded in `frozen/FREEZE.json`. This commit packages preserved inputs; its commit timestamp is not evidence of evaluation timing.
