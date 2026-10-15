# C2 Semantic Audit handoff

Checkpoint date: 2026-10-15. Branch: `sun/c2-semantic-audit` in `sun-tze/self-evolving-active-reasoning`.

## Scope and authorization

Audit the existing 20 answers from successful Job 25941208 only. Two independent blind reviewers are explicitly authorized. No new Qwen inference, GPU jobs, paid APIs, upstream changes or PR merge. Prioritize research value, preserve checkpoints, and avoid repeated passed tests and cosmetic work.

## Current status: semantic audit completed

- Original real pilot and its analysis are preserved at base `1e0798dbe71a6f040f84560d94ce8626eeae5b0d`; execution commit remains `58f7e7bb68bf4caf1b36826073685f3ceafb5fdc`.
- Reproducible rubric, selected-case causal anchors, confirmatory proposal and two independently shuffled blind packets were frozen before independent ratings. The preserved input checkpoint is `7d7478d19021f68c5b30ca31f778197fb7f3314f`; original content hashes and recorded freeze time are in `frozen/FREEZE.json`. Repository history was repackaged after evaluation, so current commit timestamps do not independently prove pre-review freezing. The rubric designer already knew pilot outcomes; this was not a pre-pilot preregistration.
- Two fresh-context agents completed all 20 ratings each. Both raw English reviews, exact evidence quotes, hashes and independence statements are saved in `experiments/sun_c2_semantic/reviews/`.
- Validation and deterministic analysis completed: `experiments/sun_c2_semantic/results/audit_v1.json`. All disagreements remain; no adjudication replaced ratings. Original scores and responses remain unchanged.
- Semantic means (0-2): A/B 0.4, C/D 0.0. Faithfulness means (0-2): A/B 0.8, C/D 0.0. Semantic writer contrast 0; dialogue contrast -0.2 on normalized scale. Only case 29 A/B correctly identify the reference mechanism. Four within-case semantic ranking reversals, all in case 29; 26 ties. See `experiments/sun_c2_semantic/REPORT.md`.
- Five initial focused tests passed. Independent code review found two validation gaps, both repaired: reverse issue/score consistency and target-file binding to pinned manifest. Two new regressions plus the affected validator test passed; seven distinct tests total, eight executions. No old C2 suites were rerun. Evidence is in `results/tests.json`, `results/tests_fixes.json`, `results/source_provenance.json` and `REVIEW.md`.
- Supplemental original-bundle target verification passed. Original rubric, packets and freeze files were not modified. Use `prepare_verified.py` for any future extraction; `prepare.py` is retained as the original frozen implementation, not the supported standalone entry point.
- All project deliverables use English. Historical numbers and machine-readable analysis remain unchanged.

## Not done / next authorized decision

1. No remaining 65-case model outputs have been generated or evaluated. Their outcome-blind reference anchors and reviewer assignments still need preparation before a future experiment. The frozen confirmatory proposal is in `confirmatory_v1.json`.
2. No human or different-model-family validation has been performed. Two independent same-model agents share systematic biases; perfect score agreement is not proof of validity.
3. A future 260-call experiment needs a separately checked token/job budget and NEW explicit user authorization. Do not submit any GPU job, expand scope, call paid APIs or rerun Qwen from this handoff.
4. The current semantic audit is complete. A new account should read this file, `REPORT.md`, the frozen rubric and machine-readable results first. Do not re-rate answers, repeat analysis or rerun passed tests merely to restore context.

## Reproducible continuation from another account

Clone the fork and checkout `sun/c2-semantic-audit`. Python 3.11+ standard library is sufficient. All selected surfaces, dialogues, reference targets and 20 verbatim responses needed for review are in `experiments/sun_c2_semantic/frozen/source_selected.json`; no MIT login or local `.c2-local` directory is required for evaluation/analysis after extraction.

The extraction command, only if the original private bundle is available, is:

```text
python -m experiments.sun_c2_semantic.prepare_verified --bundle .c2-local/pilot-final --output experiments/sun_c2_semantic/recreated_frozen
```

It refuses to overwrite frozen output. Authoritative extraction is `experiments/sun_c2_semantic/frozen/`. `FREEZE.json` contains file hashes and timing. The hash inventory is not adversarial isolation; blinding is procedural. The two blind reviewers were instructed to read only their assigned packet and rubric, and both reviews were saved before aggregation/unblinding. The GitHub checkpoint contains both raw reviews unchanged.

To reproduce saved analysis without re-rating, use a new output path:

```text
python -m experiments.sun_c2_semantic.audit --output audit_reproduction.json
```

This uses only committed files and needs no API, GPU or login. The seven focused tests, if later code changes justify running them, are `python -m unittest experiments.sun_c2_semantic.test_audit -v`. Current passed results are already committed. Detailed provenance re-verification requires the original private bundle; cross-account audit replay does not.

## Publishing notes

Published through the GitHub connector because this machine's shell could not resolve github.com. With explicit user authorization, the two C2 branch histories were repackaged after evaluation. Execution code, raw experimental data, original numerical scores, frozen evaluation files and raw reviews were preserved. The current GitHub branch is the authoritative cross-account continuation point. Existing local clones may contain divergent history: do not force-push it over the published branch. Use the current branch and the content hashes in `frozen/FREEZE.json`; distinguish original recorded freeze time from later repository packaging dates.

## Historical evidence locations

- Published immutable pilot outputs/F1: `experiments/sun_c2_real/reports/pilot_25941208_analysis.json` (SHA256 `cf049d5853ec73473f324b88a1f5164e52e39310ad999d3c06c4f3a73f920664`).
- Original pilot report: `experiments/sun_c2_real/Research_Pilot_Report_25941208.txt`; provenance/runtime result: `experiments/sun_c2_real/reports/pilot_25941208_result.json`.
- Fixed manifest SHA256: `1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548`. Selected positions 7,29,40,58,70; 5 x 4 answers.
- Local complete run logs/archive: `.c2-local/pilot2-58f7e7b/evidence/` and `pilot_25941208_evidence.tar.gz` (archive SHA256 `8bf08541c5a8ac16e79747afdfde4fbd7d034f63144f1a7956584e343c7b8a39`). These large run artifacts are not necessary to continue the semantic audit. Engaging preserved run directory is `/home/sunyz/orcd/pool/ar_harness/c2/runs/pilot_25941208`; consult historical delivery documents for staging details.
- Prior test results already saved under `experiments/sun_c2_real/reports/`: Engaging 95/95; Windows 94 pass and one platform skip. New audit test evidence is in `experiments/sun_c2_semantic/results/tests.json` and `tests_fixes.json`.

## Research limitations and next decision

The pilot outcomes were already known to the rubric designer. This is exploratory post-hoc rubric design, frozen before independent semantic ratings and before the remaining 65 outcomes. Same-model independent agents are not independent human experts. Only 5 cases: no generalization or significance claim. Primary scoring retains disagreements; no silent consensus. The remaining 65-case experiment is a proposal requiring new authorization, future outcome-blind reference anchors and budget checks.
