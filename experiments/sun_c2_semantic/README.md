# Offline C2 Semantic Audit

Uses only the 20 existing answers from Job 25941208. No inference, credentials, MIT connection or third-party Python packages are needed for continuation from GitHub. Original responses and Character F1 remain in the historical report, unchanged.

## Frozen evaluation

`rubric_v1.json` specifies three ordinal axes: core semantic correctness, consistency with referee evidence, and unsupported-claim severity. Evidence Faithfulness is `min(consistency, 2 - unsupported_severity)`. Scores range 0–2. Reference truth and dialogue support are separate: a true gold fact can still be unsupported by the recorded dialogue.

`anchors_v1.json` fixes necessary causal components for these five cases. `confirmatory_v1.json` proposes the remaining 65-case experiment; it does not authorize execution. The rubric designer had seen pilot outputs before freezing; this is explicitly a post-hoc exploratory audit.

`frozen/FREEZE.json` binds selected inputs, independently shuffled review packets, source mappings, rubric and design by SHA256. Preserved input checkpoint: `7d7478d19021f68c5b30ca31f778197fb7f3314f`. The recorded content freeze preceded independent ratings; current Git history was repackaged afterward, so its commit timestamps are not independent proof of pre-review freezing. The two reviewers must read only their packet and rubric. No condition labels, writer instructions, source labels, historical scores or original IDs appear in packets. The shared filesystem provides procedural blinding, not access isolation.

## Files and reproduction

- `frozen/source_selected.json`: complete selected surfaces, histories, reference solutions and 20 verbatim answers, with historical identities/scores.
- `frozen/packet_r1.json`, `packet_r2.json`: independently shuffled, opaque item IDs for the two reviewers.
- `reviews/r1.json`, `r2.json`: completed raw reviewer outputs. Never replace disagreements with consensus.
- `audit.py`: validates frozen data, rating coverage, ranges, evidence quotes/references, formula; computes paired contrasts, agreement and ranking reversals.
- `test_audit.py`: seven focused tests, including the independent review regressions. Original passing suites need not be rerun.
- `results/audit_v1.json`: completed machine-readable analysis; `REPORT.md` is its concise interpretation.
- `provenance.py`: original target-file hash and selected reference identity checks; completed evidence in `results/source_provenance.json`.
- `prepare_verified.py`: supported future extraction entry. The frozen original `prepare.py` is archival; its target hash check was supplemented without changing any frozen packet.

From the repository root, Python 3.11+:

```text
python -m experiments.sun_c2_semantic.audit --output audit_reproduction.json
```

The output must not already exist. This validates and analyzes the committed reviews without asking a model to rate again. Results are deterministic; a reproduction should be byte-identical to `results/audit_v1.json`. To run only new audit tests if needed:

```text
python -m unittest experiments.sun_c2_semantic.test_audit -v
```

Extraction from the original bundle is documented in root `HANDOFF.md`; it is unnecessary for replaying the audit. Do not regenerate or overwrite frozen inputs. Reviewer judgments are not guaranteed reproducible from another model invocation; the saved annotations and their deterministic analysis are reproducible.

## Interpretation

Primary answer score is the arithmetic mean of two independent reviews. Semantic/Evidence Faithfulness means are shown on 0–2; contrasts divide them by 2 to compare with F1 on 0–1. All disagreements and raw axes remain available. A strict ranking reversal requires opposite nonzero signs; ties are counted separately. Answer pairs are dependent within cases and are descriptive counts, not independent trials.

The reviewers are two fresh-context model executions, not two human experts or diverse model families. Five cases cannot establish general improvement. The remaining 65 require outcome-blind reference anchors, prospective review assignments, budget verification and new user authorization before any GPU job.
