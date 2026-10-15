# C2 Semantic Audit: existing 20-answer pilot

The semantic audit does **not support a writer-instruction improvement in these five cases**. Both reviewers identify the correct core mechanism only in case 29, conditions A and B. Character F1 sometimes ranks an incorrect explanation above a correct one. These are exploratory findings, not evidence of general superiority or failure on the benchmark.

## Results

Two fresh-context agents independently evaluated all 20 answers with shuffled opaque IDs. Conditions, writer prompts, source labels and F1 were hidden. Core reference anchors and the rubric were frozen before ratings; original recorded time and hashes are in `frozen/FREEZE.json`, preserved in checkpoint `7d7478d19021f68c5b30ca31f778197fb7f3314f`. Current repository history was repackaged after evaluation, so its commit timestamps do not independently establish pre-review freezing. The rubric designer already knew the pilot outputs; this is a post-hoc pilot audit, not pre-pilot preregistration.

Primary scores are the mean of the two original ratings. Semantic Correctness measures the required causal mechanism; Evidence Faithfulness combines referee consistency and unsupported-claim severity. They remain separate metrics.

| Condition | Dialogue / writer | Character F1 (0-1) | Semantic Correctness (0-2) | Evidence Faithfulness (0-2) | Fully correct answers |
|---|---|---:|---:|---:|---:|
| A | Bare / plain | 0.745 | 0.40 | 0.80 | 1/5 |
| B | Bare / evidence | 0.694 | 0.40 | 0.80 | 1/5 |
| C | First-selected / plain | 0.697 | 0.00 | 0.00 | 0/5 |
| D | First-selected / evidence | 0.717 | 0.00 | 0.00 | 0/5 |

For comparable contrasts, semantic and faithfulness scores below are divided by two. Contrasts are computed within each case and then averaged.

| Contrast | Character F1 | Semantic Correctness | Evidence Faithfulness |
|---|---:|---:|---:|
| Writer: ((B-A)+(D-C))/2 | -0.0159 | 0.0000 | 0.0000 |
| Dialogue: ((C-A)+(D-B))/2 | -0.0129 | -0.2000 | -0.4000 |
| Interaction: (D-C)-(B-A) | +0.0709 | 0.0000 | 0.0000 |

The positive lexical interaction is absent from both semantic metrics. The negative semantic dialogue contrast comes entirely from one case and should not be generalized. Acquisition costs differed historically; these are effects of the fixed recorded histories, not equal-compute search effects. No new significance tests were performed.

## Ranking reversals and examples

- Semantic correctness: **4 strict reversals among 30 within-case condition pairs**, all in case 29; the other 26 pairs are semantic ties. Thus the four comparable non-tied pairs all reverse, but this is one case, not four independent observations. Faithfulness also has 4 reversals, alongside 8 concordant pairs and 18 ties.
- Condition-mean ordering changes from F1 **A > D > C > B** to both semantic metrics **A = B > C = D**. B-versus-C and B-versus-D are strict mean-order reversals; A-versus-B and C-versus-D become ties.
- Case 29: B correctly explains survival through a fire-resistant suit (semantic 2, faithfulness 2), but F1 is only **0.484**. C proposes a safe room (semantic 0, faithfulness 0) and scores **0.689**; D proposes a hidden compartment and scores **0.666**. Recorded bare dialogue explicitly supports protective clothing; the first-selected history denies key safe-room claims.
- Case 7: B receives F1 **0.929** while replacing a hospital record mix-up with a deliberate deception. Both semantic reviewers assign 0. This is high lexical overlap with the wrong causal mechanism.
- Case 40: A receives F1 **0.921**, semantic 0 and faithfulness 1. Its trusted-intermediary account draws on recorded answers but misses mistaken burial/escape and the coded safety signal. Faithfulness to an unhelpful dialogue does not establish reference correctness.
- Case 70: D receives F1 **0.843** for a hidden-passage explanation; the reference mechanism is virtual reality. Both semantic and faithfulness scores are 0.

## Agreement, ambiguity and limitations

Semantic Correctness, unsupported severity and Evidence Faithfulness agree on **20/20 answers** (quadratic-weighted kappa 1.00). Referee consistency agrees on **17/20** (kappa 0.8404). Differences are preserved in `results/audit_v1.json`: case 7 A (consistency 2 versus 0; advance knowledge versus a supposed hidden prank), case 40 A (1 versus 2; speculative motives and Unknown versus denial), and case 58 B (1 versus 2; scope of a compound question about fragile valuables). These differences do not change aggregate faithfulness because unsupported severity already determines the lower score.

Both reviewers mark central unsupported claims in **14/20 answers**. They identify central contradictions in 10/20 and 11/20 respectively. These are answer-level counts, not claim-level rates.

The two reviewers are independent executions of the same model family, not independent human experts. Shared rubric anchors can create shared errors. Blinding is procedural, not filesystem access isolation, and repeated case content remains recognizable. Five cases and one deterministic generation per cell limit inference. Original histories may be noisy or internally inconsistent; this audit preserves them without repair. Core-anchor sensitivity and cross-family/human validation remain untested.

## Freeze, cost and next experiment

`confirmatory_v1.json` was frozen before these independent ratings and before any remaining 65-case outcomes. It proposes 260 existing fixed requests with normalized semantic writer effect as the single primary estimand, paired case bootstrap intervals, retained raw reviews and explicit missing-data handling. Before any future generation, freeze outcome-blind reference anchors for all 65 cases and reviewer assignments. The proposal still requires a separate verified input/job budget and new user GPU authorization; it has not been executed.

This audit used **0 new Qwen calls, 0 GPU jobs and 0 paid external API calls**. Two review agents produced 40 answer-level ratings using Work quota; that usage was not separately metered. Existing pilot costs remain 15,210 input tokens and 1,817 output tokens, unchanged. All original responses, F1 scores and run evidence are preserved.

Seven distinct focused CPU tests passed across initial and fix-specific runs. Independent code review found two validation gaps; both were corrected and rechecked. The original frozen rubric/packets and both raw reviews remain unchanged. See `REVIEW.md`, `results/source_provenance.json` and root `HANDOFF.md` for evidence and cross-account continuation. The saved analysis can be reproduced offline without invoking any model.
