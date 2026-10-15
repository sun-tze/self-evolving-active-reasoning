# C2 confirmatory evaluation: 65 held-out cases

The evidence-writer instruction did not improve semantic correctness under the frozen evaluation. Its normalized paired effect was **-0.0308 (95% CI -0.0577 to -0.0058)**. The interval lies below zero, although the prespecified confirmation rule concerned a positive effect. Character F1 slightly favored the evidence writer, with an interval spanning zero. These results concern the fixed requests, model and reference-based ratings used here.

## Design and completion

All 260 original requests completed successfully on 15 October 2026. The five pilot cases were excluded. A/B use bare dialogue; C/D use first-selected dialogue. A/C use the plain writer; B/D use the evidence writer. The model was Qwen2.5-7B-Instruct at the original pinned revision, with temperature 0, top-p 1, the original seeds and a 400-token output cap.

The rubric, 65 outcome-blind reference anchors, assignments and analysis rules were frozen before generation. Two blind model ratings per answer were collected in 26 fresh contexts, yielding **520/520 validated ratings**. Reviewers received only the rubric, review protocol and assigned packet; condition labels, original case indices, writer instructions, F1 and other ratings were withheld. Both original ratings were retained and averaged, without consensus replacement. Semantic Correctness assesses the reference causal mechanism; Evidence Faithfulness uses only the surface and recorded referee answers.

## Results

| Condition | Dialogue / writer | Character F1 (0-1) | Semantic Correctness (0-2) | Evidence Faithfulness (0-2) | Fully correct by both reviewers |
|---|---|---:|---:|---:|---:|
| A | Bare / plain | 0.681 | 0.785 | 0.662 | 7/65 |
| B | Bare / evidence | 0.683 | 0.754 | 0.669 | 8/65 |
| C | First-selected / plain | 0.693 | 0.754 | 0.331 | 15/65 |
| D | First-selected / evidence | 0.695 | 0.662 | 0.400 | 12/65 |

The semantic columns are ordinal-score means, not accuracy percentages. The fully-correct column requires both reviewers to assign 2 and is a different descriptive endpoint from the prespecified mean score.

Contrasts below normalize both semantic scales to [0,1]. Brackets are paired-case 95% bootstrap intervals, using 10,000 resamples and seed 62026. Every contrast contains all 65 cases.

| Paired contrast | Semantic Correctness | Evidence Faithfulness | Character F1 |
|---|---:|---:|---:|
| Writer: ((B-A)+(D-C))/2 | -0.0308 [-0.0577, -0.0058] | +0.0192 [-0.0173, +0.0577] | +0.0020 [-0.0167, +0.0213] |
| Dialogue: ((C-A)+(D-B))/2 | -0.0308 [-0.1288, +0.0654] | -0.1500 [-0.2654, -0.0346] | +0.0126 [-0.0349, +0.0590] |
| Interaction: (D-C)-(B-A) | -0.0308 [-0.1038, +0.0346] | +0.0308 [-0.0423, +0.1077] | +0.0001 [-0.0368, +0.0376] |
| Writer with bare dialogue: B-A | -0.0154 [-0.0538, +0.0231] | +0.0038 [-0.0500, +0.0577] | +0.0019 [-0.0212, +0.0273] |
| Writer with first-selected dialogue: D-C | -0.0462 [-0.1000, +0.0000] | +0.0346 [-0.0115, +0.0885] | +0.0021 [-0.0258, +0.0313] |
| Dialogue with plain writer: C-A | -0.0154 [-0.1192, +0.0846] | -0.1654 [-0.2846, -0.0500] | +0.0125 [-0.0379, +0.0622] |
| Dialogue with evidence writer: D-B | -0.0462 [-0.1500, +0.0577] | -0.1346 [-0.2615, -0.0077] | +0.0126 [-0.0377, +0.0634] |

The single primary endpoint was the semantic writer marginal effect, with confirmation requiring its lower interval endpoint above zero. That criterion was not met. The secondary faithfulness writer estimate was positive but uncertain. First-selected histories had lower faithfulness on this sample; their semantic and F1 dialogue intervals included zero. None of the interaction intervals excluded zero. Secondary intervals are descriptive and are not adjusted for multiple comparisons. Dialogue contrasts compare these recorded histories, whose historical acquisition costs differed; they do not establish equal-compute search efficiency.

## Metric disagreement and examples

Condition means ranked **D > C > B > A** under F1, **A > B = C > D** under Semantic Correctness, and **B > A > D > C** under Evidence Faithfulness. Five of six condition-mean pairs strictly reversed between F1 and semantic scores; one became a semantic tie.

Among 390 within-case condition pairs, semantic comparison produced **82 strict reversals across 29 cases**, 83 concordant pairs and 225 ties. Faithfulness produced **111 reversals across 37 cases**, 76 concordant pairs and 203 ties. Ties include pairs tied on both metrics. Pairs within a puzzle are dependent, so these counts are descriptive, not independent trials.

Two post-hoc illustrations were selected by the largest lexical gap among semantic reversals and the highest F1 among answers scored semantically zero:

- **Case 41:** A proposes spirits influencing a diary: F1 **0.335**, semantic **1**, faithfulness **2**. It earns partial credit, since the reference requires an enchanted diary linked to an oracle. C substitutes a person inserting predictions: F1 **0.779**, semantic **0**, faithfulness **0**. Greater lexical overlap accompanies a worse causal explanation.
- **Case 77:** B scores F1 **0.925**, but semantic **0** and faithfulness **0**. Its account of psychological influence misses both reference mechanisms: letters from Sarah's future self and preventive changes to the timeline. All four conditions score semantic zero in this case.

## Reviewer agreement and reference quality

| Rating axis | Exact agreement | Quadratic-weighted kappa |
|---|---:|---:|
| Semantic Correctness | 252/260 (96.9%) | 0.971 |
| Referee consistency | 252/260 (96.9%) | 0.949 |
| Unsupported-claim severity | 246/260 (94.6%) | 0.945 |
| Evidence Faithfulness | 246/260 (94.6%) | 0.945 |

There were **28/260 answers with a disagreement on at least one axis**, including eight semantic disagreements. All original scores and evidence-linked annotations are retained privately. High agreement does not establish accuracy: reviewers are independent executions from the same model family, not independent human experts, and can share systematic biases. Blinding is procedural; the shared filesystem and recognizable answer content do not provide technical isolation.

The eight cases flagged for reference quality before generation remain in the primary analysis. Their pooled semantic mean was 0.625/2, versus 0.754/2 for the other 57 cases. This frozen descriptive stratification is not an exclusion analysis or a causal comparison.

## Resources and failures

One L40S job completed in **588 seconds (9 minutes 48 seconds; 0.1633 GPU-hours)** with exit code 0, within the 60-minute limit. The allocation was eight CPUs and 32 GiB host memory. It made exactly **260 calls**, using **194,338 input tokens and 23,425 output tokens** (217,763 total). Output usage was 22.5% of the 104,000-token cap. There were **zero inference failures, ambiguous calls, missing responses or truncated completions**; no retries, requeues or additional jobs were used.

Model-default and shutdown-cleanup warnings were recorded without affecting successful completion. Fresh code, integrity, runtime, model-cache, tokenizer, bounded CLI and scheduler preflight checks passed before submission. No paid API was used. Institutional GPU charges are unknown; reviewer Work usage was not separately metered. Historical episode costs are reported separately in the numerical summary and include earlier finalizers; acquisition-only and complete referee-token costs are unavailable.

## Provenance, reproduction and limits

Execution used commit `67883c8bd7f9f7aacc270c0bcf91375e292bd10e` and V2 freeze SHA256 `ac354acd199bee67d952c311fd01474148e74fd5285fe9d39b3b7ee0939d6fcb`. The original manifest, requests, model settings, rubric, anchors and assignments were unchanged. The frozen semantic pipeline verified packet hashes and coverage and reproduced F1 directly from the original raw journal before aggregation.

The [numerical summary](results/summary.json) provides full-precision aggregate estimates and provenance hashes. [HANDOFF.md](../../HANDOFF.md) locates the private archives and documents offline reproduction. Raw responses, operational logs, review packets and reviewer annotations remain private. The pilot's responses and scores remain unchanged and are not pooled here.

The rubric was developed after the exploratory pilot but frozen before these 65 outcomes. Imperfect fictional reference mechanisms and noisy referee histories constrain validity. The case bootstrap holds ratings fixed and does not estimate reviewer uncertainty or generation variance. This finite case set is not a random sample of all puzzles; one greedy generation per cell and same-family reviewers limit generalization. Human or cross-family validation has not been performed. The result supports separating lexical overlap, semantic correctness and evidence faithfulness; it does not support the proposed positive writer effect under this evaluation.
