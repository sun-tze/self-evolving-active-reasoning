# GN: public evidence and final-answer readout

This contribution separates **collecting useful feedback** from **turning that feedback into a correct final answer**. It includes a small public-only constraint adapter, synthetic tests, paired-statistics reproduction, and the experiments that motivated it.

The repository already implements exact GN inference in [stage0/gn_oracle.py](../../stage0/gn_oracle.py), and [runner/gn_runner_async.py](../../runner/gn_runner_async.py) already supports belief summaries. **Exact solving and candidate-state prompting are not claimed as new methods.** The additions here are a strict public-record interface, repeated-digit guess handling, explicit abstention, a controlled final-readout comparison, and its measured limitations.

## Evidence so far

All model experiments use frozen Qwen2.5-7B-Instruct, revision `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0.7 and top_p 0.7. Each episode has 25 public-feedback interactions; only the final readout differs within the later paired studies. Weights are not trained.

### 1. Context evolution: completed and audited

One 20-step adaptation trajectory produced 19 format-accepted curator outputs but only **one actual note change**, at step 2. Evaluation used 100 paired cases and four conditions:

| Condition | Mean Q | Exact answers /100 | Record uniquely determines answer | Final outputs truncated |
|---|---:|---:|---:|---:|
| B0: no note | 0.886680 | 1 | 92 | 65 |
| B1: official static example | 0.766758 | 0 | 33 | 69 |
| S: initial note M0 | 0.808280 | 0 | 71 | 31 |
| E: final note M20 | 0.852756 | 0 | 70 | 74 |

Q averages normalized log candidate-set reduction over the 25 interaction steps; it is **not accuracy or a measurement of internal model belief**. Primary E−S Q difference: +0.044476, paired percentile 95% CI [0.020238, 0.068738], 66 wins /34 losses. E did not outperform B0 descriptively, and no final-accuracy gain was observed. Invalid curator notes were rejected while preserving the previous bytes; early-format checks became diagnostic records. These development-time amendments are part of the executed version, not an unchanged original frozen protocol.

### 2. Limited-feedback readout: completed and audited

A separate 50-case train-derived internal holdout was fixed before requests, excluding prior designated labels and declared exposed four-digit strings. It is **not a new official-test replication**. A common B0 history was followed by four final variants:

| Variant | Final procedure | EM /50 | Record-consistent /50 | Truncated /50 |
|---|---|---:|---:|---:|
| R0 | Original final prompt, 128 tokens | 0 | 0 | 32 |
| R1 | Same prompt, 512 tokens | 0 | 0 | 5 |
| R2 | Shared strong-prompt draft + neutral self-check; 256+256 caps | 0 | 0 | 0 |
| R3 | Same draft + first public-feedback contradiction; same caps | 0 | 0 | 0 |

R2 and R3 share the exact draft and revision seed. The checker does not enumerate candidates or see labels. Raw second outputs are scored, with no best-of selection or repeated correction. The original parser returns the last four-digit span, including for truncated outputs; missing spans count as wrong. There is no favorable reparsing.

Primary R3−R2: 0 percentage points, 0 wins /0 losses /50 ties. The paired bootstrap is degenerate [0,0]; supplementary simultaneous exact-binomial bounds are [-8.391,+8.391] percentage points. This is not an equivalence claim. R1 reduces truncation without observed EM improvement; it does not show that all other budgets or formats would fail. All 50 drafts and both revisions emitted exactly 8 output tokens despite larger caps. R3 changed 41/50 drafts and satisfied its cited constraint in 7/50 cases (R2: 1/50), yet none satisfied the whole record.

The allocation completed in 295 GPU seconds with 1,550 requests (50 discarded development readouts +1,500 evaluation requests), 2,093,206 input and 29,630 output tokens. Experimental API cost was $0; GPU time is reported separately.

### 3. Symbolic replay: post hoc, not a model result

The public-only solver recovered 44/50 answers correctly, abstained on six, and emitted no wrong answer. Candidate counts were 44 singletons, five pairs and one triple. Local replay took approximately 0.765 CPU wall seconds, without a model call. This was computed **after observing the null readout result**; it motivates a new test and does not rescue R3−R2.

### 4. Prospective answer-supplied readout: completed and audited

A further disjoint internal N50 was fixed before this run, with 1,996 eligible labels after declared exclusions. This is train-derived and not an official-test or pretraining-unseen claim. Baseline and assisted finals share the same new 25-turn history, 128-token cap and seed. Execution order alternates. The assisted final receives an exact candidate-state block before the unchanged final instruction. Raw model outputs are never replaced by solver answers.

| Final procedure | Correct /50 | Wrong /50 | Abstained /50 | Record-consistent /50 |
|---|---:|---:|---:|---:|
| Baseline model | 0 | 50* | 0 | 0 |
| State-assisted model | 46 | 4 | 0 | 49 |
| Direct unique-only solver | 46 | 0 | 4 | 46 |

*Baseline wrong includes one output with no parsed answer. Abstentions count as incorrect for all-case exact match. Direct-solver coverage is 46/50; its observed accuracy among answered cases is 46/46, not a universal guarantee.*

The primary paired assisted−baseline gain is **+92 percentage points**: 46 wins, zero losses, four ties. Paired bootstrap 95% CI: **[84,98] pp**. Supplementary conservative simultaneous exact-binomial bounds: **[70.644,98.201] pp**. These are within-study estimates, without an adjustment for outcome-informed sequential method development.

There were 46 singleton records and four ambiguous records (two to 64 candidates). The assisted model answered 45/46 singleton cases and 1/4 ambiguous cases correctly. The four ambiguous sets had sizes 3, 2, 4 and 2. Uniform selection would yield 1.583 expected correct answers across those four; the observed one correct does not demonstrate above-chance selection. No case exercised the above-64 count-only branch. Relative to the direct solver, one lost singleton and one gained ambiguous case offset exactly: **the model adds no aggregate EM on this sample**, while producing four wrong outputs instead of four abstentions. The practical routing choice depends on the cost of wrong answers versus abstention. The unique-only solver avoids a final model call, while still requiring the same preceding interaction history.

Baseline final outputs truncated in 31/50 cases; assisted outputs in 0/50. Final output tokens fell from 5,489 to 536, while final input tokens increased from 106,802 to 111,837. Summed final-call elapsed times were 130.441 versus 15.001 seconds under concurrency, **not end-to-end wall-clock latency**. The resolver used 0.817 thread-CPU seconds in total. These timings do not imply that the whole system is model-free or equally fast on other hardware.

Two assisted outputs mentioned multiple candidate numbers in prose and ended with an explicit final Guess. The retained diagnostic field named `state_JSON_marker_echo` only detects the substrings `candidate_count` or `candidates`; both hits are **prose mentions, not JSON echoes**. The two marker hits and the two multiple-span hits are the same two cases. Both have an explicit final Guess that agrees with the frozen parser: anonymous pair 28 was correct and pair 32 was wrong. No alternative parser was applied. The unchanged last-four-digit parser scored those explicit guesses. The parsed answer equaled the final listed candidate on 46/50 cases, 45 are correct singleton outputs and one is an ambiguous output; this is an equality check rather than proof of parser contamination or a cognitive mechanism.

The run completed in **218 GPU seconds**, using 1,350 requests, 1,641,690 input and 16,027 output tokens. Experimental API cost was $0. The independent reconstruction checker passed once in 1.919 seconds and checked all requests, public feedback, states, seeds, caps and outcome scoring. It ran under the same executor, not an independent organization. The original null results remain above.

When a record has one candidate, the tool has computed the answer. The gain is therefore **answer-supplied/tool-assisted readout**. It does not establish learned reasoning, memory evolution, solver novelty, or which component of state content, input length and presentation caused the gain. See [readout_settings.json](readout_settings.json) for the fixed settings and [results.svg](results.svg) for the result figure.

The result supports a concrete engineering change: use public-feedback constraint checking for reliable final answers. A subsequent [prospective question-efficiency study](../sun_gn_question_efficiency/README.md) has now completed: three evolved-note chains averaged 1.76 more queries than no note on 100 new paired cases (95% CI [0.19,3.28]). This negative learning result is separate from the positive tool-assisted readout result. A different performance-selected policy method is under investigation; no further benefit is claimed.

## Use the adapter

The hidden number has four distinct ASCII digits; leading zero is allowed. A guess may repeat digits. Each public record has exactly these fields:

```json
[{"turn": 1, "guess": "0011", "same": 1, "different": 1}]
```

`resolve_public(records)` enumerates all 5,040 legal hypotheses, returns an answer only if exactly one survives, and otherwise reports ambiguity or inconsistency. It accepts no secret/label fields. `public_state(records, 64)` returns the complete sorted candidate list up to 64; larger sets get an explicit count-only representation, never a selected subset. Inconsistent records raise an error for model-state delivery. Dataset/exposure exclusions never prune the solver universe.

`readout_pair.branch_readout` is a portable extraction of the evaluated final-branch routine. It accepts public history, frozen prompt text and a caller-provided boundary that records requests and enforces budgets. It neither loads a model nor submits jobs. The private cluster deployment is not included, so this folder alone is not an end-to-end reproduction of the original allocations.

## Verify without a GPU

From this directory, using standard-library Python:

```sh
python -B -m unittest discover -v
python -B verify_public_results.py
```

Nine synthetic adapter/statistics tests pass. The allocation runner additionally passed a 50-case synthetic auditor reconstruction and rejection of corrupted state; a submission-directory naming mismatch was caught before GPU submission and fixed without changing scientific code. These code checks are not real model performance evidence.

[results.json](results.json) contains aggregates and anonymized paired outcome rows so the released EM statistics can be recomputed. Raw prompts, records, private filesystem paths, authorization documents, target labels and original case identifiers are excluded. This limits independent reconstruction of raw model outputs; hashes bind retained originals but do not replace access to them. Audit code independently reconstructed results, but was executed by the same assistant. Research reviewers reviewed reported evidence, not an independent raw-data replication.

This is one model, one decoding setup and outcome-informed sequential follow-up. Restricting the internal holdout by prior exposures changes its eligible distribution. Null and positive results remain separate, and all tried variants remain visible. No general model-incapacity claim, selective winner-only report, or unadjusted family-wide confirmatory claim is warranted.
