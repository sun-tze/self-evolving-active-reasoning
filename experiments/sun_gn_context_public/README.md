# GN context adaptation — implementation ready, experiment pending

Can a small model learn useful strategy notes from its own public interactions,
then use those frozen notes to acquire more information on unseen Guessing
Numbers cases? This studies context adaptation with fixed weights, separately
from the project's CD/SP proposer-and-judge work.

## Status on 24 October 2026

| Evidence | Verified result | Limit |
|---|---|---|
| Historical real C2 | 65 cases, 260 Qwen completions, 520 ratings; one 588-second L40S job. The positive writer-effect hypothesis was not confirmed. | Separate writer experiment; not a GN result. [Report](https://github.com/sun-tze/self-evolving-active-reasoning/blob/39db2ce769a48ef4a6c8bc3053c3c84bf27db830/experiments/sun_c2_confirmatory/REPORT.md) |
| GN offline V7 | Windows: 70 passed / 6 skipped. Independent Linux run of the same 76-test suite: 71 passed / 5 skipped. Independent tokenizer verifier: 53 checks passed. | Historical checks, not rerun for publication. Bounded V7 repair review: PASS WITH LIMITS. No installed-vLLM or inference claim. |
| GN real experiment | Not run in the verified records. | Actual N, M20, Q, EM, calls, tokens and allocation costs are unobserved. |

The independently authored tokenizer verifier was executed locally by the
coordinator. Synthetic responses/usage are not model or server evidence.

## Experiment and results table

One serial twenty-episode adaptation trajectory produces M20. Development chooses
100 transfer cases, or the prespecified 50-case fallback, before adaptation.
M20 stays fixed throughout transfer; there is no learning from evaluation cases.

| Condition | Context | Real mean Q | Real final EM: count / denominator |
|---|---|---|---|
| B0 | Official base prompt | NA | NA |
| B1 | Official static guidance | NA | NA |
| S | Fixed five-line M0 | NA | NA |
| E | M20 from the adaptation trajectory | NA | NA |

Primary E−S estimate, 95% interval, common-case n and missingness: **NA**.
These are empty result slots, not zeros.

- **Model/data.** Qwen2.5-7B-Instruct model and tokenizer revision
  `a09a35458c702b33eeacc393d103063234e8bc28`; AR-Bench
  `9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1`. Train rows 0–19 adapt;
  20–39 develop; test rows 0–99 transfer, or 0–49 under the development gate.
- **Interaction/memory.** Each episode has 25 interaction calls and one final
  guess, without early stopping. The curator sees old notes and public dialogue,
  not secrets or scores; the controller uses secrets for feedback/EM. Mechanical
  validation accepts 1–12 ASCII bullet lines, ≤512 tokens. Invalid proposals keep
  old memory; no repair, retry or quality selection. Exact prompts, M0 and rules
  are in the source.
- **Measurement.** Across all 5,040 possible secrets,
  `Q = mean[t=1..25](1 − log2(|H_t|)/log2(5040))`.
  Only delivered public feedback changes H; the final guess is excluded from Q.
  Final exact match is mandatory secondary. An infeasible guess can be informative.
- **Pairing.** E−S uses four-condition complete cases in manifest order;
  10,000 paired bootstrap draws, Python seed 62026. Report coverage and full-N
  missing-case bounds. Incomplete coverage is descriptive only. Exact seeds,
  condition rotation and interval rules are preserved in the source.
- **Sampling/budget.** Uniform neutral A2: `--generation-config vllm`;
  actor temperature/top-p/output cap .7/.7/128, curator 0/1/768; top-k disabled,
  repetition penalty one. Effective settings, EOS and reviewed vLLM 0.31.0 pins
  are in the contracts.
  At most two one-L40S jobs, each ≤120 minutes including startup/save/shutdown,
  four allocation-hours total, API $0. No extra arms, jobs or automatic retries.

## Source and remaining work

[Source](../sun_gn_context/) contains 32 unchanged V7 source/configuration/test
files from `e881f629490bc702865c52bb0783555bfa207ed1`.
[The manifest](PUBLICATION_MANIFEST.json) records their hashes and provenance.
All seven test-source files are retained; fixtures are synthetic.

**This is a source reference, not a complete runnable checkout.** Original
hash-bound protocols, answer-linked manifest and historical fixture remain
private. Full execution/tests require the unchanged private V7 package; do not
substitute this README or relax checks. No weights, credentials, runtime bindings
or raw run logs are published; C2 and teammate files are unchanged.

Remaining: permitted access and existing-job checks; pinned runtime/weight CPU
preflight; separate stage releases; development/adaptation and frozen transfer;
independent raw-log recomputation; fill the real results and analyze failure cases.
The implementation is complete; these execution prerequisites are still open.

For the presentation: explain the controlled memory comparison, why Q can measure
information acquisition even when final EM is low, and what the offline checks
establish. An eventual Q gain would not prove improved internal beliefs, reliable
solving or broad generalization: E−S mixes memory content, length and extra
generation, conditional on one trajectory. Credit Zining Liu's earlier
[GN instrumentation](https://github.com/Zarreboln/self-evolving-active-reasoning/blob/84e8d210ff868ead99879eb100346ec77e331749/stage0/gn_oracle.py)
and the pinned [AR-Bench](https://github.com/tmlr-group/AR-Bench/tree/9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1)
prompts and evaluator.
