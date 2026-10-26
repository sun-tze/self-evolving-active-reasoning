# Search and evaluation costs

The learned policies used fewer queries during evaluation, but obtaining them also required training, proposal, and selection work. The following counts cover this study only; earlier experiments are excluded.

| Phase | Actor queries | Curator requests | Input tokens | Output tokens |
|---|---:|---:|---:|---:|
| Learning and selection | 11,988 | 24 | 10,087,850 | 96,895 |
| Five-arm evaluation | 9,322 | 0 | 6,233,185 | 74,586 |
| Total | 21,310 | 24 | 16,321,035 | 171,481 |

Learning comprised 36 trace-training episodes (311 queries), 96 shared baseline-selection episodes (1,044 queries), 912 challenger-selection episodes (10,633 queries), and 24 curator proposals. Evaluation comprised 200 cases for each of five arms: 1,000 episodes. Shared baselines are counted once in the totals.

One actor query is one model request for a game guess. Curator requests generate policy notes and have different input lengths and output caps. Consequently, query counts alone do not measure total compute. Token counts are recorded provider usage reconciled against the archived call ledger; they were not independently re-tokenized. Concurrency prevents allocating GPU wall time in proportion to these counts.

## Conditional amortization

The following arithmetic asks how many future cases would offset the actor queries used by **one E search**, assuming its observed evaluation saving remains unchanged. Each search is charged its full 48-case baseline, its own training and candidate selection, and six curator requests. This differs from the equal half-baseline allocation also available in the cost JSON.

| Policy | Upfront actor queries | Separate curator requests | Observed queries saved per evaluation case vs B0 | Actor-query crossover |
|---|---:|---:|---:|---:|
| E0 | 2,358 | 6 | 3.215 | 734 future cases |
| E1 | 1,833 | 6 | 3.420 | 536 future cases |

The last column is `ceil(upfront actor queries / observed saving per case)`. It excludes curator requests from the actor-query numerator. Including both curator and actor input/output tokens, and assigning every token equal weight, gives illustrative token-count crossovers of 472 cases for E0 and 332 for E1. Input and output tokens do not have equal hardware or pricing costs in general.

These per-search scenarios exclude the R comparator searches, five-arm evaluation, and previous studies. They are not amortization estimates for the entire research program. Future cases may have different costs or benefits; none of these calculations demonstrates a monetary, latency, or compute break-even point.

## Supporting files

- [descriptive-costs.json](descriptive-costs.json) contains phase totals, per-search allocations, all formulas' inputs, and source hashes.
- [learning-lineage.json](learning-lineage.json) retains all 24 proposal dispositions and policy-hash lineage. E0 changed on proposal 2; E1 on proposals 1 and 2; R0 and R1 retained their initial policies.
- [verification-result.json](verification-result.json) records the independent public arithmetic check. It reproduced all 200 paired cases' summaries, 44 interval endpoints, and predeclared flags exactly. This checks public arithmetic, not private execution provenance or the cost ledger.

The standard-library verifier reads only `anonymous-cases.json` and `results.json`; supplemental files do not affect it. From this directory, run `python -B verify_results.py --cases anonymous-cases.json --results results.json`. An optional `--output` path must name a new file.
