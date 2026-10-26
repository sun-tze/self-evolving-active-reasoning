# Offline diagnostic of the cancelled native UoT run

Job 26955564 ran native UoT (depth 1, Qwen2.5-7B-Instruct, 512-token output cap,
original 20-case order and seeds) and was cancelled after 3/20 complete cases,
483 call intents and 482 responses. It is **not** a UoT baseline result. These
files diagnose the recorded decisions only; no model, network or GPU was used.

## v1 stopped on a binding check; the cause

`diagnose_saved_scores.py` (v1) and its output `diagnostic-result-01.json` are kept
unchanged. v1 stopped with `Private-case binding differs` because it compared two
different files:

| hash | what it is |
|---|---|
| `27c45505…` | raw bytes of the package-root `private-cases.json`, bound at `run_start` |
| `b785af72…` | canonical (sorted, compact) JSON of the case table, recorded in `plan.json` |
| `44fd801c…` | the batch's re-serialized reference copy `allocation/controller/private-cases.json` |

v1 hashed the copy and compared it with the raw hash of the original. v2
(`diagnose_saved_scores_v2.py`) checks the chain the run recorded instead: run_start
raw hash = plan raw hash; plan canonical hash = canonical hash of the copy; plan
copy hash = copy bytes; admission/allocation-context bindings = run_start; planned
schedule = started episodes. The replay itself is unchanged. The original package
bytes, `uot_batch.py` and a hash-chained binding of `plan.json` are not in the
offline evidence, so the raw→canonical link is the run's own record. The content
check is the replay: every recorded prompt was re-derived from the copy's targets,
and all three saved episodes reproduced `official_result` exactly.

## What the recorded decisions show

`diagnostic-summary-02.json` and `proposal-parsing-counts-02.json` are anonymous
aggregates of the private v2 output (24 complete decisions, 72 candidates, 457
recorded simulation responses; one missing response is excluded, not read as empty).

- **Repeated questions.** The native scorer picked a guess the episode had already
  asked in 9/24 decisions. A repeated guess carries zero information, and the state
  stayed the same size. In 8 of the 9, all three parsed candidates were repeats.
- **Parsing.** The native parser keeps the first three 4-digit strings of the
  proposal response. In 14/23 proposals the first number was an already-asked guess
  restated by the model, and each of those 14 responses also contained later
  distinct-digit unasked numbers (269 in total) that were dropped.
- **Enumerative scoring.** 393/457 simulation responses stopped at the 512-token cap.
  The model listed at most 29 verified survivors while exact buckets reached 1440
  (median 1 vs. 23). Undercounts inflate the score of concentrated partitions:
  zero-information candidates received a median native score of 0.97·N, and
  largest-bucket share vs. native inflation has Spearman 0.95 when N > 50.
- **Ranking.** Native and exact-count rankings differ in 16/24 decisions, but 8 of
  the 11 changed picks are among exactly tied candidates (e.g. every first guess on
  the full 5040 set). The native pick was strictly worse on the same one-step
  objective in 3/24 decisions (gaps 0.46, 105.85 and 209.0 expected candidates).
  Shannon entropy is reported separately and is not substituted.

These are statements about recorded states and proposals. They do not estimate the
success rate or question count a repaired parser or exact scorer would achieve;
that needs new trajectories.

## Rerunning

The scripts need the private evidence archive, which is not published:

```text
python -I -B diagnose_saved_scores_v2.py --evidence EVIDENCE --reference AR_BENCH_SOURCES --output PRIVATE.json
python -I -B summarize_diagnostic.py --diagnostic PRIVATE.json --output summary.json
python -I -B probe_proposal_parsing.py --evidence EVIDENCE --output parsing.json
```

`AR_BENCH_SOURCES` must contain the hash-pinned `gn_evaluator.py`, `prompt.py` and
`utils_gn.py`. The full v2 output contains per-decision hashes and stays private.
