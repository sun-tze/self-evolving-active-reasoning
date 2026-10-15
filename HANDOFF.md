# Completed C2 confirmatory evaluation

The 65-case confirmatory study is complete: 260 successful Qwen completions and 520 validated blind semantic ratings. The primary positive writer-effect hypothesis was not confirmed. Read the [research report](experiments/sun_c2_confirmatory/REPORT.md) and [aggregate results](experiments/sun_c2_confirmatory/results/summary.json).

## Frozen execution

- Branch: `sun/c2-confirmatory-65`.
- Approved execution commit: `67883c8bd7f9f7aacc270c0bcf91375e292bd10e`.
- V2 freeze SHA256: `ac354acd199bee67d952c311fd01474148e74fd5285fe9d39b3b7ee0939d6fcb`.
- Original manifest SHA256: `1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548`.
- The single authorized job completed within every approved resource limit. No further GPU authorization exists.

The report commit adds research deliverables; it does not replace the approved executable checkpoint. Original requests, rubric, anchors, assignments and pilot scores are unchanged. The [pilot audit](experiments/sun_c2_semantic/REPORT.md) remains separate.

## Private records and offline reproduction

Both archives are retained in the private MIT pool at `~/orcd/pool/ar_harness/c2/` and in the original local workspace. Access requires the owner's MIT authentication; GitHub alone intentionally does not expose model outputs, logs or reviewer annotations.

| Archive | SHA256 |
|---|---|
| `confirmatory-25971516-inference-checkpoint.tar.gz` | `af1a65f840c06a5270ffa76f34f8e141602caea8cee81ef878efabdfa9e11d6b` |
| `confirmatory-25971516-semantic-checkpoint.tar.gz` | `cdd42bdef0aacc1778dd9e7e6e82975bdc6b7c8aa6fe437f3efc535c72b6e499` |

The inference archive contains the raw journal, F1 analysis, frozen packets, preflight records and submission receipt. The semantic archive contains both complete review streams, packet copies, results, validation hashes and an integrity inventory under `semantic_25971516/`. Preserve both archives unchanged.

Use the approved execution commit and the original bundle at `~/orcd/pool/ar_harness/c2/pilot-v1`. Set `RUN` to the preserved `runs/confirmatory_25971516` directory and `SEMANTIC` to the extracted `semantic_25971516` directory. Set `OUT` to a new filename. Reproduce the analysis without new inference or ratings:

```sh
python -m experiments.sun_c2_confirmatory.semantic analyze \
  --bundle "$BUNDLE" \
  --freeze-sha256 ac354acd199bee67d952c311fd01474148e74fd5285fe9d39b3b7ee0939d6fcb \
  --analysis "$RUN/analysis.json" --run "$RUN/inference" \
  --packets "$RUN/review_packets" --reviews "$SEMANTIC/reviews" --out "$OUT"
```

Compare the reproduced JSON with `semantic_25971516/semantic_result.json`. Do not regenerate completed answers or ratings, overwrite historical results, or repeat passed tests without a new reason.

No required execution or scoring work remains. Human or cross-family reviewer validation has not been performed and remains a limitation. Any additional experiment or GPU job requires separate authorization. Publish only reviewed English research deliverables; keep operational records and reviewer annotations private. Do not modify upstream, merge PRs or force-push history.
