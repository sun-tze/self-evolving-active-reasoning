# C2 confirmatory evaluation: remaining 65 cases

Status: design and implementation prepared; GPU execution requires explicit authorization. No outcomes from these 65 cases have been generated or inspected during preparation.

Current executable checkpoint: [CPU preflight amendment](AMENDMENT.md). The active integrity binding is `freeze_v2.json`; the original `freeze.json` remains unchanged. Use the V2 hash for all current verification, planning and authorization commands. Earlier approval is not valid for this amended code.

## Design

The population is the original 70-case unseen set minus pilot positions 7, 29, 40, 58 and 70. Each of the remaining 65 cases retains its four original requests: A = bare dialogue/plain writer, B = bare dialogue/evidence writer, C = first-selected dialogue/plain writer, D = first-selected dialogue/evidence writer. Neither histories nor generation payloads are rebuilt.

The primary estimand is the paired writer marginal effect, `((B-A)+(D-C))/2`, on Semantic Correctness normalized to [0,1]. Each answer receives two independent blind ratings; their arithmetic mean is used. Confirmation requires all 260 answers and 520 ratings, with the lower endpoint of the paired-case 95% bootstrap interval above zero. The bootstrap uses 10,000 draws and seed 62026. Evidence Faithfulness, Character F1, dialogue effects, interaction and ranking reversals are secondary descriptive outcomes. No composite score or pilot pooling is used.

`protocol.json` implements the earlier confirmatory proposal without changing the semantic scoring rubric. `anchors.json` contains 65 reference-derived anchors curated and independently reviewed before generation. Reference-quality flags are retained for positions 6, 21, 25, 42, 46, 56, 64 and 73; all remain in the primary population. Canonical fictional mechanisms are not claims about real-world science.

`assignments.json` fixes two independent review streams, each with 13 batches of five cases. Each batch receives a fresh reviewer context. Reviewers receive only `rubric_v1.json`, `review_protocol.json` and their assigned packet. Condition labels, original positions, source identities, writer instructions, F1 and other ratings are withheld. Blinding is procedural, not filesystem isolation. Same-family model reviewers can share systematic biases. Review annotations remain private; numerical ratings, disagreements and relevant evidence excerpts can be published.

## Resource envelope

| Resource | Limit |
|---|---:|
| Completion requests | 260 |
| Input tokens | 194,338 |
| Output tokens | 104,000 maximum, 400 per answer |
| Largest input | 1,216 tokens |
| GPU | One L40S |
| Slurm jobs | One; no retry or requeue |
| Allocation | 60 minutes; at most 1 GPU-hour |
| CPUs / host memory | 8 / 32 GiB |
| Writer deadline / request timeout | 3,000 seconds / 120 seconds |
| Paid external API calls | Zero |

Input tokens by condition: A 55,049; B 56,219; C 40,950; D 42,120. Counts were reproduced offline using the pinned tokenizer and equal the original budget. These are logical prompt tokens; prefix caching does not change the reporting denominator.

The model remains `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`, temperature 0, top-p 1, and original per-request seeds. The launcher preserves the successful pilot's vLLM configuration, including `--attention-backend FLASH_ATTN`. Account: `mit_general`; partition: `mit_normal_gpu`.

The successful pilot generated 1,817 output tokens in 37.306 seconds of request time. At that observed rate, the full 104,000-token ceiling alone would take approximately 35.6 minutes. The 60-minute allocation allows startup and processing overhead; it is a hard cap, not a runtime guarantee. Institutional monetary cost is unknown. Reviewer work consumes the available Work allowance and is separate from Qwen/GPU usage.

## Integrity and execution

The original manifest SHA256 is `1bfb5608ba6740f4d6024c00cffb25cc7cca347e8826b452be46a14207dea548`. The immutable `freeze.json` records the original design. `freeze_v2.json` preserves that lineage and binds the rubric, anchors, reference identities, assignments, exact request selection, token budget and amended executable files. The published commit plus the byte hash of `freeze_v2.json` identifies the current checkpoint. Any further amendment requires a new version and invalidates an earlier approval.

The submission controller checks the pinned code and freeze, requires a clean checkout, and validates the exact runtime and cache used by the launcher. It runs CPU runtime/cache/tokenizer checks, verifies the vLLM CLI, and performs Slurm `--test-only` before submission. A canonical approval ledger permits one submission attempt; the scheduler receipt binds launch and inference to one job ID. Unknown submission outcomes must be inspected, never retried. True startup/tokenizer/inference errors stop execution; warnings alone do not. Length-capped answers remain in the analysis.

Fresh cluster preflight and Slurm validation were not available at preparation time because the remote browser connection failed. They are enforced prerequisites to submission, not assumed successes.

Operational approval records, model outputs, journals, environment inventories, review packets and original annotations belong in private storage outside the checkout. The launcher preserves these under the existing pool's C2 run directory. The semantic pipeline replays the raw journal and requires exact agreement with lexical analysis before constructing packets or aggregating reviews.

## Reproduction

Use the published `sun/c2-confirmatory-65` commit with the original bundle. The bundle is retained in the MIT pool at `~/orcd/pool/ar_harness/c2/pilot-v1`. Local verification and planning do not allocate a GPU:

```sh
python -m experiments.sun_c2_confirmatory.study --bundle "$BUNDLE"
python -m experiments.sun_c2_confirmatory.run --bundle "$BUNDLE" --freeze-sha256 "$FREEZE_SHA"
```

Only after explicit approval, create a private authorization JSON containing `approved: true`, the exact freeze hash, published 40-character code commit, a unique authorization ID, and the unchanged `gpu_limits` object from `protocol.json`. Submit once through the controller:

```sh
python -m experiments.sun_c2_confirmatory.submit_once \
  --bundle "$BUNDLE" --freeze-sha256 "$FREEZE_SHA" \
  --authorization-file "$APPROVAL" --private-root "$PRIVATE_SUBMISSION" \
  --runtime-python "$HOME/code/ar_harness/conda_env/bin/python" \
  --cache "$HOME/orcd/pool/ar_harness/hf_cache"
```

The launcher produces `analysis.json` and an `inference/` journal directory. Preserve both. Construct blind packets after generation, then collect the two independent annotation streams under their frozen batch names:

```sh
python -m experiments.sun_c2_confirmatory.semantic packets \
  --bundle "$BUNDLE" --freeze-sha256 "$FREEZE_SHA" \
  --analysis "$RUN/analysis.json" --run "$RUN/inference" --out "$PACKETS"
python -m experiments.sun_c2_confirmatory.semantic analyze \
  --bundle "$BUNDLE" --freeze-sha256 "$FREEZE_SHA" \
  --analysis "$RUN/analysis.json" --run "$RUN/inference" \
  --packets "$PACKETS" --reviews "$REVIEWS" --out "$SEMANTIC_RESULT"
```

No command resumes generation or replaces missing answers. Incomplete generation or review withholds the primary conclusion. Report all paired contrasts, denominators, intervals, truncations and failures. Preserve original pilot results unchanged.

## Interpretation limits

The rubric was developed after observing the exploratory pilot, then frozen before these remaining outcomes. Reference anchors are outcome-blind but depend on imperfect canonical solutions. The finite set is not a random sample from all puzzles, and its bootstrap intervals do not establish broad benchmark generalization. A single greedy generation does not estimate generation variance. Dialogue contrasts measure the utility of these fixed histories with unequal acquisition costs; writer contrasts concern two standardized instructions. No guaranteed statistical power is claimed.
