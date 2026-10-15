# C2 real-data fixed-dialogue writer pilot

This additive component adapts the public Experiment 10 release to the C2 design.
It leaves `experiments/sun_writer_probe/` and its synthetic-only boundary unchanged.
It prepares **70 unseen cases Ã— four cells = 280 requests**, then selects five
cases without consulting outcomes for a **20-call pilot**. No new model results
are included in this branch. Raw data, requests, targets, and runtime products
stay in the ignored `.c2-local/` directory or an explicitly chosen cluster pool path.

## Design and interpretation

| Cell | Fixed recorded dialogue | Standardized writer |
|---|---|---|
| A | Bare | Plain |
| B | Bare | Evidence constrained |
| C | First-selected `cand_sp_strict1_b10g3r2_b0r1g2` | Plain |
| D | First-selected | Evidence constrained |

The two instructions are unchanged from the synthetic C2 protocol. The shared
system prompt now describes a situation puzzle and treats recorded dialogue as
evidence. This is a versioned real protocol, not a reproduction of the evolved
multi-draft finalizer. The original questioner, referee, proposer, and judge are
not rerun. New requests use Qwen2.5-7B-Instruct, temperature 0, top-p 1, 400 output
tokens, and one stable seed per case shared across the four cells. The original
questioner used temperature 0.7; the new deterministic writer is a deliberate probe.

| Quantity | Contrast | Interpretation |
|---|---|---|
| Writer effect, bare history | Bâˆ’A | Instruction effect with identical bare observations |
| Writer effect, first history | Dâˆ’C | Instruction effect with identical first observations |
| Dialogue effect, plain writer | Câˆ’A | Utility of these two acquired histories for the plain writer |
| Dialogue effect, evidence writer | Dâˆ’B | Utility for the evidence writer |
| Marginal writer effect | ((Bâˆ’A)+(Dâˆ’C))/2 | Average over the two fixed histories |
| Marginal dialogue effect | ((Câˆ’A)+(Dâˆ’B))/2 | Average over the two fixed instructions |
| Interaction | (Dâˆ’C)âˆ’(Bâˆ’A) | Change in writer effect across the histories |

These are paired descriptive comparisons. Dialogue effects are **not equal-compute
acquisition effects**. Character F1 measures lexical multiset overlap, not semantic
correctness or factual faithfulness. It cannot by itself establish a superior
reasoning system. The report uses 10,000 paired case-bootstrap resamples, seed
62026, and descriptive 95% percentile intervals. There are no significance claims,
automatic winner selection, or multiplicity-adjusted tests. Missing/failed/ambiguous
responses keep null scores; any missing component suppresses its contrast mean
and interval. Truncated completions retain their score and an explicit flag.

## Provenance and data boundary

`source_lock.json` pins the public raw archive, five exact archive members,
official dataset, official scorer, and AR-Bench revision. The stale third-party
URL in the parent project's documentation was resolved to `tmlr-group/AR-Bench`.
No upstream repository is modified.

The historical record's `index` is a **zero-based array position**. The dataset's
own `index` field has only 99 unique values in 100 rows and must never be used as
the join key. For every source record, preparation verifies the matching surface
at the historical prompt boundary, source harness/task/status, 25 exact question/
reply pairs, and a replay of the logged character F1 against that dataset position.
The 30 excluded positions must agree across the archived evaluation config,
archived evolution stream, and `numpy.random.default_rng(0).permutation(100)[:30]`.
Both published unseen means are also checked. The archive and all input files
must match their fixed hashes before any requests are produced.

Model-facing fields are allowlisted to `surface` and the original `question`/
`reply` strings, plus the shared system and writer instructions. There is no
copying of `gold`, `bottom`, key questions, story trees, `trace`, historical final
answers, scores, source labels, paths, or provenance into model messages.
Nonstandard referee replies (including `Invisible`, `After`, and a rejected-question
message) are preserved verbatim. Full hidden-story matches in observations fail
preparation. This structural boundary does not claim that observed evidence
contains no answer-relevant facts; retaining those facts is the purpose of the probe.

The bundle contains:

- `requests.jsonl`: all 280 requests, with full payload and request hashes.
- `manifest.json`: source/config/code/protocol binding, per-record and per-line
  hashes, observation hashes, exact unseen/pilot positions, and file hashes.
- `private/targets.jsonl`: offline-only scoring targets and identities.
- `private/historical_costs.json`: offline-only historical cost records.
- `audit.json`, `second_path_audit.json`, `budget.json`, `pilot_plan.json`, `READY.json`.

The runner opens only manifest, requests, and token-budget files. It sends exactly
`request["payload"]`; the HTTP stub tests compare the actual wire body with that
object. The analyzer opens targets later. Hashes establish supplied-content
consistency, not cryptographically signed historical authenticity. A supplied
manifest SHA-256 is the external integrity anchor; preserve it separately.

## Reproduce on CPU

Use Python 3.11+ and the existing environment if it already has NumPy, zstandard,
Transformers, and Jinja2. `requirements-cpu.txt` records the tested CPU versions;
install it only in a separate CPU virtual environment if needed. Do not replace
the cluster's existing vLLM environment with these dependencies.

From the repository root, the single staging command downloads only public data
and tokenizer files, verifies their hashes, generates the requests, calculates
exact chat-template token counts, and performs a second verification path:

```bash
python experiments/sun_c2_real/stage.py --inputs .c2-local --bundle .c2-local/pilot-bundle
```

Choose a fresh bundle directory; the command refuses to overwrite a bundle.
The command prints its manifest SHA-256 and saves it in `READY.json`. Tokenization
does not load model weights. Local pinned tokenizer files may be supplied with
`--tokenizer /path/to/tokenizer`. Input downloads can be reused on later runs.

Run both CPU suites, retaining an explicit platform-skip record:

```bash
python experiments/sun_c2_real/checks.py --out .c2-local/tests.json
```

On this Windows host the unchanged legacy symlink test requires an unavailable
OS privilege; the wrapper skips only that specific WinError 1314 capability.
On Linux all 54 legacy tests should execute. A skip is not counted as a pass.

Existing cluster extracts can also be used directly with `prepare.py --raw-root
~/orcd/pool/ar_harness --arbench-root ~/code/ar_harness/AR-Bench --out BUNDLE`,
provided all pinned files, including both archived configs and historical scores,
are present and byte-identical. `stage.py` is the more portable single entrypoint.

## Budget and execution gates

The outcome-independent pilot positions are **7, 29, 40, 58, 70**. The selection
rule sorts all unseen positions by SHA-256 of `c2-pilot-v1:<position>` and takes five.
It is fixed before any new output. If pilot results inform changes, those five
become development cases and the remaining 65 must be reserved for confirmation.
Requests for all 70 are prepared as requested; running `--scope all` later would
include the five pilot cases and must not be labeled a fresh 70-case confirmation.

| Scope | Cases | Model calls | Exact input tokens | Maximum output tokens |
|---|---:|---:|---:|---:|
| Pilot | 5 | 20 | 15,210 | 8,000 |
| Full prepared set | 70 | 280 | 209,548 | 112,000 |

Maximum input length is 1,097 tokens for the pilot and 1,216 for all requests,
well below the existing 16,384-token context including 400 generated tokens.
These counts use the fixed Qwen tokenizer revision in `model_lock.json`. The
runner also verifies each planned input with the **served vLLM tokenizer before
generating anything**, stopping if any token count differs.

The provided Slurm pilot requests one L40S, 8 CPUs, 32 GiB, and a hard 30-minute
allocation: at most **0.5 L40S GPU-hours**, including model startup. This is an
allocation ceiling, not a measured runtime or dollar estimate. Readiness polling
can take about seven minutes including request timeouts, followed by a 20-minute
runner deadline. Each HTTP request has an
absolute deadline enforced by closing socket I/O; Slurm supplies the outer
allocation limit. A unique job-specific served-model alias is checked before
generation, preventing a port collision from selecting an old service. The server is killed
on script exit. Cancel the allocated job with `scancel JOB_ID` if needed.

Historical questioner costs include the original finalizers and are reported
separately per source, once per case. They are not acquisition-only costs and are
not added to new writer cost. Historical referee token usage is unavailable and
stays null. New input/output usage, incomplete coverage, attempted/returned calls,
and returned-call durations are reported separately. Obtain institutional rates
before translating allocation time to money. No paid API is used.

The default runner command is a **zero-network plan**. Inference requires
`--execute`, a nonempty human `--authorization-id`, and explicit output directory.
Only an explicit loopback HTTP endpoint is accepted; use an SSH tunnel for an
existing cluster server. Redirects and environment proxies are disabled. There
are zero automatic retries. A started request without a durable result is
ambiguous and cannot be automatically resumed. Failed runs also require review
before any new run; never delete or edit their journals to force a retry.

## Deploy after review and human pilot authorization

Use a separate clone of this fork's `sun/c2-real-data-adapter` branch on Engaging,
for example `~/code/ar_harness_c2_real`; preserve `~/code/ar_harness` and teammate
branches. Stage its bundle into a dedicated pool directory with `stage.py`.
The runtime and model cache are reused from `~/code/ar_harness/conda_env` and
`~/orcd/pool/ar_harness/hf_cache`. The new Slurm file copies the existing
`engaging/serve_7b.sbatch` model, dtype, context, batch, memory and CUDA settings,
adding a pinned revision, loopback binding, shorter limit, and bounded client.
It runs offline with respect to Hugging Face: missing model cache fails rather
than downloading weights inside a GPU allocation.

After staging, use the printed manifest SHA-256. This command validates scheduling
only and does not submit a GPU job:

```bash
sbatch --test-only engaging/c2_pilot.sbatch "$PWD" /absolute/path/to/BUNDLE MANIFEST_SHA256 AUTHORIZATION_ID REVIEW_REFERENCE
```

Only after the human has approved the concrete pilot and the independent review
has been addressed, use the same command without `--test-only`. `AUTHORIZATION_ID`
and `REVIEW_REFERENCE` are evidence references, not magic approval tokens. Do not
invent them. The script enforces a clean code checkout and records commit, packages,
GPU/driver, review reference, plan, raw completions, journal, and analysis under
`~/orcd/pool/ar_harness/c2/runs/pilot_JOB_ID/`.

For an already running, approved Qwen endpoint, run `runner.py` directly with
`--bundle`, `--manifest-sha256`, `--budget`, and `--base-url` to inspect the plan;
add `--execute --authorization-id ... --out ...` only after authorization. To score
an existing run, use `analyze.py --bundle BUNDLE --manifest-sha256 SHA --run RUN
--out REPORT.json`. Inference does not require the private target files; analysis does.

The authenticated OnDemand shell is accessible. Initial inspection found no
`conda_env` or Qwen weights in the expected cache for this account. CPU preparation
and Slurm scheduling checks are tracked in `DELIVERY.md`; an actual GPU-node
vLLM startup remains untested. No GPU jobs or real-model calls have been launched.

The vLLM interface was checked against the [official CLI reference](https://docs.vllm.ai/en/latest/cli/serve/)
and [tokenization request/response source](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/tokenize/protocol.py).
Actual installed cluster-version compatibility remains an explicit deployment check.
