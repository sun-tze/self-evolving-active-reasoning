# Reproducing the arithmetic and the mechanism

These are two different checks. Recomputing the published statistics uses the actual anonymous outcomes. Running the method with the public example environment uses a new dataset and model generations; it does not reproduce the private study's cases or promise the same findings.

An additional deterministic check characterizes the two selected schedules over
the entire public 5,040-hypothesis space:

```sh
python -B public_schedule_analysis.py --output schedule-check-local.json
```

It reads only the exported learned policies, checks small brute-force feedback
partitions, and computes each prefix's exact singleton fraction. It makes no
model calls. This is post-hoc mathematics of fixed lists, not an additional
optimizer or actor replication. Unknown continuation costs remain unknown.

## Recompute the published arithmetic

From this release directory, using standard-library Python:

```sh
python -B verify_results.py --cases anonymous-cases.json --results results.json
```

Add `--output verification-local.json` to save a new report; the output file must not already exist. The verifier recomputes the paired statistics, including all four cohort-level 98.75% primary intervals and the descriptive pooled 95% intervals. It reads only the two specified JSON files and makes no model or network calls. This checks released arithmetic and pairing, not private execution provenance. `PUBLIC-SCHEMA.json` specifies row ordering, seeds and resampling.

## Replicate the mechanism on new public data

The `code/` directory contains the unchanged pure selection gate and public monitor, a portable sequential search, exact active actor/curator prompt strings, and an example environment. From `code/`, after providing your own callable model adapter:

```python
import json
from pathlib import Path
from example_environment import make_environment
from portable_search import run_experiment

protocol = json.loads(Path("protocol.json").read_text(encoding="utf8"))
repetitions, cohorts, feedback = make_environment()
result = run_experiment(
    protocol, repetitions, cohorts,
    call=my_model_adapter, feedback=feedback,
)
```

`my_model_adapter(request)` must return a dictionary containing `text` and `finish_reason` (`stop` or `length`). The request contains `model`, `messages`, `temperature`, `top_p`, `max_tokens` and `seed`. The adapter must pin the declared model revision, honor these parameters, and enforce the context, token and runtime limits appropriate to its backend. No adapter, model backend, network client, installation or deployment controls are bundled. Calling `run_experiment` invokes the adapter; importing these modules or running their synthetic tests does not invoke a model.

The example draws 332 distinct secrets deterministically from the full 5,040-secret universe, including leading zeroes. It supplies two disjoint 18-training/48-selection blocks and two 100-case evaluation cohorts. This is a **new public replication sample**, not the restricted internal sample, and overlap with private secrets is neither checked nor excluded. Feedback stays in the environment; no labels or case IDs are included in model requests. Published numerical examples and public sample labels are intentional public data.

The portable flow preserves six proposal rounds, one shared empty-note baseline per block, cached incumbent vectors, mechanical skips without replacement proposals, first-singleton stopping, the original parser, and all five frozen evaluation arms. It executes sequentially. Identifier-dependent seeds, serving concurrency and the different data distribution prevent an exact private-run replay. Results returned to the caller contain notes and transcripts; they are not automatically an anonymous publication artifact.

## Local synthetic checks and source bindings

From `code/`:

```sh
python -B -m unittest -v test_portable_search
```

These five tests use synthetic callbacks and make no model or network calls. Existing receipts in `code/checks/` record four mechanism tests passing, followed by one newly added environment test passing. The original four were not rerun when the fifth was added; the receipts preserve their version-specific test-source hashes. These are implementation checks, not research results. Publication assembly reran no tests.

`code/MANIFEST.json` binds the released code, protocol and both receipts. The selection and monitor files are byte-identical to their experimental source. The portable controller is an explicitly documented extraction, not the private deployment or execution auditor.
