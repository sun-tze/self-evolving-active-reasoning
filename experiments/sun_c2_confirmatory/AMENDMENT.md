# CPU preflight amendment

Date: 2026-10-15. Status: reviewed and CPU-validated; renewed GPU authorization required.

The attempt at `72e97fd` stopped when `vllm serve --help=all` exceeded 120 seconds on the login node. No GPU job or model inference occurred. That timeout does not establish an inference failure; the existing pinned architecture had completed the 20-call pilot.

The controller now runs an isolated, standard-library verifier with a 30-second subprocess timeout. It checks installed vLLM 0.31.0 metadata, the executable and interpreter, the console entrypoint, and the direct `--attention-backend` registration in `EngineArgs.add_cli_args`. Source reads are size-limited and hashed. The verifier does not import vLLM or torch, initialize CUDA, or start a server. This is a static compatibility check; actual startup, service identity, tokenization, GPU and inference checks remain mandatory.

## Integrity

The original `freeze.json` is preserved byte for byte. `freeze_v2.json` explicitly binds that parent and the amended executable files. Only `study.py` and `submit_once.py` changed within the original frozen inventory; all 19 protected files retain their original hashes. The new helper is included in V2. The verifier rejects changes to protected design metadata or files and rejects authorization bound to V1.

| Artifact | SHA256 |
|---|---|
| Original freeze | `732f6a7f71df04757016b265a71ea3549286305baffc4f7325e4af5e000dca9b` |
| Active V2 freeze | `ac354acd199bee67d952c311fd01474148e74fd5285fe9d39b3b7ee0939d6fcb` |
| Static verifier | `3067895896ca8445586575d9743888779bdd8d00db81eea771cdad24d37a8b22` |

The 65 cases, 260 original payloads, model revision and sampling settings, rubric, reference anchors, reviewer assignments, analysis and resource limits are unchanged. No remaining-case outputs were generated or inspected.

## Validation

- 23 focused CPU tests passed: 10 preflight/amendment regressions and 13 confirmatory regressions. Coverage includes invalid installation metadata, missing or oversized sources, inactive option mentions, timeout/failure before any scheduler call, protected-file changes and obsolete authorization.
- The exact verifier passed on MIT Engaging in 0.044 seconds, locating the registered option at installed source line 1063. Operational evidence remains in private storage.
- V2 verification passed against the original bundle: 65 cases and 260 unchanged requests.
- Independent read-only review found no blocking issues. It verified the preserved original hashes, V2 lineage, authorization rejection and fail-closed controller path. Reviewer annotations remain private.

Reproduce the focused checks without GPU allocation:

```sh
python -m unittest experiments.sun_c2_confirmatory.test_cli_preflight experiments.sun_c2_confirmatory.test_confirmatory -q
python -m experiments.sun_c2_confirmatory.study --bundle "$BUNDLE"
```

Renewed authorization must bind the published commit and V2 hash. The envelope remains one L40S job, 60 minutes, 8 CPUs, 32 GiB, 260 calls, 194,338 input tokens and 104,000 output tokens. Fresh runtime/cache/tokenizer validation and Slurm `--test-only` must pass before submission. No paid APIs, automatic retries, requeues or additional jobs are permitted. No GPU job was submitted for this amendment.
