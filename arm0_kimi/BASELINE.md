# GN context representation baseline

This package is a **representation baseline**. It is not a completed context-evolution method, and the offline receipts below are not a scientific lift.

The three arms differ only in how public query–feedback text is written into the next prompt.

| Arm | Role | What the actor is shown |
| --- | --- | --- |
| `chrono` | Secondary presentation control | Compact chronological guess and A/B feedback. Raw actor prose is dropped. |
| `ledger` | Control | Lossless numbered ledger. Q is the complete actor response. A is the exact referee `aAbB`, or `NO_VALID_GUESS`. |
| `ledger_belief` | Candidate | The same ledger, plus a host-computed `gn_belief_v1` panel. The model does not compute that object and is not given a tool for it. |

Feasible-set updates run in host Python after each valid guess. There is no learned layout, no proposer, and no tip memory. A later evolution claim would require the candidate to beat `ledger` on final-guess exact match, with a consistency diagnostic moving in the same direction. Beating `chrono` alone is not success. Diagnostics are reported and are not themselves success. This wiring does not make that claim: `scientific_lift_claimed` is false even if a future run passes the gate.

## Cases, budget, and decoder

Official runs load AR-Bench `data/gn/test.json` at commit `9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1` and keep `test.json[:n]` fixed across arms and across decoding seeds `0`, `1`, and `2`. The checked digest is `e192e5b2d0e58a62e1b4027ba736368d5dc81b2a8f973080766d5288c9804394` (800 bytes, 100 secrets). The smoke prefix is `8362`, `1058` (`arbench-test-0000`, `arbench-test-0001`). The first 20 secrets are the locked list printed by `python3 -m arm0_kimi.official_cases`. `train.json` (digest `e3467c07c1ad85c25f64faaab29c7d66a7685cc90d677386551516afd32c662b`) is rejected and is not in this tree.

The packaged fixture is a byte copy of that test file so the checksum can be checked without a submodule checkout and without a network call. A checked-out `AR-Bench/data/gn/test.json` is preferred when it is present. `--cases` defaults to `synthetic`, which resamples with the seed and is plumbing only. `--cases official` does not fall back to that generator.

Each episode is 25 queries plus one separate final response (26 actor calls). A correct mid-episode guess does not stop the episode. Exact match is the independent final guess against the hidden secret. A singleton feasible set is not substituted for that guess. Feedback is the AR-Bench pair `(same position, same digit in the wrong position)` from pinned `compare_guess`, restricted to four distinct digits.

The Qwen client talks to an OpenAI-compatible local endpoint. Defaults are `http://127.0.0.1:18000/v1`, model `Qwen/Qwen2.5-7B-Instruct`, `temperature=0.7`, `max_tokens=800`, and the episode seed. The body does not contain `reasoning_effort` or `top_p`. The bearer token is the literal `EMPTY` unless `VLLM_API_KEY` is already set; the value is not written into receipts. Moonshot and Kimi URLs are refused. `KimiActor` is unchanged: it still requires `MOONSHOT_API_KEY` or `KIMI_API_KEY`, still sends `temperature=1.0` and `reasoning_effort`, and was not called.

The belief panel has no gold, answer, or secret field. Before any guess the hidden secret is not in the prompt. After guesses, a feasible set of size at most 20 may list unlabeled candidates. That list is the host hypothesis set, not a labeled answer.

Re-invoking the same command on the same `--out` reuses completed episode files. They are not executed again and their calls are counted once. An incomplete, corrupt, or internally inconsistent file is replaced. A completed file from a different budget, backend, or case source is left in place and the run stops.

## Offline receipts

No paid API, GPU job, SSH session, or tunnel was used. The actor in these receipts is the deterministic mock. It ignores the rendered prompt and plays a feasible-first policy, so the three arms match and the gate fails closed. That tie is a property of the mock, not evidence about Qwen.

Verifier, this tree, fixture path:

`sha256_ok` true, `n_secrets` 100, `wired_into_runner` true, smoke secrets `8362` and `1058`, `locked_n20_ok` true.

```bash
python3 -m unittest discover -s arm0_kimi/tests -t .
```

`Ran 29 tests in 12.699s` — `OK`.

```bash
python3 -m arm0_kimi.smoke --backend mock --cases official --arm all \
  --n 2 --seeds 0,1,2 --max-turns 25 \
  --out /tmp/arm0_repr_baseline_smoke_n2_timed
```

`wall_seconds` 17.606219789999955. Eighteen episodes (3 arms × 2 cases × 3 seeds), 468 calls (26 per episode). Each arm: final-guess EM 1.0 (6/6), `feedback_consistency` 1.0, `repeat_guess` 0.7799999999999999. `readout_passes` false. Fail-closed reason: `candidate_does_not_beat_ledger_control`. `scientific_lift_claimed` false.

Repeating that command on the directory from the same protocol (`/tmp/arm0_repr_baseline_smoke_n2`) resumed 18 episodes, executed 0, left 18 episode files, and kept `usage.calls` at 468. The resume invocation recorded `wall_seconds` 0.006144024000036552.

```bash
python3 -m arm0_kimi.run --backend mock --cases official --arm all \
  --n 20 --seeds 0,1,2 --max-turns 1 \
  --out /tmp/arm0_repr_baseline_matrix_n20_t1
```

This checks the locked prefix only. `--max-turns 1` is not the scientific budget. `wall_seconds` 5.102912141000047. 180 episodes, 360 calls, case ids `arbench-test-0000` through `arbench-test-0019`, the same secret list on every arm and seed. `readout_passes` false. `scientific_lift_claimed` false.

## What a cluster run still has to supply

These commands are not authorized by this note and were not run.

```bash
python3 -m arm0_kimi.smoke --backend vllm --cases official --arm all \
  --n 2 --seed 0 \
  --base-url http://127.0.0.1:18000/v1 \
  --out <directory-outside-the-repository>

python3 -m arm0_kimi.run --backend vllm --cases official --arm all \
  --n 20 --seeds 0,1,2 \
  --base-url http://127.0.0.1:18000/v1 \
  --out <fresh-directory-outside-the-repository>
```

On `vllm`, omitted `--temperature` and `--max-tokens` are sent as 0.7 and 800. The full matrix is 180 episodes and 4,680 calls. It needs a local OpenAI-compatible server at the URL above. This environment has no such server. Starting one requires a GPU allocation, an Engaging login, and a tunnel. None of those steps is implemented here, and no login procedure is included. `Kimi` remains available for a separate, explicitly keyed run; it is not a substitute for the Qwen path.
