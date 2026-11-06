# Arm0 GN representation baseline

Context representation for Guess Number: `chrono`, lossless `ledger` (control), and `ledger_belief` (candidate). Read [BASELINE.md](BASELINE.md) before treating any number as a result. This is not a context-evolution method.

Python 3.10+. Standard library only.

```bash
python3 -m unittest discover -s arm0_kimi/tests -t .
python3 -m arm0_kimi.official_cases
python3 -m arm0_kimi.smoke --backend mock --cases official --arm all \
  --n 2 --seeds 0,1,2 --max-turns 25 \
  --out /tmp/arm0_repr_baseline_smoke
```

`--out` is required. `--cases` defaults to `synthetic` and stays there until `official` is explicit. Official mode checks the AR-Bench test-split digest and reuses `test.json[:n]` for every arm and seed. Repeating a command on the same `--out` resumes completed episodes without running them again or counting their calls twice.

The Qwen client is selected with `--backend vllm`. It defaults to `http://127.0.0.1:18000/v1`, `Qwen/Qwen2.5-7B-Instruct`, temperature 0.7, and 800 completion tokens, and it sends the decoding seed. It does not send `reasoning_effort` or `top_p`, and it does not accept a Moonshot URL. Nothing in this package opens that endpoint, submits a job, or logs in to a cluster.

`KimiActor` is still present. Constructing it requires `MOONSHOT_API_KEY` or `KIMI_API_KEY` and sends Kimi's own payload, including `reasoning_effort`. The offline checks patch the socket and do not call Moonshot.
