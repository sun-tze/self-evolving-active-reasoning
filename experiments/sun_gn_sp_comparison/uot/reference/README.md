# Sources of the cancelled native UoT run and the depth-1 prototype

Unchanged copies, kept so the diagnostics and the new runtime can pin what ran.

| file | sha256 | role |
|---|---|---|
| `uot_controller_entry.py` | `c6414b9b…` | controller of job 26955564 (scope, admission, completion gate) |
| `qwen_binding.py` | `22615e96…` | `QwenBoundary`: holds its lock across the transport call, so calls serialize |
| `uot_binding.py` | `f92b9c66…` | UoT role classification and the single-episode native invocation |
| `depth1_batch.py` | `e1ea6f02…` | prototype: pre-draws one UoT turn's simulation requests and reduces responses in serial order |
| `independent_test_depth1.py` | `726f6766…` | 9 acceptance tests of the prototype; `independent-test-result-01.json` is their recorded pass |

`uot_batch.py`, the tokenizer and the loopback transport of that run are not
included. The prototype test expects the pinned AR-Bench `gn_evaluator.py`,
`prompt.py` and `utils_gn.py` in a sibling `reference/` directory; they are not
redistributed here (use the `AR-Bench` submodule at `9971322f`).

The prototype shows only that planning and reduction match the original given
identical responses. It is not wired into any adapter, and its request schema
is rejected by `uot_binding` by design. `../runtime/` takes a different route
(parallel episodes, native code unchanged); see its README.
