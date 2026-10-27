# Bounded concurrent runtime for GN episodes

The cancelled UoT run made 482 model calls one at a time at about 9.6 s each:
`QwenBoundary.call` holds its lock across the transport, so nothing overlapped.
`concurrent_boundary.py` replaces that boundary for runs of independent cases.

## Design

- **Parallel episodes, unchanged native code.** `arbench_adapter.run_gn_episode`
  builds a fresh namespace and a per-episode seeded `random.Random` for every case,
  and keeps its call counter and events local. Cases therefore run in parallel
  threads while each case keeps its serial call order. `native_uot.py` makes the
  same `run_gn_episode` call as the pinned `uot_binding.run_native_uot_episode`;
  only the per-case `chat`, secret, seed and case id are supplied differently.
  The depth-1 batching prototype in `../reference/` would instead parallelize
  inside one turn, which needs a new request schema and adapter; it is not used.
- **Atomic admission, short lock.** Under one lock: halted check, per-case call
  index contiguity, context, call/input/output reservations, the request deadline
  (`now + 120 s < deadline`), a `run/case-NNN/call-NNNN` id and a durable intent
  record. Schema, role, cap and sampling checks and token counting run before it.
  The transport call runs outside the lock, at most `max_in_flight` at once, with
  one transport object per worker thread.
- **Fail closed.** The first admission, transport or integrity failure halts
  admission for the whole run. Calls already sent finish or time out and are
  recorded. Cases already finished keep their results; others end as
  `execution_failure`, `not_started` or `abandoned_running`; nothing is retried
  or imputed.
- **Cost accounting.** `actual_*` counts verified responses. A call whose transport
  raised, whose response failed the integrity check, or which is still open when
  the controller stops waiting is counted at its full reservation as uncertain:
  `conservative_output = actual_output + uncertain_reserved_output`.
- **Durable log.** `HashChainJournal` appends JSONL records that each carry the
  previous record's hash, with fsync per append. `EpisodeStore` writes each case
  result exclusively when it ends and binds its hash into the journal.
  `verify_journal` checks the chain, unique ids, per-case contiguity, no admission
  after a halt, and the open-call count.

## Tests

`python -I -B test_concurrent_boundary.py -v` (stdlib only; uses the `AR-Bench`
submodule at `9971322f` or `ARBENCH_ROOT`). 11 tests, no model or network:

- 80 fake calls of 30 ms: 2.67 s serially, 0.70 s with 4 in flight (exactly 4 observed).
- Native UoT through the pinned adapter, 6 synthetic secrets, 877 calls: serial and
  4-way runs give byte-identical per-case results and per-case request sequences.
- The boundary lock is free during every transport call.
- 8 cases racing for a 37-call cap admit exactly 37 calls.
- Deadline refusal before any transport call; transport timeout, integrity failure
  and a hung call each halt, drain and account as specified; journal tampering is
  detected; a native mid-run failure leaves no imputed result.

## What this does not show

The fake transport returns identical text whether calls overlap or not. A real
server batches concurrent requests, which can change generated text even with the
same per-request seed, so a concurrent run is a new run of the same protocol, not
a replay of the serial one. Real throughput, the transport's thread safety and the
tokenizer binding are unmeasured here. With zero fake latency the native code's own
CPU work under the GIL limits overlap; real ~10 s calls make that negligible.
The controller must close the journal right after `run_cases` returns.
