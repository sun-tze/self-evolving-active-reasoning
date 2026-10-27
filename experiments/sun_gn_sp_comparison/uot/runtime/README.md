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

## Throughput probe and feasibility gate (prepared, not run on a GPU)

`replay_throughput.py` replays the cancelled run's own request bodies through the
boundary and `loopback_transport.py` (stdlib HTTP, `127.0.0.1` only, no proxy,
redirect or retry). Phase "parallel" sends recorded calls 1–144 over 8 streams;
phase "serial" sends calls 145–162 on one stream (disjoint, so prefix caching
cannot carry over). Token counts are the recorded ones and the server's
`prompt_tokens` must equal them. It reports calls/s, latency, finish reasons and
how many generated texts equal the recorded text. A halted phase stops the probe
and yields no gate. `test_replay_throughput.py` (9 tests) covers the transport
against a local HTTP server (3.2–3.8x calls/s at 4 streams with 50 ms server
delay), error/redirect/timeout refusal and the gate arithmetic. An offline
`--fake-latency` dry run on the private recorded journal sent all 162 bodies
byte-identically, with 8 in flight and both journals verified.

The gate projects a 20-case native run at the measured 8-way rate: the larger of
total calls / rate and ceil(20/8) waves of the longest case at the measured median
latency, plus 140 s startup, for two call scenarios (recorded mean 139.3 calls per
case; 1.5x the recorded maximum, 229.5). It passes if the pessimistic projection
times 1.25 fits the allocation minus the 240 s request margin. Projections for
*assumed* speedups over the recorded 9.6 s serial call (not measurements):

| 8-way speedup | recorded mean | pessimistic | passes for 2418 s / 7200 s / 10800 s |
|---|---|---|---|
| 4x | 136 min | 223 min | no / no / no |
| 6x | 92 min | 149 min | no / no / no |
| 8x | 69 min | 112 min | no / no / yes |

So a complete native 20-case run does not fit the remaining conservative budget at
any plausible speedup; the probe is only worth running if a larger allocation for
the full run is acceptable.

## What this does not show

The fake transport returns identical text whether calls overlap or not. A real
server batches concurrent requests, which can change generated text even with the
same per-request seed, so a concurrent run is a new run of the same protocol, not
a replay of the serial one. Real throughput, the transport's thread safety and the
tokenizer binding are unmeasured here. With zero fake latency the native code's own
CPU work under the GIL limits overlap; real ~10 s calls make that negligible.
The controller must close the journal right after `run_cases` returns.
