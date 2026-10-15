# Independent code review — 2026-10-15

The user explicitly authorized a separate read-only review agent. The reviewer
inspected the adapter, source/config bindings, runner, scoring/statistics, costs,
Slurm script, tests, original C2 protocol, and historical Experiment 10 sources.
The implementing agent applied fixes; the reviewer independently rechecked them.

**Final conclusion: approved at the code level for a bounded pilot. No unresolved
blocking findings. Cluster compatibility checks and explicit human GPU
authorization are still required.** This is not a claim that a real pilot ran.

| Finding | Resolution and independent evidence |
|---|---|
| P2: socket inactivity timeout did not bound total HTTP time | Replaced urllib I/O with a local HTTP connection and absolute timer that shuts down socket I/O. Body-dribble regression passes; independent slow-header probe aborted after 188 ms for a 150 ms deadline. |
| P2: port collision could accept an old service's health response | Select a free port, check launched child before/after health, add an unpredictable per-job served-model alias, and require the marker in `/models` before tokenization or generation. Old-service rejection regression passes. |

Reviewer verification:

- 35/35 new CPU tests pass, including local HTTP integration tests.
- All 12 legacy synthetic files are byte-identical to `e16ccf7`.
- Independently loaded and hashed raw inputs without using the adapter's
  validation helper: both sources contain 100 unique positional IDs; the exact
  30 evolution-seen positions agree across archived configs and NumPy RNG.
- All 280 projections and four-cell pairings match the raw records.
- No complete hidden story from any of the 100 targets appears in outgoing
  payloads. All 100 surfaces and all 100 targets are unique as exact strings.
- The dataset has only 99 unique `index` values, validating the positional join.
- Scorer, contrasts, missingness behavior, and separation of historical and new
  costs are consistent with the protocol and its stated limitations.

Reviewed runtime file hashes:

| File | SHA-256 |
|---|---|
| `runner.py` | `38cc557744d83c4adba3112ec17a9335f92303e0c82c5cd4f53f763c20c14519` |
| `../../engaging/c2_pilot.sbatch` | `b2e71c64327a6d7c6fdd12d933b7d23eca994775473fd065e18ba719c1b1d97d` |

The reviewer did not authenticate to Engaging, run Slurm validation, load model
weights, or perform real inference. The same applies to the implementing agent
at this review point. Hash consistency is not signed historical authenticity.
Structural field isolation does not establish semantic decontamination of the
recorded dialogue. The review also corrected readiness documentation to account
for the approximately seven-minute worst-case polling period.
