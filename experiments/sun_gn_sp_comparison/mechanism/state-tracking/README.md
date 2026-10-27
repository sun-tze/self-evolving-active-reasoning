# GN: is the bottleneck acquiring information or using it?

Under the native GN protocol (25 questions, then one final guess), **information is not scarce**. Any sequence of non-repeating guesses gathers enough public feedback to pin down the secret. Uniformly random unasked guesses do this for all 5,040 secrets, but they hit the secret in only 0.48% of games. Choosing a guess that is consistent with the feedback so far on just 5% of turns raises the hit rate to 65%; at 20% it reaches 99%.

This agrees with the team's earlier GN diagnosis in [briefing.md](../../../../briefing.md): 88.9–95.7% of Qwen2.5-7B's guesses were already ruled out by its own history, and an externally computed belief state lifted the solve rate from 2% to 99%. Its consistent-guess and oracle means, 5.47 and 5.33 guesses in [experiments.md](../../../../experiments.md), match the reference below. New here are the identification result for non-repeating questions and the mixture curve.

The seven Qwen2.5-7B arms in [gn-complete](../../gn-complete/README.md) hit the secret in 0–2 of 20 cases each. Their failure therefore cannot come from the question budget being too small. It must come from asking repeated or uninformative questions, from not using the feedback they received, or both. The extraction below separates these two on the saved traces.

![Model-free reference](reference-policies.svg)

## Model-free reference (exhaustive, deterministic)

Every policy plays all 5,040 four-distinct-digit secrets. It stops when the secret is hit, or after 26 guesses (25 questions plus the final guess). A policy draws a guess from the feedback-consistent secrets with probability p; otherwise it draws a uniformly random unasked guess. No policy repeats a guess.

| Policy | Consistent guesses (measured) | Identified after 25 questions | Hit within 26 guesses | Mean / max guesses to hit |
|---|---:|---:|---:|---:|
| Random unasked guess (p = 0) | 0.9% | 100% | 0.48% (24) | 15.6 / 25 |
| p = 0.05 | 6.3% | 100% | 65.2% | 14.6 / 26 |
| p = 0.1 | 11.6% | 100% | 88.5% | 13.0 / 26 |
| p = 0.2 | 22.0% | 100% | 98.8% | 10.2 / 26 |
| p = 0.3 | 32.1% | 100% | 99.96% | 8.4 / 26 |
| p = 0.5 | 52.2% | 100% | 100% | 6.9 / 19 |
| Always a random consistent guess (p = 1) | 100% | 100% | 100% | 5.46 / 8 |
| First consistent guess, lexicographic | 100% | 100% | 100% | 5.56 / 9 |
| Consistent guess with the smallest expected remaining set | 100% | 100% | 100% | 5.32 / 8 |

Full distributions are in [reference-policies.json](reference-policies.json). "Identified" means the public feedback leaves exactly one possible secret. It measures the information acquired, not anything the actor believes. The expected-remaining-set rule is the exact partition score from the [UoT diagnostic](../../uot/diagnostics/README.md), applied to consistent guesses only.

This is a mathematical reference, not a model of LLM behavior. LLM errors are not uniformly random, and the native actor is never shown the consistent set. The figure's gray band is the observed per-arm hit range; the x position of the LLM arms is unknown until their traces are measured.

## Real-data check on three UoT cases

The same per-guess metrics were run on the three complete episodes saved by the cancelled native UoT run (private evidence, bindings verified by [diagnostics v2](../../uot/diagnostics/README.md)). Of 26 logged guesses, 9 were repeats (34.6%), and 9 of 23 informed guesses were consistent with earlier feedback (39.1%). Public feedback identified the secret after turns 7, 9 and 7. In each case the harness then appended the single remaining candidate, which is the logged "success". This matches B7.4 and only illustrates the extractor; three cases support no rate estimate.

## Measuring the seven GN arms (CPU only, no model)

The private GN evidence is a `run_compare` controller directory. Run this wherever that evidence lives:

```text
python -B gn_state.py extract --controller <controller> [--controller <second segment>] --output state-tracking-gn-complete.json
```

It runs in seconds with the standard library only. For each development, final and training episode it does the following:

1. Checks the episode bytes against the run result's SHA-256 binding.
2. Checks the native target against the bound case file.
3. Recomputes every logged feedback.
4. Requires at most one complete record per evaluation slot across segments. Missing slots are counted as missing, never as failures.

The output holds only per-(phase, checkpoint, arm) aggregates:

- the consistency rate;
- the repeat, parse-failure and non-distinct-digit rates;
- bits of information acquired after 25 questions;
- how many episodes were identified, and the median turn;
- how many were hit, and how many had a correct last guess;
- how many were identified but had a wrong last guess;
- agreement with the native success flag.

A structural leak check refuses to write any string other than arm and phase names, fixed definitions and hex digests.

Interpretations fixed before the private output is seen:

- **Mostly identified, last guess wrong:** the arms gathered enough information but did not use it, which is a state-tracking or readout failure.
- **Low identification with high repeat rates:** the arms wasted questions, as UoT did. State the share explained by repeats.
- **DC and ACE metrics flat across checkpoints 0, 3 and 6:** the evolved memories did not reach the bottleneck. Changed memory text alone is not adaptation.

None of these readings is a counterfactual success estimate. A causal claim needs a new run, sketched in [the proposal bullets](../../PROPOSAL-BULLETS.md).

## Files

- `gn_state.py`: feedback (AR-Bench `compare_guess` semantics), exact state rows, the reference policies and the private extractor. Standard library only.
- `test_gn_state.py`: 13 tests, run with `python -I -B test_gn_state.py`. They cover:
  - feedback against the hash-pinned AR-Bench `utils_gn.py` on 20,000 pairs, including repeated-digit guesses;
  - state rows against a naive recomputation;
  - hand-made and readout-failure histories, and malformed records;
  - reference determinism;
  - extraction from synthetic controllers: hash tampering, case mismatch, duplicate slots, list-form native targets and the leak check.
- `reference-policies.json`: produced by `python -B gn_state.py reference --output reference-policies.json` (about 20 s).
- `plot_reference.py`, `reference-policies.svg` and `reference-policies.png`: the figure (matplotlib). The SVG is byte-stable across renders.
