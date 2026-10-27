# Proposal bullets: context evolution for active reasoning (GN and SP)

Each bullet carries its evidence status:

- **[run]**: an executed experiment with the stated limits.
- **[offline]**: an exact computation or a re-analysis of saved records, with no new model calls.
- **[proposed]**: not yet done.

## Motivation

- **[run]** On AR-Bench Guess Number, Qwen2.5-7B hits the secret in **0–2 of 20 cases** in every arm. That holds for five prompting/search baselines and for two context-evolution methods, DC and ACE ([gn-complete](gn-complete/README.md)).
- **[run, team]** The team's earlier GN diagnosis ([briefing.md](../../briefing.md)) found that **88.9–95.7% of Qwen2.5-7B's guesses were already ruled out by its own history** ("dead guesses"). An externally computed belief state lifted the solve rate from **2% to 99%**. Presentation also mattered: dead guesses fell to 19.2% with a short explicit candidate list, against 73.0% with an abstract summary.
- **[offline]** New here: the 25-question budget is not what limits the models. Even uniformly random unasked guesses gather enough public feedback to identify the secret for **all 5,040 secrets**, yet they hit it only **0.48%** of the time. Making a feedback-consistent guess on just 5% of turns lifts the hit rate to **65%** ([state-tracking reference](mechanism/state-tracking/README.md)).
- **[offline, measured on the saved traces]** In 7 arms × 20 final cases, the feedback had determined the secret before the final guess in **101 of 140**. Yet only **5** final guesses were right, and only **0.2–5%** of guesses were consistent with earlier feedback. The models gather the information; they do not use it ([B7.6 result](mechanism/state-tracking/README.md#result-on-the-saved-traces-b76)).
  - An earlier inference here, that the questions themselves were uninformative, is not supported and has been withdrawn.
- So the gap between the LLMs and trivial policies lies in **wasting questions and failing to use feedback**. Context can change both: what the model is shown and what it keeps. This is where self-evolution could improve **question efficiency**, not just make final answers more polished.
- **[run, team log]** On circuit decoding, the team found the same direction: an always-present state panel beats asking the model to maintain state itself (A6). A strict state-only harness, in which the model still writes every query, reaches 99.5/98.5 held out (B3). Our question is whether **context evolution** can learn such a representation, and whether it transfers to GN.

## Limitations of existing methods (with evidence)

- **DC and ACE do not adapt on GN [run].**
  - Both stay at 0/5 development exact match at updates 0, 3 and 6; final scores are 0/20 (DC) and 1/20 (ACE).
  - Their memories hold strategy prose. Nothing in them maintains the episode's own feedback state, which the reference identifies as the binding constraint.
  - **[offline] The traces confirm it, and show more than a flat line.**
    - On the same five development cases, consistent guesses stay at 0–1% at updates 0, 3 and 6.
    - Repeated guesses rise: DC from 6% to 43%. Determined cases fall: DC from 5 to 3, ACE from 5 to 2.
    - In the final split, DC repeats 42% of its guesses, against 13% for ZeroShot.
    - This is one learning trajectory per method, so it is descriptive.
- **The SP "gain" is not evidence of better reasoning [run].**
  - DC and ACE raise character F1 from 0.696 to 0.835 and 0.820, but word F1 stays flat (0.311 → 0.309 / 0.320).
  - Their answers are 34–44% longer.
  - The team's A7 found that answer length alone explains 65% of character-F1 variance.
  - N = 5, the referee is same-family and human qualification is unmet.
- **UoT spends most of its budget on bookkeeping errors [offline].** Native UoT, depth 1, was cancelled at 3/20 cases.
  - Each case needed 113–153 model calls, 4–6× a single-pass baseline, at about 9.6 s each when serial.
  - In 9 of 24 recorded decisions it re-asked an already-asked number. The parser keeps the first three 4-digit strings, and 14/23 proposals began by restating history.
  - 393/457 simulation calls hit the 512-token cap. This inflated the scores of zero-information guesses.
  - All 3 "successes" were the harness appending the last remaining candidate ([UoT diagnostics](uot/diagnostics/README.md)).
- **Unreliable feedback limits learning signals [run].**
  - The SP referee fails qualification.
  - ACE's own paper reports degradation without reliable feedback.
  - A learner cannot be better than its signal.

## Future method and plan

- **[proposed] Evolve the state representation, not strategy tips.** The learnable object is how the episode's own question–feedback facts are kept and laid out: a ledger layout chosen on training cases only.
  - Actor, budget and decoding stay fixed.
  - No candidate lists, recommended questions, information scores or solvers enter the prompt.
  - The team's list-versus-summary gap shows that presentation is a strong lever. The open question is how much of it a learned layout of facts and constraints can recover without a list.
- **[proposed] Three arms on the same fresh cases:**
  - chronological history (native);
  - a fixed lossless ledger (strong static baseline);
  - the evolved layout.

  Claim evolution only if the evolved layout beats the static ledger. Beating chronological history alone shows that structured memory helps, not that evolution does.
- **[proposed] Primary outcomes:**
  - repeat rate, feedback-consistency rate and identification (the `gn_state.py` metrics);
  - native success;
  - capped cost-to-resolution, with unresolved cases set to the cap.

  Report all calls, tokens and selection cost. Keep every failed chain.
- **[proposed] Preconditions before any GPU run:**
  1. ~~Run the CPU-only extraction on the saved 7-arm traces~~: done in B7.6. The arms fail at using feedback, not at gathering it.
  2. Run the measured throughput probe and feasibility gate in [uot/runtime](uot/runtime/README.md).

  A 3-arm × 20-case pilot is about 1,560 actor calls: roughly 4 h serially, or an estimated 30–60 min with 4–8× batching. The batched figure is a projection, not a measurement.
- **[proposed] SP needs a qualified referee before any semantic claim.** Human labels on a fixed reply sample come first; baseline breadth follows.

## Baseline coverage

| AR-Bench method | GN (Qwen2.5-7B, 20 final cases, native success) | SP (Kimi K2.6, 5 final cases, character F1) |
|---|---|---|
| ZeroShot | 0/20 | 0.696 |
| FewShot | 0/20 | not run |
| FewShotInst | 1/20 | not run |
| ProactiveCoT | 1/20 | not run |
| ToT (declared parser repair) | 2/20 | not run |
| UoT (depth 1) | **partial**: 3/20 complete when cancelled; mechanism diagnosed offline | not run |
| SFT / DPO | not run: AR-Bench at `9971322f` ships no training code, recipe or checkpoints | not run (same reason) |
| DC-Cumulative (evolution) | 0/20; development 0/5 at 0, 3 and 6 updates | 0.835 |
| ACE (evolution; supervised on GN, public feedback on SP) | 1/20; development 0/5 at 0, 3 and 6 updates | 0.820 |

GN and SP use different models and metrics; never pool them. Character F1 is overlap, not accuracy. There is one learning trajectory per method.

## Figures for the slides

- `mechanism/state-tracking/gn-final-arms-state.png`: information versus its use, measured for all 7 arms.
- `mechanism/state-tracking/gn-dev-state-curves.png`: DC and ACE process metrics across updates 0, 3 and 6.
- `mechanism/state-tracking/reference-policies.png`: information versus its use, model-free.
- `gn-complete/gn-development-curves.png` and `gn-complete/gn-final-metrics.png`: the evolution curves and final arms.
- `sp/figures/sp-character-f1.svg`: SP curves, labelled as overlap scores.
- The UoT numbers above, from `uot/diagnostics/diagnostic-summary-02.json`.
