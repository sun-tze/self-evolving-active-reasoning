# Reference: official task definitions, metrics and baselines

<details>
<summary><b>Circuit Decoding (MTP) · Situation Puzzles (AR-Bench) · judge reference points</b></summary>

<dl><dd>

> Sources: the AR-Bench (ICML 2025) and Multi-Turn Puzzles papers, checked line by line on
> 2026-10-12. The authoritative statement of what each metric is, and what the papers leave
> undefined, is [`experiments/OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md).
> Metrics and names are the official ones. Quantities we added are marked **(ours)** and are never
> placed in the same column as an official metric.

#### 1. Official baselines, Circuit Decoding (MTP Figure 1)

| model | circuit-wise accuracy |
|---|---|
| Gemini-2.5-Pro | 39.8 |
| GPT-4o tier | 15.4 |
| open-weight tier | 9.3 |

MTP Table 1 reports a single metric for Circuit Decoding, *normalized circuit-wise accuracy*; the
normalization is not defined in the paper. The "Success Rate (%)" in its figures is defined
per task and never stated. We therefore report `circuit_wise_accuracy` (the paper's metric) and
`joint_truth_table_correct` (the paper's noun, our literal reading of a solved episode).

#### 2. Official baselines, Situation Puzzles (AR-Bench Table 6, zero-shot, character-level F1)

| model | F1 (char) |
|---|---|
| GPT-4o | 63 |
| Llama-3.1-405B | 55 |
| Llama-3.1-70B | 54 |
| Qwen-2.5-7B | 51 |
| QwQ-32B | 49 |
| Llama-3.1-8B | 43 |
| GPT-4o-mini | 41 |

AR-Bench's official metrics are Detective Cases accuracy, Situation Puzzles character-level F1
(`arbench.utils.utils_sp.f1_score`) and Guess Number exact match. Its official referee is
Llama-3.1-405B. Key-question recall is **(ours)**, not an AR-Bench metric.

#### 3. Judge reference points (TurtleBench-150, seed 7, zero-shot)

| judge | accuracy | source |
|---|---|---|
| Claude Opus 5 (one fresh context per case) | 88.0 | ours, `labels_opus.json` |
| Llama-3.1-405B bf16 | 82.7 | TurtleBench's archived predictions |
| **Llama-3.1-405B-INT4 (in use)** | **82.0 / 83.3 on two runs** | ours; the gap is sampling at temperature 0.7 |
| Llama-3.1-70B (2-shot) | 74.0 | archived predictions |
| GPT-4o (2-shot) | 63.3 | archived predictions |
| Qwen2.5-7B-Instruct | 50.5 | ours; invalidates the early self-judged SP numbers |

TurtleBench is the right instrument here because **AR-Bench itself used it to cross-check its LLM
judges** (its Table 13), and TurtleBench's referee prompt is the source from which AR-Bench's SP
answerer prompt was adapted: same three-way scheme, same two few-shot examples.

</dd></dl>

</details>

> ---

# Experiment log (traceable, two stages)

<details>
<summary><b>Index (id · date · wall time · headline) and recording conventions</b></summary>

<dl><dd>

| id | date | wall | headline |
|---|---|---|---|
| A1 | 10-12 | — | **Official definitions fixed.** Circuit Decoding has one official metric, circuit-wise accuracy. `joint truth table correct` is the paper's noun under our literal reading. `uniq@end`, `eff`, `zero%`, `oracle%` are **(ours)**, diagnostic only. Column names and JSON keys unified repository-wide |
| A2 | 10-06 | pending (CPU) | **An 18-query budget is sufficient.** The greedy max-entropy oracle solves 300/300 in 11.60 queries; greedy over 4 sampled queries 250/300; random fresh entry 156/300; sequential scan 58/300; uniform random 35/300. Later failures cannot be attributed to the budget |
| **A3** | 10-10–10-14 | see children | **Referee calibration (TurtleBench-150).** The official Llama-3.1-405B-INT4 at **83.3** remains the referee; none of eleven candidates is a drop-in replacement under case-by-case comparison |
| A3.1 | 10-13 | ~10 min each, 1×H200 | Eight candidates, thinking off: Qwen3.8-Flash-Next 79.3, Qwen3-235B 74.0, Qwen3-32B 74.0, Llama-3.3-70B 73.3, Qwen3.6-35B-A3B 72.7, Qwen3.8-27B 64.0, gpt-oss-120b 62.7 |
| A3.2 | 10-13 | 10 min ×2 | Thinking on: Flash-Next 79.3 → **80.7**, Qwen3.8-27B 64.0 → **78.7**. Neither reaches 83.3 |
| **A3.3** | 10-13 | 12 s, $0.0041 | **Jev 1.13 (TypeSafe, closed hosted "System One" model) scores 78.7.** Unknown recall **36.8** against the 405B's 26.3, and it is the only candidate returning a usable distribution: accuracy rises monotonically with its confidence, 50.0% below 0.2 to 95.6% above 0.8. Not a drop-in: 80.7% agreement, repairs 10 of the 405B's 25 errors, breaks 17 of its 125 correct ones. **Not deterministic**: five identical runs gave 78.7 / 80.0 / 79.3 / 80.0 / 78.0 |
| A3.4 | 10-13 | 12 s each | Jev's three documented prompting levers (structured criteria, question splitting, both) all trade 3–5 points of accuracy for 10–25 points of Unknown recall, which the probability threshold already offers without a second request |
| **A3.5** | 10-13 | — | **Macro-averaging inverts the ranking.** The gold distribution is skewed (C 55 / I 76 / U 19), so a judge that never says Unknown is barely penalised. By macro recall, **Flash-Next (thinking) 79.9 is first and the 405B 69.6 is fifth**. In the SP pilot the referee answered Unknown to **13.8%** of 4,500 questions, so roughly one question in seven falls in the class the 405B recovers worst |
| **A3.6** | 10-14 | ~1 min per model | **Reading label probabilities never changes the verdict**, matching greedy decoding to within one point; it only exposes the distribution. **Calibration is a property of the model, not of the mechanism**: ECE 0.56 for Qwen2.5-7B, 0.16 for Qwen3-32B, **0.055 for Qwen3.8-27B**, whose macro recall 69.3 matches the 405B's 69.6. The 7B has one failure mode: it answers Unknown on 99 of 150 cases, and forbidding Unknown lifts it from 36.0 to 68.0 |
| A4 | 10-10 | 7 min | **Situation Puzzles without a harness, under the official-grade referee**: F1 0.621. The earlier 0.659 was self-judged by the 7B and is withdrawn |
| **A5** | 10-10/11 | pending | **Actionability ladder, seven hand-written harnesses.** bare 12.9 / 0.5 / uniq@end 23.2 → symbolic 20.5 / 2.2 / 41.2 → enum15 23.7 / 5.8 / 41.8 → **T3-gated 31.3 / 10.2** → splits 22.3 / 5.7 / **53.7** → **oracle hint 27.3 / 6.8 / 96.5**. **Information gathering and final accuracy move apart**: the oracle hint nearly saturates uniq@end at 96.5 yet leaves joint accuracy at 6.8, so the bottleneck is the final inference, not the questioning |
| A6 | 10-11 | 1 h 08 | **EvoHarness-RL as an external baseline**: env_only (ReAct) 4.3/4.9, inline (the model calls track/commit itself) 5.7/6.9, **always_on (a STATE panel every turn) 74.8/79.0 with joint 62.0/65.0**. With the same information, **an always-present panel beats asking the model to maintain it**, in the same direction as A5 |
| **A7** | 10-21 | — | **What the SP metric measures** (re-analysis of 500 stored episodes, no new inference). **Answer length alone explains 65.0% of the variance in character-level F1 (r = +0.806)**; key-question recall gives r = +0.050 and a partial correlation of +0.068 once length is controlled. Mean F1 rises monotonically with length with no turning point, so there is **no effective length penalty**. **Controlling for length, the bare→final gain of +15.2 shrinks to +0.4…+7.2.** The task also never requires the final answer to follow from the recorded history (**step-skipping is permitted**; the degenerate baseline is not yet measured). ⚠️ The first computation returned r = 0.90 because it included 88 empty answers from a preempted run; after cleaning, 0.806 |
| **A8** | 10-15–21 | — | **Independent preregistered replication (Yangzi Sun).** Fixed dialogue × two writers, 65 cases, 260 generations, 520 blind ratings, adding Semantic Correctness and Evidence Faithfulness. **82 strict F1-versus-semantic rank reversals across 29 cases**; the evidence-constrained writer did not improve semantic correctness (−0.0308 [−0.0577, −0.0058]). Two readings we do not adopt: it evaluates the **first** generation, whose recall we had already documented as flat, and its own data shows the evolved dialogue doubling fully-correct answers (15/65 against 7/65) while the ordinal mean moves the other way |
| B1 | 10-10 20:30–23:10 | 2 h 40 | One-shot evolution `open1` (regime **open**): the proposer reaches the ceiling in one iteration; `copy_query` scores joint **80.4** and uniq@end 99.7 held out. ⚠️ Under this regime the harness may choose queries, so the score is a ceiling, not a claim |
| B2 | 10-10 21:22 – 10-11 ~01:00 | ~3 h 38 | One-shot evolution `so1` (regime **state_only**: no information gain, no split counts, no recommended queries). `veto_only` scores **91.3 / 86.2** with uniq@end 96.7 **at the same token cost as bare**. This is the shape of the claim: remove what computes for the model, keep state and constraints, and the score still rises |
| **B3** | 10-12 19:01–22:33 | 3 h 32 | **Test-time evolution with label isolation.** The strict stream goes **0/10 → 9/10 → 10/10**; 29/32 sessions clean and **every judge session clean**. Frozen evaluation on 200 held-out instances: bare 11.8 / 0.0, the label-free state-only harness **99.5 / 98.5** (98.2 on the unseen split), the labeled stream 100/100 at 215 tokens per episode. The judge rejected three harnesses that scored 10/10 because they handed the model a menu, enforcing that **the model writes every query** |
| **B4** | 10-13 05:03–07:17 | 2 h 14 | **Transfer to AR-Bench Situation Puzzles** (strict, 405B referee). Prequential F1 **63.7 → 74.5 → 79.9** over the 30-puzzle stream. On all 100 test puzzles (70 unseen), **bare 60.4 → 73.0–76.7**, above every zero-shot model in AR-Bench Table 6 including GPT-4o at 63, with the 7B still writing every question. Cost: 3–8× output tokens; the unchosen `b2r0g1` matches at half the tokens. Read together with A7 |
| **B5** | 10-13 | — | **A ruler for the TTHE judge, requiring no external benchmark.** Each run stores the candidate pool's gold, which that judge never saw. `strict6` selects a best-scoring candidate 2/3 with mean regret 0.011; `labeled6` 3/3 and 0.000; **`sp_strict1` 1/3, mean regret 0.063, worst 0.114**. The Circuit Decoding pools saturate, so the measurement is uninformative there; **on Situation Puzzles the judge is measurably noisy** |
| B6 | 10-26 | see children | GN capped-identification studies: negative note updates and selected-policy improvement; separate prior PR #2 |
| B6.1 | 10-26 | 434 s allocation | Three-chain note updates worsened capped-query cost; negative result retained |
| B6.2 | 10-26 | 784 s allocation | Two selected-policy learning blocks; paired evaluation, selection costs and limitations retained |
| B7 | 10-26 | see children | Native GN and exploratory SP method-adapter pilots; neither establishes a general method ranking |
| B7.1 | 10-26 | 5,648.25 s allocation | GN full run failed: 30/30 dev, 97/140 final, 43 missing; descriptive common prefix only |
| B7.2 | 10-26 | 2,903.56 s active | SP completed: final N=5, character F1 not accuracy; same-family referee qualification UNMET |

> **Purpose.** Every experiment must be traceable from **conditions (task, model, regime, signals,
> budget, split) → script → input → output and log → result**.
>
> **Markers.** 📁 local, verified path · ☁️ server (Engaging `~/results/ar_harness`, H100
> `/home/tmp_l`) · 🔒 controller-private (`~/.ar_harness_controller/<run>/`, never committed).
>
> **Recording rules.** Every formal experiment carries its **date** and **wall time** before its
> name. Dates are `YYYY-MM-DD`, spanning days as `MM-DD–MM-DD`, repeated runs as two dates. Clock
> times belong in the per-experiment README, not in this index. Times are taken from log start and
> end; where no log exists, from artefact mtime, and the inference is stated. When no reliable
> record exists, write `pending` — **never guess**. The last column of a result table is always the
> traceback: the script that produced that row, and its log.
>
> 🔗 **Traceability blocks** sit at the end of each leaf and hold upstream scripts, artefact paths
> and batch logs, with no duplication of the per-row tracebacks.

> **Two stages.**
> - **A1–A8, the evaluation substrate** — official definitions, the task's ceiling, referee
>   calibration, the harness-free baselines, the hand-written ladder, an external baseline, **an
>   audit of the metric itself**, and **an independent replication**. This stage decides whether the
>   numbers we report mean anything. A7's answer is that **the official F1 largely measures answer
>   length**, so the main tables must be read beside it.
> - **B1–B5, harness evolution** — one-shot evolution, test-time evolution, label isolation,
>   cross-task transfer, and the judge's own quality. This stage decides whether the mechanism holds.
>
> **Metric definitions, uniform across this document.**
> - **circuit-wise accuracy** — Circuit Decoding's **only official metric**: the fraction of the
>   three circuits whose full truth table is correct.
> - **joint truth table correct** — the complete 24-bit joint table is correct. The paper's noun,
>   our literal reading.
> - **f1_char** — Situation Puzzles' **official metric**, character-level F1 against the hidden
>   bottom (`arbench.utils.utils_sp.f1_score`).
> - **key-question recall (ours)** — the share of a puzzle's key questions the referee judges to
>   have been hit. A process metric, not official.
> - **uniq@end, eff, zero%, oracle%, rep% (ours)** — diagnostics for mechanism analysis. Never
>   reported alongside an official metric.
> - **prequential** — the score of the **incumbent** harness on a new batch, before adaptation: the
>   deployment view. **transductive** — the score of the **committed** harness on the same batch:
>   the fitting view. The two are always reported separately.
> - **ECE** — expected calibration error over ten equal-width bins. **macro recall** — the
>   unweighted mean of the three class recalls, which corrects for the skewed gold distribution.
>
> **Scope of the claim.** Training-free: **no**. Data-free: **also no** — there is a stream of
> instances. The defensible term is **label-free / transductive**, and the evidence is that strict
> matches labeled. A genuinely data-free claim would require the task family to be self-generated.

**Data and judge assets**

| item | path | note |
|---|---|---|
| Circuit Decoding instances | ☁️ `~/code/ar_harness/mt_puzzles/` | 300 released instances, 18 queries then the joint table |
| Situation Puzzles instances | ☁️ `~/code/ar_harness/AR-Bench/data/sp/` | official split; bottoms and key questions are controller-private |
| TurtleBench | ☁️ `~/TurtleBench` (both machines) | 1,531 English cases; a fixed sample of 150 (seed 7) across 32 stories |
| Per-case judge predictions | 📁 [`experiments/02_judge_calibration/turtle150_predictions/`](../experiments/02_judge_calibration/turtle150_predictions/) | eleven candidates plus Jev and the typed read, with probabilities where available |
| Controller-private records | 🔒 `~/.ar_harness_controller/<run>/` | solve jobs, full optimisation log, lineage. **Never committed** |
| Raw traces | GitHub Release `v0.1-experiments-2026-10-11` | `cd_tthe_isolated_raw`, `cd_tthe_pilots_raw`, `sp_tthe_pilot_raw` |

</dd></dl>

</details>

> ---

# Stage A — the evaluation substrate

<details>
<summary><b>A1 Official definitions and names (10-12)</b></summary>

<dl><dd>

Before any number in this log is comparable to a published one, the metric it is computed under has
to match the metric the benchmark defines. This node fixes that, and the result is
[`experiments/OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md).

**Circuit Decoding.** MTP Table 1 reports exactly one metric, *normalized circuit-wise accuracy*,
and the normalization is never defined. The "Success Rate (%)" of its figures is defined per task
and never stated. We therefore report `circuit_wise_accuracy`, the paper's metric, and
`joint_truth_table_correct`, the paper's noun under our literal reading of a solved episode.
Everything else we compute — `uniq@end`, `eff`, `zero%`, `oracle%`, `rep%` — is **(ours)** and is
never placed in a column beside an official metric.

**Situation Puzzles.** The official metric is character-level F1 against the hidden bottom
(`arbench.utils.utils_sp.f1_score`), the official referee is Llama-3.1-405B, and the companion
tasks use Detective Cases accuracy and Guess Number exact match. **Key-question recall is (ours)**,
a process metric we added; it is reported beside F1 and never in place of it.

**One correction this node produced.** An earlier sheet attributed a "successful end state" metric
to Circuit Decoding; it belongs to Word Chaining. The sheet was rewritten and all column names and
JSON keys were unified across the repository, with the stored score files migrated.

**Standing rule.** Tasks, metrics and baselines are reported under their official definitions and
official names. Limitations may be stated — [A7](#) and [A8](#) do exactly that — but the reported
tables are not substituted with a metric of our own.

<details>
<summary>🔗 Traceability (A1)</summary>

<dl><dd>

- Definitions: 📁 [`experiments/OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md)
- Scorers carrying the official keys: [`runner/cd_score.py`](../runner/cd_score.py) ·
  [`runner/cd_summarize.py`](../runner/cd_summarize.py) ·
  [`runner/cd_score_evoharness.py`](../runner/cd_score_evoharness.py)
- Index of all runs under these names: 📁 [`experiments/README.md`](../experiments/README.md)

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A2 Stage-0: hypothesis space, oracle and synthetic policies (10-06, CPU)</b></summary>

<dl><dd>

| policy | solved | queries | ex-ante eff | zero-information queries |
|---|---|---|---|---|
| greedy max-entropy (the oracle) | **300/300** | 11.60 | 1.000 | 0.0% |
| greedy over 4 sampled queries | 250/300 | 13.92 | 0.775 | 18.1% |
| random fresh entry | 156/300 | 16.51 | 0.629 | 27.7% |
| sequential scan | 58/300 | 17.41 | 0.540 | 38.1% |
| uniform random (repeats allowed) | 35/300 | 17.63 | 0.480 | 44.1% |

**The 18-query budget is sufficient**: the oracle collapses the hypothesis space to one in 11.60
queries on average. Every later failure must therefore be attributed to the policy, not to the
budget. This is the precondition for everything that follows.


<details>
<summary>🔗 Traceability (A2)</summary>

<dl><dd>

- Details: 📁 [`experiments/01_stage0_oracle/`](../experiments/01_stage0_oracle/)
- Oracle and synthetic policies: `stage0/cd_oracle.py` (CPU only, no model calls)

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A3 Referee calibration on TurtleBench-150</b></summary>

<dl><dd>

> What is calibrated here is the **referee**: the model that answers the player's questions in
> Situation Puzzles. It is not the judge inside the evolution loop that commits harnesses; that one
> has its own ruler in [B5](#). The two must be named separately throughout the paper.

<details>
<summary>A3.1–A3.2 Eleven candidates, thinking off and on (10-13)</summary>

<dl><dd>

Accuracy, then per-class recall as Correct / Incorrect / Unknown (gold counts 55 / 76 / 19).

| candidate | job | thinking off | C/I/U | thinking on | C/I/U |
|---|---|---|---|---|---|
| **Llama-3.1-405B-INT4 (in use)** | 25698041 | **83.3** | 89.1 / 93.4 / 26.3 | n/a | |
| Qwen3.8-Flash-Next-FP8 | 25718457 / 25724076 | 79.3 | 90.9 / 81.6 / 36.8 | **80.7** | 90.9 / 75.0 / 73.7 |
| Qwen3.8-27B-FP8 | 25718456 / 25724077 | 64.0 | 74.5 / 59.2 / 52.6 | **78.7** | 85.5 / 77.6 / 63.2 |
| Qwen3-235B-A22B-2507-FP8 | 25718459 | 74.0 | 83.6 / 69.7 / 63.2 | not run | |
| Qwen3-32B | 25698272 | 74.0 | 80.0 / 81.6 / 26.3 | not run | |
| Llama-3.3-70B-FP8-dynamic | 25698273 | 73.3 | 76.4 / 84.2 / 21.1 | not run | |
| Qwen3.6-35B-A3B-FP8 | 25718458 | 72.7 | 87.3 / 69.7 / 42.1 | not run | |
| gpt-oss-120b | 25698271 | 62.7 | 61.8 / 65.8 / 52.6 | not run | |

**Every candidate fails on Unknown**, the class that decides whether a player's question is
answerable from the story at all. The 405B's advantage is concentrated in Incorrect (93.4, with no
other candidate above 85); its own Unknown recall of 26.3 is second from last. Thinking buys back
precisely that class for the Qwen3.8 models (Flash-Next 36.8 → 73.7) without overtaking the 405B.

Case by case against the incumbent (125 correct, 25 wrong): the strongest candidate, Flash-Next
with thinking, repairs 13 of the 25 errors and breaks 17 of the 125 correct answers. All others are
worse. **No candidate is a drop-in**, and replacing the referee would move the Situation Puzzles
numbers for reasons unrelated to the harness, so the official 405B is retained.

⚠️ Serving notes. FP8 on H200 requires `VLLM_USE_DEEP_GEMM=0` (the DeepGEMM JIT fails against
nvcc 13.4 headers). Flash-Next needs `max_num_seqs<=57` for its Mamba cache. The 405B-INT4 requires
pipeline parallelism, since tensor parallelism triggers a FlashInfer JIT build, and its
`generation_config` omits `<|eot_id|>`, so `stop_token_ids [128001, 128008, 128009]` must be passed.

</dd></dl>
</details>

<details>
<summary>A3.3–A3.4 A closed typed-decision model as referee: Jev 1.13 (10-13)</summary>

<dl><dd>

Jev takes a `state` plus typed `questions` and returns a typed answer with probabilities
(`POST /v1/systemone`). That is this task's native shape, so one run was warranted. The three
TurtleBench judging rules were carried over verbatim as the question's `instructions` and the three
labels as its `criteria`. ⚠️ Jev accepts no system message, so **the official prompt cannot be sent
byte-for-byte**; this is the one asymmetry in the comparison.

| | Jev 1.13 | Llama-3.1-405B-INT4 |
|---|---|---|
| accuracy | 78.7 | 83.3 |
| recall C / I / U | 80.0 / 88.2 / **36.8** | 89.1 / 93.4 / 26.3 |
| wall for 150 cases | **12 s** (16 concurrent) | ~15 s on 2×H200 after a 270 s load |
| cost | **$0.0041** (98,480 input tokens at $0.042/MTok) | an H200 pair for the run |

It is the only candidate whose verdicts arrive as a distribution, and its confidence
(`1 − H(p)/ln K`) is monotone in accuracy:

| confidence | n | accuracy |
|---|---|---|
| 0.0–0.2 | 12 | 50.0% |
| 0.2–0.4 | 26 | 61.5% |
| 0.4–0.6 | 37 | 73.0% |
| 0.6–0.8 | 30 | 86.7% |
| 0.8–1.0 | 45 | 95.6% |

**None of the vendor's own prompting levers improves it** (A3.4): structured `what` / `not_for` /
`examples` criteria give 74.0, splitting into a decidability `noul` plus a binary choice 74.7, both
together 76.0 with the best macro recall at 70.9. Each buys 10–25 points of Unknown recall for 3–5
points of accuracy, which is the same trade the probability threshold already offers at no extra
request.

⚠️ **Jev is not deterministic.** Five identical requests returned 78.7 / 80.0 / 79.3 / 80.0 / 78.0,
with Unknown recall between 26.3 and 36.8 and per-case probabilities moving as well. **Differences
under about two points cannot be read from a single run.**

Not adopted, for three reasons: it is not a drop-in case by case; AR-Bench's referee is the 405B and
AR-Bench reports accuracy; and it is a closed hosted service, so every story and guess would leave
our infrastructure. Its natural role is a cheap filter, or a second opinion whose confidence selects
which verdicts to re-check.

</dd></dl>
</details>

<details>
<summary>A3.5 Macro-averaging inverts the ranking (10-13)</summary>

<dl><dd>

Accuracy weights the classes as the gold does (Correct 55, Incorrect 76, Unknown 19), so **a judge
that never says Unknown is barely penalised**. Under macro-averaged recall and macro F1:

| judge | accuracy | macro recall | macro F1 | C / I / U |
|---|---|---|---|---|
| Qwen3.8-Flash-Next (thinking) | 80.7 | **79.9** | **76.0** | 90.9 / 75.0 / 73.7 |
| Qwen3.8-27B (thinking) | 78.7 | 75.4 | 72.5 | 85.5 / 77.6 / 63.2 |
| Qwen3-235B-A22B-2507 | 74.0 | 72.2 | 68.6 | 83.6 / 69.7 / 63.2 |
| Qwen3.8-Flash-Next | 79.3 | 69.8 | 70.1 | 90.9 / 81.6 / 36.8 |
| **Llama-3.1-405B-INT4 (in use)** | **83.3** | 69.6 | 70.9 | 89.1 / 93.4 / 26.3 |
| Jev 1.13 | 78.7 | 68.3 | 69.8 | 80.0 / 88.2 / 36.8 |
| Qwen3.6-35B-A3B | 72.7 | 66.4 | 65.9 | 87.3 / 69.7 / 42.1 |
| Qwen3-32B | 74.0 | 62.6 | 63.6 | 80.0 / 81.6 / 26.3 |
| Qwen3.8-27B | 64.0 | 62.1 | 60.0 | 74.5 / 59.2 / 52.6 |
| Llama-3.3-70B | 73.3 | 60.5 | 60.8 | 76.4 / 84.2 / 21.1 |
| gpt-oss-120b | 62.7 | 60.1 | 57.7 | 61.8 / 65.8 / 52.6 |

**The effect is not marginal in play.** Across the 180 episodes and 4,500 referee answers of the SP
pilot, the referee said No 58.5%, Yes 27.6% and **Unknown 13.8%**: about one question in seven falls
in the class the 405B recovers worst. A referee that under-reports Unknown forces those into a Yes
or a No, which injects noise into the game itself, not merely into our metric.

**Disposition.** The referee is unchanged, because AR-Bench's is the 405B and accuracy is what
AR-Bench reports. But the macro table belongs in the paper beside accuracy, and once the main
results are in, the SP numbers should be re-run under Flash-Next (thinking) as a robustness check.
*Not yet run.*

</dd></dl>
</details>

<details>
<summary>A3.6 Reading label probabilities instead of sampling a word (10-14, H100)</summary>

<dl><dd>

Jev's advantage over the chat judges was not accuracy but that its verdicts arrive as a
distribution. That mechanism is not proprietary: any served model can be constrained to the three
labels and read at the first token. [`runner/typed_judge_eval.py`](../runner/typed_judge_eval.py)
does this with `structured_outputs: {"choice": [...]}` and the official referee prompt;
[`h100/typed_bench.sh`](../h100/typed_bench.sh) scores one model under three protocols.

| model | sampled, T=0.7 | sampled, T=0 | typed read | ECE | C / I / U (typed) | macro |
|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 34.7–37.3 | 34.7–35.3 | 36.0–36.7 | 0.56 | 50.9 / 17.1 / 68.4 | 45.5 |
| Qwen3-32B-FP8 (no thinking) | 72.7–74.7 | 73.3–74.0 | 73.3 | 0.16 | 80.0 / 80.3 / 26.3 | 62.2 |
| **Qwen3.8-27B-FP8 (no thinking)** | 64.7 | 70.0 | **70.7** | **0.055** | 76.4 / 68.4 / 63.2 | **69.3** |

**The typed read never changes the verdict** — it matches greedy decoding to within one point, since
the argmax is the same token — **it only exposes the distribution, and the distribution is worth
having only when it carries information.** For Qwen3-32B it does (ECE 0.16), and the threshold
becomes a usable control: taking Unknown whenever its probability exceeds 0.30 raises Unknown recall
from 26.3 to 36.8 and macro recall from 62.2 to 64.4, at a cost of 0.6 points of accuracy. For the
7B it does not: mean P(argmax) 0.945 against accuracy 0.360. **Calibration is a property of the
model, not of the mechanism** — which is also the honest reading of the Jev result.

**Qwen3.8-27B is the best-calibrated judge measured**: ECE 0.055, mildly under-confident (mean
P(argmax) 0.665 against accuracy 0.707), and macro recall 69.3, level with the 405B's 69.6 from a
27B checkpoint on one H100, because it alone does not collapse on Unknown. Its accuracy is 12.6
points lower, so it does not replace a benchmark that reports accuracy; it is the candidate for uses
where the third class matters. It is also the only model for which sampling temperature is not free:
T=0.7 costs it 5.3 points against T=0.

⚠️ **Three errors on the way, each of which produced plausible-looking numbers.** (i) The sampled
protocol defaults to T=0.7 while the typed read uses 0; measured, this is worth 0.7–1.4 points.
(ii) For a reasoning model the first token is the thinking marker, not the label; scanning later
positions then matched the letter "I" of ordinary prose as a prefix of "Incorrect", yielding **50.7%
with all 150 cases answered Incorrect at confidence 1.000**. (iii) **vLLM 0.19 silently ignores
`guided_choice`**: the request succeeds, nothing is constrained, and the numbers are identical to an
unconstrained run. The correct field is `structured_outputs`, and thinking is disabled through
`chat_template_kwargs: {"enable_thinking": false}`. The five-line probe that settled this is
[`runner/probe_guided.py`](../runner/probe_guided.py).

</dd></dl>
</details>

<details>
<summary>🔗 Traceability (A3)</summary>

<dl><dd>

- Evaluation: [`runner/turtle_judge_eval.py`](../runner/turtle_judge_eval.py) (sampled) ·
  [`runner/typed_judge_eval.py`](../runner/typed_judge_eval.py) (typed read) ·
  [`runner/jev_judge_eval.py`](../runner/jev_judge_eval.py) and
  [`runner/jev_judge_variants.py`](../runner/jev_judge_variants.py) (Jev) ·
  [`runner/probe_guided.py`](../runner/probe_guided.py) (constrained-output probe)
- Serving: ☁️ `engaging/calib.sbatch`, `engaging/serve_referee.sbatch`, `engaging/judge405.sbatch` ·
  📁 [`h100/typed_bench.sh`](../h100/typed_bench.sh)
- Predictions and logs: 📁 [`experiments/02_judge_calibration/turtle150_predictions/`](../experiments/02_judge_calibration/turtle150_predictions/)
- Details: 📁 [`experiments/02_judge_calibration/README.md`](../experiments/02_judge_calibration/README.md)
- Credentials: the Jev key is in the macOS keychain under `typesafe_token`; **the value is never committed**

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A4 Situation Puzzles without a harness, official-grade referee (10-10 21:29–21:36, 7 min)</b></summary>

<dl><dd>

F1 **0.621**. ⚠️ The previously recorded 0.659 was obtained with the **7B judging itself** and is
withdrawn — which is the reason the whole of A3 exists. All historical SP numbers are rebased on the
405B-INT4 referee.


<details>
<summary>🔗 Traceability (A4)</summary>

<dl><dd>

- Details: 📁 [`experiments/07_sp_bare_405judge/`](../experiments/07_sp_bare_405judge/)
- Referee calibration that invalidated the earlier figure: [A3](#)
- Raw traces: Release `sp405_traces.tar.gz`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A5 Actionability ladder: seven hand-written harnesses (10-10/11)</b></summary>

<dl><dd>

| harness (what is appended after each answer) | cwa | joint | uniq@end (ours) | eff (ours) | tok/ep |
|---|---|---|---|---|---|
| bare — nothing | 12.9 | 0.5 | 23.2 | 0.553 | 2389 |
| facts (a) — the model's own observations restated | 16.2 | 0.5 | 29.0 | 0.570 | 2800 |
| symbolic — remaining configurations and determined outputs | 20.5 | 2.2 | 41.2 | 0.612 | 2648 |
| enum15 — symbolic plus the full list when ≤15 remain | 23.7 | 5.8 | 41.8 | 0.612 | 2892 |
| **T3-gated** — enum15 only when a trap proxy fires | **31.3** | **10.2** | 36.8 | 0.612 | 3104 |
| splits (c) — enum15 plus 0/1 split counts per open entry | 22.3 | 5.7 | **53.7** | 0.702 | 4316 |
| oracle hint (d) — enum15 plus the max-entropy query | 27.3 | 6.8 | **96.5** | 0.951 | 4522 |

**Information gathering and final accuracy move apart**, which is the most important observation on
this track. The oracle hint drives uniq@end to 96.5 — in nearly every episode the hypothesis space
collapses to one — yet joint accuracy reaches only 6.8. **The questioning is already good enough;
the bottleneck is reading the collapsed state out as an answer.** T3-gated reaches a higher joint
score with less information, pointing at the same place.


<details>
<summary>🔗 Traceability (A5)</summary>

<dl><dd>

- Details: 📁 [`experiments/03_cd_ladder/`](../experiments/03_cd_ladder/)
- Scoring and the ladder figure: [`runner/cd_score.py`](../runner/cd_score.py) ·
  [`runner/cd_ladder_plot.py`](../runner/cd_ladder_plot.py)
- Raw traces: Release `cd_ladder_traces.tar.gz`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A6 EvoHarness-RL as an external baseline (10-11 06:28–07:36, 1 h 08)</b></summary>

<dl><dd>

| mode | run | cwa | joint | uniq@end (ours) |
|---|---|---|---|---|
| env_only (ReAct, no harness) | s0 / s1 | 4.3 / 4.9 | 0.0 / 0.0 | 29.3 / 26.0 |
| inline = EvoHarness-Base (the model calls track / commit / recall / note) | s0 / s1 | 5.7 / 6.9 | 0.7 / 1.7 | 30.0 / 31.7 |
| **always_on (a STATE panel every turn)** | s0 / s1 | **74.8 / 79.0** | **62.0 / 65.0** | **71.0 / 74.7** |

With the same state information available, **an always-present panel beats asking the model to
maintain it** (joint 62–65 against 0.7–1.7). The inline mode barely improves on ReAct, so "give the
model tools and let it keep its own notes" does not hold for a frozen 7B on this task. Together with
A5: the harness should maintain state for the model, but **must not reason for it**.


<details>
<summary>🔗 Traceability (A6)</summary>

<dl><dd>

- Details: 📁 [`experiments/06_evoharness_cd_baseline/`](../experiments/06_evoharness_cd_baseline/)
- Our Circuit Decoding adapter for the official framework: `baselines/evoharness_cd/`
- Scoring: [`runner/cd_score_evoharness.py`](../runner/cd_score_evoharness.py)
- Raw traces: Release `evoharness_cd_trajectories.tar.gz`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>A7 What the Situation Puzzles metric measures (10-21)</b></summary>

<dl><dd>

A re-analysis of records already held: the frozen evaluation of five harnesses on all 100 test
puzzles, 500 episodes, no new inference. Script
[`runner/metric_audit.py`](../runner/metric_audit.py); details
[`experiments/11_metric_audit/`](../experiments/11_metric_audit/).

| correlated with character-level F1 | r |
|---|---|
| **answer length in words** | **+0.806** |
| output tokens for the episode | +0.403 |
| **key-question recall (ours)** | **+0.050** |

Length alone explains R² = 0.650; adding recall reaches 0.652, with a recall coefficient of +0.021.
With length partialled out, recall correlates with F1 at +0.068. Mean F1 by length is monotone with
no turning point — 25–50 words 0.458, 50–75 0.634, 75–100 0.766, 100–125 0.826, 125–150 0.868 —
so **the metric carries no effective length penalty**.

**How much of the headline gain survives length control**: +0.005 in the 25–50 band, **+0.072** in
50–75, +0.043 in 75–100, +0.004 in 100–125, against a pooled difference of **+0.152**. The final
harness writes 91 words where bare writes 61, and most of the headline gap is that difference. The
advantage shrinking at longer answers is consistent both with the advantage being small and with F1
saturating at length; **F1 alone cannot separate the two**, which is what semantic grading is for.

**Three consequences for the benchmark.** The metric **rewards verbosity**, and our harness was
pushed along exactly that gradient, from 670 to 5,263 output tokens per episode. **Step-skipping is
permitted**: nothing requires the final answer to follow from the recorded question-and-answer
history, so an agent may waste all 25 questions and write a long answer from the surface alone — the
degenerate baseline that would quantify this needs the gold bottoms and is *not yet measured*. And
**the environment carries its own noise**: the referee is right about 83.3% of the time, so roughly
one answer in six that the questioner receives may be wrong.

⚠️ **Recorded error.** The first computation returned r = 0.90 because it included 88 empty answers
from the run interrupted by the referee's preemption; those points sit at (0 words, 0 F1). After
removing them, 0.806. The direction is unchanged; the figure was overstated once.

**No change to what we report.** AR-Bench's metric is character-level F1 and the main tables keep
it, per [`OFFICIAL_DEFINITIONS.md`](../experiments/OFFICIAL_DEFINITIONS.md). A7 is the limitations
section, not a replacement metric.

</dd></dl>

</details>

<details>
<summary><b>A8 Independent replication: a fixed-dialogue writer probe (10-15–21)</b></summary>

<dl><dd>

A preregistered experiment by Yangzi Sun separating **questioning** from **final answering**: our
recorded dialogues (bare and first-generation) are frozen and paired with two writers (plain and
evidence-constrained), giving four cells over 65 cases, 260 generations and 520 blind ratings, with
two new ordinal metrics, Semantic Correctness and Evidence Faithfulness.
[Report](https://github.com/sun-tze/self-evolving-active-reasoning/blob/sun/c2-confirmatory-65/experiments/sun_c2_confirmatory/REPORT.md)

| condition | F1 | semantic (0–2) | faithfulness (0–2) | fully correct by both reviewers |
|---|---|---|---|---|
| A bare / plain | 0.681 | 0.785 | 0.662 | 7/65 |
| B bare / evidence | 0.683 | 0.754 | 0.669 | 8/65 |
| C first-selected / plain | 0.693 | 0.754 | 0.331 | **15/65** |
| D first-selected / evidence | 0.695 | 0.662 | 0.400 | 12/65 |

**What we adopt.** F1 and semantic correctness reverse rank 82 times across 29 cases, so lexical
overlap and causal understanding can come apart entirely. The evidence-constrained instruction did
not improve semantic correctness (−0.0308, 95% CI [−0.0577, −0.0058]), plausibly because a situation
puzzle's solution is by construction *not* in the evidence: pinning the model to confirmed facts
suppresses the abductive step the task requires. This reaches the same conclusion as [A7](#) by an
independent route.

**What we do not adopt.** (i) It evaluates the **first** generation, whose key-question recall we
had already documented as flat; by the final harness, recall rises from 57.6 to 66.4 on the unseen
split. (ii) The unchosen candidate `b2r0g1` reaches a higher F1 than the first generation with
**half** the tokens and +10.5 points of recall, which a pure length account does not explain.
(iii) Its own data shows the evolved dialogue doubling the count of answers both reviewers scored
fully correct, 15/65 against 7/65, while the ordinal mean moves the other way: removing the
full-credit cases leaves a mean of 0.38 for the evolved dialogue against 0.64 for bare, so the
evolved condition is **bimodal** where bare accumulates partial credit.

**Methodological boundaries**, as the report itself states. It does not reproduce our final-answer
mechanism (three drafts, scoring, three checks), so no cell corresponds to the system we actually
run. The reviewers are two independent executions of the same model family, so 96.9% agreement does
not establish accuracy. We add one further confound: faithfulness may be dominated by **how much
evidence there is to contradict** — a weaker dialogue yields fewer constraints and therefore fewer
opportunities to violate them, so the measure should be normalised per recorded constraint.

**Next step, not yet run.** Blind semantic grading of the **complete** harness answers for bare, the
final harness and `b2r0g1`, by a cross-family judge (Opus 5 scores 88.0 on TurtleBench against the
405B's 83.3), with a 30-answer human-verified subsample, reporting both the mean and the
fully-correct rate.

</dd></dl>

</details>

> ---

# Stage B — harness evolution

<details>
<summary><b>B1 / B2 One-shot evolution: regime open against state_only</b></summary>

<dl><dd>

<details>
<summary>B1 `open1`, regime open (10-10 20:30–23:10, 2 h 40)</summary>

<dl><dd>

The proposer **reaches the ceiling in one iteration**: `copy_query` scores joint **80.4** with
uniq@end 99.7 held out. ⚠️ Under the open regime the harness may select queries directly, so this
number is a **ceiling, not a claim**. It is superseded by B2.

</dd></dl>
</details>

<details>
<summary>B2 `so1`, regime state_only (10-10 21:22 – 10-11 ~01:00, ~3 h 38)</summary>

<dl><dd>

Rule: the harness may maintain and display state and may refuse, but **may not compute information
gain, report split counts or recommend queries**. `veto_only` scores circuit-wise accuracy **91.3**
and joint **86.2** with uniq@end 96.7, **at the same token cost as bare**.

**This is the shape of the claim.** Remove the part that computes on the model's behalf, keep the
constraints and the state, and the score still rises — without spending more tokens. A5's T3-gated
and A6's always_on point the same way.

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>B3 / B3 Test-time harness evolution and label isolation</b></summary>

<dl><dd>

> In test-time harness evolution, instances arrive in batches; after each batch a proposer writes
> candidate harnesses and a judge decides whether to commit one **without seeing gold**. Two streams
> are run: **strict** (run status only) and **labeled** (status plus outcomes), so that
> strict ≈ labeled is evidence for the label-free claim. The proposer and judge are Claude Opus 5.5
> inside Claude Code; the questioner is a frozen Qwen2.5-7B.


<details>
<summary>B3 Stream, audit and frozen evaluation (10-12 19:01–22:33, 3 h 32)</summary>

<dl><dd>

**Stream `strict6`** (B=10, G=3, R=2):

| batch | incoming → committed | cwa | joint | transductive joint | pool | wall |
|---|---|---|---|---|---|---|
| 0 | bare → b0r1g1 | 7 | 0 | 100 | 5 | 2577 s |
| 1 | b0r1g1 → b1r1g0 | 97 | 90 | 90 | 6 | 1080 s |
| 2 | b1r1g0 → b2r0g1 | 100 | 100 | 100 | 6 | 2188 s |

**Stream `labeled6`** reaches 100/100 in batch 0 and commits no change thereafter.

**Frozen evaluation on 200 held-out instances**, no proposer and no judge: bare circuit-wise accuracy
11.8 and joint 0.0; the strict final `b2r0g1` **99.5 / 98.5** (98.2 on the 170 unseen); `b1r1g0`
99.3 / 98.0; the labeled final 100/100 at **215 tokens per episode**.

**Audit.** 29 of 32 sessions clean, and **every judge session clean**. The judge rejected three
harnesses that scored 10/10 because they presented the model with a menu or a ranking, enforcing
that **the model writes every query**. This is the dividing line for whether the mechanism holds, not
a matter of style.

**strict ≈ labeled** is the direct evidence for the label-free claim: both streams finish at the
ceiling, and the strict one needs one additional batch.

</dd></dl>
</details>

<details>
<summary>🔗 Traceability (B3 / B3)</summary>

<dl><dd>

- Loop: [`meta/cd_tthe/optimize.py`](../meta/cd_tthe/optimize.py) ·
  [`proposer.py`](../meta/cd_tthe/proposer.py) (prompts, `audit_session`, subscription-limit retry) ·
  [`solve_worker.py`](../meta/cd_tthe/solve_worker.py) (reveal levels) ·
  [`audit.py`](../meta/cd_tthe/audit.py) (static audit)
- Evaluation: [`meta/cd_tthe/evaluate.py`](../meta/cd_tthe/evaluate.py) ·
  [`trajectory.py`](../meta/cd_tthe/trajectory.py)
- Details: 📁 [`09_cd_tthe_isolated`](../experiments/09_cd_tthe_isolated/)
- Transcripts in each folder's `dialogues/`; verdicts in `judge_verdicts/`; audits in
  `session_audits.jsonl`
- Raw traces: Release `cd_tthe_isolated_raw.tar.zst`

</dd></dl>
</details>

</dd></dl>

</details>

<details>
<summary><b>B4 Cross-task transfer: AR-Bench Situation Puzzles (10-13 05:03–07:17, 2 h 14)</b></summary>

<dl><dd>

The same loop on a task with **no enumerable hypothesis space**. The questioner is the frozen
Qwen2.5-7B; the referee and key-question judge are the official Llama-3.1-405B-INT4; the protocol is
AR-Bench's own 25 yes/no questions followed by a written answer. A 30-puzzle stream (seed 0),
B=10, G=3, R=2, **strict** signals: run status and referee answers only, no F1.

**Stream** (prequential = the incumbent on the new batch, before adaptation):

| batch | incoming → committed | F1 (char) | key-q recall | transductive F1 | pool |
|---|---|---|---|---|---|
| 0 | bare → b0r1g2 | 63.7 | 55 | 79 | 7 |
| 1 | b0r1g2 → b1r1g0 | 74.5 | 65 | 67 | 5 |
| 2 | b1r1g0 → b2r1g0 | 79.9 | 72 | 69 | 6 |

**Frozen evaluation on all 100 test puzzles**, 70 of which the stream never used:

| harness | F1, 100 | F1, unseen 70 | key-q recall, 100 | tok/ep |
|---|---|---|---|---|
| bare (AR-Bench's official zero-shot loop) | 60.4 | 61.4 | 59.7 | 670 |
| b0 committed | 76.7 | 75.7 | 59.0 | 4486 |
| b1 not selected (`b1r0g0`) | 73.0 | 73.8 | 64.3 | 4685 |
| b1 committed | 76.2 | 75.9 | 63.2 | 4944 |
| **b2 not selected (`b2r0g1`)** | 76.3 | **77.5** | 68.2 | **2112** |
| b2 committed (final) | 75.6 | 75.4 | **68.7** | 5263 |

**Label-free evolution from ten failure traces lifts the frozen 7B from 60.4 to 73–77 on puzzles it
has never seen**, above every zero-shot model in AR-Bench's own table including GPT-4o at 63, with
the 7B still writing every question. Key-question recall rises only in the later generations
(59 → 68), so **the first generation improves the answer, not the questioning**. The cost is 3–8×
the output tokens; the unchosen `b2r0g1` matches at half. **Read together with [A7](#)**, which
shows that a large part of the F1 column is answer length.

**What the harness learned** (the batch-0 winner, from the adversarial-audit proposer). Each turn
the model receives one fresh message: the puzzle, the referee's answers grouped as CONFIRMED /
RULED OUT / NOT IN THE STORY, and a focus. It first tests the assumptions a reader takes for granted,
since a situation puzzle usually turns on one of them being false; it then maintains a single
**working story** that must contain every confirmed fact, whose unconfirmed parts are tested one at a
time; after a run without a Yes it asks one broad question on an aspect not yet covered. Every
candidate question passes format checks — one fact, yes/no, no "or", not about beliefs, no invented
names, not a near-repeat — and the final answer is rewritten from the confirmed facts. Later
generations added vetoes on repeats and on leads the referee has already denied, a ban on entities
refused twice, and a ban on definite reference to unconfirmed guesses.

⚠️ **The judge is noisy here.** At batch 1 it committed the pool's lowest-F1 candidate (0.674 against
the incumbent's 0.745) and at batch 2 it passed over the best (0.802). On ten-puzzle batches those
gold differences are within noise, which **is itself the finding**; quantified in [B5](#). Session
audits: 18 of 21 clean, the three discards being miscounted relative paths that resolved outside the
folder but did not exist; the audit now checks existence.

</dd></dl>

<details>
<summary>🔗 Traceability (B4)</summary>

<dl><dd>

- Task adapters: [`meta/cd_tthe/tasks.py`](../meta/cd_tthe/tasks.py) ·
  [`sp_bridge.py`](../meta/cd_tthe/sp_bridge.py) (bottoms and key questions private) ·
  [`sp_harness_base.py`](../meta/cd_tthe/sp_harness_base.py) ·
  [`sp_signals.py`](../meta/cd_tthe/sp_signals.py) ·
  [`agents_sp/bare.py`](../meta/cd_tthe/agents_sp/bare.py) (the official zero-shot loop)
- Details: 📁 [`10_sp_tthe_pilot`](../experiments/10_sp_tthe_pilot/), including `judge_verdicts/`,
  `proposal_cards/`, `candidates/`, `eval_test100_scores.json`, `trajectory.png` and
  [`dialogues/`](../experiments/10_sp_tthe_pilot/dialogues/) (four puzzles, bare against evolved)
- Raw traces: Release `sp_tthe_pilot_raw.tar.zst` (17 MB)
- Cost: 21 proposer and judge sessions, about **$12** at subscription list price, 2 h 14 min

</dd></dl>
</details>

</details>

<details>
<summary><b>B5 A ruler for the judge inside the loop (10-13)</b></summary>

<dl><dd>

TurtleBench measures the **referee**. The **judge** that decides which harness is deployed next had
no instrument. It needs no external benchmark: every run stores the candidate pool's gold scores,
which that judge never saw, beside the candidate it committed.
Script: [`runner/judge_pick_eval.py`](../runner/judge_pick_eval.py).

| run | task | committed a best-scoring candidate | mean regret | worst | mean rank |
|---|---|---|---|---|---|
| `strict6_b10g3r2` | CD | 2/3 | 0.011 | 0.033 | 2.00 |
| `labeled6_b10g3r2` | CD | 3/3 | 0.000 | 0.000 | 1.00 |
| `sp_strict1_b10g3r2` | SP | **1/3** | **0.063** | **0.114** | 3.67 |

Regret is expressed in each task's own metric. The Circuit Decoding pools saturate — six candidates
at 1.000 make every choice correct — so the measurement says little there. **Situation Puzzles is
where it bites**: the judge gave up 0.075 and 0.114 F1 on its two poor calls, and at batch 1 it
committed the worst of five.

**This belongs in the paper.** The label-free judge is a component of the mechanism we claim, and on
the harder task it is measurably noisy. Part of the cause is ten-puzzle batches, where those gold
gaps are inside the noise — which is also a finding.

**External instruments that would fit, none yet run.** The referee's decision is three-way grounded
inference, which is well benchmarked outside puzzle games: **FEVER** (SUPPORTED / REFUTED / NOT
ENOUGH INFO, the same three labels with NEI playing Unknown), **ANLI and MNLI** (entailment /
contradiction / neutral, with no game idiom, separating reasoning from format), and **SQuAD 2.0's
unanswerable half** (isolating the decidability sub-decision alone). **JudgeBench** and
**RewardBench 2** evaluate the other kind of judge, choosing between candidate responses; closer in
spirit but a different decision, so they would at most be a sanity check. The first three are public
and need no login.

</dd></dl>

</details>

> ---

<details>
<summary><b>B6 2026-10-26, see children — GN context and information acquisition</b></summary>

<dl><dd>

```text
B6 GN context / mechanism analysis — frozen Qwen2.5-7B-Instruct
├── B6.1 Note updates without performance selection: negative result
├── B6.2 Propose → evaluate → retain: selected-policy improvement
│   ├── Two trace-guided learning blocks
│   └── Paired fitness-only search controls
└── GN/SP baseline and literature-method comparisons are reported separately in B7
```

**Metric boundary.** Capped identification turns **(ours)** means the first query after which exact public feedback leaves one possible secret, capped at 25; unresolved cases remain at 25. The actor does not see candidate lists/counts, and there is no final-answer call. This measures information sufficiency, not the actor's internal belief or AR-Bench's official final-answer exact match. It extends the questioning/readout distinction in A8 through a separate GN experiment; it does not reproduce the complete SP harness or resolve A8's full-system limitations.

Wall times below are recorded scheduler parent allocation elapsed times on one L40S, including startup and shutdown. They exclude preparation and CPU audits; they are not researcher work time.

<details>
<summary>B6.1 2026-10-26, 434 s — three-chain note updates</summary>

<dl><dd>

Three chains share 18 training cases with different fixed orders/seeds; each makes six updates. All three final notes, no note and a separate rules-only note are tested on the same 100 internal evaluation cases. No performance-based update selection is used.

| contrast | mean capped-query difference (ours; lower is better) | 95% paired interval | traceback: script → input → result/log |
|---|---:|---|---|
| mean(E0,E1,E2) − no note | **+1.7633** | [0.1933, 3.2767] | [verify_results.py](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_question_efficiency/verify_results.py) → [anonymous cases](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_question_efficiency/anonymous-cases.json) → [report, audit reference and figure](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_question_efficiency/README.md). Original execution journal is private. |

The tested updates worsened efficiency and average identification coverage. The most favorable individual chain is not selected as a success. Intervals condition on three realized chains sharing one training set.

</dd></dl>
</details>

<details>
<summary>B6.2 2026-10-26, 784 s — selected text policies</summary>

<dl><dd>

Two learning blocks pair trace-guided E with fitness-only R. Each search has six proposal opportunities and uses its block's same 48-case selection bank; promotion requires at least one fewer mean capped query without reduced identification count. All four final policies and no note are then frozen and tested on two 100-case cohorts.

| contrast | mean capped-query difference (ours; lower is better) | interval | traceback: script → input → result/log |
|---|---:|---|---|
| pooled mean(E0,E1) − no note | **−3.3175** | [−4.2150, −2.4575], 95% descriptive | [verify_results.py](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/verify_results.py) → [anonymous cases](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/anonymous-cases.json) → [results](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/results.json) / [verification log](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/verification-result.json). Original execution journal is private. |

All four within-study primary 98.75% intervals favored E over no note or R. The two cohorts evaluate the same frozen policies, not additional learning runs. The notes are useful fixed query schedules, not evidence of broad adaptive reasoning. R also receives performance-based selection, although neither realized R search promoted a proposal. Training compute is not matched; lower evaluation query cost is not lower total cost. The cross-study change does not isolate selection alone as the cause, and current-study multiplicity control does not cover the earlier outcome-informed research history.

</dd></dl>
</details>

<details>
<summary>🔗 Traceability (B6)</summary>

<dl><dd>

- Reference branches: [negative study](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_question_efficiency/README.md) · [selected-policy study and figure](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/README.md).
- Selection mechanism and frozen prompts: [portable code](https://github.com/sun-tze/self-evolving-active-reasoning/tree/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/code/) · [all proposal dispositions](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/learning-lineage.json) · [cost accounting](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/COSTS.md).
- [Reproduction guide](https://github.com/sun-tze/self-evolving-active-reasoning/blob/c1defaf4a911d9973a43e424beb357d74f64379e/experiments/sun_gn_selected_policy/REPRODUCE.md) separates exact public arithmetic from fresh-data method replication. Execution audits and public arithmetic checks passed within the same team; neither is external replication. Public arithmetic does not independently reconstruct private run provenance.
- Delivery: [team PR #2](https://github.com/Zarreboln/self-evolving-active-reasoning/pull/2), separate prior contribution; its files are not duplicated here.

</dd></dl>
</details>

**Related comparison.** B7 records the later incomplete GN comparison and completed exploratory SP pilot. Their native final-answer/overlap metrics are separate from the capped-identification metric above.

</dd></dl>
</details>

<details>
<summary><b>B7 2026-10-26, see children — native GN and exploratory SP method-adapter pilots</b></summary>

<dl><dd>

```text
B7 Native task algorithms plus DC/ACE memory cores
├── B7.1 GN: failed full comparison; complete dev curve and partial final prefix
└── B7.2 SP: completed small exploratory run; referee qualification remains UNMET
```

These bounded pilots use pinned upstream memory cores with task adapters. They do not reproduce the complete papers/CLIs or resolve the requirements for a full baseline matrix. Each method has one realized learning trajectory. This section is separate from B6's custom capped-identification studies and does not pool their metrics.

<details>
<summary>B7.1 2026-10-26, 5,648.25 s allocation — incomplete GN comparison</summary>

<dl><dd>

Qwen2.5-7B-Instruct compared ZeroShot, FewShot, FewShotInst, an explicitly repaired ToT parser, ProactiveCoT, DC-Cumulative and supervised ACE with a 512-token actor cap. DC receives public interactions; ACE's training reflector sees the target. The supervision conditions differ.

The original run **failed** at request admission. All 30 development slots completed, but only **97/140 final slots** did: one final episode was interrupted and 42 never started. Missing scores are not zero. Separate partial forensic checks passed for available artifacts and native scoring; this is not a full-run audit PASS. Upstream DC/ACE merge was not independently reimplemented.

| descriptive observation | denominator / boundary | traceback: script → input → result |
|---|---|---|
| DC and ACE final-actor exact match both 0/5 at updates 0, 3 and 6 | Same five reused development cases, one trajectory/method; not 15 independent cases | [verifier](../experiments/sun_gn_sp_comparison/gn/verify_results.py) → [anonymous rows](../experiments/sun_gn_sp_comparison/gn/anonymous-numerics.json) → [results and SVG](../experiments/sun_gn_sp_comparison/gn/README.md) |
| Common final prefix: 13 cases × seven methods | Posthoc complete prefix; six additional completed rows remain in the coverage count; time/order censoring prevents clean ranking | [coverage and limits](../experiments/sun_gn_sp_comparison/gn/integrity-summary.json) |
| DC: 6/6 curator outputs capped; after first update, 0/5 body changes excluding Version line | Descriptive extraction, not isolated cap/extractor/prompt/model diagnosis | [numeric diagnostics](../experiments/sun_gn_sp_comparison/gn/results.json) |
| ACE: initial wrong → target-informed reflection correct → after update wrong, on all six training cases | Dependent supervised stages, not heldout gains and not proof that merging caused loss | [method/provenance](../experiments/sun_gn_sp_comparison/METHODS.md) |

Native success, explicit final-actor exact match and any-time correct guesses are distinct. Unassessed final answers stay null. No significance test or confidence interval is assigned to the incomplete final prefix. Allocation time includes startup/shutdown and is not researcher work time.

</dd></dl>
</details>

<details>
<summary>B7.2 2026-10-26, 2,903.56 s active / 5,670.70 s full wall — unqualified SP pilot</summary>

<dl><dd>

All roles used Kimi K2.6. Three training, two development and five final cases were disjoint. DC/ACE development was measured at steps 0/1/3; final memories were fixed at step 3, not selected by the best development result. The run completed 33 episodes, six updates and 1,692 generation calls across three segments. Two earlier failures remain recorded; the active-time allowance was explicitly amended from 45 to 60 minutes. Active time excludes repair/wait time.

| method | development character F1, step 0 → 1 → 3 (same N=2) | final character F1 (N=5) | diagnostic word F1 (N=5) |
|---|---|---:|---:|
| ZeroShot | not scheduled | 0.6962 | 0.3109 |
| DC-Cumulative | 0.8855 → 0.8890 → 0.7134 | 0.8353 | 0.3090 |
| ACE-Public-Feedback-SP | 0.8672 → 0.8313 → 0.8857 | 0.8200 | 0.3197 |

Traceback: [standard-library verifier](../experiments/sun_gn_sp_comparison/sp/numeric/verify_results.py) → [27 anonymous evaluation rows](../experiments/sun_gn_sp_comparison/sp/numeric/anonymous-episodes.json) → [results, SVG and protocol](../experiments/sun_gn_sp_comparison/sp/README.md).

Mechanical audit was PASS_WITH_LIMITS (38,260 checks, zero issues). **Referee qualification remains UNMET.** Character F1 measures literal overlap, not accuracy. Flat word F1 does not prove that the character difference is spurious or caused by length. Same-family judging, five final cases, one learning trajectory and source quality prevent general ranking/semantic-solving claims.

Both learners see public interaction only. ACE-Public-Feedback-SP is a named adaptation: one no-ground-truth reflection and one curator/merge; no correctness-triggered regeneration. It is not the supervised ACE condition above. All three DC curators and ACE reflectors reached their caps, as did one ACE curator; truncations were retained.

A fixed 30-reply post-run AI/source review found **23 supported, 1 contradicted, 6 unresolved**. It is not human gold, a random sample or an error-rate estimate; original scores are unchanged. [Aggregate review](../experiments/sun_gn_sp_comparison/sp/feedback-source-review.json). Generation usage-price estimate: **$1.97990950**, excluding unverified estimator fees and settlement effects; not an all-fee invoice. [Usage](../experiments/sun_gn_sp_comparison/sp/figures/sp-actual-usage.json).

</dd></dl>
</details>

**Next falsifiable comparison, not yet run.** Freeze a finite bank of full ACE snapshots and compare the initial, latest and separately validation-selected snapshot on common fresh heldout cases with the same monitor and explicit total cost. The current observations motivate this test but do not establish why an update failed. Complete GN coverage, suitable SP feedback qualification and replicated learning trajectories remain outstanding.

[Minimal result package and source references](../experiments/sun_gn_sp_comparison/README.md). Exact public arithmetic is reproducible; private run provenance was checked within the execution team, not externally replicated. Raw cases, targets, prompts, transcripts and coordination records are not published.

</dd></dl>
</details>

# Operations: the two machines and the traps

<details>
<summary><b>Capability boundaries and known pitfalls</b></summary>

<dl><dd>

| | MIT ORCD / Engaging | standalone H100 (`ssh h100`) |
|---|---|---|
| GPUs | H200 141 GB, at most 2 on `mit_normal_gpu` | 4×H100 80 GB, **shared with other users**, chosen at launch by free memory |
| Queue and preemption | queued; the 405B runs on a preemptable partition and was preempted twice on 10-13 | no queue, no preemption; a model that fits serves in 1–3 minutes |
| Software | current vLLM | `venv_cu128` (vLLM 0.19.1, torch 2.11+cu128); driver 570.195.03 is CUDA 12.8, so the **ceiling is vLLM 0.26.0** |
| Does not fit | — | Llama-3.1-405B-INT4 (needs two cards), Qwen3.8-Flash-Next (~126 GB resident), Qwen3-235B |

- **Engaging requires Duo.** Public key and Duo are both required and no token can be stored, so the
  ControlMaster connection must be re-established by hand after it drops.
- **Only `venv_cu128` works on the H100.** The other virtual environment carries a CUDA 13 build of
  torch and fails with "NVIDIA driver is too old".
- **`ninja` is an inference-time dependency.** Models with Gated DeltaNet (Qwen3.8) compile their
  attention kernel on the first forward pass. Without it the server starts, passes `/health`, then
  returns 500 on every request as the engine dies.
- **A model's name is not its architecture.** Qwen3.8-27B's Hugging Face `architectures` field reads
  `Qwen3_5ForConditionalGeneration`, which vLLM 0.19.1 already registers; searching the registry for
  "3.8" gives the wrong answer. Check `config.json`.
- **Never give `pgrep -f` or `pkill -f` a pattern contained in your own command line.** It matches
  the invoking shell. This happened twice: once deadlocking a waiter against itself, once killing
  the session that issued it.
- **vLLM 0.19 silently ignores `guided_choice`**, succeeding while constraining nothing.
- **Claude Code's five-hour subscription window voids sessions silently**, so the proposer detects
  the limit and `--resume-from BATCH ROUND` continues the stream.
- 2026-10-21: the H100's SSH host key changed. **Not accepted**; pending out-of-band confirmation
  that the machine was rebuilt.

Machine notes: 📁 [`h100/README.md`](../h100/README.md).

</dd></dl>

</details>


<details>
<summary>B7.3 2026-10-26, 2,161 s suffix allocation — GN final matrix completed</summary>

<dl><dd>

This addendum preserves B7.1's failed original and partial report. A separately authorized suffix completed only the previously unissued work: **30/30 development and 140/140 final slots**, with 12 updates. The final memories remain fixed at update 6. Cached original responses are not new inference: 4,032 original + 1,096 new = **5,128 actual calls**.

DC-Cumulative scored **0/20** native final success and supervised ACE **1/20**. Both learners had **0/5 development exact match at updates 0, 3 and 6**, on the same five reused cases. Completing the missing cells resolves coverage, but does not establish a learning benefit. Full-text changes are not evidence of improved heldout behavior, and these small counts do not support general method ranking. The final-actor, native-success and singleton/query-efficiency endpoints must remain distinct.

[Complete anonymous results and figures](../experiments/sun_gn_sp_comparison/gn-complete/README.md) · [Interpretation and limits](../experiments/sun_gn_sp_comparison/gn-complete/INTERPRETATION.md) · [Standard-library arithmetic verifier](../experiments/sun_gn_sp_comparison/gn-complete/verify_results.py). This was posthoc segmented completion after partial outcomes were visible, not a fresh replication. The original run and two earlier failed audit records remain unchanged; the separate corrected audit passed integrity checks.

**Practical value of the negative result:** neither learner showed development exact-match improvement, so successful memory generation alone is not a reason to expand compute. The completed matrix also prevents the 43 previously missing final cells from being silently treated as zeros or omitted from the conclusion. It does not isolate a prompt, truncation, extraction, merge, or model cause.

The historical [singleton/query studies in PR 2](https://github.com/Zarreboln/self-evolving-active-reasoning/pull/2) use different protocols and endpoints. UoT, SFT, DPO, qualified SP feedback and replicated learning trajectories remain outstanding. Existing SP results remain exploratory with an unqualified same-family referee. No new experiment or method-selection result is claimed by this addendum.

</dd></dl>
</details>



<details>
<summary>B7.4 2026-10-26, native UoT cancelled at 3/20 — offline diagnosis and a bounded concurrent runtime</summary>

<dl><dd>

Native UoT (depth 1, Qwen2.5-7B-Instruct, 512-token cap, original 20-case order and seeds) was cancelled at the owner's request after about 4,780 s with **3/20 cases complete**, 483 call intents and 482 responses. It is not a UoT baseline result, and no 20-case audit was run. Calls were strictly serial (about 9.6 s each) because the model boundary held its lock across the transport.

**Offline diagnosis of the recorded decisions (no model calls).** The first diagnostic stopped on a case-binding check that compared the batch's re-serialized case copy with the raw hash of the package original. The corrected version checks the recorded raw → canonical → copy chain and replays every recorded prompt, reproducing all three saved results exactly. Over 24 complete decisions:

- 9 native picks re-asked an already-asked guess, which carries zero information; in 8 of them all three parsed candidates were repeats.
- The native parser keeps the first three 4-digit strings. In 14/23 proposals the first number was restated history; those responses contained 269 later unasked numbers that were dropped.
- 393/457 simulation responses hit the 512-token cap. The model listed at most 29 survivors where exact buckets reached 1,440, so uninformative, concentrated partitions received inflated scores.
- The three completed cases are official successes only through the harness's automatic singleton; no actor query hit the target.

[Diagnostic, aggregates and limits](../experiments/sun_gn_sp_comparison/uot/diagnostics/README.md). These describe recorded states only, not counterfactual success or question savings.

**Runtime.** A replacement boundary runs independent cases in parallel with the native code unchanged: atomic short-lock admission, bounded in-flight calls, fail-closed halt, uncertain-cost accounting and a hash-chained journal. Offline tests give byte-identical serial and 4-way native results on synthetic cases. A replay throughput probe and gate are prepared but not run. At any plausible 8-way speedup a full 20-case native run needs about 69–112 min (recorded mean vs pessimistic), more than the remaining conservative budget. [Runtime, probe and gate](../experiments/sun_gn_sp_comparison/uot/runtime/README.md).

</dd></dl>
</details>
