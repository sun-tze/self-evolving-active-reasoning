"""Offline public-evidence diagnostics, never imported by the model boundary.

Credits: the full-space feasibility/entropy instrumentation follows Zining's
stage0/gn_oracle.py and runner/score_arbench_log.py at scaffold 84e8d21.
This implementation uses direct multiset feedback to cover repeated guesses,
all actual call slots, and the new prespecified Q rather than matrix efficiency.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations
from functools import lru_cache
from math import log2
import random
from statistics import mean

from .core import Assets, IntegrityError, NotNumberError, feedback, parse_guess
from .freeze import CONDITIONS

SPACE = tuple("".join(p) for p in permutations("0123456789", 4))


@lru_cache(maxsize=65536)
def candidate_feedback(candidate: str, guess: str):
    return feedback(candidate,guess)


def score_episode(record: dict, assets: Assets) -> dict:
    if record.get("completed") is not True or len(record["slots"]) != 26:
        raise IntegrityError("Only complete 26-call episodes may be scored")
    alive = set(SPACE)
    sizes, rows, seen, first_unique = [], [], set(), None
    hit = False
    final_em = False
    invalid = truncated = total_input = total_output = 0
    public = []
    for index, slot in enumerate(record["slots"], 1):
        final = index == 26
        if slot["slot"] != index or slot["kind"] != ("final" if final else "interaction"):
            raise IntegrityError("Slot numbering/kind")
        public.append(dict(role="user", content=assets.prompts["final_guess_prompt"] if final else assets.prompts["guess_prompt"].format(turn=index)))
        public.append(dict(role="assistant", content=slot["raw"]))
        try:
            parsed = parse_guess(slot["raw"])
        except NotNumberError:
            parsed = None
        if parsed != slot["guess"] or slot["malformed"] != (parsed is None):
            raise IntegrityError("Parser/log mismatch")
        invalid += parsed is None
        response = slot["response"]
        if response["text"] != slot["raw"]:
            raise IntegrityError("Response/raw mismatch")
        truncated += response["finish_reason"] == "length"
        total_input += response["prompt_tokens"]
        total_output += response["completion_tokens"]
        if final:
            if slot["delivered_feedback"] is not None or "delivered_message" in slot:
                raise IntegrityError("Final feedback must never be delivered")
            final_em = parsed == record["secret"]
            continue
        before = len(alive)
        row = dict(slot=index, n_before=before, malformed=parsed is None)
        if parsed is None:
            if slot["delivered_feedback"] is not None or slot["delivered_message"] != assets.prompts["refine_prompt"]:
                raise IntegrityError("Malformed slot feedback")
        else:
            observed = tuple(slot["delivered_feedback"])
            expected_text = assets.prompts["eval_prompt"].format(same_pos=observed[0], diff_pos=observed[1])
            if expected_text != slot["delivered_message"] or feedback(record["secret"], parsed) != observed:
                raise IntegrityError("Delivered feedback consistency failure")
            parts = Counter(candidate_feedback(candidate, parsed) for candidate in alive)
            entropy = -sum((n / before) * log2(n / before) for n in parts.values())
            after = {candidate for candidate in alive if candidate_feedback(candidate, parsed) == observed}
            if not after:
                raise IntegrityError("Empty feasible set; no clipping permitted")
            if before > 1:
                row.update(repeated_query=parsed in seen, infeasible=parsed not in alive,
                           zero_information=len(after) == before, expected_entropy=entropy)
            seen.add(parsed)
            alive = after
            hit |= parsed == record["secret"]
        public.append(dict(role="user", content=slot["delivered_message"]))
        if len(alive) == 1 and first_unique is None:
            first_unique = index
        sizes.append(len(alive))
        row["n_after"] = len(alive)
        row["normalized_accumulation"] = 1 - log2(len(alive)) / log2(5040)
        rows.append(row)
    if public != record["public_messages"]:
        raise IntegrityError("Incomplete or altered public transcript")
    return dict(case_id=record["case_id"], condition=record["condition"], Q=mean(r["normalized_accumulation"] for r in rows),
                final_EM=final_em, hit_any=hit, remaining_candidates=len(alive), first_unique_slot=first_unique,
                invalid_outputs=invalid, truncated_outputs=truncated, actor_calls=26,
                input_tokens=total_input, output_tokens=total_output, sizes=sizes, diagnostics=rows)


def percentile(sorted_values, p):
    at = (len(sorted_values) - 1) * p
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (at - lo)


def aggregate(planned_ids: list[str], scores: list[dict]) -> dict:
    if len(set(planned_ids)) != len(planned_ids) or not planned_ids:
        raise IntegrityError("Planned IDs must be unique and nonempty")
    planned = set(planned_ids)
    manifest_ids=[r["case_id"] for r in Assets.load().manifest["rows"] if r["stage"]=="evaluation"]
    if planned<=set(manifest_ids):
        planned_ids=[case for case in manifest_ids if case in planned]
    elif any(case.startswith(Assets.load().manifest["benchmark_commit"]+"/") for case in planned):
        raise IntegrityError("Non-evaluation benchmark case in aggregate")
    else:
        # Synthetic unit-test IDs have no benchmark manifest. Their stable
        # lexical order is test-only; live evaluation always takes the path above.
        planned_ids=sorted(planned_ids)
    cells = {}
    for rec in scores:
        key = rec["case_id"], rec["condition"]
        if key[0] not in planned or key[1] not in CONDITIONS or key in cells:
            raise IntegrityError("Duplicate or unplanned aggregate cell")
        if not 0 <= rec["Q"] <= 1:
            raise IntegrityError("Q outside definition")
        cells[key] = rec
    common = [i for i in planned_ids if all((i, c) in cells for c in CONDITIONS)]
    differences = [cells[i, "E"]["Q"] - cells[i, "S"]["Q"] for i in common]
    n, N = len(common), len(planned_ids)
    ci = None
    if n:
        rng = random.Random(62026)
        bootstrap = sorted(mean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(10000))
        ci = [percentile(bootstrap, .025), percentile(bootstrap, .975)]
    bounds = [(sum(differences)-(N-n))/N, (sum(differences)+(N-n))/N]
    table = {}
    for c in CONDITIONS:
        complete = [cells[i, c] for i in planned_ids if (i, c) in cells]
        paired = [cells[i, c] for i in common]
        table[c] = dict(completed=len(complete), paired_n=n,
                        paired_Q=mean(x["Q"] for x in paired) if paired else None,
                        completed_final_EM=sum(x["final_EM"] for x in complete),
                        paired_final_EM=sum(x["final_EM"] for x in paired),
                        completed_hit_any=sum(x["hit_any"] for x in complete),
                        invalid_outputs=sum(x["invalid_outputs"] for x in complete),
                        truncated_outputs=sum(x["truncated_outputs"] for x in complete))
    return dict(planned_N=N, paired_n=n, paired_case_ids=common,
                missing_cells=[dict(case_id=i,condition=c) for i in planned_ids for c in CONDITIONS if (i,c) not in cells],
                conditions=table,paired_delta_Q=differences,delta_Q=mean(differences) if differences else None, CI95=ci,
                full_N_bounds=bounds, descriptive_only=n<N,
                positive_primary_supported=bool(n==N and ci and ci[0]>0),
                wins=sum(x>0 for x in differences), losses=sum(x<0 for x in differences), ties=sum(x==0 for x in differences),
                bootstrap_seed=62026, bootstrap_replicates=10000,
                interpretation="Conditional on this single trajectory and seed scheme; Q measures public evidence accumulation.")
