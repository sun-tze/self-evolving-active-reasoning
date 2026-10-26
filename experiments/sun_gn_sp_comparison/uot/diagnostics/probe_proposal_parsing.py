"""Count how native UoT proposal parsing interacts with restated guess history.

The native parser keeps the first three 4-digit strings of a proposal response
(_gn_extract_multiple_guesses: re.findall(r"\\d{4}")[:3]). This probe counts, for
the saved complete episodes of the cancelled run, how often those three slots are
guesses the episode had already asked, and how many other distinct-digit,
not-yet-asked 4-digit strings appear later in the same response and are dropped.
A dropped string is not necessarily an intended proposal; only counts are
reported. Response text, guesses and targets are never written.

Standard library only; no model, network or GPU use. Only --output is written.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def probe(evidence):
    allocation = evidence / "allocation"
    journal = read_jsonl(allocation / "runtime-journal.jsonl")
    for previous, current in zip(journal, journal[1:]):
        if current["previous_sha256"] != previous["sha256"]:
            raise ValueError("Runtime previous-hash linkage differs")
    intents = [r["payload"] for r in journal if r["event"] == "call_intent"]
    responses = {r["payload"]["call_id"]: r["payload"]["response"]
                 for r in journal if r["event"] == "call_response"}
    controller = allocation / "controller"
    saved = [r["payload"] for r in read_jsonl(controller / "journal.jsonl")
             if r["kind"] == "episode_saved"]
    counts = dict(proposals=0, proposals_truncated_at_cap=0,
                  proposals_with_asked_guess_in_parsed_slots=0, asked_guess_parsed_slots=0,
                  proposals_all_parsed_slots_asked=0, proposals_first_number_is_asked_guess=0,
                  proposals_with_asked_slot_and_later_unasked_numbers=0,
                  later_unasked_distinct_digit_numbers_dropped=0)
    for save in saved:
        path = controller / save["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != save["sha256"]:
            raise ValueError("Saved episode hash differs")
        asked_sequence = [g[0] for g in
                          json.loads(path.read_text(encoding="utf-8"))["official_result"]["guess_list"]]
        proposals = [i for i in intents[save["call_start_exclusive"]:save["call_end_inclusive"]]
                     if i["role"] == "uot_proposal"]
        if len(proposals) > len(asked_sequence):
            raise ValueError("More proposals than asked guesses")
        for turn, intent in enumerate(proposals):
            response = responses[intent["call_id"]]
            asked = set(asked_sequence[:turn])
            numbers = re.findall(r"\d{4}", response["text"])
            parsed = numbers[:3]
            later = {n for n in numbers[3:]
                     if n not in asked and n not in parsed and len(set(n)) == 4}
            hits = sum(n in asked for n in parsed)
            counts["proposals"] += 1
            counts["proposals_truncated_at_cap"] += response["finish_reason"] == "length"
            counts["proposals_with_asked_guess_in_parsed_slots"] += hits > 0
            counts["asked_guess_parsed_slots"] += hits
            counts["proposals_all_parsed_slots_asked"] += len(parsed) == 3 and hits == 3
            counts["proposals_first_number_is_asked_guess"] += bool(numbers) and numbers[0] in asked
            counts["proposals_with_asked_slot_and_later_unasked_numbers"] += hits > 0 and bool(later)
            counts["later_unasked_distinct_digit_numbers_dropped"] += len(later)
    return {
        "status": "ANONYMOUS_PROPOSAL_PARSING_COUNTS",
        "scope": "saved complete episodes of the cancelled native depth-1 run; counts only",
        "saved_episodes": len(saved),
        "parser": "native first three 4-digit strings",
        "counts": counts,
        "limitations": [
            "A later unasked number is not necessarily an intended proposal.",
            "No repaired parser was run; no counterfactual trajectory is implied.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite " + str(args.output))
    result = probe(args.evidence)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
