"""Frozen prompts, official-compatible GN adapter, mechanical memory validation.

The model boundary receives only chat messages and generation parameters. It
does not receive a data path, controller secret, scores, or candidate sets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable

from .freeze import ROOT, seed, sha

CONTRACT_SHA = "66e190533ae646f450a288d222678481e6037df8e7594baa997d319f4460d238"
PROTOCOL_SHA = "73668cd7d02a4d6e0e0dc0e4f79d770ed45b55ae759a982c9bce256bdb974a55"
ADDENDUM_SHA = "813cf19bd8a97b035c1b29ab999315cca6abab1dd1e8ac13e1e1ec14e269684b"
URL_RE = re.compile(r"(?:[a-z][a-z0-9+.-]*://|www\.|\b[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.(?:[a-z]{2,})(?:\b|/))", re.IGNORECASE)


class IntegrityError(RuntimeError):
    pass


class NotNumberError(Exception):
    """The approved patch: missing regex match raises instead of returning []."""


def parse_guess(raw: str) -> str:
    matches = re.findall(r"\d{4}", raw)
    if not matches:
        raise NotNumberError()
    return "".join(str(int(c)) for c in matches[-1])


def feedback(secret: str, guess: str) -> tuple[int, int]:
    if len(secret) != len(guess):
        raise ValueError("Lists must have the same length")
    same = sum(a == b for a, b in zip(secret, guess))
    common = sum((Counter(secret) & Counter(guess)).values())
    return same, common - same


@dataclass(frozen=True)
class Assets:
    prompts: dict
    m0: str
    manifest: dict

    @classmethod
    def load(cls, root: Path = ROOT):
        frozen = root / "frozen"
        raw = (frozen / "contract.json").read_bytes()
        if sha(raw) != CONTRACT_SHA or sha((root / "PROTOCOL.md").read_bytes()) != PROTOCOL_SHA:
            raise IntegrityError("Frozen contract/protocol changed")
        if sha((root / "PROTOCOL_ADDENDUM_1.md").read_bytes())!=ADDENDUM_SHA:
            raise IntegrityError("Paired-order addendum changed")
        from .sampling import policy
        policy()
        contract = json.loads(raw)
        for name, digest in contract["files"].items():
            if sha((frozen / name).read_bytes()) != digest:
                raise IntegrityError(f"Frozen asset changed: {name}")
        prompts = json.loads((frozen / "prompts.json").read_bytes())
        for name, digest in contract["prompt_value_hashes"].items():
            if sha(prompts[name]) != digest:
                raise IntegrityError(f"Prompt changed: {name}")
        return cls(prompts, (frozen / "M0.txt").read_bytes().decode("utf-8"),
                   json.loads((frozen / "manifest.json").read_bytes()))

    def system(self, condition: str, memory: str = "") -> str:
        if condition not in ("B0", "B1", "S", "E"):
            raise ValueError("Unknown condition")
        key = "propose_template_with_1_shot_inst" if condition == "B1" else "propose_template"
        base = self.prompts[key].format(turn=25)
        if condition in ("S", "E") and memory:
            return base + self.prompts["header"] + memory
        return base

    def curator_messages(self, memory: str, public_messages: list[dict]) -> list[dict]:
        # Build from the allowlisted public interface rather than serializing a controller record.
        clean = []
        for m in public_messages:
            if set(m) != {"role", "content"} or m["role"] not in ("user", "assistant") or not isinstance(m["content"], str):
                raise IntegrityError("Non-public field in curator interface")
            clean.append(dict(m))
        transcript = json.dumps(clean, ensure_ascii=True, separators=(",", ":"))
        return [{"role": "user", "content": self.prompts["curator"].format(
            rules=self.prompts["Game_rule"], memory=memory, transcript=transcript)}]


def validate_memory(raw: str, finish_reason: str, count: Callable[[str], int]) -> dict:
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.split("\n")
    reasons = []
    if not 1 <= len(lines) <= 12:
        reasons.append("line_count")
    if not normalized:
        reasons.append("empty")
    if any(not line.startswith("- ") or not line[2:].strip() for line in lines):
        reasons.append("bullet_format")
    if any(any(not 32 <= ord(c) <= 126 for c in line) for line in lines):
        reasons.append("non_printable_ascii")
    if re.search(r"[0-9]", normalized):
        reasons.append("digits")
    if "`" in normalized:
        reasons.append("backtick")
    if "|" in normalized:
        reasons.append("pipe")
    if URL_RE.search(normalized):
        reasons.append("url")
    tokens = count(normalized)
    if tokens > 512:
        reasons.append("token_cap")
    if finish_reason == "length":
        reasons.append("incomplete")
    return dict(normalized=normalized, accepted=not reasons, reasons=reasons,
                tokens=tokens, finish_reason=finish_reason, raw_sha256=sha(raw),
                normalized_sha256=sha(normalized))


def run_episode(assets: Assets, actor, *, case_id: str, secret: str, condition: str,
                memory: str, journal) -> dict:
    """Controller-only secret, exactly 25 interaction calls and one final.

    No diagnostics are computed here. `actor` has no filesystem access supplied
    by this API. Infrastructure faults propagate and halt the stage.
    """
    from time import monotonic
    started = monotonic()
    if len(secret) != 4 or len(set(secret)) != 4 or not all(c in "0123456789" for c in secret):
        raise IntegrityError("Invalid controller secret")
    system = assets.system(condition, memory)
    messages = [{"role": "system", "content": system}]
    slots = []
    for slot in range(1, 27):
        final = slot == 26
        messages.append({"role": "user", "content": assets.prompts["final_guess_prompt"] if final else assets.prompts["guess_prompt"].format(turn=slot)})
        response = actor.call(messages, seed(case_id, slot), "actor", dict(case_id=case_id, condition=condition, slot=slot))
        raw = response["text"]
        messages.append({"role": "assistant", "content": raw})
        record = dict(slot=slot, kind="final" if final else "interaction", raw=raw,
                      seed=seed(case_id, slot), response=response)
        try:
            guess = parse_guess(raw)
        except NotNumberError:
            record.update(guess=None, malformed=True, delivered_feedback=None)
            if not final:
                messages.append({"role": "user", "content": assets.prompts["refine_prompt"]})
                record["delivered_message"] = assets.prompts["refine_prompt"]
        else:
            record.update(guess=guess, malformed=False, repeated_digits=len(set(guess)) < 4,
                          delivered_feedback=None)
            if not final:
                s, d = feedback(secret, guess)
                rendered = assets.prompts["eval_prompt"].format(same_pos=s, diff_pos=d)
                messages.append({"role": "user", "content": rendered})
                record.update(delivered_feedback=[s, d], delivered_message=rendered)
        slots.append(record)
        journal.append("slot", dict(case_id=case_id, condition=condition, **record))
    result = dict(case_id=case_id, condition=condition, secret=secret, slots=slots,
                  public_messages=messages[1:], system_sha256=sha(system), memory_sha256=sha(memory),
                  completed=True, elapsed_seconds=monotonic() - started)
    journal.append("episode_complete", result)
    return result


def update_memory(assets: Assets, curator, *, step: int, case_id: str, memory: str,
                  public_messages: list[dict], count, journal, directory: Path) -> tuple[str, dict]:
    if not 1 <= step <= 20:
        raise ValueError("Adaptation update outside frozen range")
    messages = assets.curator_messages(memory, public_messages)
    response = curator.call(messages, seed(case_id, 0, "curator"), "curator", dict(case_id=case_id, step=step))
    validation = validate_memory(response["text"], response["finish_reason"], count)
    updated = validation["normalized"] if validation["accepted"] else memory
    record = dict(step=step, case_id=case_id, raw=response["text"], response=response,
                  validation=validation, old_sha256=sha(memory), new_sha256=sha(updated))
    journal.append("memory_update", record)
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive checkpoint creation prevents silent replacement or candidate selection.
    with (directory / f"M{step}.txt").open("xb") as f:
        f.write(updated.encode("utf-8"))
    with (directory / f"update_{step:02d}.json").open("x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=True, sort_keys=True, indent=2)
        f.write("\n")
    return updated, record
