"""Owner-private fsynced journal and bounded model boundary; no retry path."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time

from .core import IntegrityError
from .freeze import sha


def normalize_json_wire(value):
    """Use JSON's decoded representation before hashing, rejecting key aliases.

    JSON object keys are strings. Hashing a Python dict with numeric keys can
    sort it differently from its later decoded form. Normalize once at the
    writer boundary, then hash and persist that exact normalized object.
    """
    def unique_object(pairs):
        result={}
        for key,item in pairs:
            if key in result:
                raise IntegrityError("Ambiguous JSON object keys after normalization")
            result[key]=item
        return result
    return json.loads(json.dumps(value,ensure_ascii=True,allow_nan=False,separators=(",", ":")),
                      object_pairs_hook=unique_object)


class Journal:
    def __init__(self, path: Path, metadata: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("x", encoding="utf-8", newline="\n")
        self.previous = "0" * 64
        self.index = 0
        self.lock = threading.Lock()
        self.append("run_start", metadata)

    def append(self, event: str, payload: dict):
        with self.lock:
            row = normalize_json_wire(dict(index=self.index, event=event, payload=payload,
                                           previous_sha256=self.previous, time_ns=time.time_ns()))
            row["sha256"] = sha(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
            self.handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.previous = row["sha256"]
            self.index += 1

    def close(self):
        self.handle.close()


def read_journal(path: Path) -> list[dict]:
    previous = "0" * 64
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        digest = row.pop("sha256")
        if row["index"] != index or row["previous_sha256"] != previous or sha(json.dumps(row, sort_keys=True, ensure_ascii=True, separators=(",", ":"))) != digest:
            raise IntegrityError("Journal integrity failure")
        row["sha256"] = digest
        rows.append(row)
        previous = digest
    if not rows or rows[0]["event"] != "run_start":
        raise IntegrityError("Missing journal header")
    return rows


class CallBoundary:
    """Reservation, intent, transport, response, validation; no automatic retries.

    All shared counters and request dispatch are serial under the lock. This
    implementation supports safe serial calls; episode concurrency is added by
    independent boundaries only when sharing the same ledger (not a new cap).
    """
    def __init__(self, transport, tokenizer, journal: Journal, *, actor_limit: int,
                 curator_limit: int, input_limit: int, output_limit: int,
                 deadline: float, reserve_seconds: float):
        self.transport, self.tokenizer, self.journal = transport, tokenizer, journal
        self.actor_limit, self.curator_limit = actor_limit, curator_limit
        self.input_limit, self.output_limit = input_limit, output_limit
        self.deadline, self.reserve_seconds = deadline, reserve_seconds
        self.counts = dict(actor=0, curator=0)
        self.reserved_input = self.reserved_output = 0
        self.actual_input = self.actual_output = 0
        self.halted = False
        self.lock = threading.Lock()

    def call(self, messages: list[dict], seed: int, role: str, metadata: dict) -> dict:
        if role not in ("actor", "curator"):
            raise ValueError("Invalid model role")
        cap = 128 if role == "actor" else 768
        input_cap = 8064 if role == "actor" else 7424
        count = self.tokenizer.chat_count(messages)
        with self.lock:
            if self.halted:
                raise IntegrityError("Boundary halted; calls cannot be retried")
            if count > input_cap or count + cap > 8192:
                self.halted = True
                self.journal.append("stop", dict(reason="context_overflow", **metadata))
                raise IntegrityError("Context overflow; no truncation")
            limit = self.actor_limit if role == "actor" else self.curator_limit
            if (self.counts[role] >= limit or self.reserved_input + count > self.input_limit or
                    self.reserved_output + cap > self.output_limit or
                    time.monotonic() + self.reserve_seconds > self.deadline):
                self.halted = True
                self.journal.append("stop", dict(reason="budget", **metadata))
                raise IntegrityError("Hard resource limit")
            self.counts[role] += 1
            self.reserved_input += count
            self.reserved_output += cap
            request = dict(model="Qwen/Qwen2.5-7B-Instruct", messages=deepcopy(messages),
                           temperature=.7 if role == "actor" else 0, top_p=.7 if role == "actor" else 1,
                           max_tokens=cap, seed=seed)
            from .sampling import validate_wire
            validate_wire(request,role)
            call_id = f"{role}-{self.counts[role]}"
            self.journal.append("call_intent", dict(call_id=call_id, role=role, metadata=metadata,
                                                    input_tokens_counted=count, request=request))
        started = time.monotonic()
        try:
            raw = self.transport(deepcopy(request))
        except BaseException as error:
            with self.lock:
                self.halted = True
                self.journal.append("call_uncertain", dict(call_id=call_id, error_type=type(error).__name__, metadata=metadata))
            raise
        with self.lock:
            # Persist exactly what returned before interpreting it. No secret-bearing error strings.
            self.journal.append("call_response", dict(call_id=call_id, response=raw, elapsed_seconds=time.monotonic()-started))
            try:
                if set(raw) != {"text", "finish_reason", "prompt_tokens", "completion_tokens", "response_id"}:
                    raise IntegrityError("Response schema")
                if not isinstance(raw["text"], str) or not isinstance(raw["response_id"], str) or not raw["response_id"]:
                    raise IntegrityError("Response content/id")
                if raw["finish_reason"] not in ("stop", "length"):
                    raise IntegrityError("Unexpected service finish reason")
                if type(raw["prompt_tokens"]) is not int or raw["prompt_tokens"] != count:
                    raise IntegrityError("Service/tokenizer input count mismatch")
                if type(raw["completion_tokens"]) is not int or not 0 <= raw["completion_tokens"] <= cap:
                    raise IntegrityError("Service output token count")
            except Exception:
                self.halted = True
                self.journal.append("stop", dict(reason="response_integrity", call_id=call_id))
                raise
            self.actual_input += raw["prompt_tokens"]
            self.actual_output += raw["completion_tokens"]
            result = dict(raw, elapsed_seconds=time.monotonic()-started, call_id=call_id)
            return result
