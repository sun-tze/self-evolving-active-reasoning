"""Bounded local vLLM client. Default is a zero-network plan, never inference.

No target loading, APIs requiring keys, redirects, proxies, or automatic retries.
An append-only started/result journal prevents reissuing ambiguous interrupted calls.
"""
import argparse
from datetime import datetime, timezone
import http.client
import os
from pathlib import Path
import random
import socket
import threading
import time
import urllib.parse
from common import (HERE, MODEL, canonical, decode, digest, load_bundle, read_json,
                    read_jsonl, require, sha, write_json)


def endpoint(value):
    p = urllib.parse.urlsplit(value)
    require(p.scheme == "http" and p.hostname in ("127.0.0.1", "::1") and p.port is not None
            and not p.username and not p.password and p.path == "/v1" and not p.query and not p.fragment,
            "only explicit loopback http://127.0.0.1:PORT/v1 endpoints are supported; use an SSH tunnel")
    return value


def http_json(url, body=None, timeout=120):
    p = urllib.parse.urlsplit(url)
    require(p.scheme == "http" and p.hostname in ("127.0.0.1", "::1") and p.port
            and not p.username and not p.password and not p.query and not p.fragment, "invalid local HTTP URL")
    require(timeout > 0, "HTTP deadline exceeded")
    # http.client does not follow redirects or consult proxy environment variables.
    connection = http.client.HTTPConnection(p.hostname, p.port, timeout=timeout)
    expired = threading.Event()
    active_socket = []
    def abort():
        expired.set()
        if active_socket:
            try:
                active_socket[0].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    timer = threading.Timer(timeout, abort)
    timer.daemon = True
    timer.start()
    try:
        connection.connect()
        active_socket.append(connection.sock)
        require(not expired.is_set(), "HTTP deadline exceeded during connection")
        connection.request("GET" if body is None else "POST", p.path,
                           body=None if body is None else canonical(body),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        require(response.status == 200, "HTTP status " + str(response.status) + "; redirects forbidden")
        raw = response.read(4 * 1024 * 1024 + 1)
        if expired.is_set():
            raise TimeoutError("absolute HTTP deadline exceeded")
    except Exception as exc:
        if expired.is_set():
            raise TimeoutError("absolute HTTP deadline exceeded") from exc
        raise
    finally:
        timer.cancel()
        connection.close()
    require(len(raw) <= 4 * 1024 * 1024, "oversized server response")
    return decode(raw)


def parse_completion(raw, request):
    require(type(raw) is dict and raw.get("model") == MODEL, "server model mismatch")
    choices = raw.get("choices")
    require(type(choices) is list and len(choices) == 1, "expected one completion")
    choice = choices[0]
    text = choice["message"]["content"]
    finish = choice["finish_reason"]
    require(type(text) is str and finish in ("stop", "length"), "invalid text or finish reason")
    require(not choice["message"].get("tool_calls"), "unexpected tool call")
    usage = raw.get("usage")
    result = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    if usage is not None:
        require(type(usage) is dict, "invalid usage")
        for key, remote in [("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens"), ("total_tokens", "total_tokens")]:
            val = usage.get(remote)
            require(val is None or type(val) is int and val >= 0, "invalid token usage")
            result[key] = val
        if all(v is not None for v in result.values()):
            require(result["input_tokens"] + result["output_tokens"] == result["total_tokens"], "usage sum mismatch")
        require(result["output_tokens"] is None or result["output_tokens"] <= request["payload"]["max_tokens"], "output cap exceeded")
    return {"text": text, "finish_reason": finish, "truncated": finish == "length", "usage": result}


def schedule(rows, indices):
    # Same-case blocks with seeded randomized cell order; no score-based selection.
    rng = random.Random(62026)
    order = sorted(indices)
    rng.shuffle(order)
    result = []
    for i in order:
        block = sorted([r for r in rows if r["index"] == i], key=lambda r: r["cell"])
        rng.shuffle(block)
        result.extend(block)
    return result


def plan(bundle, manifest_sha, budget_file, scope, base_url, max_calls, max_input, max_output, server_marker=None):
    manifest, rows = load_bundle(bundle, manifest_sha)
    budget = read_json(budget_file)
    require(budget["format"] == "c2_token_budget_v1" and budget["manifest_sha256"] == manifest_sha, "budget binding mismatch")
    require(budget["model_lock"] == read_json(HERE / "model_lock.json"), "budget tokenizer mismatch")
    counts = budget["request_input_tokens"]
    require(set(counts) == {r["request_id"] for r in rows} and
            all(type(n) is int and 0 < n <= 15984 for n in counts.values()), "invalid token counts")
    selected = schedule(rows, manifest["pilot_indices"] if scope == "pilot" else manifest["unseen_indices"])
    totals = {"calls": len(selected), "input_tokens": sum(counts[r["request_id"]] for r in selected),
              "max_output_tokens": sum(r["payload"]["max_tokens"] for r in selected)}
    require(totals["calls"] <= max_calls and totals["input_tokens"] <= max_input and
            totals["max_output_tokens"] <= max_output, "planned run exceeds approved caps")
    doc = {"format": "c2_run_plan_v1", "manifest_sha256": manifest_sha, "budget_sha256": sha(Path(budget_file).read_bytes()),
           "scope": scope, "indices": sorted({r["index"] for r in selected}), "endpoint": endpoint(base_url),
           "server_marker": server_marker,
           "schedule": [r["request_id"] for r in selected], "totals": totals,
           "model_lock": budget["model_lock"], "runner_sha256": sha(Path(__file__).read_bytes()),
           "common_sha256": sha((HERE / "common.py").read_bytes()), "retries": 0}
    return doc, selected, counts


def append_event(path, obj):
    with Path(path).open("ab") as f:
        f.write(canonical(obj) + b"\n")
        f.flush()
        os.fsync(f.fileno())


def journal_state(events, selected):
    allowed = {r["request_id"]: r for r in selected}
    started, finished = set(), set()
    for event in events:
        rid = event["request_id"]
        require(rid in allowed and event["payload_sha256"] == allowed[rid]["payload_sha256"], "journal request mismatch")
        if event["event"] == "started":
            require(rid not in started, "duplicate attempted request")
            started.add(rid)
        elif event["event"] == "result":
            require(rid in started and rid not in finished, "unpaired/duplicate result")
            require(event["status"] in ("completed", "failed"), "invalid result status")
            finished.add(rid)
        else:
            raise ValueError("unknown journal event")
    return started, finished


def execute(doc, selected, counts, out, authorization, max_seconds, timeout, resume=False):
    require(type(authorization) is str and authorization.strip(), "explicit human authorization reference required")
    require(type(max_seconds) is int and max_seconds > 0 and 0 < timeout <= max_seconds, "invalid time caps")
    out = Path(out)
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "run.json", {"plan": doc, "plan_sha256": digest(doc), "authorization": authorization,
                    "max_seconds": max_seconds, "timeout": timeout,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
    run = read_json(out / "run.json")
    require(run["plan"] == doc and run["plan_sha256"] == digest(doc) and run["authorization"] == authorization
            and run["max_seconds"] == max_seconds and run["timeout"] == timeout, "resume configuration mismatch")
    lock = out / "RUNNING.lock"
    # A stale lock must be inspected manually; never bypass an active/uncertain run.
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        journal = out / "journal.jsonl"
        events = read_jsonl(journal) if journal.exists() else []
        started, finished = journal_state(events, selected)
        require(started == finished, "ambiguous interrupted request: refusing automatic retry")
        require(not any(e.get("status") == "failed" for e in events), "failed run requires review; no retries")
        # Deadline persists across resumes; authorization cannot silently renew time.
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(run["created_utc"])).total_seconds()
        remaining = max_seconds - elapsed
        require(remaining > 0, "run deadline exceeded")
        deadline = time.monotonic() + remaining
        def call(url, body=None):
            left = deadline - time.monotonic()
            require(left > 0, "run deadline exceeded")
            return http_json(url, body, min(timeout, left))
        models = call(doc["endpoint"] + "/models")
        require(any(x.get("id") == MODEL for x in models["data"]), "expected model not served")
        if doc.get("server_marker"):
            require(any(x.get("id") == doc["server_marker"] for x in models["data"]),
                    "wrong vLLM service: unique job marker not served")
        # vLLM's tokenizer, not a character heuristic, is the final input-budget authority.
        for r in selected:
            tokenized = call(doc["endpoint"][:-3] + "/tokenize",
                            {"model": MODEL, "messages": r["payload"]["messages"], "add_generation_prompt": True})
            require(type(tokenized.get("count")) is int and tokenized["count"] == counts[r["request_id"]], "server/local tokenizer mismatch")
        for r in selected:
            if r["request_id"] in finished:
                continue
            require(time.monotonic() < deadline, "run deadline exceeded")
            base = {"request_id": r["request_id"], "payload_sha256": r["payload_sha256"]}
            append_event(journal, dict(base, event="started", utc=datetime.now(timezone.utc).isoformat()))
            t0 = time.monotonic()
            raw = None
            try:
                # The network body is EXACTLY the allowlisted payload. No envelope/provenance/targets.
                raw = call(doc["endpoint"] + "/chat/completions", r["payload"])
                parsed = parse_completion(raw, r)
                require(parsed["usage"]["input_tokens"] in (None, counts[r["request_id"]]), "returned prompt usage mismatch")
                append_event(journal, dict(base, event="result", status="completed", raw=raw,
                             parsed=parsed, error=None, elapsed_s=time.monotonic() - t0))
            except Exception as exc:
                append_event(journal, dict(base, event="result", status="failed", raw=raw, parsed=None,
                             error=type(exc).__name__ + ": " + str(exc), elapsed_s=time.monotonic() - t0))
                raise
        return len(selected)
    finally:
        lock.unlink()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--budget", required=True)
    p.add_argument("--scope", choices=["pilot", "all"], default="pilot")
    p.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    p.add_argument("--server-marker", help="unique extra served-model alias of the launched Slurm service")
    p.add_argument("--max-calls", type=int, default=20)
    p.add_argument("--max-input-tokens", type=int, default=50000)
    p.add_argument("--max-output-tokens", type=int, default=8000)
    p.add_argument("--max-seconds", type=int, default=1200)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--authorization-id")
    p.add_argument("--out")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    doc, selected, counts = plan(a.bundle, a.manifest_sha256, a.budget, a.scope, a.base_url,
                                a.max_calls, a.max_input_tokens, a.max_output_tokens, a.server_marker)
    if not a.execute:
        print(canonical(doc).decode())
    else:
        require(a.out is not None, "--out required")
        execute(doc, selected, counts, a.out, a.authorization_id, a.max_seconds, a.timeout, a.resume)
