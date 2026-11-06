"""Mock actor, key-gated Kimi transport, and local OpenAI-compatible Qwen transport.

Nothing in this module opens a socket at import time. ``KimiActor`` still
refuses to be constructed without a Moonshot key and still sends Kimi's
``reasoning_effort``. ``QwenActor`` does not use that key, does not send
``reasoning_effort`` or ``top_p``, and refuses Moonshot/Kimi URLs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener, urlopen

from .domain import DeterministicBelief

DEFAULT_QWEN_BASE_URL = "http://127.0.0.1:18000/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_QWEN_TEMPERATURE = 0.7
DEFAULT_QWEN_MAX_TOKENS = 800
FORBIDDEN_QWEN_HOST_MARKERS = ("moonshot", "kimi.ai", "kimi.com")


@dataclass
class ChatResponse:
    text: str
    usage: dict[str, Any]
    request_id: str | None = None
    finish_reason: str | None = None


@dataclass
class UsageLedger:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    requests: list[dict] = field(default_factory=list)

    def add(self, usage: dict[str, Any], *, arm: str, case_id: str, turn: int) -> None:
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") or usage.get("cached_tokens") or 0
        row = {
            "arm": arm, "case_id": case_id, "turn": turn,
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
            "cached_input_tokens": int(cached),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }
        self.calls += 1
        self.input_tokens += row["input_tokens"]
        self.output_tokens += row["output_tokens"]
        self.cached_input_tokens += row["cached_input_tokens"]
        self.requests.append(row)

    def public(self, arm: str | None = None) -> dict:
        rows = [row for row in self.requests if arm is None or row["arm"] == arm]
        return {"calls": len(rows),
                "input_tokens": sum(row["input_tokens"] for row in rows),
                "output_tokens": sum(row["output_tokens"] for row in rows),
                "cached_input_tokens": sum(row["cached_input_tokens"] for row in rows),
                "total_tokens": sum(row["total_tokens"] for row in rows)}


class MockActor:
    """Deterministic feasible-first actor for plumbing. Not an LLM result.

    It reads the host belief object and ignores the rendered prompt, so a mock
    matrix cannot measure a representation difference.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.usage = UsageLedger()

    def chat(self, messages: list[dict], *, seed: int, arm: str, case_id: str,
             turn: int, belief: DeterministicBelief | None = None, final: bool = False) -> ChatResponse:
        del messages, seed, final
        belief = belief or DeterministicBelief()
        alive = belief.feasible
        feasible_unasked = [candidate for candidate in alive if candidate not in belief.asked]
        guess = feasible_unasked[0] if feasible_unasked else alive[0]
        self.usage.add({"prompt_tokens": 0, "completion_tokens": 0},
                       arm=arm, case_id=case_id, turn=turn)
        return ChatResponse(text=f"Mock feasible choice.\nGuess: {guess}",
                            usage={"prompt_tokens": 0, "completion_tokens": 0},
                            finish_reason="stop")


class KimiActor:
    """Small stdlib HTTP client for Moonshot. Preserved; not the Qwen path.

    Timeout bounds each attempt. Authorization never enters records.
    Temperature is the historical Kimi value 1.0, and ``reasoning_effort`` is sent.
    """

    DEFAULT_REASONING_EFFORT = "low"
    ALLOWED_REASONING_EFFORT = frozenset({"low", "high", "max"})

    def __init__(self, model: str, base_url: str = "https://api.moonshot.ai/v1",
                 timeout: float = 600.0, max_tokens: int = 4096,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT) -> None:
        self.api_key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Live Kimi backend requires MOONSHOT_API_KEY (or KIMI_API_KEY alias); "
                "refusing to call without a key. Source the local alias only in your shell."
            )
        if reasoning_effort not in self.ALLOWED_REASONING_EFFORT:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(self.ALLOWED_REASONING_EFFORT)}"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.usage = UsageLedger()

    def chat(self, messages: list[dict], *, seed: int, arm: str, case_id: str,
             turn: int, belief: DeterministicBelief | None = None, final: bool = False) -> ChatResponse:
        del belief, final
        payload = {"model": self.model, "messages": messages, "seed": seed,
                   "temperature": 1.0, "max_tokens": self.max_tokens,
                   "reasoning_effort": self.reasoning_effort}
        request = Request(self.base_url + "/chat/completions", method="POST",
                          data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json",
                                   "Authorization": "Bearer " + self.api_key,
                                   "User-Agent": "arm0-kimi-local/0.1"})
        body = None
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    body = json.load(response)
                break
            except HTTPError as exc:
                exc.read()
                raise RuntimeError(f"Kimi HTTP {exc.code}; credentials never logged") from None
            except (TimeoutError, socket.timeout, URLError) as exc:
                last_err = exc
                if attempt == 2:
                    raise RuntimeError(f"Kimi transport failed after 3 attempts: {exc}") from None
                time.sleep(2.0 * (attempt + 1))
        if body is None:
            raise RuntimeError(f"Kimi transport failed: {last_err}") from None
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        self.usage.add(usage, arm=arm, case_id=case_id, turn=turn)
        return ChatResponse(text=choice["message"].get("content") or "", usage=usage,
                            request_id=body.get("id"), finish_reason=choice.get("finish_reason"))


def _refuse_qwen_url(base_url: str) -> None:
    lowered = base_url.lower()
    if any(marker in lowered for marker in FORBIDDEN_QWEN_HOST_MARKERS):
        raise RuntimeError(
            "QwenActor refuses Moonshot/Kimi URLs; use KimiActor for that backend"
        )


class QwenActor:
    """OpenAI-compatible stdlib client for a local vLLM endpoint.

    ``chat`` matches ``KimiActor`` so the episode loop can stay one function.
    The request body is model, messages, seed, temperature, and max_tokens.
    ``reasoning_effort`` and ``top_p`` are omitted. Retries are zero.
    Environment proxies are ignored so a host proxy cannot redirect localhost.
    """

    def __init__(
        self,
        model: str = DEFAULT_QWEN_MODEL,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        timeout: float = 600.0,
        max_tokens: int = DEFAULT_QWEN_MAX_TOKENS,
        temperature: float = DEFAULT_QWEN_TEMPERATURE,
        api_key: str | None = None,
        transport: Callable[[Request], dict] | None = None,
    ) -> None:
        _refuse_qwen_url(base_url)
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be a number")
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = float(temperature)
        # Literal EMPTY matches the team's Engaging vLLM clients unless a key
        # is already in the environment. The value is never written to records.
        self.api_key = api_key if api_key is not None else (os.environ.get("VLLM_API_KEY") or "EMPTY")
        self.usage = UsageLedger()
        self._transport = transport

    def payload(self, messages: list[dict], seed: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "seed": seed,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _default_transport(self, request: Request) -> dict:
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                body = json.load(response)
        except HTTPError as exc:
            exc.read()
            raise RuntimeError(f"Qwen HTTP {exc.code}; credentials never logged") from None
        except (TimeoutError, socket.timeout, URLError) as exc:
            raise RuntimeError(f"Qwen transport failed (retries=0): {exc}") from None
        if not isinstance(body, dict):
            raise RuntimeError("Qwen response was not a JSON object")
        return body

    def chat(
        self,
        messages: list[dict],
        *,
        seed: int,
        arm: str,
        case_id: str,
        turn: int,
        belief: DeterministicBelief | None = None,
        final: bool = False,
    ) -> ChatResponse:
        del belief, final
        payload = self.payload(messages, seed)
        request = Request(
            self.base_url + "/chat/completions",
            method="POST",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "arm0-qwen-local/0.2",
            },
        )
        transport = self._transport or self._default_transport
        body = transport(request)
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("Qwen response missing choices") from None
        usage = body.get("usage") or {}
        self.usage.add(usage, arm=arm, case_id=case_id, turn=turn)
        return ChatResponse(
            text=message.get("content") or "",
            usage=usage,
            request_id=body.get("id"),
            finish_reason=choice.get("finish_reason"),
        )


def describe_qwen() -> dict[str, Any]:
    return {
        "draft": False,
        "wired": True,
        "http_on_import": False,
        "default_base_url": DEFAULT_QWEN_BASE_URL,
        "default_model": DEFAULT_QWEN_MODEL,
        "default_temperature": DEFAULT_QWEN_TEMPERATURE,
        "default_max_tokens": DEFAULT_QWEN_MAX_TOKENS,
        "sends_reasoning_effort": False,
        "sends_top_p": False,
        "requires_moonshot_key": False,
        "retries": 0,
        "chat_signature": "matches KimiActor.chat",
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(payload)
    temporary.replace(path)
