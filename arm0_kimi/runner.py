"""Matched-case runner for the GN representation baseline.

Arms share secrets, seeds, budgets, the referee, and the final-guess metric.
``chrono``, ``ledger``, and ``ledger_belief`` differ only in host-written
prompt text. Seeds on the official path are decoding seeds. Completed episode
files are reused; they are not run again and they are not counted twice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

from .client import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MAX_TOKENS,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_TEMPERATURE,
    ChatResponse,
    KimiActor,
    MockActor,
    QwenActor,
    utc_now,
    write_json,
)
from .domain import DeterministicBelief, LocalReferee, Turn, extract_guess, make_cases
from .official_cases import load_official_cases
from .prompts import ARMS, prompt_for, system
from .readout import representation_baseline_readout

STUDY_LABEL = "representation_baseline"


@dataclass(frozen=True)
class RunConfig:
    arm: str
    n: int
    seeds: tuple[int, ...]
    backend: str = "mock"
    case_source: str = "synthetic"
    model: str | None = None
    max_turns: int = 25
    temperature: float | None = None
    max_tokens: int | None = None
    list_cap: int = 20
    base_url: str | None = None
    gn_json: Path | None = None
    reasoning_effort: str = "low"  # sent only by KimiActor
    timeout: float = 600.0

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError("arm must be chrono, ledger, or ledger_belief")
        if self.backend not in {"mock", "kimi", "vllm"}:
            raise ValueError("backend must be mock, kimi, or vllm")
        if self.case_source not in {"synthetic", "official"}:
            raise ValueError("case_source must be synthetic or official")
        if type(self.n) is not int or self.n < 1:
            raise ValueError("n must be a positive int")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        seeds = tuple(self.seeds)
        if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be unique ints")
        if self.reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, or max")
        if self.timeout < 600.0:
            raise ValueError("timeout must be >= 600")
        if self.max_turns < 1:
            raise ValueError("Turn budget must be positive")
        if self.backend == "vllm":
            model = self.model or DEFAULT_QWEN_MODEL
            base_url = self.base_url or DEFAULT_QWEN_BASE_URL
            temperature = DEFAULT_QWEN_TEMPERATURE if self.temperature is None else self.temperature
            max_tokens = DEFAULT_QWEN_MAX_TOKENS if self.max_tokens is None else self.max_tokens
        elif self.backend == "kimi":
            model = self.model or "kimi-k3.1-pro"
            base_url = self.base_url or "https://api.moonshot.ai/v1"
            if self.temperature not in (None, 1.0):
                raise ValueError(
                    "KimiActor sends temperature 1.0; set temperature only on the vllm backend"
                )
            temperature = 1.0
            max_tokens = 4096 if self.max_tokens is None else self.max_tokens
        else:
            model = "deterministic_mock"
            base_url = self.base_url or "https://api.moonshot.ai/v1"
            temperature = 1.0 if self.temperature is None else self.temperature
            max_tokens = 4096 if self.max_tokens is None else self.max_tokens
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("temperature must be a number")
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive int")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "temperature", float(temperature))
        object.__setattr__(self, "max_tokens", max_tokens)
        if self.gn_json is not None:
            object.__setattr__(self, "gn_json", Path(self.gn_json))


def sent_temperature(config: RunConfig) -> float:
    """Temperature placed on the wire. Kimi stays at its historical 1.0."""
    if config.backend == "kimi":
        return 1.0
    return float(config.temperature)


def model_label(config: RunConfig) -> str:
    if config.backend == "mock":
        return "deterministic_mock"
    return config.model


@dataclass
class Episode:
    arm: str
    seed: int
    case: dict[str, str]
    config: RunConfig
    history: list[Turn]
    belief: DeterministicBelief
    final: ChatResponse | None = None
    final_guess: str | None = None
    calls: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.calls = []

    @property
    def queries(self) -> int:
        return len(self.history)

    @property
    def repeat(self) -> float | None:
        valid = [turn.guess for turn in self.history if turn.guess is not None]
        if not valid:
            return None
        return sum(guess in valid[:index] for index, guess in enumerate(valid)) / len(valid)

    @property
    def consistency(self) -> float | None:
        valid = [turn.guess for turn in self.history if turn.guess is not None]
        if not valid:
            return None
        belief = DeterministicBelief()
        flags = []
        for turn in self.history:
            if turn.guess is not None:
                flags.append(turn.guess in belief.feasible)
                assert turn.feedback is not None
                belief.observe(turn.guess, turn.feedback)
        return sum(flags) / len(flags)

    def record(self) -> dict[str, Any]:
        alive = self.belief.feasible
        # Final-guess exact match only. A singleton feasible set is not written
        # in as the answer, and a correct mid-episode query does not count.
        final_guess = self.final_guess
        em = final_guess == self.case["secret"]
        return {
            "schema_version": "arm0_representation_baseline_episode_v1",
            "study_label": STUDY_LABEL,
            "context_evolution": False,
            "arm": self.arm,
            "seed": self.seed,
            "case_id": self.case["case_id"],
            "secret": self.case["secret"],
            "case_source": self.config.case_source,
            "backend": self.config.backend,
            "model": model_label(self.config),
            "status": "complete" if self.final is not None else "incomplete",
            "primary_EM": em,
            "native_success_definition": (
                "final independent actor guess exactly equals hidden secret; "
                "no singleton substitution; mid-episode hits do not end the episode"
            ),
            "final_guess": final_guess,
            "final_raw_response": self.final.text if self.final else None,
            "determined_before_final": len(alive) == 1,
            "determined_but_wrong_final": len(alive) == 1 and not em,
            "repeat_guess": self.repeat,
            "feedback_consistency": self.consistency,
            "invalid_query_count": sum(turn.guess is None for turn in self.history),
            "valid_query_count": sum(turn.guess is not None for turn in self.history),
            "n_queries": self.queries,
            "n_alive_end": len(alive),
            "public_feedback": [
                {
                    "turn": turn.turn,
                    "raw_response": turn.raw_response,
                    "guess": turn.guess,
                    "feedback": asdict(turn.feedback) if turn.feedback else None,
                }
                for turn in self.history
            ],
            "calls": self.calls,
            "protocol": _protocol_blob(self.config, self.case),
        }


def _protocol_blob(config: RunConfig, case: dict[str, str]) -> dict[str, Any]:
    if config.case_source == "official":
        seed_policy = (
            "decoding seed on every chat call; official test.json[:n] is shared "
            "across arms and seeds and is not resampled"
        )
    else:
        seed_policy = (
            "synthetic random.Random(seed).sample plumbing; not the scientific path"
        )
    return {
        "max_turns": config.max_turns,
        "temperature": sent_temperature(config),
        "temperature_sent": config.backend in {"vllm", "kimi"},
        "max_tokens_per_call": config.max_tokens,
        "reasoning_effort": config.reasoning_effort if config.backend == "kimi" else None,
        "top_p_sent": False,
        "seed_policy": seed_policy,
        "case_source": config.case_source,
        "official_sha256": case.get("sha256"),
        "representation": "baseline",
        "context_evolution": False,
        "queries_plus_final": config.max_turns + 1,
    }


def _request(actor: Any, episode: Episode, *, final: bool, turn: int) -> ChatResponse:
    user = prompt_for(episode.arm, episode.config.max_turns, episode.history, final=final)
    # The belief panel is host text. The model is not asked to compute it.
    if episode.arm == "ledger_belief":
        user = user + "\n\n" + episode.belief.panel(episode.config.list_cap)
    messages = [
        {"role": "system", "content": system(episode.config.max_turns)},
        {"role": "user", "content": user},
    ]
    response = actor.chat(
        messages,
        seed=episode.seed,
        arm=episode.arm,
        case_id=episode.case["case_id"],
        turn=turn,
        belief=episode.belief,
        final=final,
    )
    episode.calls.append({
        "turn": turn,
        "final": final,
        "messages": messages,
        "response": response.text,
        "request_id": response.request_id,
        "finish_reason": response.finish_reason,
        "usage": response.usage,
    })
    return response


def run_episode(arm: str, case: dict[str, str], seed: int, actor: Any,
                config: RunConfig) -> dict[str, Any]:
    referee = LocalReferee(case["secret"])
    episode = Episode(
        arm=arm, seed=seed, case=case, config=config,
        history=[], belief=DeterministicBelief(),
    )
    for index in range(1, config.max_turns + 1):
        response = _request(actor, episode, final=False, turn=index)
        guess = extract_guess(response.text)
        answer = referee.respond(guess) if guess is not None else None
        episode.history.append(Turn(turn=index, raw_response=response.text, guess=guess, feedback=answer))
        if answer is not None and guess is not None:
            episode.belief.observe(guess, answer)
    episode.final = _request(actor, episode, final=True, turn=config.max_turns + 1)
    episode.final_guess = extract_guess(episode.final.text)
    return episode.record()


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"episodes": 0}
    def mean(key: str) -> float | None:
        values = [record[key] for record in records if record[key] is not None]
        return sum(values) / len(values) if values else None
    return {
        "episodes": len(records),
        "EM": mean("primary_EM"),
        "EM_successes": sum(record["primary_EM"] for record in records),
        "repeat_guess": mean("repeat_guess"),
        "feedback_consistency": mean("feedback_consistency"),
        "determined_before_final": mean("determined_before_final"),
        "determined_but_wrong_final": mean("determined_but_wrong_final"),
        "invalid_query_count": sum(record["invalid_query_count"] for record in records),
    }


def usage_from_records(records: list[dict]) -> dict[str, int]:
    """Count saved calls once. Resumed episodes contribute the calls already on disk."""
    totals = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "total_tokens": 0,
    }
    for record in records:
        for call in record.get("calls") or []:
            usage = call.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens") or usage.get("cached_tokens") or 0
            totals["calls"] += 1
            totals["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            totals["output_tokens"] += int(usage.get("completion_tokens") or 0)
            totals["cached_input_tokens"] += int(cached)
            totals["total_tokens"] += int(usage.get("total_tokens") or 0)
    return totals


def episode_path(out_dir: Path, arm: str, seed: int, case_id: str) -> Path:
    return out_dir / "episodes" / f"{arm}__s{seed}__{case_id}.json"


def inspect_episode(path: Path, config: RunConfig, case: dict[str, str], seed: int) -> tuple[str, dict | None, str]:
    """Classify an on-disk episode as reuse, rerun, or conflict.

    ``reuse`` keeps a completed matching episode without calling the actor.
    ``rerun`` replaces an incomplete, corrupt, or internally inconsistent file.
    ``conflict`` stops: a completed file from another protocol is left in place.
    """
    if not path.exists():
        return "rerun", None, "missing"
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "rerun", None, "unreadable"
    if not isinstance(record, dict):
        return "rerun", None, "unreadable"
    if record.get("arm") not in (None, config.arm):
        return "conflict", record, "arm"
    if record.get("case_id") not in (None, case["case_id"]):
        return "conflict", record, "case_id"
    if record.get("seed") not in (None, seed):
        return "conflict", record, "seed"
    if "secret" in record and record.get("secret") != case["secret"]:
        return "conflict", record, "secret"
    proto = record.get("protocol") if isinstance(record.get("protocol"), dict) else None
    if record.get("status") == "complete" and proto is None:
        return "conflict", record, "protocol"
    if proto is None:
        return "rerun", record, "missing_protocol"
    mismatches = []
    expected = {
        "max_turns": config.max_turns,
        "temperature": sent_temperature(config),
        "max_tokens_per_call": config.max_tokens,
        "case_source": config.case_source,
        "official_sha256": case.get("sha256"),
    }
    for key, value in expected.items():
        if proto.get(key) != value:
            mismatches.append(key)
    if record.get("backend") != config.backend:
        mismatches.append("backend")
    if record.get("model") != model_label(config):
        mismatches.append("model")
    if record.get("case_source") != config.case_source:
        mismatches.append("case_source")
    if mismatches:
        why = ",".join(mismatches)
        if record.get("status") == "complete":
            return "conflict", record, why
        return "rerun", record, why
    if record.get("status") != "complete":
        return "rerun", record, "incomplete"
    calls = record.get("calls")
    if not isinstance(calls, list) or len(calls) != config.max_turns + 1:
        return "rerun", record, "call_count"
    final_raw = record.get("final_raw_response")
    if not isinstance(final_raw, str):
        return "rerun", record, "final_response"
    if record.get("final_guess") != extract_guess(final_raw):
        return "rerun", record, "final_guess"
    if bool(record.get("primary_EM")) != (record.get("final_guess") == case["secret"]):
        return "rerun", record, "em"
    return "reuse", record, "complete"


def _early_stop_reason(records: list[dict], config: RunConfig) -> str | None:
    """After a Kimi arm finishes: stop remaining arms on a broken decoder."""
    if config.backend != "kimi" or not records:
        return None
    total_q = sum(int(record.get("n_queries") or 0) for record in records)
    invalid = sum(int(record.get("invalid_query_count") or 0) for record in records)
    if total_q > 0 and (invalid / total_q) > 0.20:
        return f"invalid_query_rate={invalid}/{total_q}={invalid / total_q:.3f} > 0.20"
    calls = [call for record in records for call in (record.get("calls") or [])]
    if not calls:
        return None
    length_empty = sum(
        1 for call in calls
        if call.get("finish_reason") == "length" and not (call.get("response") or "").strip()
    )
    if length_empty / len(calls) > 0.50:
        return (
            f"finish_reason=length + empty content dominates "
            f"({length_empty}/{len(calls)}); not bumping max_tokens"
        )
    return None


def _assert_matched(configs: list[RunConfig]) -> None:
    if not configs:
        raise ValueError("at least one arm config is required")
    arms = [config.arm for config in configs]
    if len(arms) != len(set(arms)):
        raise ValueError("duplicate arm in one matrix")
    first = configs[0]
    for config in configs[1:]:
        if (config.n, config.seeds, config.backend, config.case_source) != (
            first.n, first.seeds, first.backend, first.case_source
        ):
            raise ValueError("arms must share n, seeds, backend, and case source")
        if (config.model, config.temperature, config.max_tokens, config.max_turns) != (
            first.model, first.temperature, first.max_tokens, first.max_turns
        ):
            raise ValueError("arms must share model, temperature, and budgets")
        if (config.base_url, config.gn_json) != (first.base_url, first.gn_json):
            raise ValueError("arms must share the endpoint and the case file")


def build_actor(config: RunConfig, seed: int) -> Any:
    if config.backend == "mock":
        return MockActor(seed)
    if config.backend == "kimi":
        return KimiActor(
            config.model,
            base_url=config.base_url,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            reasoning_effort=config.reasoning_effort,
        )
    if config.backend == "vllm":
        return QwenActor(
            model=config.model,
            base_url=config.base_url,
            timeout=config.timeout,
            max_tokens=config.max_tokens,
            temperature=sent_temperature(config),
        )
    raise ValueError(config.backend)


def run_matrix(configs: list[RunConfig], out_dir: Path) -> dict:
    """Run or resume a matched matrix. Episode files are the source of counts."""
    _assert_matched(configs)
    official_cases = None
    if configs[0].case_source == "official":
        # One list, before any seed loop. Failure here does not call make_cases.
        official_cases = load_official_cases(configs[0].n, json_path=configs[0].gn_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []
    resumed = 0
    executed = 0
    seen: set[tuple] = set()
    started = time.monotonic()
    for config in configs:
        records: list[dict] = []
        for seed in config.seeds:
            cases = official_cases if official_cases is not None else make_cases(config.n, seed)
            actor = None
            for case in cases:
                key = (config.arm, seed, case["case_id"])
                if key in seen:
                    raise RuntimeError(f"duplicate episode key {key}")
                seen.add(key)
                path = episode_path(out_dir, config.arm, seed, case["case_id"])
                disposition, prior, why = inspect_episode(path, config, case, seed)
                if disposition == "conflict":
                    raise RuntimeError(
                        f"Refusing to reuse or overwrite {path}: existing episode does not "
                        f"match this protocol ({why}). Use a fresh --out."
                    )
                if disposition == "reuse":
                    assert prior is not None
                    records.append(prior)
                    all_records.append(prior)
                    resumed += 1
                    print(
                        f"{config.arm} seed={seed} case={case['case_id']} resume "
                        f"EM={int(bool(prior['primary_EM']))} alive={prior['n_alive_end']} "
                        f"repeat={prior['repeat_guess']} consistency={prior['feedback_consistency']}",
                        flush=True,
                    )
                    continue
                if actor is None:
                    actor = build_actor(config, seed)
                start = time.monotonic()
                record = run_episode(config.arm, case, seed, actor, config)
                record["elapsed_seconds"] = time.monotonic() - start
                records.append(record)
                all_records.append(record)
                executed += 1
                write_json(path, record)
                print(
                    f"{config.arm} seed={seed} case={case['case_id']} run "
                    f"EM={int(record['primary_EM'])} alive={record['n_alive_end']} "
                    f"repeat={record['repeat_guess']} consistency={record['feedback_consistency']}",
                    flush=True,
                )
        summary = aggregate(records)
        summary["usage"] = usage_from_records(records)
        write_json(out_dir / f"summary_{config.arm}.json", summary)
        stop_reason = _early_stop_reason(records, config)
        if stop_reason:
            note = (
                f"# EARLY_STOP\n\n"
                f"- arm_completed: {config.arm}\n"
                f"- reason: {stop_reason}\n"
                f"- remaining_arms_skipped: {[item.arm for item in configs[configs.index(config) + 1:]]}\n"
                f"- out_dir: {out_dir}\n"
                f"- note: max_tokens left at {config.max_tokens}; do not bump to 8192 without a separate decision.\n"
            )
            (out_dir / "EARLY_STOP.md").write_text(note)
            print(f"EARLY_STOP after arm={config.arm}: {stop_reason}", flush=True)
            configs[:] = configs[: configs.index(config) + 1]
            break
    arms = [config.arm for config in configs]
    by_arm = {}
    for config in configs:
        arm_records = [record for record in all_records if record["arm"] == config.arm]
        by_arm[config.arm] = aggregate(arm_records)
        by_arm[config.arm]["usage"] = usage_from_records(arm_records)
    per_seed: dict[str, dict] = {}
    for seed in configs[0].seeds:
        per_seed[str(seed)] = {
            arm: aggregate([
                record for record in all_records
                if record["seed"] == seed and record["arm"] == arm
            ])
            for arm in arms
        }
    readout = representation_baseline_readout(by_arm, per_seed=per_seed)
    if configs[0].case_source == "official":
        case_generator = (
            "official AR-Bench data/gn/test.json[:n] at "
            "9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1; "
            "fixed across arms and decoding seeds; train.json not loaded"
        )
    else:
        case_generator = (
            "synthetic random.Random(seed).sample of 5040 valid secrets; "
            "plumbing only; not the scientific path"
        )
    result = {
        "schema_version": "arm0_representation_baseline_run_v1",
        "study_label": STUDY_LABEL,
        "context_evolution": False,
        "scientific_lift_claimed": False,
        "mock_is_not_a_representation_result": configs[0].backend == "mock",
        "backend": configs[0].backend,
        "model": model_label(configs[0]),
        "case_source": configs[0].case_source,
        "case_generator": case_generator,
        "official_sha256": official_cases[0]["sha256"] if official_cases else None,
        "case_ids": [case["case_id"] for case in official_cases] if official_cases else None,
        "n_per_arm_seed": configs[0].n,
        "matched_seeds": list(configs[0].seeds),
        "decoding_seeds_only": configs[0].case_source == "official",
        "arms": arms,
        "control": "ledger",
        "candidate": "ledger_belief",
        "temperature_sent": sent_temperature(configs[0]) if configs[0].backend in {"vllm", "kimi"} else None,
        "max_tokens_sent": configs[0].max_tokens if configs[0].backend == "vllm" else (
            configs[0].max_tokens if configs[0].backend == "kimi" else None
        ),
        "reasoning_effort_sent": configs[0].reasoning_effort if configs[0].backend == "kimi" else None,
        "top_p_sent": False,
        "resumed_episodes": resumed,
        "executed_episodes": executed,
        "started_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "by_arm": by_arm,
        "per_seed": per_seed,
        "readout": readout,
        "usage": usage_from_records(all_records),
        "pricing_estimate_usd": None,
        "pricing_note": "Not an invoice. This offline path does not price a live endpoint.",
    }
    write_json(out_dir / "run_summary.json", result)
    return result
