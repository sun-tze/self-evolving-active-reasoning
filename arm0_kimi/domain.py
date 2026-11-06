"""Pure, standard-library Guess Number domain and external belief tracking."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
import json
import random
import re

CANDIDATES = tuple("".join(p) for p in permutations("0123456789", 4))
CANDIDATE_SET = frozenset(CANDIDATES)
GUESS_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def valid_guess(text: str) -> bool:
    return isinstance(text, str) and text in CANDIDATE_SET


def extract_guess(text: str) -> str | None:
    """Match AR-Bench's last-four-digit extraction, then require unique digits."""
    matches = GUESS_RE.findall(text or "")
    if not matches:
        return None
    guess = matches[-1]
    return guess if valid_guess(guess) else None


def feedback(guess: str, secret: str) -> tuple[int, int]:
    if not valid_guess(guess) or not valid_guess(secret):
        raise ValueError("GN requires four distinct digits (leading zero allowed)")
    correct_position = sum(a == b for a, b in zip(guess, secret))
    common = len(set(guess) & set(secret))
    return correct_position, common - correct_position


@dataclass(frozen=True)
class Feedback:
    correct_position: int
    wrong_position: int

    def __post_init__(self) -> None:
        a, b = self.correct_position, self.wrong_position
        if type(a) is not int or type(b) is not int or not (0 <= a <= 4 and 0 <= b <= 4):
            raise ValueError("Invalid GN feedback")
        if a + b > 4 or (a, b) == (3, 1):
            raise ValueError("Impossible GN feedback")

    def render(self) -> str:
        return f"{self.correct_position}A{self.wrong_position}B"


@dataclass(frozen=True)
class Turn:
    turn: int
    raw_response: str
    guess: str | None
    feedback: Feedback | None

    def raw_pair(self) -> tuple[str, str]:
        """Lossless Q/A item: the actor text and public numeric feedback."""
        return self.raw_response, self.feedback.render() if self.feedback else "NO_VALID_GUESS"


class DeterministicBelief:
    """Exact feasible set, exclusions, and consistency; never invokes a model."""

    def __init__(self) -> None:
        self.history: list[tuple[str, Feedback]] = []

    def observe(self, guess: str, answer: Feedback) -> None:
        if not valid_guess(guess):
            raise ValueError("Belief accepts only valid guesses")
        self.history.append((guess, answer))

    @property
    def feasible(self) -> tuple[str, ...]:
        alive = CANDIDATES
        for guess, answer in self.history:
            observed = asdict(answer)
            alive = tuple(c for c in alive if asdict(Feedback(*feedback(guess, c))) == observed)
        return alive

    @property
    def asked(self) -> frozenset[str]:
        return frozenset(g for g, _ in self.history)

    def object(self, list_cap: int = 20) -> dict:
        alive = self.feasible
        result = {
            "schema": "gn_belief_v1",
            "feasible_count": len(alive),
            "feasible": list(alive) if len(alive) <= list_cap else None,
            "feasible_omitted": len(alive) > list_cap,
            "exclusions": {
                "asked": sorted(self.asked),
                "ruled_out_count": len(CANDIDATES) - len(alive),
                "digits_not_in_any_feasible_secret": [
                    d for d in "0123456789" if not any(d in c for c in alive)
                ],
            },
            "consistency": {
                "history_nonempty": bool(alive),
                "next_guess_rule": "Use a feasible, unasked guess; final guess must be feasible.",
            },
            "pinned_positions": {
                str(i + 1): alive[0][i] for i in range(4)
                if alive and len({c[i] for c in alive}) == 1
            },
        }
        return result

    def panel(self, list_cap: int = 20) -> str:
        return "DETERMINISTIC_BELIEF\n" + json.dumps(
            self.object(list_cap), sort_keys=True, separators=(",", ":")
        )


class LocalReferee:
    """Mock GN environment using the exact AR-Bench A/B feedback rule."""

    def __init__(self, secret: str) -> None:
        if not valid_guess(secret):
            raise ValueError("Invalid secret")
        self.secret = secret

    def respond(self, guess: str) -> Feedback:
        return Feedback(*feedback(guess, self.secret))


def make_cases(n: int, seed: int) -> list[dict[str, str]]:
    """Fresh synthetic local draft cases; not a claim to replay private AR-Bench IDs."""
    if type(n) is not int or not 1 <= n <= len(CANDIDATES):
        raise ValueError("n must be between 1 and 5040")
    rng = random.Random(seed)
    secrets = rng.sample(CANDIDATES, n)
    return [{"case_id": f"synthetic-{seed}-{i:04d}", "secret": secret}
            for i, secret in enumerate(secrets)]
