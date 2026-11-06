"""Arm-isomorphic task wording plus the lossless renderer."""
from __future__ import annotations

from .domain import Turn

ARMS = ("chrono", "ledger", "ledger_belief")
SYSTEM = """You play Guess Number. The secret has four DISTINCT digits from 0 through 9; leading zero is allowed.
For query t, return a query and a four-digit guess. The referee answers A = correct digit in correct position and B = correct digit in wrong position.
There are exactly MAX_TURNS queries, followed by one separate final response. Do not stop early if you guess correctly.
For every query and the final response, end your response with exactly one four-digit candidate in this format: Guess: 0123
Use public feedback rather than repeating a guess or guessing a value ruled out by earlier feedback."""


def system(max_turns: int) -> str:
    return SYSTEM.replace("MAX_TURNS", str(max_turns))


def render_ledger(turns: list[Turn]) -> str:
    """Every turn, original actor text, and exact feedback; no truncation/summary."""
    lines = ["LOSSLESS_NUMBERED_LEDGER (Q = complete original actor response; A = exact referee feedback)"]
    for turn in turns:
        q, a = turn.raw_pair()
        lines.append(f"[{turn.turn:02d}] Q: {q}")
        lines.append(f"[{turn.turn:02d}] A: {a}")
    return "\n".join(lines)


def prompt_for(arm: str, max_turns: int, turns: list[Turn], *, final: bool) -> str:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    if final:
        marker = "FINAL RESPONSE after all queries. Use all evidence and return your best secret."
    else:
        marker = f"QUERY {len(turns) + 1} of {max_turns}. Propose the next candidate."
    if arm == "chrono":
        if not turns:
            context = "No public guesses have been made."
        else:
            # Chrono intentionally loses original actor prose; it mirrors native public history.
            context = "CHRONOLOGICAL HISTORY\n" + "\n".join(
                f"Guess {t.turn}: {t.guess or 'INVALID'}; feedback: {t.feedback.render() if t.feedback else 'INVALID'}"
                for t in turns
            )
        return context + "\n\n" + marker
    return render_ledger(turns) + "\n\n" + marker
