"""Compatibility shim. ``QwenActor`` lives in :mod:`arm0_kimi.client`.

Running this module prints the static description and does not open a socket.
"""
from .client import (  # noqa: F401
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MAX_TOKENS,
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_TEMPERATURE,
    QwenActor,
    describe_qwen,
)

DEFAULT_BASE_URL = DEFAULT_QWEN_BASE_URL
DEFAULT_MODEL = DEFAULT_QWEN_MODEL


def describe() -> dict:
    return describe_qwen()


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2, sort_keys=True))
