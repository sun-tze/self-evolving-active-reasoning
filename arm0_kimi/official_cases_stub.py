"""Compatibility shim. The wired loader is :mod:`arm0_kimi.official_cases`.

Importing this module does not read a dataset and does not call a model.
"""
from .official_cases import (  # noqa: F401
    EXPECTED_TEST_SHA256,
    LOCKED_N20_SECRETS,
    SMOKE_N2_SECRETS,
    default_test_json,
    load_official_cases,
    sha256_file,
    verify_official_file,
)

# Historical name used by the draft stub.
DEFAULT_TEST_JSON = default_test_json()


def verify_local_official_file() -> dict:
    report = verify_official_file()
    report["wired_into_runner"] = True
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(verify_local_official_file(), indent=2, sort_keys=True))
