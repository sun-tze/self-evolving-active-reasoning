"""SHA-checked loader for the official AR-Bench GN test split.

Scientific runs take ``test.json[:n]`` once and reuse that list for every arm
and every decoding seed. The seed is not an argument: it must not resample
secrets. ``train.json`` is rejected by digest even if a caller points at it.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os

from .domain import valid_guess

# AR-Bench @ 9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1 data/gn/test.json
EXPECTED_TEST_SHA256 = "e192e5b2d0e58a62e1b4027ba736368d5dc81b2a8f973080766d5288c9804394"
# Same commit, data/gn/train.json. Not held-out. Never a legal official source.
EXPECTED_TRAIN_SHA256 = "e3467c07c1ad85c25f64faaab29c7d66a7685cc90d677386551516afd32c662b"
AR_BENCH_COMMIT = "9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1"

# First 20 of test.json. A silent edit of the prefix fails closed even after the digest check.
LOCKED_N20_SECRETS = (
    "8362", "1058", "9340", "9763", "3182", "8709", "3851", "2045", "7294", "7280",
    "5804", "3520", "1054", "2907", "4291", "9416", "3815", "4683", "9164", "2039",
)
SMOKE_N2_SECRETS = ("8362", "1058")

_PACKAGE = Path(__file__).resolve().parent
PACKAGE_FIXTURE = _PACKAGE / "fixtures" / "arbench_gn_test.json"
SUBMODULE_TEST_JSON = _PACKAGE.parent / "AR-Bench" / "data" / "gn" / "test.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_test_json() -> Path:
    """Prefer an explicit path, then a checked-out submodule, then the packaged fixture.

    The fixture is a byte copy of the pinned file so offline tests do not fetch it.
    """
    override = os.environ.get("AR0_GN_TEST_JSON")
    if override:
        return Path(override)
    if SUBMODULE_TEST_JSON.is_file():
        return SUBMODULE_TEST_JSON
    return PACKAGE_FIXTURE


def load_official_cases(
    n: int,
    *,
    json_path: Path | None = None,
    expected_sha256: str = EXPECTED_TEST_SHA256,
) -> list[dict[str, str]]:
    """Return the first ``n`` official test secrets as case dicts.

    The same list is the matched sample for every arm and every seed.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive int")
    path = Path(json_path) if json_path is not None else default_test_json()
    if not path.is_file():
        raise FileNotFoundError(
            f"Official GN test.json not found: {path}. "
            "Pass --gn-json or check out the AR-Bench submodule at "
            f"{AR_BENCH_COMMIT}. Refusing to fall back to synthetic cases."
        )
    digest = sha256_file(path)
    if digest == EXPECTED_TRAIN_SHA256:
        raise ValueError(
            "Refusing AR-Bench GN train.json: it is not the held-out test split"
        )
    if digest != expected_sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: got {digest}, expected {expected_sha256}"
        )
    secrets = json.loads(path.read_text())
    if not isinstance(secrets, list) or not all(isinstance(item, str) for item in secrets):
        raise ValueError("Official GN JSON must be a list of secret strings")
    if any(not valid_guess(secret) for secret in secrets):
        raise ValueError("Official GN secrets must be 4 distinct digits (leading zero allowed)")
    if n > len(secrets):
        raise ValueError(f"n={n} exceeds official file length {len(secrets)}")
    if n >= 2 and tuple(secrets[:2]) != SMOKE_N2_SECRETS:
        raise ValueError("Official smoke prefix drifted from 8362, 1058")
    if n >= 20 and tuple(secrets[:20]) != LOCKED_N20_SECRETS:
        raise ValueError("First 20 official secrets drifted from the locked protocol list")
    chosen = secrets[:n]
    return [
        {
            "case_id": f"arbench-test-{index:04d}",
            "secret": secret,
            "source": "arbench_gn_test",
            "sha256": digest,
        }
        for index, secret in enumerate(chosen)
    ]


def verify_official_file(path: Path | None = None) -> dict:
    """Read-only checksum report. Does not call a model."""
    target = Path(path) if path is not None else default_test_json()
    report = {
        "path": str(target),
        "exists": target.is_file(),
        "arbench_commit": AR_BENCH_COMMIT,
        "expected_sha256": EXPECTED_TEST_SHA256,
        "wired_into_runner": True,
        "train_json_held_out": False,
    }
    if not target.is_file():
        report["sha256_ok"] = False
        return report
    digest = sha256_file(target)
    secrets = json.loads(target.read_text())
    report.update({
        "sha256": digest,
        "sha256_ok": digest == EXPECTED_TEST_SHA256,
        "n_secrets": len(secrets) if isinstance(secrets, list) else None,
        "bytes": target.stat().st_size,
        "is_train_split": digest == EXPECTED_TRAIN_SHA256,
    })
    if report["sha256_ok"]:
        report["smoke_n2_secrets"] = list(secrets[:2])
        report["matrix_n20_secrets"] = list(secrets[:20])
        report["locked_n20_ok"] = tuple(secrets[:20]) == LOCKED_N20_SECRETS
    return report


if __name__ == "__main__":
    print(json.dumps(verify_official_file(), indent=2, sort_keys=True))
