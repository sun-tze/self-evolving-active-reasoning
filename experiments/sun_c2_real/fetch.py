"""Download public pinned research artifacts, safely extract only five exact members."""
import argparse
from pathlib import Path
import tarfile
import urllib.request
from common import HERE, read_json, require, sha


def download(url, path, expected):
    if path.exists():
        require(sha(path.read_bytes()) == expected, "existing download hash mismatch")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        raw = r.read(50 * 1024 * 1024 + 1)
    require(len(raw) <= 50 * 1024 * 1024 and sha(raw) == expected, "download size/hash mismatch")
    path.write_bytes(raw)


def fetch(destination):
    import zstandard
    root = Path(destination)
    lock = read_json(HERE / "source_lock.json")
    archive = root / "sp_tthe_pilot_raw.tar.zst"
    download(lock["release_url"], archive, lock["archive_sha256"])
    wanted = {v["path"]: v["sha256"] for v in lock["members"].values()}
    found = set()
    with archive.open("rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as z, tarfile.open(fileobj=z, mode="r|") as tar:
        for m in tar:
            if m.name not in wanted:
                continue
            require(m.isfile() and m.name not in found and m.size <= 40 * 1024 * 1024, "unsafe/duplicate member")
            raw = tar.extractfile(m).read()
            require(sha(raw) == wanted[m.name], "archive member hash mismatch")
            dest = root / "raw" / m.name
            require(dest.resolve().is_relative_to((root / "raw").resolve()), "member path escapes root")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            found.add(m.name)
    require(found == set(wanted), "missing archive members")
    for entry in lock["arbench_files"].values():
        url = "https://raw.githubusercontent.com/tmlr-group/AR-Bench/" + lock["arbench_commit"] + "/" + entry["path"]
        download(url, root / "AR-Bench" / entry["path"], entry["sha256"])
    print("Pinned public inputs verified; no model calls.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=".c2-local")
    fetch(p.parse_args().out)
