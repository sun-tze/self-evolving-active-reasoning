"""Regression coverage for the external model-cache integrity boundary."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import MODEL
from runtime_preflight import verify_snapshot


class RuntimeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        values = {"config.json": b"{}", "generation_config.json": b"{}",
                  "model.safetensors.index.json": json.dumps({"weight_map": {"tensor": "part.safetensors"}}).encode(),
                  "part.safetensors": b"abcdefgh", "tokenizer.json": b"tokenizer"}
        self.weights = {"model": MODEL, "revision": "fixed", "files": {}}
        for name, raw in values.items():
            (self.root/name).write_bytes(raw)
            self.weights["files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        self.lock = {"model": MODEL, "revision": "fixed", "files": {
                     "tokenizer.json": self.weights["files"]["tokenizer.json"]["sha256"]}}

    def test_valid_snapshot_matches_external_lock(self):
        self.assertEqual(len(verify_snapshot(self.root, self.lock, self.weights)), 5)

    def test_nonempty_truncated_shard_rejected(self):
        (self.root/"part.safetensors").write_bytes(b"a")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            verify_snapshot(self.root, self.lock, self.weights)

    def test_same_size_corrupt_shard_rejected(self):
        (self.root/"part.safetensors").write_bytes(b"abcdefgX")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_snapshot(self.root, self.lock, self.weights)

    def test_changed_index_rejected_before_parsing(self):
        path = self.root/"model.safetensors.index.json"
        path.write_bytes(path.read_bytes().replace(b"part", b"fake"))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            verify_snapshot(self.root, self.lock, self.weights)

    def test_lock_omitting_required_shard_rejected(self):
        weights = copy.deepcopy(self.weights)
        del weights["files"]["part.safetensors"]
        with self.assertRaisesRegex(ValueError, "incomplete model file lock"):
            verify_snapshot(self.root, self.lock, weights)


if __name__ == "__main__":
    unittest.main()
