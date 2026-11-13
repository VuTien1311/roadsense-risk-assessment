import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_weights", ROOT / "scripts" / "download_weights.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadWeightsTests(unittest.TestCase):
    """No network: the 'remote' file is a local file:// URL."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.base = Path(self.folder.name)
        self.payload = b"pretend checkpoint " * 1000
        self.remote = self.base / "remote.bin"
        self.remote.write_bytes(self.payload)
        self.root = self.base / "repo"
        self.root.mkdir()

    def tearDown(self):
        self.folder.cleanup()

    def manifest(self, **changes):
        entry = {"path": "weights/model.pth", "model": "test", "bytes": len(self.payload),
                 "sha256": hashlib.sha256(self.payload).hexdigest(), "url": self.remote.as_uri(), "license": "test"}
        entry.update(changes)
        path = self.base / "manifest.json"
        path.write_text(json.dumps({"models": [entry]}), encoding="utf-8")
        return ["--root", str(self.root), "--manifest", str(path)]

    def test_download_verify_and_keep(self):
        self.assertEqual(downloader.main(self.manifest()), 0)
        target = self.root / "weights" / "model.pth"
        self.assertEqual(target.read_bytes(), self.payload)
        self.remote.unlink()  # a second run must not need the source again
        self.assertEqual(downloader.main(self.manifest()), 0)

    def test_wrong_hash_is_rejected_and_nothing_is_left(self):
        self.assertEqual(downloader.main(self.manifest(sha256="0" * 64)), 1)
        self.assertEqual(list((self.root / "weights").glob("*")), [])

    def test_wrong_size_is_rejected(self):
        self.assertEqual(downloader.main(self.manifest(bytes=len(self.payload) + 1)), 1)
        self.assertFalse((self.root / "weights" / "model.pth").exists())

    def test_existing_wrong_file_is_never_overwritten(self):
        target = self.root / "weights" / "model.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"something else")
        self.assertEqual(downloader.main(self.manifest()), 1)
        self.assertEqual(target.read_bytes(), b"something else")

    def test_list_downloads_nothing(self):
        self.assertEqual(downloader.main(self.manifest() + ["--list"]), 0)
        self.assertFalse((self.root / "weights").exists())

    def test_manifest_has_a_verifiable_source_for_every_paper_model(self):
        models = json.loads((ROOT / "weights_manifest.json").read_text(encoding="utf-8"))["models"]
        self.assertEqual(len(models), 3)
        for entry in models:
            self.assertTrue(entry["url"].startswith("https://"), entry["path"])
            self.assertEqual(len(entry["sha256"]), 64, entry["path"])
            self.assertGreater(entry["bytes"], 0, entry["path"])
            self.assertTrue(entry["license"], entry["path"])


if __name__ == "__main__":
    unittest.main()
