"""Regression tests for the release scanner's public-documentation exception."""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check_release.py"
PUBLIC_IMAGE = "docs/assets/railmind-teaser.jpg"


class ReleaseCheckTests(unittest.TestCase):
    def scan(self, files):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            shutil.copy2(CHECKER, root / "scripts" / "check_release.py")
            (root / "third_party").mkdir()
            (root / "third_party" / "source_lock.json").write_text("{}", encoding="utf-8")
            for relative, data in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            result = subprocess.run(
                [sys.executable, str(root / "scripts" / "check_release.py")],
                capture_output=True, text=True, check=False,
            )
            return result.returncode, json.loads(result.stdout)

    def test_approved_public_image_passes(self):
        code, report = self.scan({PUBLIC_IMAGE: (REPO / PUBLIC_IMAGE).read_bytes()})
        self.assertEqual(code, 0, report)
        self.assertEqual(report["status"], "PASS")

    def test_replacing_approved_image_is_rejected(self):
        code, report = self.scan({PUBLIC_IMAGE: b"unreviewed replacement"})
        self.assertNotEqual(code, 0)
        self.assertTrue(any("asset hash mismatch" in error for error in report["errors"]))

    def test_unlisted_image_in_documentation_is_rejected(self):
        code, report = self.scan({"docs/assets/scenes/private.png": b"private image"})
        self.assertNotEqual(code, 0)
        self.assertTrue(any("Private data" in error for error in report["errors"]))

    def test_same_filename_outside_approved_path_is_rejected(self):
        code, report = self.scan({"samples/railmind-teaser.jpg": (REPO / PUBLIC_IMAGE).read_bytes()})
        self.assertNotEqual(code, 0)
        self.assertTrue(any("Private data" in error for error in report["errors"]))

    def test_supervision_records_remain_blocked(self):
        code, report = self.scan({"examples/supervision.jsonl": b"{}\n"})
        self.assertNotEqual(code, 0)
        self.assertTrue(any("Private data" in error for error in report["errors"]))

    def test_model_payload_remains_blocked(self):
        code, report = self.scan({"examples/checkpoint.pth": b"weights"})
        self.assertNotEqual(code, 0)
        self.assertTrue(any("Non-source payload" in error for error in report["errors"]))

    def test_credentials_remain_blocked(self):
        fake_key = b"sk-" + b"A" * 24
        code, report = self.scan({"examples/config.txt": fake_key})
        self.assertNotEqual(code, 0)
        self.assertTrue(any("Potential credential" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
