import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "update_checksums.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, timeout=60,
    )


def _write_manifest(path: Path, payload: Path) -> None:
    path.write_text(json.dumps({
        "TestTool": {"url": payload.as_uri(), "sha256": ""},
    }), encoding="utf-8")


def test_update_checksums_writes_and_checks(tmp_path: Path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    manifest = tmp_path / "binaries.json"
    _write_manifest(manifest, payload)

    check = _run("--check", "--binaries", str(manifest))
    assert check.returncode == 1
    assert "drift: TestTool url" in check.stdout

    update = _run("--binaries", str(manifest))
    assert update.returncode == 0
    assert "updated: TestTool url" in update.stdout
    digest = hashlib.sha256(b"payload").hexdigest()
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert stored["TestTool"]["sha256"] == digest

    check = _run("--check", "--binaries", str(manifest))
    assert check.returncode == 0
    assert "0 mismatched" in check.stdout


def test_update_checksums_reports_drift(tmp_path: Path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"one")
    manifest = tmp_path / "binaries.json"
    _write_manifest(manifest, payload)
    assert _run("--binaries", str(manifest)).returncode == 0

    payload.write_bytes(b"two")
    check = _run("--check", "--binaries", str(manifest))
    assert check.returncode == 1
    assert "drift: TestTool url" in check.stdout
