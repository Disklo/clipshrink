# Refresh the SHA-256 values in binaries.json after an upstream release.
# Run with --check to report drift without writing.

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DEFAULT_BINARIES = Path(__file__).resolve().parent.parent / "binaries.json"

URL_FIELDS = (
    ("url", "sha256"),
    ("mirror", "sha256_mirror"),
    ("url_linux", "sha256_linux"),
    ("mirror_linux", "sha256_mirror_linux"),
)


def url_hash(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "clipshrink/2.0"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response:
        for chunk in iter(lambda: response.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Refresh SHA-256 checksums in binaries.json.")
    parser.add_argument("--binaries", type=Path, default=DEFAULT_BINARIES,
                        help="path to binaries.json (default: %(default)s)")
    parser.add_argument("--check", action="store_true",
                        help="report drift without rewriting the file")
    args = parser.parse_args(argv)

    try:
        data = json.loads(args.binaries.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {args.binaries}: {exc}")
        return 1
    if not isinstance(data, dict):
        print(f"error: {args.binaries} must contain a JSON object")
        return 1

    checked = changed = failed = 0
    for label, entry in data.items():
        if not isinstance(entry, dict):
            continue
        for url_field, sha_field in URL_FIELDS:
            url = entry.get(url_field, "")
            if not isinstance(url, str) or not url:
                continue
            checked += 1
            try:
                actual = url_hash(url)
            except Exception as exc:
                print(f"error: {label} {url_field}: {exc}")
                failed += 1
                continue
            expected = entry.get(sha_field, "")
            if expected == actual:
                print(f"ok: {label} {url_field}")
                continue
            changed += 1
            if args.check:
                print(f"drift: {label} {url_field}: "
                      f"{expected or '(missing)'} -> {actual}")
            else:
                entry[sha_field] = actual
                print(f"updated: {label} {url_field}")

    if args.check:
        print(f"{checked} checked, {changed} mismatched, {failed} failed.")
        return 1 if changed or failed else 0
    if changed:
        args.binaries.write_text(json.dumps(data, indent=2) + "\n",
                                 encoding="utf-8")
        print(f"{checked} checked, {changed} updated, {failed} failed.")
    else:
        print(f"{checked} checked, no changes, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
