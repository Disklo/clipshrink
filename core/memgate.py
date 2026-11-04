from __future__ import annotations

import functools
import os
import re
import sys
from pathlib import Path

# share of OS-reported available memory tool children may claim
_BUDGET_FRACTION = 0.7
# below this, tiny machines get one probe at a time
_MIN_BUDGET = 512 * 1024 * 1024
# uncapped magick decodes to HDRI float32, ~16 B/px
_PROBE_BYTES_PER_PIXEL = 16
_PROBE_FIXED_OVERHEAD = 512 * 1024 * 1024
# 2 GiB held a 658 MP source to 1.86 GiB byte-identical, encode-only.
# geometry ops (-resize) are not byte-stable under cap.
_MAGICK_CAP_BYTES = 2 * 1024 * 1024 * 1024
_MAGICK_DISK_LIMIT = "64GiB"

# planning budget override, e.g. "12g", "512m", bare bytes
_MAX_RAM_ENV = 'CLIPSHRINK_MAX_RAM'
# opt-in hard RAM cap for every magick command. spilled geometry can
# shift by +-1 LSB, but stays bounded.
_LIMITS_ENV = 'CLIPSHRINK_MAGICK_LIMITS'

_SIZE_SUFFIX = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}


def parse_size(text: str) -> int:
    match = re.fullmatch(r"\s*(\d+)\s*([kmg])?\s*", text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"invalid size: {text!r}")
    value = int(match.group(1))
    suffix = match.group(2)
    if suffix is not None:
        value *= _SIZE_SUFFIX[suffix.lower()]
    return value


def _mem_available() -> int | None:
    try:
        with open("/proc/meminfo", "rb") as f:
            for line in f:
                if line.startswith(b"MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    if sys.platform == "win32":
        # os.sysconf is POSIX-only, env override still works
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def budget_bytes(available: int | None = None) -> int | None:
    # leave the rest to page cache, pipeline and WM
    override = os.environ.get(_MAX_RAM_ENV)
    if override is not None:
        try:
            return max(parse_size(override), _MIN_BUDGET)
        except ValueError:
            pass  # a typo shouldn't crash a run, just measure
    if available is None:
        available = _mem_available()
    if available is None:
        return None
    return max(int(available * _BUDGET_FRACTION), _MIN_BUDGET)


def probe_worker_cap(width: int, height: int, requested: int,
                     budget: int | None = None) -> int:
    # serializing probes changes wall-clock only
    if requested <= 1 or width <= 0 or height <= 0:
        return max(requested, 1)
    if budget is None:
        budget = budget_bytes()
    if budget is None:
        return requested
    per_probe = (_PROBE_BYTES_PER_PIXEL * width * height
                 + _PROBE_FIXED_OVERHEAD)
    return max(min(requested, budget // per_probe), 1)


def _mounts_text() -> str:
    try:
        with open("/proc/mounts", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


@functools.lru_cache(maxsize=64)
def is_tmpfs(path: str) -> bool:
    # spilling to tmpfs means allocating RAM, non-Linux is False
    mounts = _mounts_text()
    if not mounts:
        return False
    resolved = os.path.realpath(path)
    # longest mount wins, nested non-tmpfs overrides tmpfs
    best_len, best_is_tmpfs = 0, False
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = os.path.realpath(parts[1])
        if resolved.startswith(mount_point) and len(mount_point) >= best_len:
            best_len = len(mount_point)
            best_is_tmpfs = parts[2] == "tmpfs"
    return best_is_tmpfs


def _spill_candidates() -> list[str]:
    candidates = [
        "/var/tmp",
        os.path.expanduser("~/.cache"),
        os.getcwd(),
    ]
    env_candidates = [os.environ.get("CLIPSHRINK_TMPDIR"),
                      os.environ.get("XDG_CACHE_HOME")]
    return [c for c in env_candidates if c] + candidates


@functools.lru_cache(maxsize=1)
def tempdir() -> str | None:
    # None = system default. a 658 MP --slow run once filled a 7.8 GiB /tmp.
    try:
        return magick_spill_dir()
    except OSError:
        return None


def apply_process_tempdir() -> None:
    # libvips disc threshold and magick cache follow TMPDIR
    target = tempdir()
    if target is not None:
        os.environ["TMPDIR"] = target


def magick_spill_dir(temp_dir: str | None = None) -> str | None:
    # route spill only, limits stay untouched
    if temp_dir is None:
        temp_dir = os.environ.get("TMPDIR") or "/tmp"
    if not is_tmpfs(temp_dir):
        return None
    for candidate in _spill_candidates():
        if not candidate:
            continue
        if is_tmpfs(candidate):
            continue
        probe = Path(candidate) / ".clipshrink-spill-probe"
        try:
            probe.write_text("")
            probe.unlink()
            return candidate
        except OSError:
            continue
    return None


def _magick_limits_opt_in() -> bool:
    return os.environ.get(_LIMITS_ENV, "").strip() in {"1", "true", "yes"}


# encode-only shapes are byte-identical under caps, others spill only
def _caps_allowed(cmd: list[str]) -> bool:
    if _magick_limits_opt_in():
        return True
    if len(cmd) >= 2 and cmd[1] == "identify":
        return True
    return any("jpeg:extent=" in arg
               or arg == "jpeg:optimize-coding=true"
               for arg in cmd[1:])


def magick_env(cmd: list[str]) -> dict[str, str]:
    # additions only, empty = inherit
    env: dict[str, str] = {}
    spill = magick_spill_dir()
    if spill is not None:
        env["MAGICK_TMPDIR"] = spill
    if _caps_allowed(cmd):
        env["MAGICK_MEMORY_LIMIT"] = "2GiB"
        env["MAGICK_MAP_LIMIT"] = "2GiB"
        env["MAGICK_DISK_LIMIT"] = _MAGICK_DISK_LIMIT
    return env
