from __future__ import annotations

import contextlib
import threading
from datetime import datetime
from typing import TextIO

_lock = threading.Lock()
_verbose = False
_debug = False
_log_file: TextIO | None = None


def configure(verbose: bool, debug: bool, log_path: str | None) -> None:
    global _verbose, _debug, _log_file
    new_file: TextIO | None = None
    if log_path:
        new_file = open(log_path, 'a', encoding="utf-8", errors="replace")  # noqa: SIM115
    with _lock:
        if _log_file is not None:
            with contextlib.suppress(OSError):
                _log_file.close()
        _verbose = verbose or debug
        _debug = debug
        _log_file = new_file


def is_verbose() -> bool:
    return _verbose


def is_debug() -> bool:
    return _debug


def log(message: str) -> None:
    _emit(message, console=_verbose)


def debug(message: str) -> None:
    _emit(message, console=_debug)


def _emit(message: str, console: bool) -> None:
    with _lock:
        if console:
            print(message)
        if _log_file is not None:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                _log_file.write(f"[{stamp}] {message}\n")
                _log_file.flush()
            except (OSError, ValueError):
                pass  # a broken log file shouldn't kill the run


def close() -> None:
    global _log_file
    with _lock:
        if _log_file is not None:
            with contextlib.suppress(OSError):
                _log_file.close()
            _log_file = None


_STDERR_TAIL_LIMIT = 2000


def log_command(cmd: list[str], returncode: int | None = None,
                stderr: str = "") -> None:
    # no returncode = starting. on failure append the stderr tail.
    if not _debug:
        return
    if returncode is None:
        debug(f"$ {' '.join(cmd)}")
        return
    line = f"$ {' '.join(cmd)} -> exit {returncode}"
    if returncode != 0 and stderr:
        tail = stderr[-_STDERR_TAIL_LIMIT:]
        if len(stderr) > _STDERR_TAIL_LIMIT:
            tail = f"...{tail}"
        line = f"{line}\n{tail}"
    debug(line)
