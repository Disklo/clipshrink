import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

from . import log as log_mod
from . import memgate


def _magick_kwargs(cmd: list[str], kwargs: dict[str, Any]) -> dict[str, Any]:
    # route magick's spill dir and shape-gated caps in one place
    if cmd and Path(cmd[0]).stem == "magick":
        extra = memgate.magick_env(cmd)
        if extra:
            kwargs = dict(kwargs)
            kwargs["env"] = {**os.environ, **extra,
                             **kwargs.get("env", {})}
    return kwargs


class AbortedException(Exception):
    pass


# how often cancellable jobs poll their CancelToken
_CANCEL_POLL = 0.1


class CancelToken:
    # dominated probes cancel this, subprocesses poll, pyvips encodes discard
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


_job_token = threading.local()


@contextlib.contextmanager
def cancellable(token: CancelToken | None) -> Iterator[None]:
    previous = getattr(_job_token, "token", None)
    _job_token.token = token
    try:
        yield
    finally:
        _job_token.token = previous


def _current_cancel_token() -> CancelToken | None:
    token = getattr(_job_token, "token", None)
    return token if isinstance(token, CancelToken) else None


def _terminate_child(proc: subprocess.Popen[Any]) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()


def _wait_cancelable(proc: subprocess.Popen[Any], token: CancelToken | None) -> int:
    if token is None:
        return int(proc.wait())
    while True:
        try:
            status = proc.wait(timeout=_CANCEL_POLL)
            break
        except subprocess.TimeoutExpired:
            if token.cancelled:
                _terminate_child(proc)
                status = proc.wait()
                break
    return int(status)


_aborted = False
_cmd_pid = 0
_cmd_pids: set[int] = set()
_pids_lock = threading.Lock()
_tls = threading.local()

# every external child takes a slot, probes and batch workers share one cap
MAX_CONCURRENT_JOBS = min(4, max(2, os.cpu_count() or 2))
_job_slots = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


# thread-local, batch workers shouldn't clobber diagnostics
def set_last_error(message: str) -> None:
    _tls.last_error = message


def get_last_error() -> str:
    return getattr(_tls, "last_error", "")


@contextlib.contextmanager
def job_slot() -> Iterator[None]:
    while True:
        if _aborted:
            raise AbortedException("Processing interrupted.")
        if _job_slots.acquire(timeout=0.1):
            break
    try:
        yield
    finally:
        _job_slots.release()


def on_interrupt(signum: int, frame: FrameType | None) -> None:
    global _aborted
    _aborted = True
    with _pids_lock:
        pids = set(_cmd_pids)
        if _cmd_pid and _cmd_pid not in pids:
            pids.add(_cmd_pid)
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)


def register_signal_handlers() -> None:
    signal.signal(signal.SIGINT, on_interrupt)


def reset_abort() -> None:
    global _aborted
    _aborted = False


def is_aborted() -> bool:
    return _aborted


def _track_pid(pid: int) -> None:
    global _cmd_pid
    with _pids_lock:
        _cmd_pids.add(pid)
        _cmd_pid = pid


def _untrack_pid(pid: int) -> None:
    global _cmd_pid
    with _pids_lock:
        _cmd_pids.discard(pid)
        _cmd_pid = next(iter(_cmd_pids), 0)


def run_killable(cmd: list[str], **kwargs: Any) -> int:
    with job_slot():
        kwargs.setdefault("stdin", subprocess.DEVNULL)
        log_mod.log_command(cmd)
        proc = subprocess.Popen(cmd, **_magick_kwargs(cmd, kwargs))
        _track_pid(proc.pid)
        try:
            return _wait_cancelable(proc, _current_cancel_token())
        finally:
            _untrack_pid(proc.pid)


def run_capture(cmd: list[str], text: bool = True,
                poll_callback: Callable[[float], None] | None = None,
                poll_interval: float = 5.0,
                **kwargs: Any) -> subprocess.CompletedProcess[str]:
    # poll_callback gets elapsed seconds every interval
    with job_slot():
        popen_kwargs = _magick_kwargs(cmd, {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
            "text": text,
            **kwargs,
        })
        proc = subprocess.Popen(cmd, **popen_kwargs)
        token = _current_cancel_token()
        _track_pid(proc.pid)
        start = time.monotonic()
        try:
            if poll_callback is None and token is None:
                out, err = proc.communicate()
            else:
                # polled wait for live timers and cancelling dominated probes
                slice_ = poll_interval if poll_callback is not None else _CANCEL_POLL
                if token is not None:
                    slice_ = min(slice_, _CANCEL_POLL)
                out, err = b"", b""
                while True:
                    try:
                        out, err = proc.communicate(timeout=slice_)
                        break
                    except subprocess.TimeoutExpired:
                        if poll_callback is not None:
                            poll_callback(time.monotonic() - start)
                        if token is not None and token.cancelled:
                            _terminate_child(proc)
            log_mod.log_command(cmd, returncode=proc.returncode,
                                stderr=err if isinstance(err, str) else "")
            return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
        finally:
            _untrack_pid(proc.pid)


def _abort_if_interrupted() -> None:
    # only the signal handler decides what counts as an interrupt
    if _aborted:
        print("\nScript interrupted. Cleaning up temporary files...")
        raise AbortedException("Processing interrupted.")
