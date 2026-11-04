from __future__ import annotations

import subprocess
from collections.abc import Callable, Generator
from typing import Any

import pytest

from core import log, utils
from core.backend import BackendInfo

RunHandler = Callable[[list[str]], tuple[int, str, str]]


@pytest.fixture(autouse=True)
def reset_process_globals() -> Generator[None, None, None]:
    utils.reset_abort()
    with utils._pids_lock:
        utils._cmd_pids.clear()
        utils._cmd_pid = 0
    utils.set_last_error("")
    log.configure(verbose=False, debug=False, log_path=None)
    yield
    log.configure(verbose=False, debug=False, log_path=None)
    utils.reset_abort()


@pytest.fixture
def make_backend() -> Callable[..., BackendInfo]:
    def factory(**overrides: Any) -> BackendInfo:
        fields: dict[str, Any] = {
            "os_type": "X11", "vips": False, "magick": True,
            "pngquant": False, "realesrgan": False, "waifu2x": False,
            "jpegli": False, "mozjpeg": False, "avifenc": False,
            "oxipng": False,
        }
        fields.update(overrides)
        return BackendInfo(**fields)

    return factory


@pytest.fixture
def fake_run_capture(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[list[str]]]:
    def install(handler: RunHandler, target: Any) -> list[list[str]]:
        calls: list[list[str]] = []

        def stub(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(list(cmd))
            returncode, out, err = handler(cmd)
            return subprocess.CompletedProcess(cmd, returncode, out, err)

        monkeypatch.setattr(target, "run_capture", stub)
        return calls

    return install
