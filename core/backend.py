from __future__ import annotations

import contextlib
import functools
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .deps import bin_path_dirs


def _prepend_local_bin_path() -> None:
    # before pyvips import, let the binding dlopen libvips from bin/
    bin_path = Path(__file__).resolve().parent.parent / "bin"
    for path_dir in reversed(bin_path_dirs(bin_path)):
        os.environ["PATH"] = os.pathsep.join([str(path_dir), os.environ.get("PATH", "")])
    # bin/vips keeps libvips in bin/vips/lib
    vips_lib = bin_path / "vips" / "lib"
    if not sys.platform.startswith("win") and vips_lib.is_dir():
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            [str(vips_lib), os.environ.get("LD_LIBRARY_PATH", "")])


_prepend_local_bin_path()

try:
    import pyvips
except (ImportError, OSError):
    # binding present but libvips won't load
    pyvips = None


def reload_pyvips() -> bool:
    # otherwise --venv --install-deps stays on magick until the next run
    global pyvips
    if pyvips is not None:
        return True
    importlib.invalidate_caches()
    try:
        pyvips = importlib.import_module("pyvips")
    except (ImportError, OSError):
        return False
    # keep the module graph acyclic
    from . import contentscan, converters
    contentscan.set_pyvips(pyvips)
    converters.set_pyvips(pyvips)
    return True


@dataclass(frozen=True)
class BackendInfo:
    os_type: str            # "Windows" | "macOS" | "Wayland" | "X11"
    vips: bool
    magick: bool
    pngquant: bool
    realesrgan: bool
    waifu2x: bool
    jpegli: bool = False
    mozjpeg: bool = False   # real mozjpeg cjpeg, not stock libjpeg-turbo
    avifenc: bool = False
    oxipng: bool = False

    @property
    def primary_backend(self) -> str:
        return "vips" if self.vips else "magick"

    @property
    def has_any_upscaler(self) -> bool:
        return self.realesrgan or self.waifu2x

    @property
    def has_any_backend(self) -> bool:
        return self.vips or self.magick


def _os_check() -> str:
    match sys.platform:
        case "win32" | "cygwin":
            return "Windows"
        case "darwin":
            return "macOS"
        case "linux" | "linux2" | "freebsd" | "openbsd" | "netbsd":
            return "Wayland" if os.environ.get("WAYLAND_DISPLAY") else "X11"
    return "unknown"


def _check_cli(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _check_mozjpeg() -> bool:
    # libjpeg-turbo also ships cjpeg, but only mozjpeg prints its name.
    if not _check_cli("cjpeg"):
        return False
    try:
        result = subprocess.run(
            ["cjpeg", "-version"],
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = f"{result.stdout} {result.stderr}".lower()
    return "mozjpeg" in output


@functools.cache
def detect_backends() -> BackendInfo:
    vips_ok = False
    if pyvips is not None:
        with contextlib.suppress(Exception):
            pyvips.Image.black(1, 1)
            vips_ok = True

    return BackendInfo(
        os_type=_os_check(),
        vips=vips_ok,
        magick=_check_cli("magick"),
        pngquant=_check_cli("pngquant"),
        realesrgan=_check_cli("realesrgan-ncnn-vulkan"),
        waifu2x=_check_cli("waifu2x-ncnn-vulkan"),
        jpegli=_check_cli("cjpegli"),
        mozjpeg=_check_mozjpeg(),
        avifenc=_check_cli("avifenc"),
        oxipng=_check_cli("oxipng"),
    )


_BACKEND: BackendInfo | None = None


def set_backend(backend: BackendInfo) -> None:
    global _BACKEND
    _BACKEND = backend


def get_backend() -> BackendInfo:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = detect_backends()
    return _BACKEND