import contextlib
import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

from .converters import to_dib_bytes
from .errors import ClipshrinkError
from .fileops import MIME_TO_EXT


class ClipboardError(ClipshrinkError):
    pass


# browsers copy PNG, most macOS apps TIFF
_MACOS_READ_TYPES: tuple[tuple[str, str], ...] = (
    ("NSPasteboardTypePNG", "image/png"),
    ("NSPasteboardTypeTIFF", "image/tiff"),
)


def _mime_for_format(fmt: str | None) -> str:
    return {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "AVIF": "image/avif",
        "JXL": "image/jxl",
        "HEIC": "image/heic",
        "HEIF": "image/heif",
    }.get(fmt.upper() if fmt else "", "image/png")


def _run_check(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(cmd, **kwargs)
    except FileNotFoundError as e:
        raise ClipboardError(f"Clipboard utility '{cmd[0]}' not found.") from e


class _LinuxClipboardSpec(TypedDict):
    list_cmd: list[str]
    list_tool: str
    split_types: Callable[[str], list[str]]
    read_cmd: Callable[[str], list[str]]
    write_cmd: Callable[[str], list[str]]
    write_tool: str


# same flow, different commands and type-list format
_LINUX_CLIPBOARD_BACKENDS: dict[str, _LinuxClipboardSpec] = {
    "Wayland": {
        "list_cmd": ["wl-paste", "--list-types"],
        "list_tool": "wl-paste",
        "split_types": lambda out: out.split(),
        "read_cmd": lambda mime: ["wl-paste", "--type", mime],
        "write_cmd": lambda mime: ["wl-copy", "--type", mime],
        "write_tool": "wl-copy",
    },
    "X11": {
        "list_cmd": ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"],
        "list_tool": "xclip",
        "split_types": lambda out: out.splitlines(),
        "read_cmd": lambda mime: ["xclip", "-selection", "clipboard", "-t", mime, "-o"],
        "write_cmd": lambda mime: ["xclip", "-selection", "clipboard", "-t", mime, "-i"],
        "write_tool": "xclip",
    },
}


def _linux_get(spec: _LinuxClipboardSpec) -> tuple[bytes, str]:
    result = _run_check(spec["list_cmd"], capture_output=True, text=True)
    if result.returncode != 0:
        raise ClipboardError(f"Failed to list clipboard types ({spec['list_tool']}).")
    image_types = [t for t in spec["split_types"](result.stdout.strip()) if t in MIME_TO_EXT]
    if not image_types:
        raise ClipboardError("No image found in clipboard.")
    chosen = image_types[0]
    data = _run_check(spec["read_cmd"](chosen), capture_output=True)
    if data.returncode != 0:
        raise ClipboardError(f"Failed to read image from clipboard ({spec['list_tool']}).")
    return data.stdout, chosen


def _linux_set(spec: _LinuxClipboardSpec, image: bytes, mime_type: str) -> None:
    result = _run_check(spec["write_cmd"](mime_type), input=image)
    if result.returncode != 0:
        raise ClipboardError(f"Failed to copy image to clipboard ({spec['write_tool']}).")


def _clipboard_import(module: str, package: str, hint: str) -> Any:
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ClipboardError(
            f"{package} is required for clipboard support on {hint}; "
            "run 'clipshrink --install-deps' to install it."
        ) from exc


def _macos_get(pb: Any, appkit: Any) -> tuple[bytes, str]:
    for type_name, mime in _MACOS_READ_TYPES:
        data = pb.dataForType_(getattr(appkit, type_name))
        if data:
            return data.bytes(), mime
    raise ClipboardError("No image found in clipboard.")


def get_clipboard_image(os_type: str) -> tuple[bytes, str]:
    if os_type in _LINUX_CLIPBOARD_BACKENDS:
        return _linux_get(_LINUX_CLIPBOARD_BACKENDS[os_type])

    if os_type == "Windows":
        win32clipboard = _clipboard_import("win32clipboard", "pywin32", "Windows")
        win32con = _clipboard_import("win32con", "pywin32", "Windows")
        win32clipboard.OpenClipboard()
        try:
            if not win32clipboard.IsClipboardFormatAvailable(win32con.CF_DIB):
                raise ClipboardError("No image found in clipboard.")
            data = win32clipboard.GetClipboardData(win32con.CF_DIB)
            return bytes(data), "image/bmp"
        finally:
            win32clipboard.CloseClipboard()

    if os_type == "macOS":
        appkit = _clipboard_import("AppKit", "pyobjc", "macOS")
        pb = appkit.NSPasteboard.generalPasteboard()
        return _macos_get(pb, appkit)

    raise ClipboardError(f"Clipboard not supported on OS '{os_type}'.")


def set_clipboard_image(image: bytes, os_type: str, fmt: str | None = None) -> None:
    # unknown formats fall back to PNG
    mime_type = _mime_for_format(fmt)
    if os_type in _LINUX_CLIPBOARD_BACKENDS:
        _linux_set(_LINUX_CLIPBOARD_BACKENDS[os_type], image, mime_type)
    elif os_type == "Windows":
        win32clipboard = _clipboard_import("win32clipboard", "pywin32", "Windows")
        win32con = _clipboard_import("win32con", "pywin32", "Windows")
        # CF_DIB wants raw pixels, temp file lets magick sniff
        suffix = MIME_TO_EXT.get(mime_type, ".png")
        fd, src_name = tempfile.mkstemp(prefix="clipshrink_clip_", suffix=suffix)
        os.close(fd)
        src_path = Path(src_name)
        src_path.write_bytes(image)
        try:
            dib = to_dib_bytes(src_path)
        finally:
            with contextlib.suppress(OSError):
                src_path.unlink()
        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_DIB, dib)
            if (fmt or "").upper() == "PNG":
                cf_png = win32clipboard.RegisterClipboardFormat("PNG")
                win32clipboard.SetClipboardData(cf_png, image)
        finally:
            win32clipboard.CloseClipboard()
    elif os_type == "macOS":
        appkit = _clipboard_import("AppKit", "pyobjc", "macOS")
        foundation = _clipboard_import("Foundation", "pyobjc", "macOS")
        pb = appkit.NSPasteboard.generalPasteboard()
        pb.clearContents()
        nsdata = foundation.NSData.dataWithBytes_length_(image, len(image))
        pb.setData_forType_(nsdata, appkit.NSPasteboardTypePNG)
    else:
        raise ClipboardError(f"Clipboard not supported on OS '{os_type}'.")
