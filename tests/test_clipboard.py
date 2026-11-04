import subprocess

import pytest

from core import clipboard
from core.clipboard import ClipboardError, _mime_for_format


def _ok(stdout="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, "")


def test_mime_for_format_defaults_to_png():
    assert _mime_for_format(None) == "image/png"
    assert _mime_for_format("") == "image/png"
    assert _mime_for_format("TIFF") == "image/png"


def test_mime_for_format_mapping():
    assert _mime_for_format("PNG") == "image/png"
    assert _mime_for_format("jpeg") == "image/jpeg"
    assert _mime_for_format("JPG") == "image/jpeg"
    assert _mime_for_format("WebP") == "image/webp"
    assert _mime_for_format("GIF") == "image/gif"
    assert _mime_for_format("AVIF") == "image/avif"


def test_clipboard_error_is_exception():
    assert issubclass(ClipboardError, Exception)


def test_linux_get_wayland_commands(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if "--list-types" in cmd:
            return _ok("text/plain image/png image/jpeg\n")
        return _ok("PNGDATA")

    monkeypatch.setattr(clipboard, "_run_check", fake_run)
    spec = clipboard._LINUX_CLIPBOARD_BACKENDS["Wayland"]
    data, mime = clipboard._linux_get(spec)
    assert mime == "image/png"
    assert data == "PNGDATA"
    assert calls[0][0] == ["wl-paste", "--list-types"]
    assert calls[1][0] == ["wl-paste", "--type", "image/png"]


def test_linux_get_x11_commands(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "TARGETS" in cmd:
            return _ok("image/png\nimage/jpeg\n")
        return _ok(b"XDATA")

    monkeypatch.setattr(clipboard, "_run_check", fake_run)
    spec = clipboard._LINUX_CLIPBOARD_BACKENDS["X11"]
    data, mime = clipboard._linux_get(spec)
    assert mime == "image/png"
    assert data == b"XDATA"
    assert calls[1] == ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]


def test_linux_get_no_image_raises(monkeypatch):
    monkeypatch.setattr(clipboard, "_run_check",
                        lambda cmd, **kwargs: _ok("text/plain\n"))
    with pytest.raises(ClipboardError, match="No image found"):
        clipboard._linux_get(clipboard._LINUX_CLIPBOARD_BACKENDS["X11"])


def test_linux_set_commands(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _ok()

    monkeypatch.setattr(clipboard, "_run_check", fake_run)
    clipboard._linux_set(clipboard._LINUX_CLIPBOARD_BACKENDS["Wayland"],
                         b"PNG", "image/png")
    assert calls[0][0] == ["wl-copy", "--type", "image/png"]
    assert calls[0][1]["input"] == b"PNG"
    clipboard._linux_set(clipboard._LINUX_CLIPBOARD_BACKENDS["X11"],
                         b"PNG", "image/jpeg")
    assert calls[1][0] == ["xclip", "-selection", "clipboard",
                           "-t", "image/jpeg", "-i"]


class _FakeNSData:
    def __init__(self, payload):
        self._payload = payload

    def bytes(self):
        return self._payload


class _FakeAppKit:
    NSPasteboardTypePNG = "public.png-type"
    NSPasteboardTypeTIFF = "public.tiff-type"


class _FakePasteboard:
    def __init__(self, contents):
        self._contents = contents
        self.requested = []

    def dataForType_(self, utype):
        self.requested.append(utype)
        return self._contents.get(utype)


def test_macos_get_reads_png():
    pb = _FakePasteboard({"public.png-type": _FakeNSData(b"PNGDATA")})
    data, mime = clipboard._macos_get(pb, _FakeAppKit)
    assert (data, mime) == (b"PNGDATA", "image/png")
    assert pb.requested == ["public.png-type"]


def test_macos_get_falls_back_to_tiff():
    pb = _FakePasteboard({"public.tiff-type": _FakeNSData(b"TIFFDATA")})
    data, mime = clipboard._macos_get(pb, _FakeAppKit)
    assert (data, mime) == (b"TIFFDATA", "image/tiff")
    assert pb.requested == ["public.png-type", "public.tiff-type"]


def test_macos_get_prefers_png_over_tiff():
    pb = _FakePasteboard({
        "public.png-type": _FakeNSData(b"PNGDATA"),
        "public.tiff-type": _FakeNSData(b"TIFFDATA"),
    })
    data, mime = clipboard._macos_get(pb, _FakeAppKit)
    assert (data, mime) == (b"PNGDATA", "image/png")


def test_macos_get_no_image_raises():
    pb = _FakePasteboard({})
    with pytest.raises(ClipboardError, match="No image found"):
        clipboard._macos_get(pb, _FakeAppKit)


class _FakeWin32Clipboard:
    def __init__(self):
        self.calls = []

    def OpenClipboard(self):
        self.calls.append("open")

    def EmptyClipboard(self):
        self.calls.append("empty")

    def SetClipboardData(self, fmt, data):
        self.calls.append(("set", fmt, bytes(data)))

    def RegisterClipboardFormat(self, name):
        return 49666

    def CloseClipboard(self):
        self.calls.append("close")


def _fake_win32(monkeypatch):
    import sys
    import types

    win32clipboard = _FakeWin32Clipboard()
    win32con = types.SimpleNamespace(CF_DIB=8)
    monkeypatch.setitem(sys.modules, "win32clipboard", win32clipboard)
    monkeypatch.setitem(sys.modules, "win32con", win32con)
    monkeypatch.setattr(clipboard, "to_dib_bytes", lambda path: b"DIBDATA")
    return win32clipboard, win32con


def test_windows_set_converts_to_dib_and_registers_png(monkeypatch, tmp_path):
    win32clipboard, win32con = _fake_win32(monkeypatch)

    clipboard.set_clipboard_image(b"PNGDATA", "Windows", "PNG")

    assert ("set", win32con.CF_DIB, b"DIBDATA") in win32clipboard.calls
    assert ("set", 49666, b"PNGDATA") in win32clipboard.calls
    assert win32clipboard.calls[0] == "open"
    assert win32clipboard.calls[-1] == "close"


def test_windows_set_jpeg_skips_png_format(monkeypatch, tmp_path):
    win32clipboard, win32con = _fake_win32(monkeypatch)

    clipboard.set_clipboard_image(b"JPGDATA", "Windows", "JPEG")

    assert ("set", win32con.CF_DIB, b"DIBDATA") in win32clipboard.calls
    assert not any(isinstance(c, tuple) and c[1] == 49666
                   for c in win32clipboard.calls)


def test_windows_set_dib_conversion_failure_raises(monkeypatch, tmp_path):
    import sys
    import types

    win32clipboard = _FakeWin32Clipboard()
    win32con = types.SimpleNamespace(CF_DIB=8)
    monkeypatch.setitem(sys.modules, "win32clipboard", win32clipboard)
    monkeypatch.setitem(sys.modules, "win32con", win32con)

    def boom(path):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(clipboard, "to_dib_bytes", boom)
    with pytest.raises(RuntimeError, match="conversion failed"):
        clipboard.set_clipboard_image(b"PNGDATA", "Windows", "PNG")
    assert win32clipboard.calls == []
