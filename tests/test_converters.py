import subprocess
import sys
from pathlib import Path

import pytest

from core import converters
from core.converters import (
    avif_save,
    filter_to_vips_kernel,
    get_image_format,
    heic_save,
    jp2_save,
    jpegli_save,
    jxl_save,
    mozjpeg_save,
    tiff_jpeg_save,
    vips_supports_format,
    webp_save,
)


def _ok_result():
    return subprocess.CompletedProcess([], 0, "", "")


def test_jpegli_save_builds_command(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    monkeypatch.setattr(converters, "_to_ppm", lambda i, o: True)
    monkeypatch.setattr(converters.utils, "run_capture",
                        lambda cmd, **k: captured.update(cmd=cmd) or _ok_result())
    out = tmp_path / "out.jpg"
    assert jpegli_save(src, out, 85, subsampling="4:4:4") is True
    cmd = captured["cmd"]
    assert cmd[0] == "cjpegli"
    assert cmd[1:4] == ["-q", "85", "-c"]
    assert cmd[4] == "444"
    ppm = [a for a in cmd if a.endswith(".ppm")]
    assert len(ppm) == 1
    assert not Path(ppm[0]).exists()


def test_jpegli_save_missing_binary_returns_false(monkeypatch, tmp_path):
    from core import converters

    def no_binary(cmd, **kwargs):
        raise FileNotFoundError("cjpegli")

    monkeypatch.setattr(converters, "_to_ppm", lambda i, o: True)
    monkeypatch.setattr(converters.utils, "run_capture", no_binary)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    assert jpegli_save(src, tmp_path / "out.jpg", 85) is False
    assert converters.utils.get_last_error()


def test_mozjpeg_save_builds_command(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    monkeypatch.setattr(converters, "_to_ppm", lambda i, o: True)
    monkeypatch.setattr(converters.utils, "run_capture",
                        lambda cmd, **k: captured.update(cmd=cmd) or _ok_result())
    out = tmp_path / "out.jpg"
    assert mozjpeg_save(src, out, 90, subsampling="4:2:2") is True
    cmd = captured["cmd"]
    assert cmd[0] == "cjpeg"
    assert "-quality" in cmd and cmd[cmd.index("-quality") + 1] == "90"
    assert "-sample" in cmd and cmd[cmd.index("-sample") + 1] == "2x1"
    assert "-progressive" in cmd
    assert cmd[cmd.index("-outfile") + 1] == str(out)


def test_avif_save_prefers_avifenc(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    monkeypatch.setattr(converters.shutil, "which",
                        lambda c: "/usr/bin/avifenc" if c == "avifenc" else None)
    monkeypatch.setattr(converters, "run_capture",
                        lambda cmd, **k: captured.update(cmd=cmd) or _ok_result())
    out = tmp_path / "out.avif"
    assert avif_save(src, out, 80, speed=1) is True
    cmd = captured["cmd"]
    assert cmd[0] == "avifenc"
    assert cmd[1:5] == ["-q", "80", "--speed", "1"]


def test_avif_save_uses_hardware_codec_when_available(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    monkeypatch.setattr(converters.shutil, "which",
                        lambda c: "/usr/bin/avifenc" if c == "avifenc" else None)
    monkeypatch.setattr(converters, "_avif_hw_codec", lambda: "nvenc")
    monkeypatch.setattr(converters, "run_capture",
                        lambda cmd, **k: captured.update(cmd=cmd) or _ok_result())
    out = tmp_path / "out.avif"
    assert avif_save(src, out, 80) is True
    cmd = captured["cmd"]
    assert cmd[0] == "avifenc"
    assert cmd[cmd.index("--codec") + 1] == "nvenc"


def test_avif_save_falls_back_to_software_when_hw_encode_fails(monkeypatch, tmp_path):
    from core import converters

    captured = []
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    def fake_run(cmd, **k):
        captured.append(cmd)
        if "--codec" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "nvenc error")
        return _ok_result()

    monkeypatch.setattr(converters.shutil, "which",
                        lambda c: "/usr/bin/avifenc" if c == "avifenc" else None)
    monkeypatch.setattr(converters, "_avif_hw_codec", lambda: "nvenc")
    monkeypatch.setattr(converters, "run_capture", fake_run)
    out = tmp_path / "out.avif"
    assert avif_save(src, out, 80) is True
    assert "--codec" in captured[0]
    assert "--codec" not in captured[1]


def test_avif_save_prefers_software_when_requested(monkeypatch, tmp_path):
    from core import converters

    captured = []
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    def fake_run(cmd, **k):
        captured.append(cmd)
        return _ok_result()

    monkeypatch.setattr(converters.shutil, "which",
                        lambda c: "/usr/bin/avifenc" if c == "avifenc" else None)
    monkeypatch.setattr(converters, "_avif_hw_codec", lambda: "nvenc")
    monkeypatch.setattr(converters, "run_capture", fake_run)
    out = tmp_path / "out.avif"
    assert avif_save(src, out, 80, prefer_software=True) is True
    assert len(captured) == 1
    assert "--codec" not in captured[0]


def test_avif_save_software_falls_back_to_hardware_when_requested(monkeypatch, tmp_path):
    from core import converters

    captured = []
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    def fake_run(cmd, **k):
        captured.append(cmd)
        if "--codec" in cmd:
            return _ok_result()
        return subprocess.CompletedProcess(cmd, 1, "", "software error")

    monkeypatch.setattr(converters.shutil, "which",
                        lambda c: "/usr/bin/avifenc" if c == "avifenc" else None)
    monkeypatch.setattr(converters, "_avif_hw_codec", lambda: "nvenc")
    monkeypatch.setattr(converters, "run_capture", fake_run)
    out = tmp_path / "out.avif"
    assert avif_save(src, out, 80, prefer_software=True) is True
    assert len(captured) == 2
    assert "--codec" not in captured[0]
    assert captured[1][captured[1].index("--codec") + 1] == "nvenc"


def test_avif_hw_probe_skips_unavailable_codecs(monkeypatch, tmp_path):
    from core import converters

    converters._avif_hw_codec.cache_clear()
    try:
        def fake_run(cmd, **k):
            if "--codec" in cmd and "nvenc" in cmd:
                out = Path(next(a for a in cmd if a.endswith(".avif")))
                out.write_bytes(b"\x00" * 8)
                return _ok_result()
            return subprocess.CompletedProcess(cmd, 1, "", "codec unavailable")

        monkeypatch.setattr(converters.sys, "platform", "linux")
        monkeypatch.setattr(converters, "run_capture", fake_run)
        assert converters._avif_hw_codec() == "nvenc"
    finally:
        converters._avif_hw_codec.cache_clear()


def test_avif_save_falls_back_to_vips_when_avifenc_missing(monkeypatch, tmp_path):
    from core import converters

    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    calls = []

    monkeypatch.setattr(converters.shutil, "which", lambda c: None)

    class FakeImage:
        def __init__(self, *a, **k):
            pass

        @classmethod
        def new_from_file(cls, path):
            return cls()

        def write_to_file(self, path, **kwargs):
            calls.append((path, kwargs))

    monkeypatch.setattr(converters.pyvips, "Image", FakeImage)
    assert avif_save(src, tmp_path / "out.avif", 80) is True
    assert calls[0][1]["Q"] == 80


def test_webp_save_falls_back_to_magick(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    def raise_exc(*a, **k):
        raise Exception("vips unavailable")

    monkeypatch.setattr(converters.pyvips.Image, "new_from_file", raise_exc)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: captured.update(cmd=cmd) or _ok_result())
    out = tmp_path / "out.webp"
    assert webp_save(src, out, 75, strip_metadata=True) is True
    cmd = captured["cmd"]
    assert cmd[0] == "magick"
    assert "-quality" in cmd and cmd[cmd.index("-quality") + 1] == "75"
    assert cmd[-1] == str(out)


def _vips_save_builds_command(saver, name):
    def test_fn(monkeypatch, tmp_path):
        from core import converters

        src = tmp_path / "in.png"
        src.write_bytes(b"x")
        calls = []

        class FakeImage:
            def __init__(self, *a, **k):
                pass

            @classmethod
            def new_from_file(cls, path):
                return cls()

            def write_to_file(self, path, **kwargs):
                calls.append((path, kwargs))

        monkeypatch.setattr(converters.pyvips, "Image", FakeImage)
        monkeypatch.setattr(converters.heif, "heic_hw_save",
                            lambda *a, **k: False)
        out = tmp_path / f"out.{name}"
        assert saver(src, out, 80, strip_metadata=True) is True
        assert calls[0][1]["Q"] == 80
        assert calls[0][1]["strip"] is True
    return test_fn


test_jxl_save_vips_path = _vips_save_builds_command(jxl_save, "jxl")
test_jp2_save_vips_path = _vips_save_builds_command(jp2_save, "jp2")
test_heic_save_vips_path = _vips_save_builds_command(heic_save, "heic")


def test_heic_save_uses_hevc_compression(monkeypatch, tmp_path):
    from core import converters

    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    calls = []

    class FakeImage:
        @classmethod
        def new_from_file(cls, path):
            return cls()

        def write_to_file(self, path, **kwargs):
            calls.append((path, kwargs))

    monkeypatch.setattr(converters.pyvips, "Image", FakeImage)
    monkeypatch.setattr(converters.heif, "heic_hw_save",
                        lambda *a, **k: False)
    assert heic_save(src, tmp_path / "out.heic", 80) is True
    assert calls[0][1]["compression"] == "hevc"


def test_heic_save_prefers_software_when_requested(monkeypatch, tmp_path):
    from core import converters

    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    order = []

    class FakeImage:
        @classmethod
        def new_from_file(cls, path):
            return cls()

        def write_to_file(self, path, **kwargs):
            order.append("vips")

    monkeypatch.setattr(converters.pyvips, "Image", FakeImage)
    monkeypatch.setattr(converters.heif, "heic_hw_save",
                        lambda *a, **k: order.append("hw") or True)
    assert converters.heic_save(src, tmp_path / "out.heic", 80,
                                prefer_software=True) is True
    assert order == ["vips"]


def test_heic_save_prefers_hardware_by_default(monkeypatch, tmp_path):
    from core import converters

    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    order = []

    class FakeImage:
        @classmethod
        def new_from_file(cls, path):
            return cls()

        def write_to_file(self, path, **kwargs):
            order.append("vips")

    monkeypatch.setattr(converters.pyvips, "Image", FakeImage)
    monkeypatch.setattr(converters.heif, "heic_hw_save",
                        lambda *a, **k: order.append("hw") or True)
    assert converters.heic_save(src, tmp_path / "out.heic", 80) is True
    assert order == ["hw"]


def test_heic_save_software_falls_back_to_hardware(monkeypatch, tmp_path):
    from core import converters

    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    class BrokenImage:
        @classmethod
        def new_from_file(cls, path):
            raise RuntimeError("vips unavailable")

    calls = []
    monkeypatch.setattr(converters.pyvips, "Image", BrokenImage)
    monkeypatch.setattr(converters.heif, "heic_hw_save",
                        lambda *a, **k: calls.append(1) or True)
    assert converters.heic_save(src, tmp_path / "out.heic", 80,
                                prefer_software=True) is True
    assert len(calls) == 1


def test_tiff_jpeg_save_vips_path(monkeypatch, tmp_path):
    from core import converters

    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    calls = []

    class FakeImage:
        @classmethod
        def new_from_file(cls, path):
            return cls()

        def write_to_file(self, path, **kwargs):
            calls.append((path, kwargs))

    monkeypatch.setattr(converters.pyvips, "Image", FakeImage)
    assert tiff_jpeg_save(src, tmp_path / "out.tiff", 80) is True
    assert calls[0][1] == {"compression": "jpeg", "Q": 80}


def test_tiff_jpeg_save_magick_fallback(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.png"
    src.write_bytes(b"x")

    def raise_exc(*a, **k):
        raise Exception("vips unavailable")

    monkeypatch.setattr(converters.pyvips.Image, "new_from_file", raise_exc)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: captured.update(cmd=cmd) or _ok_result())
    assert tiff_jpeg_save(src, tmp_path / "out.tiff", 80) is True
    cmd = captured["cmd"]
    assert "-compress" in cmd and cmd[cmd.index("-compress") + 1] == "JPEG"
    assert "-quality" in cmd and cmd[cmd.index("-quality") + 1] == "80"


def test_filter_names_map_to_kernels():
    cases = {
        "lanczos": "lanczos3",
        "Lanczos": "lanczos3",
        "nearest": "nearest",
        "point": "nearest",
        "box": "box",
        "catrom": "cubic",
        "bicubic": "cubic",
        "cubic": "cubic",
        "mitchell": "mitchell",
    }
    for name, kernel in cases.items():
        assert filter_to_vips_kernel(name) == kernel


def test_unknown_filter_defaults_to_lanczos3():
    assert filter_to_vips_kernel("nonexistent-filter") == "lanczos3"


def test_vips_supported_formats():
    assert vips_supports_format("PNG")
    assert vips_supports_format("jpeg")
    assert vips_supports_format("webp")
    assert not vips_supports_format("psd")
    assert not vips_supports_format("pdf")


def test_get_image_format_returns_png(tmp_path):
    pyvips = pytest.importorskip("pyvips")
    image_path = tmp_path / "test.png"
    pyvips.Image.black(10, 10).write_to_file(str(image_path))
    assert get_image_format(image_path) == "PNG"


def test_get_frame_count_uses_vips_n_pages(tmp_path, monkeypatch):
    from core import converters

    class FakeVips:
        class Image:
            @classmethod
            def new_from_file(cls, path):
                return cls()

            def get(self, field):
                return 12

    monkeypatch.setattr(converters, "pyvips", FakeVips)
    image_path = tmp_path / "anim.gif"
    image_path.write_bytes(b"x")
    assert converters.get_frame_count(image_path) == 12


def test_get_frame_count_falls_back_to_magick(tmp_path, monkeypatch):
    from core import converters

    class NoPages:
        class Image:
            @classmethod
            def new_from_file(cls, path):
                return cls()

            def get(self, field):
                raise Exception("field not found")

    captured = {}
    monkeypatch.setattr(converters, "pyvips", NoPages)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: captured.update(cmd=cmd)
                        or subprocess.CompletedProcess(cmd, 0, "3", ""))
    image_path = tmp_path / "anim.gif"
    image_path.write_bytes(b"x")
    assert converters.get_frame_count(image_path) == 3
    assert captured["cmd"] == ["magick", "identify", "-format", "%n\n", str(image_path)]


def test_get_frame_count_sans_vips_uses_magick(tmp_path, monkeypatch):
    from core import converters

    captured = {}
    monkeypatch.setattr(converters, "pyvips", None)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: captured.update(cmd=cmd)
                        or subprocess.CompletedProcess(cmd, 0, "7", ""))
    image_path = tmp_path / "anim.gif"
    image_path.write_bytes(b"x")
    assert converters.get_frame_count(image_path) == 7
    assert captured["cmd"] == ["magick", "identify", "-format", "%n\n", str(image_path)]


def test_get_frame_count_checks_magick_when_vips_reports_one_page(
        tmp_path, monkeypatch):
    from core import converters

    class OnePage:
        class Image:
            @classmethod
            def new_from_file(cls, path):
                return cls()

            def get(self, field):
                return 1

    monkeypatch.setattr(converters, "pyvips", OnePage)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: subprocess.CompletedProcess(cmd, 0, "3", ""))
    image_path = tmp_path / "anim.avif"
    image_path.write_bytes(b"x")
    assert converters.get_frame_count(image_path) == 3


def test_get_frame_count_trusts_vips_when_magick_cannot_read(
        tmp_path, monkeypatch):
    from core import converters

    class OnePage:
        class Image:
            @classmethod
            def new_from_file(cls, path):
                return cls()

            def get(self, field):
                return 1

    monkeypatch.setattr(converters, "pyvips", OnePage)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "no"))
    image_path = tmp_path / "still.bin"
    image_path.write_bytes(b"x")
    assert converters.get_frame_count(image_path) == 1


def test_get_frame_count_uses_avifdec_for_animated_avif(tmp_path, monkeypatch):
    from core import converters

    class OnePage:
        class Image:
            @classmethod
            def new_from_file(cls, path):
                return cls()

            def get(self, field):
                return 1

    monkeypatch.setattr(converters, "pyvips", OnePage)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: subprocess.CompletedProcess(cmd, 0, "1", ""))
    monkeypatch.setattr(converters.shutil, "which",
                        lambda tool: "/usr/bin/avifdec" if tool == "avifdec" else None)
    monkeypatch.setattr(
        converters, "run_capture",
        lambda cmd: subprocess.CompletedProcess(
            cmd, 0, "", "Image Sequence Frames: (4 expected frames)"))
    image_path = tmp_path / "anim.avif"
    image_path.write_bytes(b"x")
    assert converters.get_frame_count(image_path) == 4


def test_get_image_format_handles_repeated_animation_output(tmp_path, monkeypatch):
    from core import converters

    monkeypatch.setattr(converters, "pyvips", None)
    monkeypatch.setattr(
        converters, "_run_magick",
        lambda cmd: subprocess.CompletedProcess(cmd, 0, "AVIF\nAVIF\nAVIF\n", ""))
    image_path = tmp_path / "anim.avif"
    image_path.write_bytes(b"x")
    assert converters.get_image_format(image_path) == "AVIF"


def test_is_apng_detects_actl_chunk(tmp_path):
    from core import converters

    def chunk(tag, data):
        import struct
        import zlib
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data)))

    ihdr = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"acTL", b"\x00\x00\x00\x01\x00\x00\x00\x00")
           + chunk(b"IDAT", b"") + chunk(b"IEND", b""))
    image_path = tmp_path / "anim.png"
    image_path.write_bytes(png)
    assert converters._is_apng(image_path) is True
    assert converters._magick_spec(image_path) == f"APNG:{image_path}"

    still = tmp_path / "still.png"
    still.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
                      + chunk(b"IDAT", b"") + chunk(b"IEND", b""))
    assert converters._is_apng(still) is False
    assert converters._magick_spec(still) == str(still)


def test_convert_animation_builds_full_sequence_command(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.avif"
    out = tmp_path / "out.gif"
    src.write_bytes(b"x")

    def fake_run(cmd):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"g")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(converters, "_run_magick", fake_run)
    assert converters.convert_animation(src, out) is True
    cmd = captured["cmd"]
    assert cmd[0] == "magick"
    assert "-coalesce" in cmd
    assert cmd[-3:] == ["-layers", "Optimize", str(out)]


def test_convert_animation_tiff_uses_jpeg_compression(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.gif"
    out = tmp_path / "out.tiff"
    src.write_bytes(b"x")

    def fake_run(cmd):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(converters, "_run_magick", fake_run)
    converters.convert_animation(src, out, quality=42)
    cmd = captured["cmd"]
    assert cmd[cmd.index("-compress") + 1] == "JPEG"
    assert cmd[cmd.index("-quality") + 1] == "42"


def test_gif_resize_apng_output_uses_apng_coder(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.gif"
    out = tmp_path / "out.apng"
    src.write_bytes(b"x")

    def fake_run(cmd):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(converters, "_run_magick", fake_run)
    converters.gif_resize(src, out, "50%")
    assert captured["cmd"][-3:] == ["-layers", "Optimize", f"APNG:{out}"]


def test_gif_resize_builds_coalesce_resize_command(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.gif"
    src.write_bytes(b"x")

    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: captured.update(cmd=cmd) or _ok_result())
    out = tmp_path / "out.gif"
    assert converters.gif_resize(src, out, "50%", kernel="Mitchell",
                                 strip_metadata=True) is True
    cmd = captured["cmd"]
    assert cmd[:3] == ["magick", str(src), "-coalesce"]
    assert "-filter" in cmd and cmd[cmd.index("-filter") + 1] == "Mitchell"
    assert "-resize" in cmd and cmd[cmd.index("-resize") + 1] == "50%"
    assert "-strip" in cmd
    assert cmd[-3:] == ["-layers", "Optimize", str(out)]


def test_gif_resize_nearest_uses_sample(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.gif"
    src.write_bytes(b"x")

    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: captured.update(cmd=cmd) or _ok_result())
    out = tmp_path / "out.gif"
    assert converters.gif_resize(src, out, "400x300!", nearest=True) is True
    cmd = captured["cmd"]
    assert "-sample" in cmd and cmd[cmd.index("-sample") + 1] == "400x300!"
    assert "-resize" not in cmd
    assert "-filter" not in cmd


def test_gif_resize_failure_raises(monkeypatch, tmp_path):
    from core import converters

    def fail(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "magick boom")

    monkeypatch.setattr(converters, "_run_magick", fail)
    src = tmp_path / "in.gif"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError):
        converters.gif_resize(src, tmp_path / "out.gif", "50%")


def _resize_setup(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    monkeypatch.setattr(converters, "pyvips", None)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: captured.update(cmd=cmd) or _ok_result())
    return converters, src, captured


def test_resize_exact_builds_magick_command(monkeypatch, tmp_path):
    converters, src, captured = _resize_setup(monkeypatch, tmp_path)
    out = tmp_path / "out.png"
    assert converters.resize_exact(src, out, 640, 480, kernel="Mitchell") is True
    cmd = captured["cmd"]
    assert cmd[:2] == ["magick", str(src)]
    assert "-filter" in cmd and cmd[cmd.index("-filter") + 1] == "Mitchell"
    assert "-resize" in cmd and cmd[cmd.index("-resize") + 1] == "640x480!"
    assert cmd[-1] == str(out)


def test_resize_fit_width_and_height_commands(monkeypatch, tmp_path):
    converters, src, captured = _resize_setup(monkeypatch, tmp_path)
    out = tmp_path / "out.png"
    converters.resize_fit_width(src, out, 800)
    assert captured["cmd"][captured["cmd"].index("-resize") + 1] == "800x"
    converters.resize_fit_height(src, out, 600)
    assert captured["cmd"][captured["cmd"].index("-resize") + 1] == "x600"


def test_resize_pct_builds_magick_command(monkeypatch, tmp_path):
    converters, src, captured = _resize_setup(monkeypatch, tmp_path)
    out = tmp_path / "out.png"
    converters.resize_pct(src, out, 50)
    assert captured["cmd"][captured["cmd"].index("-resize") + 1] == "50%"


def test_sample_pct_uses_sample_without_filter(monkeypatch, tmp_path):
    converters, src, captured = _resize_setup(monkeypatch, tmp_path)
    out = tmp_path / "out.png"
    converters.sample_pct(src, out, 50)
    cmd = captured["cmd"]
    assert "-sample" in cmd and cmd[cmd.index("-sample") + 1] == "50%"
    assert "-filter" not in cmd
    assert "-resize" not in cmd


def test_resize_strip_metadata_appends_strip(monkeypatch, tmp_path):
    converters, src, captured = _resize_setup(monkeypatch, tmp_path)
    out = tmp_path / "out.png"
    converters.resize_pct(src, out, 50, strip_metadata=True)
    assert "-strip" in captured["cmd"]


def test_resize_failure_raises(monkeypatch, tmp_path):
    from core import converters

    def fail(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "magick boom")

    monkeypatch.setattr(converters, "pyvips", None)
    monkeypatch.setattr(converters, "_run_magick", fail)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="Failed to resize"):
        converters.resize_exact(src, tmp_path / "out.png", 640, 480)


def test_resize_exact_vips_returns_false_without_vips(monkeypatch, tmp_path):
    monkeypatch.setattr(converters, "pyvips", None)
    assert converters.resize_exact_vips(tmp_path / "in.png", tmp_path / "out.png",
                                        10, 10) is False


def test_resize_exact_vips_uses_vscale_keyword(monkeypatch, tmp_path):
    calls = {}

    class FakeImage:
        width = 100
        height = 100

        def resize(self, scale, **kwargs):
            calls["args"] = (scale, kwargs)
            return self

        def write_to_file(self, path, **kwargs):
            calls["out"] = (path, kwargs)

    fake_pyvips = type(sys)("fake_pyvips")
    fake_pyvips.Image = type("Image", (), {
        "new_from_file": staticmethod(lambda path: FakeImage()),
    })
    monkeypatch.setattr(converters, "pyvips", fake_pyvips)
    assert converters.resize_exact_vips(tmp_path / "in.png", tmp_path / "out.png",
                                        50, 40) is True
    scale, kwargs = calls["args"]
    assert scale == 0.5
    assert kwargs["vscale"] == 0.4
    assert kwargs["kernel"] == "lanczos3"


def test_jpeg_save_vips_path_uses_subsample_mode(monkeypatch, tmp_path):
    calls = {}

    class FakeImage:
        def write_to_file(self, path, **kwargs):
            calls["kwargs"] = kwargs
            calls["path"] = path

    def fake_new(path, **kwargs):
        calls["access"] = kwargs.get("access")
        return FakeImage()

    fake = type(sys)("fake_pyvips")
    fake.Image = type("Image", (), {"new_from_file": staticmethod(fake_new)})
    monkeypatch.setattr(converters, "pyvips", fake)
    assert converters.jpeg_save(tmp_path / "in.png", tmp_path / "out.jpg", 85,
                                subsampling="4:2:0", optimize=True,
                                progressive=True) is True
    assert calls["access"] == "sequential"
    assert calls["kwargs"] == {"Q": 85, "subsample_mode": "on",
                               "optimize_coding": True, "interlace": True}


def test_jpeg_save_444_maps_to_subsample_mode_off(monkeypatch, tmp_path):
    calls = {}

    class FakeImage:
        def write_to_file(self, path, **kwargs):
            calls["kwargs"] = kwargs

    fake = type(sys)("fake_pyvips")
    fake.Image = type("Image", (), {
        "new_from_file": staticmethod(lambda path, **k: FakeImage())})
    monkeypatch.setattr(converters, "pyvips", fake)
    assert converters.jpeg_save(tmp_path / "in.png", tmp_path / "out.jpg", 85,
                                subsampling="4:4:4") is True
    assert calls["kwargs"]["subsample_mode"] == "off"


def test_jpeg_save_422_falls_back_to_magick(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(converters, "run_capture",
                        lambda cmd: (captured.setdefault("cmd", cmd),
                                     subprocess.CompletedProcess(cmd, 0, "", ""))[1])
    assert converters.jpeg_save(tmp_path / "in.png", tmp_path / "out.jpg", 85,
                                subsampling="4:2:2") is True
    assert "-sampling-factor" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-sampling-factor") + 1] == "4:2:2"


def test_to_png_lossless_magick_selects_first_frame(monkeypatch, tmp_path):
    from core import converters

    captured = {}
    src = tmp_path / "anim.gif"
    out = tmp_path / "out.png"
    src.write_bytes(b"x")

    def fake_run(cmd):
        captured["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(converters, "pyvips", None)
    monkeypatch.setattr(converters, "_run_magick", fake_run)
    assert converters.to_png_lossless(src, out) is True
    assert captured["cmd"][1] == f"{src}[0]"


def test_to_png_lossless_ffmpeg_rescue(monkeypatch, tmp_path):
    src = tmp_path / "in.heic"
    src.write_bytes(b"x")

    def fail_magick(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "libheif error")

    def fake_run_capture(cmd, **kwargs):
        assert cmd[0] == "ffmpeg" and "-nostdin" in cmd
        Path(cmd[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(converters, "_run_magick", fail_magick)
    monkeypatch.setattr(converters.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(converters, "run_capture", fake_run_capture)
    out = tmp_path / "out.png"
    assert converters.to_png_lossless(src, out) is True


def test_to_png_lossless_ffmpeg_rescue_skips_multiframe(monkeypatch, tmp_path):
    src = tmp_path / "anim.heic"
    src.write_bytes(b"x")

    class FakeImage:
        def get(self, name):
            return 2

    fake = type(sys)("fake_pyvips")
    fake.Image = type("Image", (), {
        "new_from_file": staticmethod(lambda path, **k: FakeImage())})
    monkeypatch.setattr(converters, "pyvips", fake)
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "err"))
    monkeypatch.setattr(converters.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(converters, "run_capture",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "", ""))
    with pytest.raises(RuntimeError):
        converters.to_png_lossless(src, tmp_path / "out.png")


def test_to_png_lossless_all_backends_fail(monkeypatch, tmp_path):
    src = tmp_path / "in.heic"
    src.write_bytes(b"x")
    monkeypatch.setattr(converters, "_run_magick",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "err"))
    monkeypatch.setattr(converters.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError):
        converters.to_png_lossless(src, tmp_path / "out.png")


def test_to_dib_bytes_strips_bmp_file_header(monkeypatch, tmp_path):
    from core.converters import to_dib_bytes

    captured = {}

    def fake_run_magick(cmd):
        captured["cmd"] = cmd
        bmp = Path(cmd[-1].split(":", 1)[1])
        bmp.write_bytes(b"BM" + b"\x00" * 12 + b"HEADER+PIXELS")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(converters, "_run_magick", fake_run_magick)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    dib = to_dib_bytes(src)
    assert captured["cmd"][0] == "magick"
    assert captured["cmd"][1] == str(src)
    assert captured["cmd"][2].startswith("BMP3:")
    assert dib == b"HEADER+PIXELS"
    assert list(tmp_path.glob("*.bmp")) == []


def test_to_dib_bytes_failure_raises_backend_error(monkeypatch, tmp_path):
    from core.converters import BackendError, to_dib_bytes

    def fake_run_magick(cmd):
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(converters, "_run_magick", fake_run_magick)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    with pytest.raises(BackendError, match="DIB conversion failed"):
        to_dib_bytes(src)


def test_to_dib_bytes_rejects_non_bmp_output(monkeypatch, tmp_path):
    from core.converters import BackendError, to_dib_bytes

    def fake_run_magick(cmd):
        bmp = Path(cmd[-1].split(":", 1)[1])
        bmp.write_bytes(b"not a bitmap at all, definitely longer than 14 bytes")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(converters, "_run_magick", fake_run_magick)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    with pytest.raises(BackendError, match="no bitmap data"):
        to_dib_bytes(src)
