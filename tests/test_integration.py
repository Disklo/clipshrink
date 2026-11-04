from __future__ import annotations

import shutil
import struct
import subprocess
import zlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from core.backend import detect_backends


def _have(*tools: str) -> bool:
    return all(shutil.which(tool) for tool in tools)


MAGICK = _have("magick")


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def write_noise_png(path: Path, width: int = 800, height: int = 600) -> int:
    payload = zlib.crc32(b"seed").to_bytes(4, "big")
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            h = (zlib.crc32(payload + struct.pack(">HH", x, y)))
            rows += bytes(((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        + _png_chunk(b"IEND", b""))
    return path.stat().st_size


@pytest.fixture
def real_backend_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import core.deps as deps

    monkeypatch.setattr(deps, "load_config", lambda: {
        "first_run_completed": True, "warn_on_local_bins": False,
        "skip_venv": True,
    })
    monkeypatch.setattr(deps, "save_config", lambda config, path=None: None)
    yield
    detect_backends.cache_clear()


def run_main(*argv: str) -> int:
    import clipshrink

    return clipshrink.main(*argv)


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_png_noise_shrinks_below_budget(
        tmp_path: Path, real_backend_config: None) -> None:
    src = tmp_path / "in.png"
    write_noise_png(src)
    out = tmp_path / "out.png"

    assert run_main("-i", str(src), "-o", str(out), "-s", "100") == 0

    assert out.is_file()
    assert out.stat().st_size <= 100 * 1024
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_jpeg_target_conversion(
        tmp_path: Path, real_backend_config: None) -> None:
    src = tmp_path / "in.png"
    write_noise_png(src)
    out = tmp_path / "out.jpg"

    assert run_main("-i", str(src), "-o", str(out),
                    "-F", "JPEG", "-s", "120") == 0

    assert out.is_file()
    assert out.stat().st_size <= 120 * 1024
    assert out.read_bytes()[:2] == b"\xff\xd8"


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_webp_target_conversion(
        tmp_path: Path, real_backend_config: None) -> None:
    src = tmp_path / "in.png"
    write_noise_png(src)
    out = tmp_path / "out.webp"

    assert run_main("-i", str(src), "-o", str(out),
                    "-F", "WEBP", "-s", "80") == 0

    assert out.is_file()
    assert out.stat().st_size <= 80 * 1024
    data = out.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_avif_target_conversion(
        tmp_path: Path, real_backend_config: None) -> None:
    src = tmp_path / "in.png"
    write_noise_png(src, 320, 240)
    out = tmp_path / "out.avif"

    assert run_main("-i", str(src), "-o", str(out),
                    "-F", "AVIF", "-s", "20") == 0

    assert out.is_file()
    assert out.stat().st_size <= 20 * 1024
    data = out.read_bytes()
    assert data[4:8] == b"ftyp" and b"avif" in data[8:20]


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_raw_bypass_is_byte_identical(
        tmp_path: Path, real_backend_config: None) -> None:
    src = tmp_path / "in.png"
    size = write_noise_png(src, 40, 30)
    out = tmp_path / "copy.png"

    assert run_main("-i", str(src), "-o", str(out), "-r") == 0

    assert out.is_file()
    assert out.stat().st_size == size
    assert out.read_bytes() == src.read_bytes()


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_dry_run_reports_without_writing(
        tmp_path: Path, real_backend_config: None,
        capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "in.png"
    write_noise_png(src)
    out = tmp_path / "out.png"

    assert run_main("-i", str(src), "-o", str(out), "-s", "100",
                    "--dry-run") == 0

    stdout = capsys.readouterr().out
    assert "Dry run (no output written)" in stdout
    assert "source:" in stdout
    assert "target: PNG" in stdout
    assert "plan:" in stdout
    assert not out.exists()


def _animated_gif(path: Path) -> None:
    subprocess.run(
        ["magick", "-size", "64x48", "xc:red", "-delay", "10",
         "-size", "64x48", "xc:blue", "-delay", "10", "-loop", "0",
         str(path)],
        check=True, capture_output=True,
    )


def _frame_count(path: Path) -> int:
    result = subprocess.run(
        ["magick", "identify", "-format", "%n\\n", str(path)],
        check=True, capture_output=True, text=True,
    )
    return int(result.stdout.splitlines()[0])


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_animated_gif_to_avif_preserves_frames(
        tmp_path: Path, real_backend_config: None) -> None:
    src = tmp_path / "anim.gif"
    out = tmp_path / "out.avif"
    _animated_gif(src)

    assert run_main("-i", str(src), "-o", str(out),
                    "-F", "AVIF", "-s", "200") == 0

    assert out.is_file()
    assert out.read_bytes()[4:8] == b"ftyp"
    assert _frame_count(out) > 1


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_animated_avif_to_gif_preserves_frames(
        tmp_path: Path, real_backend_config: None) -> None:
    gif = tmp_path / "anim.gif"
    src = tmp_path / "anim.avif"
    out = tmp_path / "out.gif"
    _animated_gif(gif)
    try:
        subprocess.run(["magick", str(gif), str(src)],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip("ImageMagick cannot write animated AVIF here")

    assert run_main("-i", str(src), "-o", str(out),
                    "-F", "GIF", "-s", "200") == 0

    assert out.is_file()
    assert out.read_bytes()[:3] == b"GIF"
    assert _frame_count(out) > 1


@pytest.mark.integration
@pytest.mark.skipif(not MAGICK, reason="ImageMagick (magick) not installed")
def test_animated_avif_to_webp_preserves_frames(
        tmp_path: Path, real_backend_config: None) -> None:
    gif = tmp_path / "anim.gif"
    src = tmp_path / "anim.avif"
    out = tmp_path / "out.webp"
    _animated_gif(gif)
    try:
        subprocess.run(["magick", str(gif), str(src)],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip("ImageMagick cannot write animated AVIF here")

    assert run_main("-i", str(src), "-o", str(out),
                    "-F", "WEBP", "-s", "200") == 0

    assert out.is_file()
    data = out.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    assert _frame_count(out) > 1
