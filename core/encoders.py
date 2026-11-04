from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from . import converters, utils
from .backend import BackendInfo, detect_backends


class SpeedProfile(TypedDict):
    avif_speed: int
    pngquant_speed: int
    oxipng_level: int | None
    search_step: int | None
    color_step: int | None
    max_iterations: int


def speed_profile(speed: str) -> SpeedProfile:
    # --slowest is explicit in core/presets.py
    profiles: dict[str, SpeedProfile] = {
        "default": {
            "avif_speed": 8,
            "pngquant_speed": 3,
            "oxipng_level": 0,
            "search_step": 3,
            "color_step": 16,
            "max_iterations": 40,
        },
        "slow": {
            "avif_speed": 4,
            "pngquant_speed": 1,
            "oxipng_level": 2,
            "search_step": None,
            "color_step": None,
            "max_iterations": 100,
        },
        "fast": {
            "avif_speed": 10,
            "pngquant_speed": 5,
            "oxipng_level": None,
            "search_step": 10,
            "color_step": 32,
            "max_iterations": 10,
        },
    }
    return profiles[speed]


def pngquant_compress(
    input_path: Path,
    output_path: Path,
    speed: int = 1,
    no_dithering: bool = False,
) -> bool:
    cmd = ["pngquant", "--force", "--speed", str(speed)]
    if no_dithering:
        cmd.append("--nofs")
    cmd.extend(["--output", str(output_path), "--", str(input_path)])
    try:
        result = utils.run_capture(cmd)
    except FileNotFoundError:
        utils.set_last_error("pngquant is not installed.")
        return False
    if result.returncode != 0:
        utils.set_last_error((result.stderr or "").strip() or (result.stdout or "").strip())
        return False
    return True


def oxipng_compress(
    input_path: Path,
    output_path: Path,
    level: int = 2,
    strip_metadata: bool = False,
) -> bool:
    cmd = ["oxipng", "-o", str(level), "--strip", "all" if strip_metadata else "safe",
           "--out", str(output_path), "--", str(input_path)]
    try:
        result = utils.run_capture(cmd)
    except FileNotFoundError:
        utils.set_last_error("oxipng is not installed.")
        return False
    if result.returncode != 0:
        utils.set_last_error((result.stderr or "").strip() or (result.stdout or "").strip())
        return False
    return True


# Probe contract: encode input at a quality knob. Codecs without chroma
# ignore subsampling. BackendInfo is captured at build time.
EncodeFn = Callable[..., bool]


@dataclass(frozen=True)
class CodecEncoder:
    name: str
    encode: EncodeFn
    extent: Callable[..., bool] | None = None
    subsampling_supported: bool = False
    suffix: str = ".jpg"


def _make_encoder(name: str, suffix: str, saver: Callable[..., bool],
                  **fixed: Any) -> CodecEncoder:
    def encode(input_path: Path, output_path: Path, quality: int,
               subsampling: str = "4:2:0", strip_metadata: bool = False) -> bool:
        try:
            return saver(input_path, output_path, quality,
                         strip_metadata=strip_metadata, **fixed)
        except RuntimeError as e:
            utils.set_last_error(str(e))
            return False

    return CodecEncoder(name=name, encode=encode, suffix=suffix)


def _jpeg_extent(input_path: Path, output_path: Path, max_bytes: int,
                 subsampling: str = "4:2:0", strip_metadata: bool = False) -> bool:
    return converters.jpeg_save_extent(input_path, output_path, max_bytes, subsampling, strip_metadata)


def _make_jpeg_encoder(backend: BackendInfo | None, use_extent: bool) -> CodecEncoder:
    # progressive probes are smaller at clipshrink sizes
    def encode(input_path: Path, output_path: Path, quality: int,
               subsampling: str = "4:2:0", strip_metadata: bool = False) -> bool:
        return jpeg_quality_reduce(
            input_path, output_path, quality,
            strip_metadata=strip_metadata, subsampling=subsampling,
            optimize=True, progressive=True, backend=backend,
        )

    return CodecEncoder("jpeg", encode,
                        extent=_jpeg_extent if use_extent else None,
                        subsampling_supported=True)


def make_lossy_encoders(backend: BackendInfo, slowest: bool = False,
                        speed: str = "default", avif_speed: int | None = None,
                        prefer_software: bool | None = None) -> dict[str, CodecEncoder]:
    use_extent = speed != "slow" or not (backend.jpegli or backend.mozjpeg)
    if avif_speed is None:
        avif_speed = speed_profile(speed)["avif_speed"]
    if prefer_software is None:
        prefer_software = slowest
    return {
        "JPEG": _make_jpeg_encoder(backend, use_extent),
        "WEBP": _make_encoder("webp", ".webp", converters.webp_save),
        "AVIF": _make_encoder("avif", ".avif", converters.avif_save,
                              speed=1 if slowest else avif_speed,
                              prefer_software=prefer_software),
        "JXL": _make_encoder("jxl", ".jxl", converters.jxl_save),
        "JP2": _make_encoder("jp2", ".jp2", converters.jp2_save),
        "HEIC": _make_encoder("heic", ".heic", converters.heic_save,
                              prefer_software=prefer_software),
        "TIFF": _make_encoder("tiff", ".tiff", converters.tiff_jpeg_save),
    }


def jpeg_quality_reduce(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
    subsampling: str = "4:2:0",
    optimize: bool = True,
    progressive: bool = True,
    backend: BackendInfo | None = None,
) -> bool:
    if backend is None:
        backend = detect_backends()
    if backend.jpegli:
        return converters.jpegli_save(
            input_path, output_path, quality, subsampling=subsampling,
            strip_metadata=strip_metadata,
        )
    if backend.mozjpeg:
        return converters.mozjpeg_save(
            input_path, output_path, quality, subsampling=subsampling,
            strip_metadata=strip_metadata,
        )
    try:
        return converters.jpeg_save(
            input_path,
            output_path,
            quality,
            subsampling=subsampling,
            optimize=optimize,
            progressive=progressive,
            strip_metadata=strip_metadata,
        )
    except RuntimeError as e:
        utils.set_last_error(str(e))
        return False


def color_palette_reduce(
    input_path: Path,
    output_path: Path,
    num_colors: int,
    strip_metadata: bool = False,
) -> bool:
    # PNG sources prefer pngquant, then magick -colors
    fmt = ""
    with contextlib.suppress(RuntimeError):
        fmt = converters.get_image_format(input_path)

    if fmt == "PNG":
        try:
            result = utils.run_capture([
                "pngquant", "--force", "--colors", str(num_colors),
                "--output", str(output_path), "--", str(input_path),
            ])
        except FileNotFoundError:
            result = None
        if result is not None and result.returncode == 0:
            return True

    cmd = ["magick", str(input_path)]
    if strip_metadata:
        cmd.append("-strip")
    cmd.extend(["+dither", "-colors", str(num_colors), "-depth", "8", str(output_path)])
    result = utils.run_capture(cmd)
    if result.returncode != 0:
        utils.set_last_error((result.stderr or "").strip() or (result.stdout or "").strip())
        return False
    return True


def animated_gif_reduce(
    input_path: Path,
    output_path: Path,
    num_colors: int,
    strip_metadata: bool = False,
) -> bool:
    # one global color table, per-frame palettes inflate ~5x
    cmd = [
        "magick", converters._magick_spec(input_path), "-coalesce",
        "-define", "quantize:global-colormap=true",
        "+dither", "-colors", str(num_colors), "-depth", "8",
    ]
    if strip_metadata:
        cmd.append("-strip")
    cmd.extend(["-layers", "Optimize", converters._magick_spec(output_path)])
    try:
        result = utils.run_capture(cmd)
    except FileNotFoundError:
        utils.set_last_error("ImageMagick is not installed.")
        return False
    if result.returncode != 0:
        utils.set_last_error((result.stderr or "").strip() or (result.stdout or "").strip())
        return False
    return True


def animated_quality_reduce(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
) -> bool:
    # extension selects the codec, each probe re-encodes from lossless
    target = converters.normalize_format(output_path.suffix.lstrip("."))
    cmd = ["magick", converters._magick_spec(input_path), "-coalesce"]
    if target == "TIFF":
        cmd.extend(["-compress", "JPEG"])
    if strip_metadata:
        cmd.append("-strip")
    cmd.extend(["-quality", str(quality),
                "-layers", "Optimize", converters._magick_spec(output_path)])
    try:
        result = utils.run_capture(cmd)
    except FileNotFoundError:
        utils.set_last_error("ImageMagick is not installed.")
        return False
    if result.returncode != 0:
        utils.set_last_error((result.stderr or "").strip() or (result.stdout or "").strip())
        return False
    return True
