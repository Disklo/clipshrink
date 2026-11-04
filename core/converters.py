import contextlib
import functools
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Literal

from . import heif, utils
from .errors import BackendError
from .utils import get_last_error, run_capture, set_last_error

try:
    import pyvips
except (ImportError, OSError):
    # binding present but libvips won't load
    pyvips = None


def set_pyvips(module: ModuleType) -> None:
    global pyvips
    pyvips = module


def _run_magick(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # child can die silently (signal, OOM, full tmpfs), never return empty
    result = run_capture(cmd)
    if result.returncode != 0:
        message = ((result.stderr or "").strip() or (result.stdout or "").strip())
        if not message:
            code = result.returncode
            if code < 0:
                message = f"magick killed by signal {-code}"
            else:
                message = f"magick exited with code {code}"
        set_last_error(message)
    return result


def to_dib_bytes(image_path: Path) -> bytes:
    # CF_DIB wants raw pixels, BMP3 uses plain BITMAPINFOHEADER
    fd, bmp_name = tempfile.mkstemp(prefix="clipshrink_dib_", suffix=".bmp")
    os.close(fd)
    bmp_path = Path(bmp_name)
    try:
        result = _run_magick(["magick", str(image_path), f"BMP3:{bmp_path}"])
        if result.returncode != 0:
            raise BackendError(
                f"clipboard DIB conversion failed: "
                f"{get_last_error() or 'magick reported an error'}")
        data = bmp_path.read_bytes()
    finally:
        with contextlib.suppress(OSError):
            bmp_path.unlink()
    if not data.startswith(b"BM") or len(data) <= 14:
        raise BackendError("clipboard DIB conversion produced no bitmap data")
    return data[14:]


def normalize_format(fmt: str) -> str:
    return {
        "JPG": "JPEG",
        "TIF": "TIFF",
        "JPE": "JPEG",
    }.get(fmt.upper(), fmt.upper())


def get_image_width(image_path: Path) -> int:
    try:
        image = pyvips.Image.new_from_file(str(image_path))
        return int(image.width)
    except Exception:
        result = _run_magick(["magick", "identify", "-format", "%w\n", str(image_path)])
        if result.returncode != 0 or not result.stdout.strip():
            raise BackendError(f"Failed to get image width: {get_last_error()}") from None
        return int(result.stdout.strip().splitlines()[0])


def get_image_height(image_path: Path) -> int:
    try:
        image = pyvips.Image.new_from_file(str(image_path))
        return int(image.height)
    except Exception:
        result = _run_magick(["magick", "identify", "-format", "%h\n", str(image_path)])
        if result.returncode != 0 or not result.stdout.strip():
            raise BackendError(f"Failed to get image height: {get_last_error()}") from None
        return int(result.stdout.strip().splitlines()[0])


def _has_png_magic(image_path: Path) -> bool:
    try:
        with open(image_path, "rb") as f:
            return f.read(8) == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def _is_apng(image_path: Path) -> bool:
    # PNG reader stops at first frame unless it goes through APNG
    if image_path.suffix.lower() == ".apng":
        return True
    try:
        with open(image_path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return False
            while True:
                header = f.read(8)
                if len(header) < 8:
                    return False
                length = int.from_bytes(header[:4], "big")
                chunk_type = header[4:8]
                if chunk_type == b"acTL":
                    return True
                if chunk_type in (b"IDAT", b"IEND"):
                    return False
                f.seek(length + 4, 1)   # data + CRC
    except OSError:
        return False


def _magick_spec(image_path: Path) -> str:
    return f"APNG:{image_path}" if _is_apng(image_path) else str(image_path)


def _vips_image_format(image_path: Path) -> str | None:
    # None = ask magick: no pyvips, load failed, unmapped loader,
    # or a non-PNG sniffed as pngload
    if pyvips is None:
        return None
    with contextlib.suppress(Exception):
        image = pyvips.Image.new_from_file(str(image_path))
        loader = image.get("vips-loader")
        formats = {
            "pngload" : "PNG",
            "jxlload": "JXL",
            "jpegload": "JPEG",
            "jpgload": "JPEG",
            "jp2load": "JP2",
            "webpload": "WEBP",
            "heicload": "HEIC",
            "tiffload": "TIFF",
            "tifload": "TIFF",
            "gifload": "GIF",
            "bmpload": "BMP",
            "svgload": "SVG",
            "vipsload": "VIPS",
            "vload": "VIPS",
        }
        if loader in formats:
            if loader == "pngload" and not _has_png_magic(image_path):
                return None
            return formats[loader]
        if loader == "heifload":
            try:
                if image.get("heif-compression") == "av1":
                    return "AVIF"
                return "HEIC"
            except Exception:
                return "HEIC"
        return None
    return None


def get_image_format(image_path: Path) -> str:
    fmt = _vips_image_format(image_path)
    if fmt is not None:
        return fmt
    result = _run_magick(["magick", "identify", "-format", "%m\n", str(image_path)])
    if result.returncode != 0 or not result.stdout.strip():
        raise BackendError(f"Failed to get image format: {get_last_error()}") from None
    # magick repeats format per frame, first line is the container's
    return normalize_format(result.stdout.strip().splitlines()[0].strip())


def get_image_filesize(image_path: Path) -> int:
    try:
        return image_path.stat().st_size
    except OSError as e:
        raise BackendError(f"Failed to get image filesize: {e}") from e


def _avifdec_frame_count(image_path: Path) -> int:
    # magick reports one page for some AVIF animations, avifdec --info is
    # authoritative. 0 if not AVIF or unavailable.
    if shutil.which("avifdec") is None:
        return 0
    try:
        result = run_capture(["avifdec", "--info", str(image_path)])
    except OSError:
        return 0
    text = f"{result.stderr or ''} {result.stdout or ''}"
    for pattern in (r"Image Sequence Frames:\s*\((\d+) expected frames\)",
                    r"(\d+)\s+frames"):
        match = re.search(pattern, text)
        if m := match:
            return max(1, int(m.group(1)))
    return 0


def get_frame_count(image_path: Path) -> int:
    # vips n-pages first, magick catches multi-frame sources vips calls stills.
    vips_count: int | None = None
    if pyvips is not None:
        with contextlib.suppress(Exception):
            image = pyvips.Image.new_from_file(str(image_path))
            vips_count = max(1, int(image.get("n-pages")))
            if vips_count > 1:
                return vips_count
    result = _run_magick(["magick", "identify", "-format", "%n\n",
                          _magick_spec(image_path)])
    if result.returncode != 0 or not result.stdout.strip():
        if vips_count is not None:
            # vips read a still, magick may not know the container.
            return vips_count
        if image_path.suffix.lower() == ".avif":
            avif_count = _avifdec_frame_count(image_path)
            if avif_count > 1:
                return avif_count
        raise BackendError(f"Failed to get frame count: {get_last_error()}") from None
    try:
        magick_count = max(1, int(result.stdout.strip().splitlines()[0]))
    except ValueError:
        if vips_count is not None:
            return vips_count
        if image_path.suffix.lower() == ".avif":
            avif_count = _avifdec_frame_count(image_path)
            if avif_count > 1:
                return avif_count
        raise BackendError(f"Failed to get frame count: {get_last_error()}") from None
    count = max(vips_count or 1, magick_count)
    if count <= 1 and image_path.suffix.lower() == ".avif":
        count = max(count, _avifdec_frame_count(image_path))
    return count


IM_FILTER_MAP = {
    "lanczos3": "Lanczos",
    "lanczos2": "Lanczos2",
    "nearest": "Point",
    "linear": "Triangle",
    "cubic": "Catrom",
    "bicubic": "Catrom",
    "box": "Box",
    "mitchell": "Mitchell",
    "point": "Point",
    "lanczos": "Lanczos",
    "catrom": "Catrom",
}


def _ffmpeg_decode_to_png(input_path: Path, output_path: Path) -> bool:
    # some HEIF files beat libde265 but decode in ffmpeg
    # still-only, so animations keep all frames
    if shutil.which("ffmpeg") is None:
        return False
    if pyvips is not None:
        with contextlib.suppress(Exception):
            if int(pyvips.Image.new_from_file(str(input_path)).get("n-pages")) > 1:
                return False
    try:
        result = run_capture([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-i", str(input_path), "-frames:v", "1",
            "-pix_fmt", "rgb24", str(output_path),
        ])
    except Exception:
        return False
    return (result.returncode == 0 and output_path.exists()
            and output_path.stat().st_size > 0)


def to_png_lossless(input_path: Path, output_path: Path) -> bool:
    # bit depth is preserved, 16-bit stays 16-bit
    try:
        image = pyvips.Image.new_from_file(str(input_path))
        image.write_to_file(str(output_path))
        return True
    except Exception:
        # [0] selects the first frame, otherwise magick writes numbered PNGs
        cmd = ["magick", f"{_magick_spec(input_path)}[0]", str(output_path)]
        result = _run_magick(cmd)
        if result.returncode != 0:
            if _ffmpeg_decode_to_png(input_path, output_path):
                return True
            raise BackendError(f"Failed to convert input to PNG: {get_last_error()}") from None
        return True


def _im_filter_name(kernel: str) -> str:
    return IM_FILTER_MAP.get(kernel.lower(), kernel)


def _resize_generic(
    input_path: Path,
    output_path: Path,
    mode: Literal["exact", "fit_width", "fit_height", "pct", "sample"],
    width: int | None = None,
    height: int | None = None,
    percentage: int | None = None,
    kernel: str = "Lanczos",
    strip_metadata: bool = False,
    error_verb: str = "resize",
) -> bool:
    try:
        image = pyvips.Image.new_from_file(str(input_path))
        vips_kernel = "nearest" if mode == "sample" else filter_to_vips_kernel(kernel)
        if mode == "exact":
            if width is None or height is None:
                raise ValueError("exact resize requires width and height")
            # vscale must be a keyword, pyvips resize takes one positional arg.
            resized = image.resize(width / image.width,
                                   vscale=height / image.height,
                                   kernel=vips_kernel)
        elif mode == "fit_width":
            if width is None:
                raise ValueError("fit_width resize requires width")
            resized = image.resize(width / image.width, kernel=vips_kernel)
        elif mode == "fit_height":
            if height is None:
                raise ValueError("fit_height resize requires height")
            resized = image.resize(height / image.height, kernel=vips_kernel)
        else:
            if percentage is None:
                raise ValueError("percentage resize requires percentage")
            resized = image.resize(percentage / 100, kernel=vips_kernel)
        resized.write_to_file(str(output_path), strip=strip_metadata)
        return True
    except Exception:
        cmd = ["magick", str(input_path)]
        if mode == "sample":
            cmd.extend(["-sample", f"{percentage}%"])
        else:
            cmd.extend(["-filter", _im_filter_name(kernel)])
            if mode == "exact":
                cmd.extend(["-resize", f"{width}x{height}!"])
            elif mode == "fit_width":
                cmd.extend(["-resize", f"{width}x"])
            elif mode == "fit_height":
                cmd.extend(["-resize", f"x{height}"])
            else:
                cmd.extend(["-resize", f"{percentage}%"])
        if strip_metadata:
            cmd.append("-strip")
        cmd.append(str(output_path))
        result = _run_magick(cmd)
        if result.returncode != 0:
            raise BackendError(f"Failed to {error_verb} image: {get_last_error()}") from None
        return True


def resize_exact(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    kernel: str = "Lanczos",
    strip_metadata: bool = False,
) -> bool:
    return _resize_generic(input_path, output_path, "exact", width=width,
                           height=height, kernel=kernel,
                           strip_metadata=strip_metadata)


def resize_exact_vips(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    kernel: str = "Lanczos",
    strip_metadata: bool = False,
) -> bool:
    # no magick fallback, exact resize decodes HDRI at ~10 GiB for 658 MP
    if pyvips is None:
        return False
    try:
        image = pyvips.Image.new_from_file(str(input_path))
        resized = image.resize(width / image.width,
                               vscale=height / image.height,
                               kernel=filter_to_vips_kernel(kernel))
        resized.write_to_file(str(output_path), strip=strip_metadata)
        return True
    except Exception:
        return False


def resize_fit_width(
    input_path: Path,
    output_path: Path,
    target_width: int,
    kernel: str = "Lanczos",
    strip_metadata: bool = False,
) -> bool:
    return _resize_generic(input_path, output_path, "fit_width", width=target_width,
                           kernel=kernel, strip_metadata=strip_metadata)


def resize_fit_height(
    input_path: Path,
    output_path: Path,
    target_height: int,
    kernel: str = "Lanczos",
    strip_metadata: bool = False,
) -> bool:
    return _resize_generic(input_path, output_path, "fit_height", height=target_height,
                           kernel=kernel, strip_metadata=strip_metadata)


def resize_pct(
    input_path: Path,
    output_path: Path,
    percentage: int,
    kernel: str = "Lanczos",
    strip_metadata: bool = False,
) -> bool:
    return _resize_generic(input_path, output_path, "pct", percentage=percentage,
                           kernel=kernel, strip_metadata=strip_metadata)


def sample_pct(
    input_path: Path,
    output_path: Path,
    percentage: int,
    strip_metadata: bool = False,
) -> bool:
    return _resize_generic(input_path, output_path, "sample", percentage=percentage,
                           strip_metadata=strip_metadata, error_verb="sample")


def gif_resize(
    input_path: Path,
    output_path: Path,
    geometry: str,
    kernel: str = "Lanczos",
    nearest: bool = False,
    strip_metadata: bool = False,
) -> bool:
    # coalesce to full-size frames, then re-optimize deltas. nearest uses -sample
    cmd = ["magick", _magick_spec(input_path), "-coalesce"]
    if nearest:
        cmd.extend(["-sample", geometry])
    else:
        cmd.extend(["-filter", _im_filter_name(kernel), "-resize", geometry])
    if strip_metadata:
        cmd.append("-strip")
    cmd.extend(["-layers", "Optimize", _magick_spec(output_path)])
    result = _run_magick(cmd)
    if result.returncode != 0:
        raise BackendError(f"Failed to resize animation: {get_last_error()}") from None
    return True


SUBSAMPLING_FACTORS = {
    "4:4:4": "4:4:4",
    "4:2:2": "4:2:2",
    "4:2:0": "4:2:0",
}


def jpeg_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    subsampling: str = "4:2:0",
    optimize: bool = True,
    progressive: bool = False,
    strip_metadata: bool = False,
) -> bool:
    try:
        save_kwargs = {
            "Q": quality,
            # vips does 4:2:0 (on) and 4:4:4 (off), 4:2:2 falls to magick
            "subsample_mode": {"4:2:0": "on", "4:4:4": "off"}.get(subsampling)
            or None,
        }
        if save_kwargs["subsample_mode"] is None:
            raise ValueError(f"vips cannot express {subsampling} subsampling")
        if optimize:
            save_kwargs["optimize_coding"] = True
        if progressive:
            save_kwargs["interlace"] = True
        if strip_metadata:
            save_kwargs["strip"] = True
        # sequential load lets jpegsave stream
        image = pyvips.Image.new_from_file(str(input_path), access="sequential")
        image.write_to_file(str(output_path), **save_kwargs)
        return True
    except Exception:
        cmd = [
            "magick",
            str(input_path),
            "-quality",
            str(quality),
            "-sampling-factor",
            SUBSAMPLING_FACTORS.get(subsampling, "4:2:0"),
        ]
        if optimize:
            cmd.extend(["-define", "jpeg:optimize-coding=true"])
        cmd.extend(["-interlace", "Plane" if progressive else "None"])
        if strip_metadata:
            cmd.append("-strip")
        cmd.append(str(output_path))
        result = _run_magick(cmd)
        if result.returncode != 0:
            raise BackendError(f"Failed to save JPEG image: {get_last_error()}") from None
        return True


def jpeg_save_extent(
    input_path: Path,
    output_path: Path,
    max_bytes: int,
    subsampling: str = "4:2:0",
    strip_metadata: bool = False,
) -> bool:
    # jpeg:extent searches quality internally, caller checks size
    cmd = [
        "magick",
        str(input_path),
        "-sampling-factor",
        SUBSAMPLING_FACTORS.get(subsampling, "4:2:0"),
        "-define",
        f"jpeg:extent={max_bytes}",
    ]
    if strip_metadata:
        cmd.append("-strip")
    cmd.append(str(output_path))
    result = _run_magick(cmd)
    return result.returncode == 0


def convert_format(
    input_path: Path,
    output_path: Path,
    target_format: str,
    quality: int = 95,
    strip_metadata: bool = False,
) -> bool:
    try:
        image = pyvips.Image.new_from_file(str(input_path))
        kwargs: dict[str, object] = {}
        if strip_metadata:
            kwargs["strip"] = True
        if target_format.upper() in {"JPEG", "JPG", "WEBP"}:
            kwargs["Q"] = quality
        if target_format.upper() == "BMP":
            raise ValueError("BMP handled by magick (palette)")
        image.write_to_file(str(output_path), **kwargs)
        return True
    except Exception:
        cmd = ["magick", str(input_path)]
        if strip_metadata:
            cmd.append("-strip")
        if target_format.upper() in {"JPEG", "JPG", "WEBP"}:
            cmd.extend(["-quality", str(quality)])
        if target_format.upper() == "BMP":
            # 8-bit palette BMP is ~3x smaller than 24-bit, -compress None
            # keeps 1 B/px (RLE8 expands on noise, input is already palette)
            cmd.extend(["-depth", "8", "-type", "palette", "-compress", "None"])
        cmd.append(str(output_path))
        result = _run_magick(cmd)
        if result.returncode != 0:
            raise BackendError(f"Failed to convert image format: {get_last_error()}") from None
        return True


def convert_animation(
    input_path: Path,
    output_path: Path,
    quality: int = 95,
    strip_metadata: bool = False,
) -> bool:
    # -layers Optimize keeps deltas compact (JXL keeps the sequence)
    target = normalize_format(output_path.suffix.lstrip("."))
    cmd = ["magick", _magick_spec(input_path), "-coalesce"]
    if target == "TIFF":
        # JPEG-in-TIFF gives animation a quality knob
        cmd.extend(["-compress", "JPEG", "-quality", str(quality)])
    elif target != "GIF":
        cmd.extend(["-quality", str(quality)])
    if strip_metadata:
        cmd.append("-strip")
    cmd.extend(["-layers", "Optimize", _magick_spec(output_path)])
    result = _run_magick(cmd)
    if result.returncode != 0:
        raise BackendError(
            f"Failed to convert animation to {target or 'target format'}: "
            f"{get_last_error()}") from None
    return True


def _vips_quality_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool,
    label: str,
    *,
    vips_extra: dict[str, object] | None = None,
    magick_extra: tuple[str, ...] = (),
) -> bool:
    try:
        image = pyvips.Image.new_from_file(str(input_path))
        save_kwargs: dict[str, object] = {"Q": quality, **(vips_extra or {})}
        if strip_metadata:
            save_kwargs["strip"] = True
        image.write_to_file(str(output_path), **save_kwargs)
        return True
    except Exception:
        cmd = ["magick", str(input_path)]
        if strip_metadata:
            cmd.append("-strip")
        cmd.extend(magick_extra)
        cmd.extend(["-quality", str(quality), str(output_path)])
        result = _run_magick(cmd)
        if result.returncode != 0:
            raise BackendError(f"Failed to save {label}: {get_last_error()}") from None
        return True


def webp_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
) -> bool:
    return _vips_quality_save(input_path, output_path, quality, strip_metadata,
                              "WebP image")


# hardware AV1 per platform: VideoToolbox, NVENC, oneVPL, AMF
_AVIF_HW_CODECS_BY_OS = {
    "darwin": ("mtl",),
    "win32": ("nvenc", "vpl", "amf"),
    "cygwin": ("nvenc", "vpl", "amf"),
    "linux": ("nvenc", "vpl", "amf"),
}


def _tiny_png_bytes() -> bytes:
    width = height = 64

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data)))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _probe_avif_hw_codec(codec: str) -> bool:
    fd, in_name = tempfile.mkstemp(prefix="clipshrink_avif_probe_", suffix=".png")
    os.close(fd)
    out_path = Path(in_name).with_suffix(".avif")
    try:
        Path(in_name).write_bytes(_tiny_png_bytes())
        result = run_capture([
            "avifenc", "--codec", codec, "-q", "30",
            "--speed", "10", "-o", str(out_path), in_name,
        ])
        return result.returncode == 0 and out_path.stat().st_size > 0
    except Exception:
        return False
    finally:
        for path in (Path(in_name), out_path):
            with contextlib.suppress(OSError):
                path.unlink()


@functools.cache
def _avif_hw_codec() -> str:
    for codec in _AVIF_HW_CODECS_BY_OS.get(sys.platform, ()):
        if _probe_avif_hw_codec(codec):
            return codec
    return ""


def _avifenc_encode(input_path: Path, output_path: Path, quality: int,
                    speed: int, hw_codec: str) -> subprocess.CompletedProcess[str]:
    # -q is monotone in size. deprecated --min/--max quantizer bounds
    # saturate at 63 and silently cap quality.
    cmd = ["avifenc", "-q", str(quality), "--speed", str(speed)]
    if hw_codec:
        cmd.extend(["--codec", hw_codec])
    cmd.extend(["-o", str(output_path), str(input_path)])
    return _run_magick(cmd)


def avif_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
    speed: int = 4,
    prefer_software: bool = False,
) -> bool:
    # avifenc (hardware when available), then vips, then magick.
    # --slowest prefers software: aom/svt-av1 beat hardware per byte.
    if shutil.which("avifenc") is not None:
        hw_codec = _avif_hw_codec()
        codecs = []
        if hw_codec and not prefer_software:
            codecs.append(hw_codec)
        codecs.append("")
        if hw_codec and prefer_software:
            codecs.append(hw_codec)
        for codec in codecs:
            result = _avifenc_encode(input_path, output_path, quality, speed, codec)
            if result.returncode == 0:
                return True
        return False
    try:
        image = pyvips.Image.new_from_file(str(input_path))
        save_kwargs = {"Q": quality}
        if strip_metadata:
            save_kwargs["strip"] = True
        image.write_to_file(str(output_path), **save_kwargs)
        return True
    except Exception:
        cmd = ["magick", str(input_path)]
        if strip_metadata:
            cmd.append("-strip")
        cmd.extend(["-quality", str(quality), str(output_path)])
        result = _run_magick(cmd)
        if result.returncode != 0:
            raise BackendError(f"Failed to save AVIF image: {get_last_error()}") from None
        return True


def jxl_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
) -> bool:
    return _vips_quality_save(input_path, output_path, quality, strip_metadata,
                              "JXL image")


def jp2_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
) -> bool:
    return _vips_quality_save(input_path, output_path, quality, strip_metadata,
                              "JP2 image")


def heic_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
    prefer_software: bool = False,
) -> bool:
    # hardware HEVC first, then vips (needs libheif+HEVC), then magick.
    # --slowest prefers software: hardware HEVC is worse per byte.
    def _try_vips() -> bool:
        image = pyvips.Image.new_from_file(str(input_path))
        save_kwargs = {"Q": quality, "compression": "hevc"}
        if strip_metadata:
            save_kwargs["strip"] = True
        image.write_to_file(str(output_path), **save_kwargs)
        return True

    def _try_hw() -> bool:
        return bool(heif.heic_hw_save(input_path, output_path, quality))

    encoders = (_try_vips, _try_hw) if prefer_software else (_try_hw, _try_vips)
    for encoder in encoders:
        with contextlib.suppress(Exception):
            if encoder():
                return True
    cmd = ["magick", str(input_path)]
    if strip_metadata:
        cmd.append("-strip")
    cmd.extend(["-quality", str(quality), str(output_path)])
    result = _run_magick(cmd)
    if result.returncode != 0:
        raise BackendError(f"Failed to save HEIC image: {get_last_error()}") from None
    return True


def tiff_jpeg_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    strip_metadata: bool = False,
) -> bool:
    # JPEG-in-TIFF is the only quality knob for photos, lossless barely shrinks
    return _vips_quality_save(input_path, output_path, quality, strip_metadata,
                              "JPEG-compressed TIFF",
                              vips_extra={"compression": "jpeg"},
                              magick_extra=("-compress", "JPEG"))


def _to_ppm(input_path: Path, output_path: Path) -> bool:
    try:
        image = pyvips.Image.new_from_file(str(input_path))
        image.colourspace("srgb").cast("uchar").write_to_file(str(output_path))
        return True
    except Exception:
        cmd = ["magick", str(input_path), "-depth", "8", str(output_path)]
        result = _run_magick(cmd)
        if result.returncode != 0:
            raise BackendError(f"Failed to convert to PPM: {get_last_error()}") from None
        return True


_JPEGLI_SUBSAMPLING = {"4:4:4": "444", "4:2:2": "422", "4:2:0": "420"}


def _cjpeg_like_save(
    binary: str,
    input_path: Path,
    output_path: Path,
    quality: int,
    make_cmd: Callable[[str, str], list[str]],
    missing_message: str,
) -> bool:
    # PPM strips metadata, callers' strip_metadata is a no-op here
    fd, name = tempfile.mkstemp(prefix="clipshrink_ppm_", suffix=".ppm")
    os.close(fd)
    ppm = Path(name)
    try:
        if not _to_ppm(input_path, ppm):
            utils.set_last_error(f"Failed to prepare PPM for {binary}.")
            return False
        result = utils.run_capture([binary, *make_cmd(str(ppm), str(output_path))])
        if result.returncode != 0:
            utils.set_last_error((result.stderr or "").strip() or (result.stdout or "").strip())
            return False
        return True
    except FileNotFoundError:
        utils.set_last_error(missing_message)
        return False
    finally:
        with contextlib.suppress(OSError):
            ppm.unlink()


def jpegli_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    subsampling: str = "4:2:0",
    strip_metadata: bool = False,
) -> bool:
    def make_cmd(ppm: str, out: str) -> list[str]:
        return ["-q", str(quality),
                "-c", _JPEGLI_SUBSAMPLING.get(subsampling, "420"),
                ppm, out]

    return _cjpeg_like_save("cjpegli", input_path, output_path, quality,
                            make_cmd, "cjpegli is not installed.")


_JPEG_SAMPLE_FACTORS = {"4:4:4": "1x1", "4:2:2": "2x1", "4:2:0": "2x2"}


def mozjpeg_save(
    input_path: Path,
    output_path: Path,
    quality: int,
    subsampling: str = "4:2:0",
    strip_metadata: bool = False,
) -> bool:
    def make_cmd(ppm: str, out: str) -> list[str]:
        return ["-quality", str(quality),
                "-sample", _JPEG_SAMPLE_FACTORS.get(subsampling, "2x2"),
                "-optimize", "-progressive",
                "-outfile", out, ppm]

    return _cjpeg_like_save("cjpeg", input_path, output_path, quality,
                            make_cmd, "mozjpeg cjpeg is not installed.")


VIPS_KERNEL_MAP = {
    "nearest": "nearest",
    "point": "nearest",
    "lanczos": "lanczos3",
    "lanczos2": "lanczos2",
    "box": "box",
    "catrom": "cubic",
    "bicubic": "cubic",
    "cubic": "cubic",
    "mitchell": "mitchell",
    "linear": "linear",
    "triangle": "linear",
}


def filter_to_vips_kernel(im_filter: str) -> str:
    return VIPS_KERNEL_MAP.get(im_filter.lower(), "lanczos3")


VIPS_SUPPORTED_FORMATS = {"PNG", "JXL", "JPEG", "JP2", "WEBP", "TIFF", "GIF", "BMP", "SVG", "VIPS", "AVIF"}


def vips_supports_format(fmt: str) -> bool:
    return normalize_format(fmt) in VIPS_SUPPORTED_FORMATS
