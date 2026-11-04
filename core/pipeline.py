import math
import os
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypedDict, cast

from . import converters, log, memgate, utils
from .args import Args
from .backend import BackendInfo, get_backend
from .backend import set_backend as set_backend
from .clipboard import ClipboardError, get_clipboard_image, set_clipboard_image
from .compressors import (
    _resolve_knob_floor,
    animated_gif_reduce,
    animated_lossy_optimize,
    calibrate_palette_ssim_target,
    calibrate_ssim_target,
    color_palette_reduce,
    iterative_animated_gif_optimize,
    iterative_art_optimize,
    iterative_lossy_optimize,
    make_lossy_encoders,
    oxipng_compress,
    pngquant_compress,
    slowest_animated_gif_optimize,
    slowest_art_optimize,
    slowest_lossy_optimize,
)
from .contentscan import detect_palette_friendly
from .converters import (
    convert_animation,
    convert_format,
    get_image_filesize,
    get_image_format,
    get_image_height,
    get_image_width,
    normalize_format,
    resize_exact,
    resize_fit_height,
    resize_fit_width,
    resize_pct,
    sample_pct,
    to_png_lossless,
)
from .errors import BackendError
from .fileops import IMAGE_EXTENSIONS
from .formats import (
    ANIMATED_TARGET_FORMATS,
    ANIMATION_SOURCE_FORMATS,
    LOSSY_FORMATS,
    PALETTE_CORRECTION_FORMATS,
    default_target_fraction,
)
from .presets import (
    SearchSettings,
    resolve_image_mode_settings,
    resolve_search_settings,
)
from .upscalers import (
    compute_nn_scale_factor,
    detect_upscale_needed,
    realesrgan_upscale,
    waifu2x_upscale,
)
from .validation import (
    ValidationError as ValidationError,
)
from .validation import (
    _Toggles,
)
from .validation import (
    resolve_image_mode as resolve_image_mode,
)
from .validation import (
    resolve_processing_toggles as resolve_processing_toggles,
)
from .validation import (
    validate_flags as validate_flags,
)

# thread-local: workers must not delete each other's temps
_temps_local = threading.local()

_CONVERSION_QUALITY = 95

# below this a neural output counts as a failure
_NN_OUTPUT_MIN_BYTES = 1024

# GIF correction converges quickly
_GIF_CORRECTION_RETRIES = 3


def _format_size(num_bytes: int) -> str:
    def _trim(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    if num_bytes >= 1024 * 1024:
        return f"{_trim(num_bytes / (1024 * 1024))} MB"
    if num_bytes >= 1024:
        return f"{_trim(num_bytes / 1024)} KB"
    return f"{num_bytes} bytes"


def _get_backend() -> BackendInfo:
    # tests monkeypatch this name
    return get_backend()


def _active_temps() -> set[Path]:
    return cast(set[Path], _temps_local.__dict__.setdefault("active", set()))


def _make_temp(suffix: str) -> Path:
    # real disk when default temp is tmpfs, multi-GB copies shouldn't eat RAM
    fd, name = tempfile.mkstemp(prefix="clipshrink_", suffix=suffix,
                                dir=memgate.tempdir())
    os.close(fd)
    path = Path(name)
    _active_temps().add(path)
    return path


def _cleanup_temps() -> None:
    temps = _temps_local.__dict__.get("active")
    if not temps:
        return
    for path in list(temps):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        temps.discard(path)


@contextmanager
def temp_files() -> Iterator[None]:
    try:
        yield
    finally:
        _cleanup_temps()


@dataclass(frozen=True)
class _Config:
    img_mode: str
    dithering: bool
    nearest: bool
    snap: bool
    realesrgan_model: str
    max_size_bytes: int
    max_width: int | None
    max_height: int | None
    integer_scaling: int | None
    filter: str
    nn_scaler: str | None
    format: str | None
    upscale: bool
    manual_skip: bool
    quality: int | None
    min_quality: int | None
    strip_metadata: bool
    subsampling: str
    raw: bool
    slowest: bool
    speed: str
    ssim_target: float | None
    search: SearchSettings


class _Measurements(TypedDict):
    width: int
    height: int
    format: str
    size: int


# keyed by target format, shrinkability belongs to the output codec
def _resolve_max_size(args: Args, measured_size: int | None,
                      target_format: str | None = None) -> int:
    # 8 MB fallback when callers build a Namespace without the flag
    if args.max_size is not None:
        return int(args.max_size)
    if measured_size is not None:
        fraction = default_target_fraction(target_format)
        return max(1, round(measured_size * fraction))
    return 8 * 1024 * 1024


def _build_config(args: Args,
                  measured_size: int | None = None,
                  target_format: str | None = None) -> _Config:
    mode = resolve_image_mode(args)
    # presets fill knobs, explicit flags win. neural pre-pass forces Lanczos.
    mode_settings = resolve_image_mode_settings(args, mode)
    max_size_bytes = _resolve_max_size(args, measured_size, target_format)

    filt = mode_settings.resample_filter

    nn_scaler = None
    if args.realesrgan or (args.filter and args.filter.lower() == "realesrgan"):
        nn_scaler = "realesrgan"
    elif args.waifu2x or (args.filter and args.filter.lower() == "waifu2x"):
        nn_scaler = "waifu2x"
    if nn_scaler is not None:
        filt = "Lanczos"

    return _Config(
        img_mode=mode,
        dithering=mode_settings.dithering,
        nearest=mode_settings.nearest,
        snap=mode_settings.snap,
        realesrgan_model=mode_settings.realesrgan_model,
        max_size_bytes=max_size_bytes,
        max_width=args.max_width,
        max_height=args.max_height,
        integer_scaling=args.integer_scaling,
        filter=filt,
        nn_scaler=nn_scaler,
        format=converters.normalize_format(args.format) if args.format else None,
        upscale=args.upscale,
        manual_skip=args.manual_skipping,
        quality=args.quality,
        min_quality=args.min_quality,
        strip_metadata=args.strip_metadata,
        subsampling=args.subsampling if args.subsampling else "4:2:0",
        raw=args.raw,
        slowest=args.slowest,
        speed="fast" if args.fast else "slow" if (args.slow or args.slowest) else "default",
        ssim_target=getattr(args, "ssim_target", None),
        search=resolve_search_settings(args),
    )


def _resolve_target_format(orig_format: str, cfg: _Config) -> str:
    return cfg.format if cfg.format is not None else orig_format


# OS transport formats, not output targets
_CLIPBOARD_TRANSPORT_FORMATS = ("DIB", "BMP", "TIFF")

# raw bytes need the right extension, magick picks decoders by extension.
# Windows CF_DIB is headerless BMP, hence .dib.
_CLIPBOARD_MIME_EXT = {
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/avif": ".avif",
    "image/jxl": ".jxl",
    "image/heic": ".heic",
    "image/heif": ".heif",
    "image/bmp": ".dib",
    "image/png": ".png",
    "image/tiff": ".tiff",
}


def _clipboard_target_format(measured_format: str, cfg: _Config, args: Args) -> str:
    # -o out.jpeg declares JPEG, else DIB/TIFF become PNG
    if cfg.format is not None:
        return cfg.format
    if args.output:
        ext = Path(args.output).suffix.lstrip(".").upper()
        if ext and f".{ext.lower()}" in IMAGE_EXTENSIONS:
            return normalize_format(ext)
    if measured_format not in _CLIPBOARD_TRANSPORT_FORMATS:
        return measured_format
    return "PNG"


def _apply_ssim_target(current: Path, cfg: _Config, backend: BackendInfo,
                       target_format: str, original_size: int) -> _Config:
    # K-ary only, calibration encodes erase the grid speed advantage
    if cfg.ssim_target is None or cfg.search.search_mode != "kary":
        return cfg
    encoders = make_lossy_encoders(backend, slowest=cfg.slowest, speed=cfg.speed,
                                   avif_speed=cfg.search.avif_speed,
                                   prefer_software=cfg.search.prefer_software)
    encoder = encoders[target_format]
    start_quality = cfg.quality if cfg.quality is not None else 100
    # ignore preset -Q floor, a floor above the crossing overshoots
    # (SSIM(85) = 0.99 for 0.95). explicit -Q applies.
    min_quality = cfg.min_quality if cfg.min_quality is not None else 1
    print(f"-> Calibrating --ssim-target {cfg.ssim_target:g} ({target_format})...")
    budget = calibrate_ssim_target(
        current, encoder, threshold=cfg.ssim_target,
        subsampling=cfg.subsampling, strip_metadata=cfg.strip_metadata,
        min_quality=min_quality, max_quality=start_quality,
    )
    if budget is None:
        print(f"Warning: SSIM >= {cfg.ssim_target:g} is unreachable even at "
              "maximum quality; keeping the size target "
              f"({_format_size(cfg.max_size_bytes)}).")
        return cfg
    budget = min(budget, max(1, original_size - 1))
    print(f"-> SSIM target calibrated: budget {_format_size(budget)}.")
    return replace(cfg, max_size_bytes=budget)


# content-aware PNG gets this floor instead of the size fraction
_AUTO_SSIM_TARGET = 0.95


def _apply_palette_ssim_target(current: Path, cfg: _Config,
                               original_size: int, *, auto: bool) -> _Config:
    # auto runs in grid mode too, min(calibrated, fraction) only shrinks
    if cfg.ssim_target is None:
        return cfg
    if not auto and cfg.search.search_mode != "kary":
        return cfg
    start_colors = cfg.quality if cfg.quality is not None else 256
    # skip preset color floor, keep explicit -Q
    min_colors = cfg.min_quality if cfg.min_quality is not None else 2
    label = "Content-aware PNG" if auto else "--ssim-target"
    print(f"-> Calibrating {label} SSIM floor {cfg.ssim_target:g} (PNG)...")
    calibrated = calibrate_palette_ssim_target(
        current, threshold=cfg.ssim_target,
        strip_metadata=cfg.strip_metadata,
        min_colors=min_colors, max_colors=start_colors,
    )
    if calibrated is None:
        if auto:
            print(f"-> Content-aware: SSIM >= {cfg.ssim_target:g} is not "
                  "reachable for this image; keeping the size target.")
        else:
            print(f"Warning: SSIM >= {cfg.ssim_target:g} is unreachable even "
                  "at the full color count; keeping the size target "
                  f"({_format_size(cfg.max_size_bytes)}).")
        return cfg
    budget, crossing_colors = calibrated
    # seed at the crossing so first-fit publishes the calibrated answer.
    # explicit -q wins.
    seed = cfg.quality if cfg.quality is not None else crossing_colors
    budget = (min(budget, cfg.max_size_bytes) if auto
              else min(budget, max(1, original_size - 1)))
    print(f"-> SSIM floor calibrated: budget {_format_size(budget)} "
          f"({crossing_colors} colors).")
    return replace(cfg, max_size_bytes=budget, quality=seed)


def measure_image(image_path: Path) -> _Measurements:
    try:
        return _Measurements(
            width=get_image_width(image_path),
            height=get_image_height(image_path),
            format=get_image_format(image_path),
            size=get_image_filesize(image_path),
        )
    except RuntimeError as e:
        raise BackendError(f"Cannot read image: {e}") from e


def check_bypass(image_path: Path, cfg: _Config,
                 measurements: _Measurements | None = None,
                 target_format: str | None = None) -> bool:
    # measurements describe the original, not the normalized working copy
    if cfg.raw:
        return True

    if measurements is None:
        measurements = measure_image(image_path)
    fmt = measurements["format"]
    size = measurements["size"]
    resolved_target = target_format if target_format is not None else _resolve_target_format(fmt, cfg)

    flag_any = (
        cfg.max_width is not None
        or cfg.max_height is not None
        or cfg.integer_scaling is not None
    )
    is_upscale = detect_upscale_needed(
        measurements["width"], measurements["height"], size,
        cfg.max_width, cfg.max_height, cfg.integer_scaling, cfg.upscale,
    )
    needs_resize = flag_any or is_upscale

    return bool(not is_upscale and not needs_resize and size <= cfg.max_size_bytes and fmt == resolved_target)


# pyvips caches by filename, never overwrite a loaded file (stale dims).
# phases write fresh temps, the caller swaps "current".
def _resize_step(source: Path, apply: Callable[[Path, Path], Any]) -> tuple[Path, int, int]:
    tmp = _make_temp(".png")
    apply(source, tmp)
    return tmp, get_image_width(tmp), get_image_height(tmp)


def _upscale_pass(current: Path, cfg: _Config, width: int, height: int,
                  orig_width: int, is_integer_scaling: bool,
                  integer_scale_val: int | None) -> tuple[Path, int, int]:
    max_w, max_h = cfg.max_width, cfg.max_height
    scale_val = integer_scale_val
    if max_w is not None and max_h is not None:
        if width != max_w or height != max_h:
            if cfg.nearest:
                wp = int((max_w / width) * 100)
                hp = int((max_h / height) * 100)
                if wp == hp:
                    current, width, height = _resize_step(
                        current, lambda src, out: sample_pct(src, out, wp))
                else:
                    current, width, height = _resize_step(
                        current,
                        lambda src, out: resize_exact(src, out, max_w, max_h, "Point"))
            else:
                current, width, height = _resize_step(
                    current,
                    lambda src, out: resize_exact(src, out, max_w, max_h, cfg.filter))
    elif max_w is not None and width != max_w:
        if cfg.nearest:
            current, width, height = _resize_step(
                current, lambda src, out: sample_pct(
                    src, out, int((max_w / width) * 100)))
        else:
            current, width, height = _resize_step(
                current, lambda src, out: resize_fit_width(src, out, max_w, cfg.filter))
    elif max_h is not None and height != max_h:
        if cfg.nearest:
            current, width, height = _resize_step(
                current, lambda src, out: sample_pct(
                    src, out, int((max_h / height) * 100)))
        else:
            current, width, height = _resize_step(
                current, lambda src, out: resize_fit_height(src, out, max_h, cfg.filter))
    elif is_integer_scaling and scale_val is not None:
        if cfg.nn_scaler is not None:
            target_w = int(orig_width * scale_val / 100)
            if width != target_w:
                print(f"-> Fine-tuning neural output to exact scale {scale_val}%...")
                current, width, height = _resize_step(
                    current, lambda src, out: resize_fit_width(
                        src, out, target_w, cfg.filter))
        elif cfg.nearest:
            current, width, height = _resize_step(
                current, lambda src, out: sample_pct(src, out, scale_val))
        else:
            current, width, height = _resize_step(
                current, lambda src, out: resize_pct(src, out, scale_val, cfg.filter))
    return current, width, height


def _downscale_pass(current: Path, cfg: _Config, width: int, height: int,
                    is_integer_scaling: bool,
                    integer_scale_val: int | None) -> tuple[Path, int, int]:
    max_w, max_h = cfg.max_width, cfg.max_height
    scale_val = integer_scale_val
    if max_w is not None and max_h is not None:
        if cfg.nearest:
            current, width, height = _resize_step(
                current,
                lambda src, out: resize_exact(src, out, max_w, max_h, "Point"))
        else:
            current, width, height = _resize_step(
                current,
                lambda src, out: resize_exact(src, out, max_w, max_h, cfg.filter))
    elif max_w is not None and width > max_w:
        if cfg.nearest:
            print(f"-> Iteratively halving (50%) until width <= {max_w}...")
            while width > max_w and not utils._aborted:
                current, width, height = _resize_step(
                    current, lambda src, out: sample_pct(src, out, 50))
                utils._abort_if_interrupted()
        else:
            current, width, height = _resize_step(
                current, lambda src, out: resize_fit_width(src, out, max_w, cfg.filter))
    elif max_h is not None and height > max_h:
        if cfg.nearest:
            print(f"-> Iteratively halving (50%) until height <= {max_h}...")
            while height > max_h and not utils._aborted:
                current, width, height = _resize_step(
                    current, lambda src, out: sample_pct(src, out, 50))
                utils._abort_if_interrupted()
        else:
            current, width, height = _resize_step(
                current, lambda src, out: resize_fit_height(src, out, max_h, cfg.filter))
    elif is_integer_scaling and scale_val is not None:
        if cfg.nearest:
            current, width, height = _resize_step(
                current, lambda src, out: sample_pct(src, out, scale_val))
        else:
            current, width, height = _resize_step(
                current, lambda src, out: resize_pct(src, out, scale_val, cfg.filter))
    return current, width, height


def run_resize_phase(image_path: Path, cfg: _Config,
                     measurements: _Measurements, toggles: _Toggles) -> Path:
    if not toggles.resize:
        return image_path

    _get_backend()
    current = image_path
    width = measurements["width"]
    height = measurements["height"]
    orig_width = width
    orig_height = height

    is_upscale = detect_upscale_needed(
        orig_width, orig_height, measurements["size"],
        cfg.max_width, cfg.max_height, cfg.integer_scaling, cfg.upscale,
    )

    is_integer_scaling = cfg.integer_scaling is not None
    integer_scale_val = cfg.integer_scaling
    if cfg.max_width is None and cfg.max_height is None and cfg.integer_scaling is None:
        if is_upscale:
            is_integer_scaling = True
            integer_scale_val = 200
        elif cfg.nearest:
            is_integer_scaling = True

    if cfg.nn_scaler is not None:
        nn_scale = compute_nn_scale_factor(
            orig_width, orig_height, cfg.max_width, cfg.max_height, cfg.integer_scaling
        )
        nn_tmp = _make_temp(".png")
        if cfg.nn_scaler == "waifu2x":
            if nn_scale == 3:
                nn_scale = 4
            elif nn_scale > 4:
                nn_scale = 8
            print(f"-> Applying neural upscaler: waifu2x (target scale {nn_scale})...")
            ok = waifu2x_upscale(current, nn_tmp, scale=nn_scale, noise_level=1)
        else:
            print("-> Applying neural upscaler: realesrgan (4x)...")
            ok = realesrgan_upscale(current, nn_tmp, mode=cfg.realesrgan_model)
        utils._abort_if_interrupted()
        if ok and nn_tmp.exists() and get_image_filesize(nn_tmp) > _NN_OUTPUT_MIN_BYTES:
            current = nn_tmp
            width = get_image_width(current)
            height = get_image_height(current)
            print(f"   neural output dimensions: {width} x {height}")
        else:
            print("Warning: neural scaler failed. Falling back to regular scaling.")
            if utils.get_last_error():
                print(f"   Neural-upscaler error: {utils.get_last_error()}")
                utils.set_last_error("")

    if is_upscale:
        current, width, height = _upscale_pass(
            current, cfg, width, height, orig_width,
            is_integer_scaling, integer_scale_val)
    else:
        current, width, height = _downscale_pass(
            current, cfg, width, height,
            is_integer_scaling, integer_scale_val)

    measurements["size"] = get_image_filesize(current)
    measurements["width"] = width
    measurements["height"] = height
    return current


def run_pngquant_phase(image_path: Path, cfg: _Config, toggles: _Toggles,
                       orig_format: str = "PNG", target_format: str = "PNG") -> Path:
    # non-PNG source/target quantizes before lossy encoding, capping quality
    if not toggles.pngquant:
        return image_path
    if get_image_filesize(image_path) <= cfg.max_size_bytes:
        return image_path
    if orig_format != "PNG" or target_format != "PNG":
        return image_path

    backend = _get_backend()
    if not backend.pngquant:
        print("Warning: pngquant not installed. Skipping compression.")
        return image_path

    print(f"-> Image > {_format_size(cfg.max_size_bytes)}. Running pngquant compression...")
    tmp = _make_temp(".png")
    no_dithering = not cfg.dithering
    ok = pngquant_compress(image_path, tmp, speed=cfg.search.pngquant_speed,
                           no_dithering=no_dithering)
    utils._abort_if_interrupted()
    if ok:
        oxipng_level = cfg.search.oxipng_level
        if backend.oxipng and oxipng_level is not None:
            oxi = _make_temp(".png")
            if oxipng_compress(tmp, oxi, level=oxipng_level,
                               strip_metadata=cfg.strip_metadata):
                result = oxi
            else:
                utils.set_last_error("")
                result = tmp
        else:
            result = tmp
        only_pngquant = (toggles.pngquant
                         and not toggles.resize and not toggles.iterative)
        if get_image_filesize(result) <= cfg.max_size_bytes or only_pngquant:
            return result
        # don't keep a non-fitting pre-pass, continue from lossless
        print("Warning: pngquant pre-pass did not reach the target; "
              "continuing from the lossless source.")
        return image_path
    print("Warning: pngquant compression failed.")
    return image_path


def _run_lossy_optimize(image_path: Path, out_tmp: Path, cfg: _Config,
                        backend: BackendInfo, target_format: str) -> None:
    if target_format == "AVIF" and not backend.avifenc:
        print("Warning: avifenc not installed. AVIF output falls back to libvips/ImageMagick.")
    encoders = make_lossy_encoders(backend, slowest=cfg.slowest, speed=cfg.speed,
                                   avif_speed=cfg.search.avif_speed,
                                   prefer_software=cfg.search.prefer_software)
    encoder = encoders[target_format]
    search = cfg.search
    start_quality = cfg.quality if cfg.quality is not None else 100
    if cfg.ssim_target is not None:
        # SSIM floor is the quality bar, preset -Q would force resolution
        # loss. explicit -Q still applies.
        min_quality = cfg.min_quality if cfg.min_quality is not None else 1
    else:
        min_quality = _resolve_knob_floor(cfg.min_quality, cfg.quality,
                                          search.min_quality_floor)
    if search.frontier:
        slowest_lossy_optimize(
            image_path, out_tmp, cfg.max_size_bytes, {target_format: encoder},
            start_quality=start_quality, min_quality=min_quality,
            strip_metadata=cfg.strip_metadata, subsampling=cfg.subsampling,
            filter=cfg.filter,
            max_iterations=search.max_iterations,
            extra_fit_levels=search.frontier_levels,
            frontier_ceiling=search.frontier_ceiling,
            deep_step=search.deep_step, ssim=search.ssim,
            compare_subsampling=search.compare_subsampling,
            # Frontier ranks at original resolution, floor picks the smallest.
            ssim_floor=cfg.ssim_target,
        )
    else:
        iterative_lossy_optimize(
            image_path, out_tmp, cfg.max_size_bytes, encoder,
            start_quality=start_quality, min_quality=min_quality,
            strip_metadata=cfg.strip_metadata, subsampling=cfg.subsampling,
            use_extent_fast_path=(
                target_format == "JPEG"
                and cfg.ssim_target is None
                and (search.search_mode == "grid"
                     or not (backend.jpegli or backend.mozjpeg))
            ),
            max_iterations=search.max_iterations,
            search_step=search.search_step,
            filter=cfg.filter,
        )


def run_iterative_phase(image_path: Path, cfg: _Config,
                        measurements: _Measurements, toggles: _Toggles,
                        target: str | None = None) -> Path:
    if not toggles.iterative:
        return image_path
    if get_image_filesize(image_path) <= cfg.max_size_bytes:
        return image_path

    print(f"-> Image still > {_format_size(cfg.max_size_bytes)}. Iteratively optimizing...")

    backend = _get_backend()
    target_format = target if target is not None else _resolve_target_format(measurements["format"], cfg)
    search = cfg.search

    custom_val = cfg.quality
    custom_min = cfg.min_quality
    out_tmp = _make_temp("." + target_format.lower())

    if target_format in LOSSY_FORMATS:
        _run_lossy_optimize(image_path, out_tmp, cfg, backend, target_format)
    else:
        start_colors = custom_val if custom_val is not None else 256
        if cfg.ssim_target is not None:
            # SSIM floor replaces preset color floor, explicit -Q applies
            min_colors = _resolve_knob_floor(custom_min, custom_val, 2)
        else:
            min_colors = _resolve_knob_floor(custom_min, custom_val,
                                             search.min_colors_floor)
        if search.frontier:
            slowest_art_optimize(
                image_path, out_tmp, cfg.max_size_bytes,
                pixel_art_mode=cfg.nearest, pixel_art_snap=cfg.snap,
                start_colors=start_colors, min_colors=min_colors,
                strip_metadata=cfg.strip_metadata, filter=cfg.filter,
                container_corrected=(target_format in PALETTE_CORRECTION_FORMATS),
                rank_width_first=(target_format == "GIF"),
                max_iterations=search.max_iterations,
                extra_fit_levels=search.frontier_levels,
                frontier_ceiling=search.frontier_ceiling,
                ssim=search.ssim, deep_step=search.deep_step,
                # rank at original resolution, floor picks smallest passing.
                # PNG only, container-corrected targets skip SSIM.
                ssim_floor=(cfg.ssim_target
                            if target_format == "PNG" else None),
            )
        else:
            iterative_art_optimize(
                image_path, out_tmp, cfg.max_size_bytes,
                pixel_art_mode=cfg.nearest, pixel_art_snap=cfg.snap,
                start_colors=start_colors, min_colors=min_colors,
                strip_metadata=cfg.strip_metadata, filter=cfg.filter,
                max_iterations=search.max_iterations,
                search_step=search.color_step,
            )

    if target_format == "PNG" and backend.oxipng:
        oxipng_level = search.oxipng_level
        if oxipng_level is not None:
            oxi = _make_temp(".png")
            if oxipng_compress(out_tmp, oxi, level=oxipng_level,
                               strip_metadata=cfg.strip_metadata):
                out_tmp = oxi
            else:
                utils.set_last_error("")

    measurements["size"] = get_image_filesize(out_tmp)
    return out_tmp


def run_format_conversion(image_path: Path, cfg: _Config,
                          measurements: _Measurements,
                          target: str | None = None) -> Path:
    target_format = cfg.format if cfg.format is not None else (target or measurements["format"])
    if get_image_format(image_path) == target_format:
        return image_path

    current = image_path
    print(f"-> Converting final image to target format: {target_format}...")
    tmp = _make_temp("." + target_format.lower())
    convert_format(current, tmp, target_format, quality=_CONVERSION_QUALITY, strip_metadata=cfg.strip_metadata)
    utils._abort_if_interrupted()
    current = tmp
    size = get_image_filesize(current)

    if size > cfg.max_size_bytes:
        if target_format in LOSSY_FORMATS:
            current = _lossy_fallback_optimize(image_path, cfg, target_format)
            size = get_image_filesize(current)
        elif target_format in PALETTE_CORRECTION_FORMATS:
            current = _palette_correction_optimize(image_path, current, cfg, target_format)
            size = get_image_filesize(current)

    measurements["size"] = get_image_filesize(current)
    return current


def _lossy_fallback_optimize(image_path: Path, cfg: _Config,
                             target_format: str) -> Path:
    out_tmp = _make_temp("." + target_format.lower())
    _run_lossy_optimize(image_path, out_tmp, cfg, _get_backend(), target_format)
    return out_tmp


def _palette_correction_optimize(source: Path, current: Path, cfg: _Config,
                                 target_format: str) -> Path:
    # BMP/VIPS are uncompressed, one resize meets budget. GIF needs ratio
    # correction.
    budget = cfg.max_size_bytes
    size = get_image_filesize(current)
    search = cfg.search
    start_colors = cfg.quality if cfg.quality is not None else 256
    min_colors = _resolve_knob_floor(cfg.min_quality, cfg.quality,
                                     search.min_colors_floor)

    if size <= budget:
        return current

    if target_format in ("BMP", "VIPS"):
        scale_pct = round(math.sqrt(budget / size * 0.95) * 100)
        scale_pct = max(1, min(scale_pct, 99))
        print(f"   {target_format} is uncompressed; resizing to {scale_pct}%...")
        resized = _make_temp(".png")
        if cfg.nearest:
            sample_pct(source, resized, scale_pct)
        else:
            resize_pct(source, resized, scale_pct, cfg.filter)
        utils._abort_if_interrupted()
        png_out = _make_temp(".png")
        color_palette_reduce(resized, png_out, start_colors, cfg.strip_metadata)
        utils._abort_if_interrupted()
        converted = _make_temp("." + target_format.lower())
        convert_format(png_out, converted, target_format, quality=_CONVERSION_QUALITY,
                       strip_metadata=cfg.strip_metadata)
        utils._abort_if_interrupted()
        return converted

    # GIF expands the intermediate, budget needs ratio correction
    for _ in range(_GIF_CORRECTION_RETRIES):
        if size <= budget:
            break
        png_size = get_image_filesize(source)
        if png_size <= 0:
            break
        adjusted = int(budget * png_size / max(size, 1) * 0.9)
        if adjusted <= 0 or adjusted >= budget:
            break
        print(f"   Container expands {png_size / max(size, 1):.2f}x; "
              f"re-optimizing with budget {adjusted} bytes...")
        png_out = _make_temp(".png")
        if search.frontier:
            slowest_art_optimize(
                source, png_out, adjusted,
                pixel_art_mode=cfg.nearest, pixel_art_snap=cfg.snap,
                start_colors=start_colors,
                min_colors=search.min_colors_floor,
                strip_metadata=cfg.strip_metadata, filter=cfg.filter,
                container_corrected=True,
                rank_width_first=True,
                max_iterations=search.max_iterations,
                extra_fit_levels=search.frontier_levels,
                frontier_ceiling=search.frontier_ceiling,
                ssim=search.ssim, deep_step=search.deep_step,
            )
        else:
            iterative_art_optimize(
                source, png_out, adjusted,
                pixel_art_mode=cfg.nearest, pixel_art_snap=cfg.snap,
                start_colors=start_colors, min_colors=min_colors,
                strip_metadata=cfg.strip_metadata, filter=cfg.filter,
                max_iterations=search.max_iterations,
                search_step=search.color_step,
            )
        utils._abort_if_interrupted()
        converted = _make_temp("." + target_format.lower())
        convert_format(png_out, converted, target_format, quality=_CONVERSION_QUALITY,
                       strip_metadata=cfg.strip_metadata)
        utils._abort_if_interrupted()
        source = png_out
        current = converted
        size = get_image_filesize(converted)
    return current


def _is_animated(image_path: Path) -> bool:
    # read failures count as a still
    try:
        if converters._has_png_magic(image_path):
            # APNG is the only animated PNG, chunk scan avoids magick
            return converters._is_apng(image_path)
        return converters.get_frame_count(image_path) > 1
    except (RuntimeError, OSError):
        return False


def _is_animated_gif(image_path: Path) -> bool:
    return _is_animated(image_path)


def _animated_resize(image_path: Path, cfg: _Config,
                     measurements: _Measurements,
                     suffix: str = ".gif") -> Path:
    # APNG is lossless, non-GIF targets start at full color
    width = measurements["width"]
    height = measurements["height"]
    nearest = cfg.nearest
    kernel = "Point" if nearest else cfg.filter
    target = None
    if cfg.max_width is not None and cfg.max_height is not None:
        if width != cfg.max_width or height != cfg.max_height:
            target = f"{cfg.max_width}x{cfg.max_height}!"
    elif cfg.max_width is not None and width != cfg.max_width:
        target = f"{cfg.max_width}x"
    elif cfg.max_height is not None and height != cfg.max_height:
        target = f"x{cfg.max_height}"
    elif cfg.integer_scaling is not None:
        target = f"{cfg.integer_scaling}%"
    else:
        is_upscale = detect_upscale_needed(
            width, height, measurements["size"],
            cfg.max_width, cfg.max_height, cfg.integer_scaling, cfg.upscale,
        )
        if is_upscale:
            target = "200%"
    if target is None:
        return image_path
    tmp = _make_temp(suffix)
    converters.gif_resize(image_path, tmp, target, kernel=kernel, nearest=nearest,
                          strip_metadata=cfg.strip_metadata)
    utils._abort_if_interrupted()
    current = tmp
    measurements["width"] = converters.get_image_width(current)
    measurements["height"] = converters.get_image_height(current)
    return current


def _animated_baseline(source: Path, target_format: str, cfg: _Config,
                       source_format: str | None = None) -> Path:
    if source_format is None:
        source_format = get_image_format(source)
    if source_format == target_format:
        return source
    tmp = _make_temp("." + target_format.lower())
    if target_format == "GIF":
        start_colors = cfg.quality if cfg.quality is not None else 256
        if not animated_gif_reduce(source, tmp, start_colors,
                                   strip_metadata=cfg.strip_metadata):
            raise BackendError(
                f"Failed to convert animation to GIF: {utils.get_last_error()}")
    else:
        convert_animation(source, tmp, quality=_CONVERSION_QUALITY,
                          strip_metadata=cfg.strip_metadata)
    utils._abort_if_interrupted()
    return tmp


def run_animated_gif_phase(image_path: Path, cfg: _Config,
                           measurements: _Measurements, toggles: _Toggles,
                           target_format: str = "GIF") -> Path:
    # neural upscalers and pngquant don't support animation
    current = image_path
    working_format = measurements["format"]
    animated_kind = ("animated GIFs" if measurements["format"] == "GIF"
                     else "animated images")

    if cfg.nn_scaler is not None:
        print(f"Warning: Neural upscalers (realesrgan/waifu2x) do not support "
              f"{animated_kind}; skipping the neural pass.")

    if toggles.resize:
        # GIF->GIF keeps GIF, other animated paths use APNG for full color
        resize_suffix = (".gif" if target_format == "GIF"
                         and measurements["format"] == "GIF" else ".apng")
        current = _animated_resize(current, cfg, measurements, resize_suffix)
        if current != image_path:
            working_format = "GIF" if resize_suffix == ".gif" else "PNG"

    if toggles.pngquant and not toggles.iterative:
        print(f"Warning: pngquant cannot compress {animated_kind}; skipping.")

    baseline = _animated_baseline(current, target_format, cfg,
                                  source_format=working_format)
    baseline_size = get_image_filesize(baseline)

    if toggles.iterative and baseline_size > cfg.max_size_bytes:
        label = "GIF" if target_format == "GIF" else target_format
        print(f"-> Animated {label} > {_format_size(cfg.max_size_bytes)}. "
              f"Optimizing{' the palette' if target_format == 'GIF' else ''}...")
        search = cfg.search
        out_tmp = _make_temp("." + target_format.lower())
        if target_format == "GIF":
            start_colors = cfg.quality if cfg.quality is not None else 256
            min_colors = _resolve_knob_floor(cfg.min_quality, cfg.quality,
                                             search.min_colors_floor)
            if search.frontier:
                slowest_animated_gif_optimize(
                    current, out_tmp, cfg.max_size_bytes,
                    pixel_art_mode=cfg.nearest, pixel_art_snap=cfg.snap,
                    start_colors=start_colors, min_colors=min_colors,
                    strip_metadata=cfg.strip_metadata, filter=cfg.filter,
                    max_iterations=search.max_iterations,
                    extra_fit_levels=search.frontier_levels,
                    frontier_ceiling=search.frontier_ceiling,
                    ssim=search.ssim, deep_step=search.deep_step,
                    entry_size_guard=False,
                )
            else:
                iterative_animated_gif_optimize(
                    current, out_tmp, cfg.max_size_bytes,
                    pixel_art_mode=cfg.nearest, pixel_art_snap=cfg.snap,
                    start_colors=start_colors, min_colors=min_colors,
                    strip_metadata=cfg.strip_metadata, filter=cfg.filter,
                    max_iterations=search.max_iterations,
                    search_step=search.color_step,
                    entry_size_guard=False,
                )
        else:
            start_quality = cfg.quality if cfg.quality is not None else 100
            min_quality = _resolve_knob_floor(cfg.min_quality, cfg.quality,
                                              search.min_quality_floor)
            animated_lossy_optimize(
                current, out_tmp, cfg.max_size_bytes, target_format,
                start_quality=start_quality, min_quality=min_quality,
                pixel_art_mode=cfg.nearest, pixel_art_snap=cfg.snap,
                strip_metadata=cfg.strip_metadata, filter=cfg.filter,
                max_iterations=search.max_iterations,
                search_step=search.search_step,
                frontier=search.frontier,
                extra_fit_levels=search.frontier_levels,
                frontier_ceiling=search.frontier_ceiling,
                deep_step=search.deep_step, ssim=search.ssim,
            )
        utils._abort_if_interrupted()
        if working_format == target_format:
            # same-format search, every restore candidate is already valid
            current = out_tmp
        else:
            try:
                produced_format = get_image_format(out_tmp)
            except RuntimeError:
                produced_format = ""
            aliases = {"HEIC": {"HEIC", "HEIF"}, "HEIF": {"HEIC", "HEIF"}}.get(
                target_format, {target_format})
            if produced_format in aliases:
                current = out_tmp
            else:
                # failed round can leave source bytes in the target temp,
                # that would mislabel the container
                print("Warning: Animation optimization produced no "
                      f"{target_format} output; using the converted baseline.")
                current = baseline
    else:
        current = baseline

    measurements["size"] = get_image_filesize(current)
    return current


# GIF-era name is canonical, this is the generic alias.
run_animated_phase = run_animated_gif_phase


def _emit_output(current: Path, args: Args, target_format: str,
                 backend: BackendInfo) -> None:
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current, out)
        print(f"Saving output to: {out}")
    else:
        data = current.read_bytes()
        set_clipboard_image(data, backend.os_type, target_format)
        print("Copying output to clipboard.")


def _format_kib(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    return f"{size / 1024:.1f} KB"


def _print_dry_run_report(
    input_path: Path | None,
    source_mime: str,
    measurements: _Measurements,
    cfg: _Config,
    args: Args,
    target_format: str,
    toggles: Any,
    is_upscale: bool,
    would_bypass: bool,
) -> None:
    # dry run doesn't encode or resize, just measures
    steps: list[str] = []
    if would_bypass:
        steps.append("bypass: already fits the budget in the target format"
                     if not args.raw else "bypass (--raw)")
    else:
        if is_upscale:
            steps.append(f"neural upscale ({cfg.nn_scaler})" if cfg.nn_scaler
                         else "upscale")
        elif toggles.resize:
            targeted = (cfg.max_width is not None or cfg.max_height is not None
                        or cfg.integer_scaling is not None)
            steps.append("resize (dimension targets)" if targeted
                         else "resize (search-driven downscale, only if needed)")
        if toggles.pngquant and measurements["format"] == "PNG" \
                and target_format == "PNG":
            steps.append("pngquant pre-pass")
        if toggles.iterative:
            family = ("lossy quality search" if target_format in LOSSY_FORMATS
                      else "palette/color search")
            mode = ("frontier" if cfg.search.frontier
                    else cfg.search.search_mode)
            steps.append(f"iterative optimize ({family}, {mode})")
        if target_format != measurements["format"]:
            steps.append(f"convert to {target_format}")

    budget = (f"explicit budget {_format_kib(cfg.max_size_bytes)}"
              if args.max_size is not None
              else f"per-format fraction budget {_format_kib(cfg.max_size_bytes)}")
    source_label = (str(input_path) if input_path is not None
                    else f"clipboard ({source_mime or 'unknown'})")
    print("Dry run (no output written):")
    print(f"  source: {source_label}, "
          f"{measurements['width']}x{measurements['height']} "
          f"{measurements['format']}, {_format_kib(measurements['size'])}")
    print(f"  target: {target_format}, {budget}")
    if cfg.ssim_target is not None:
        print(f"  quality floor: SSIM >= {cfg.ssim_target} (budget refined by calibration)")
    print(f"  output: {args.output or 'clipboard'}")
    if steps:
        print("  plan:")
        for step in steps:
            print(f"    - {step}")
    else:
        print("  plan: (nothing to do)")


def process_image(input_path: Path | None, args: Args) -> dict[str, Any]:
    with temp_files():
        return _process_image(input_path, args)


def _process_image(input_path: Path | None, args: Args) -> dict[str, Any]:
    backend = _get_backend()
    cfg = _build_config(args)
    utils.set_last_error("")
    source_mime = ""
    orig_path = None

    try:
        log.log(f"Pipeline: {input_path or 'clipboard image'} | target format "
                f"{cfg.format or 'same as source'} | budget {cfg.max_size_bytes} bytes")
        if input_path is not None:
            print(f"Processing local file: {input_path}")
            orig_path = input_path
            measurements = measure_image(input_path)
            animated_source = (measurements["format"] in ANIMATION_SOURCE_FORMATS
                               and _is_animated(input_path))
            target_format = _resolve_target_format(measurements["format"], cfg)
            animated = (animated_source
                        and target_format in ANIMATED_TARGET_FORMATS)
            if animated:
                # frames must survive, keep the original animation
                current = input_path
            else:
                current = _make_temp(".png")
                if converters._has_png_magic(input_path):
                    shutil.copyfile(input_path, current)
                else:
                    # normalize to PNG so extension-sensitive tools see valid input
                    to_png_lossless(input_path, current)
        else:
            print("Processing image from clipboard...")
            data, source_mime = get_clipboard_image(backend.os_type)
            raw_ext = _CLIPBOARD_MIME_EXT.get(source_mime, ".png")
            current = _make_temp(raw_ext)
            current.write_bytes(data)
            orig_path = _make_temp(raw_ext)
            orig_path.write_bytes(data)
            measurements = measure_image(current)
            # macOS TIFF is transport only, treat as PNG for bypass
            if source_mime == "image/tiff" and measurements["format"] == "TIFF":
                measurements["format"] = "PNG"
            animated_source = (measurements["format"] in ANIMATION_SOURCE_FORMATS
                               and _is_animated(current))
            target_format = _clipboard_target_format(measurements["format"], cfg, args)
            animated = (animated_source
                        and target_format in ANIMATED_TARGET_FORMATS)
            # animated: keep raw bytes, otherwise normalize to PNG
            if not animated and not converters._has_png_magic(current):
                norm = _make_temp(".png")
                to_png_lossless(current, norm)
                current = norm

        # no -s: target is a per-format fraction of this image's size
        if args.max_size is None:
            cfg = _build_config(args, measurements["size"], target_format)

        if measurements["width"] <= 0 or measurements["height"] <= 0:
            raise BackendError("Cannot read image dimensions.")

        if animated_source and not animated:
            kind = "GIF" if measurements["format"] == "GIF" else "image"
            print(f"Warning: source is an animated {kind}; converting to a still "
                  f"image ({target_format}) keeps only the first frame.")

        is_upscale = detect_upscale_needed(
            measurements["width"], measurements["height"], measurements["size"],
            cfg.max_width, cfg.max_height, cfg.integer_scaling, cfg.upscale,
        )

        if cfg.upscale:
            if cfg.max_width is not None and cfg.max_width <= measurements["width"]:
                raise ValidationError("Upscale mode requested, but max-width <= original width.")
            if cfg.max_height is not None and cfg.max_height <= measurements["height"]:
                raise ValidationError("Upscale mode requested, but max-height <= original height.")
            if cfg.integer_scaling is not None and cfg.integer_scaling <= 100:
                raise ValidationError("Upscale mode requested, but integer-scaling is 100% or less.")
        if cfg.manual_skip and not is_upscale:
            raise ValidationError("--manual-skipping can only be used with upscaling.")

        toggles = resolve_processing_toggles(args, is_upscale)
        if cfg.ssim_target is not None and not toggles.iterative:
            raise ValidationError(
                "--ssim-target requires the iterative optimization phase "
                "(upscaling disables it unless --manual-skipping is given).")

        if args.dry_run:
            _print_dry_run_report(
                input_path, source_mime, measurements, cfg, args,
                target_format, toggles, is_upscale,
                check_bypass(current, cfg, measurements=measurements,
                             target_format=target_format),
            )
            return {"success": True, "final_size": measurements["size"],
                    "bypassed": False}

        if check_bypass(current, cfg, measurements=measurements,
                        target_format=target_format):
            if args.raw:
                print("Raw mode active. Bypassing all processing.")
            else:
                print("Image is already under target size, correct format, and bounds.")
            # emit original bytes, working copy may be a normalized PNG
            _emit_output(orig_path, args, target_format, backend)
            return {"success": True, "final_size": get_image_filesize(orig_path), "bypassed": True}

        # calibrate on the lossless copy, one budget drives every phase
        auto_ssim = False
        if cfg.ssim_target is not None and (
                animated or (target_format not in LOSSY_FORMATS
                             and target_format != "PNG")):
            print("Warning: --ssim-target applies to lossy still-image or "
                  "PNG targets; using the size target.")
        elif cfg.ssim_target is not None and target_format == "PNG":
            cfg = _apply_palette_ssim_target(current, cfg, measurements["size"],
                                             auto=False)
        elif cfg.ssim_target is not None:
            cfg = _apply_ssim_target(current, cfg, backend, target_format,
                                     measurements["size"])
        elif (args.max_size is None
              and not args.fast
              and not animated
              and target_format == "PNG"
              and toggles.iterative
              and detect_palette_friendly(current)):
            # palette-friendly PNG stills get the SSIM floor, auto budget only
            # shrinks. detector is fail-open.
            print("-> Content-aware: palette-friendly PNG detected "
                  "(screenshot/graphics-like).")
            cfg = _apply_palette_ssim_target(
                current, replace(cfg, ssim_target=_AUTO_SSIM_TARGET),
                measurements["size"], auto=True)
            auto_ssim = True

        target_note = ""
        if args.max_size is None:
            if cfg.ssim_target is not None and not auto_ssim:
                target_note = f" (SSIM >= {cfg.ssim_target:g} vs original)"
            elif auto_ssim:
                target_note = f" ({_format_size(cfg.max_size_bytes)}, content-aware)"
            else:
                pct = round(default_target_fraction(target_format) * 100)
                target_note = (f" ({_format_size(cfg.max_size_bytes)} = "
                               f"{pct}% of original)")
        print(
            f"Initial: {measurements['size'] / (1024 * 1024):.2f} MB "
            f"({measurements['width']}x{measurements['height']}). "
            f"Mode: {cfg.img_mode}. Target: {target_format}{target_note}. Processing..."
        )

        if animated:
            # still phases would collapse frames, animated gets the whole job
            current = run_animated_gif_phase(current, cfg, measurements,
                                             toggles, target_format)
        else:
            current = run_resize_phase(current, cfg, measurements, toggles)
            current = run_pngquant_phase(current, cfg, toggles,
                                         orig_format=measurements["format"],
                                         target_format=target_format)
            current = run_iterative_phase(current, cfg, measurements, toggles, target_format)
            current = run_format_conversion(current, cfg, measurements, target_format)

        final_size = get_image_filesize(current)
        if final_size <= cfg.max_size_bytes:
            print(f"Success! Final size: {final_size / (1024 * 1024):.2f} MB.")
        else:
            print(f"Warning: Reached optimization limits. Final size: {final_size / (1024 * 1024):.2f} MB.")

        utils._abort_if_interrupted()

        log.log(f"Pipeline done: final size {final_size} bytes")
        _emit_output(current, args, target_format, backend)
        return {"success": True, "final_size": final_size, "bypassed": False}
    except utils.AbortedException:
        return {"success": False, "final_size": None, "bypassed": False}
    except (ValidationError, ClipboardError, RuntimeError) as e:
        log.debug(f"Pipeline error: {e}")
        print(f"Error: {e}")
        if utils.get_last_error():
            print(f"   Reason: {utils.get_last_error()}")
        return {"success": False, "final_size": None, "bypassed": False}
