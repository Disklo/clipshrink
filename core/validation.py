from __future__ import annotations

import argparse  # noqa: F401  (re-exported for duck-typed callers)
from dataclasses import dataclass

from .args import Args
from .backend import get_backend
from .converters import normalize_format
from .errors import ClipshrinkError
from .formats import LOSSY_FORMATS


class ValidationError(ClipshrinkError):
    pass


@dataclass(frozen=True)
class _Toggles:
    resize: bool
    pngquant: bool
    iterative: bool


def validate_flags(args: Args) -> None:
    if args.integer_scaling is not None and (
        args.max_width is not None or args.max_height is not None
    ):
        raise ValidationError("--integer-scaling cannot be used with --max-width or --max-height")
    if args.quality is not None and args.min_quality is not None and args.quality < args.min_quality:
        raise ValidationError(f"-q ({args.quality}) must be >= -Q ({args.min_quality})")
    if args.quality is not None and not (1 <= args.quality <= 256):
        raise ValidationError("-q must be between 1 and 256")
    if args.min_quality is not None and not (1 <= args.min_quality <= 256):
        raise ValidationError("-Q must be between 1 and 256")
    if args.max_size is not None and args.max_size < 1:
        raise ValidationError("--max-size must be a positive size (kilobytes by default, or with a k/KB/m/MB suffix)")
    jobs = getattr(args, "jobs", None)
    if jobs is not None and jobs < 1:
        raise ValidationError("--jobs must be at least 1")
    if sum(bool(getattr(args, f, False)) for f in ("fast", "slow", "slowest")) > 1:
        raise ValidationError("--slow, --fast and --slowest are mutually exclusive")
    if getattr(args, "venv", None) and getattr(args, "venv_skip", False):
        raise ValidationError("--venv and --venv-skip are mutually exclusive")

    # --ssim-target replaces size objective, conflicting flags are out
    ssim_target = getattr(args, "ssim_target", None)
    if ssim_target is not None:
        if not 0.0 < ssim_target < 1.0:
            raise ValidationError("--ssim-target must be between 0 and 1 (exclusive), e.g. 0.95")
        if args.max_size is not None:
            raise ValidationError("--ssim-target and -s/--max-size are mutually exclusive "
                                  "(both define the size objective)")
        if not (args.slowest or args.slow):
            raise ValidationError(
                "--ssim-target requires --slow or --slowest (K-ary search modes; "
                "the calibration encodes would erase the grid modes' speed advantage)")
        if args.raw:
            raise ValidationError("--ssim-target cannot be combined with -r/--raw")
        if args.conversion_only:
            raise ValidationError("--ssim-target cannot be combined with --conversion-only")
        if args.skip_iterative or args.only_resize or args.only_pngquant:
            raise ValidationError("--ssim-target requires the iterative optimization phase")
        _fmt = normalize_format(args.format) if args.format else None
        if _fmt is not None and _fmt not in LOSSY_FORMATS and _fmt != "PNG":
            raise ValidationError(
                "--ssim-target requires a lossy or PNG target format "
                f"(-F {args.format} is not)")

    # preset knobs are ordinary flags, same range checks
    for flag, val, lo, hi in (
        ("--search-step", getattr(args, "search_step", None), 1, 100),
        ("--color-step", getattr(args, "color_step", None), 1, 256),
        ("--avif-speed", getattr(args, "avif_speed", None), 0, 10),
        ("--pngquant-speed", getattr(args, "pngquant_speed", None), 1, 10),
        ("--oxipng-level", getattr(args, "oxipng_level", None), 0, 6),
        ("--frontier-ceiling", getattr(args, "frontier_ceiling", None), 1, 256),
        ("--deep-step", getattr(args, "deep_step", None), 1, 99),
    ):
        if val is not None and not (lo <= val <= hi):
            raise ValidationError(f"{flag} must be between {lo} and {hi}")
    for flag, val in (
        ("--max-iterations", getattr(args, "max_iterations", None)),
        ("--frontier-levels", getattr(args, "frontier_levels", None)),
    ):
        if val is not None and val < 1:
            raise ValidationError(f"{flag} must be at least 1")
    if getattr(args, "compare_subsampling", False) and not getattr(args, "frontier", False):
        raise ValidationError("--compare-subsampling requires --frontier")
    if getattr(args, "no_oxipng", False) and getattr(args, "oxipng_level", None) is not None:
        raise ValidationError("--no-oxipng cannot be combined with --oxipng-level")

    backend = get_backend()
    uses_realesrgan = args.realesrgan or (args.filter and args.filter.lower() == "realesrgan")
    uses_waifu2x = args.waifu2x or (args.filter and args.filter.lower() == "waifu2x")
    if uses_realesrgan and not backend.realesrgan:
        raise ValidationError("realesrgan-ncnn-vulkan is not installed.")
    if uses_waifu2x and not backend.waifu2x:
        raise ValidationError("waifu2x-ncnn-vulkan is not installed.")
    if args.conversion_only and not args.format:
        raise ValidationError("--conversion-only requires --format")
    if (uses_realesrgan or uses_waifu2x) and resolve_image_mode(args) == "pixel_art":
        raise ValidationError("Neural upscalers cannot be used in Pixel Art mode.")


def resolve_image_mode(args: Args) -> str:
    if args.pixel_art:
        return "pixel_art"
    if args.drawing:
        return "drawing"
    return "photo"


def resolve_processing_toggles(args: Args, is_upscale: bool) -> _Toggles:
    # Upscaling skips pngquant and iterative unless --manual-skipping.
    run_resize = run_pngquant = run_iter = True
    if args.skip_resize:
        run_resize = False
    if args.skip_pngquant:
        run_pngquant = False
    if args.skip_iterative:
        run_iter = False

    if args.only_resize:
        run_resize, run_pngquant, run_iter = True, False, False
    elif args.only_pngquant:
        run_resize, run_pngquant, run_iter = False, True, False
    elif args.only_iterative:
        run_resize, run_pngquant, run_iter = False, False, True

    if args.raw:
        run_resize = run_pngquant = run_iter = False
    if args.conversion_only and args.format:
        run_resize = run_pngquant = run_iter = False

    if is_upscale and not args.manual_skipping:
        run_pngquant = False
        run_iter = False

    return _Toggles(resize=run_resize, pngquant=run_pngquant, iterative=run_iter)
