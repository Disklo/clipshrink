from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .args import Args
from .encoders import speed_profile
from .validation import resolve_image_mode


@dataclass(frozen=True)
class SearchSettings:
    search_step: int | None          # lossy quality grid step (None = K-ary)
    color_step: int | None           # palette color grid step (None = K-ary)
    max_iterations: int
    avif_speed: int
    pngquant_speed: int
    oxipng_level: int | None         # oxipng -o level (None = skip)
    frontier: bool
    frontier_levels: int
    frontier_ceiling: int | None     # None = 90 quality or palette start colors
    deep_step: int                   # frontier step (%) when sqrt saturates
    ssim: bool
    compare_subsampling: bool        # JPEG probes 4:4:4 and 4:2:0 (frontier)
    prefer_software: bool            # AVIF/HEIC software encoders before hw
    min_quality_floor: int           # -Q default for lossy (85, 30 in slowest)
    min_colors_floor: int            # -Q default for palette (128, 2 in slowest)

    @property
    def search_mode(self) -> Literal["kary", "grid"]:
        # read through this, don't test step fields directly
        return "kary" if self.search_step is None else "grid"


def _from_profile(
    name: str,
    *,
    frontier: bool,
    frontier_levels: int,
    frontier_ceiling: int | None,
    deep_step: int,
    ssim: bool,
    compare_subsampling: bool,
    prefer_software: bool,
    min_quality_floor: int,
    min_colors_floor: int,
) -> SearchSettings:
    # keyword-only so mypy checks every preset
    prof = speed_profile(name)
    return SearchSettings(
        search_step=prof["search_step"],
        color_step=prof["color_step"],
        max_iterations=prof["max_iterations"],
        avif_speed=prof["avif_speed"],
        pngquant_speed=prof["pngquant_speed"],
        oxipng_level=prof["oxipng_level"],
        frontier=frontier,
        frontier_levels=frontier_levels,
        frontier_ceiling=frontier_ceiling,
        deep_step=deep_step,
        ssim=ssim,
        compare_subsampling=compare_subsampling,
        prefer_software=prefer_software,
        min_quality_floor=min_quality_floor,
        min_colors_floor=min_colors_floor,
    )


PRESETS: dict[str, SearchSettings] = {
    "default": _from_profile(
        "default",
        frontier=False, frontier_levels=3, frontier_ceiling=None,
        deep_step=75, ssim=True, compare_subsampling=False,
        prefer_software=False, min_quality_floor=85, min_colors_floor=128,
    ),
    "slow": _from_profile(
        "slow",
        frontier=False, frontier_levels=3, frontier_ceiling=None,
        deep_step=75, ssim=True, compare_subsampling=False,
        prefer_software=False, min_quality_floor=85, min_colors_floor=128,
    ),
    "fast": _from_profile(
        "fast",
        frontier=False, frontier_levels=3, frontier_ceiling=None,
        deep_step=75, ssim=True, compare_subsampling=False,
        prefer_software=False, min_quality_floor=85, min_colors_floor=128,
    ),
    "slowest": SearchSettings(
        search_step=None,
        color_step=None,
        max_iterations=200,
        avif_speed=1,
        pngquant_speed=1,
        oxipng_level=2,
        frontier=True,
        frontier_levels=3,
        # None = mode default (90 quality, full color for palette).
        # pinning 90 here would leak into the palette frontier.
        frontier_ceiling=None,
        deep_step=75,
        ssim=True,
        compare_subsampling=True,
        prefer_software=True,
        min_quality_floor=30,
        min_colors_floor=2,
    ),
}


def resolve_preset(args: Args) -> str:
    if getattr(args, "slowest", False):
        return "slowest"
    if getattr(args, "fast", False):
        return "fast"
    if getattr(args, "slow", False):
        return "slow"
    return "default"


def _pick(args: Args, name: str, base: Any) -> Any:
    val = getattr(args, name, None)
    return base if val is None else val


def resolve_search_settings(args: Args, preset: str | None = None) -> SearchSettings:
    # --no-oxipng clears the level, knob floors are preset-only
    if preset is None:
        preset = resolve_preset(args)
    base = PRESETS[preset]
    return SearchSettings(
        search_step=_pick(args, "search_step", base.search_step),
        color_step=_pick(args, "color_step", base.color_step),
        max_iterations=_pick(args, "max_iterations", base.max_iterations),
        avif_speed=_pick(args, "avif_speed", base.avif_speed),
        pngquant_speed=_pick(args, "pngquant_speed", base.pngquant_speed),
        oxipng_level=(None if getattr(args, "no_oxipng", False)
                      else _pick(args, "oxipng_level", base.oxipng_level)),
        frontier=_pick(args, "frontier", base.frontier),
        frontier_levels=_pick(args, "frontier_levels", base.frontier_levels),
        frontier_ceiling=_pick(args, "frontier_ceiling", base.frontier_ceiling),
        deep_step=_pick(args, "deep_step", base.deep_step),
        ssim=_pick(args, "ssim", base.ssim),
        compare_subsampling=_pick(args, "compare_subsampling", base.compare_subsampling),
        prefer_software=_pick(args, "prefer_software", base.prefer_software),
        min_quality_floor=base.min_quality_floor,
        min_colors_floor=base.min_colors_floor,
    )


def preset_flags(preset: str) -> list[str]:
    # --slow/--slowest leave step knobs unset for K-ary
    if preset == "default":
        return []
    base = PRESETS["default"]
    s = PRESETS[preset]
    flags: list[str] = []

    def add(name: str, flag: str) -> None:
        val, bval = getattr(s, name), getattr(base, name)
        if val != bval and not (val is None and bval is not None):
            flags.append(f"--{flag}={val}")

    def add_bool(name: str, flag: str) -> None:
        val, bval = getattr(s, name), getattr(base, name)
        if val != bval:
            flags.append(f"--{flag}" if val else f"--no-{flag}")

    add("search_step", "search-step")
    add("color_step", "color-step")
    add("max_iterations", "max-iterations")
    add("avif_speed", "avif-speed")
    add("pngquant_speed", "pngquant-speed")
    if s.oxipng_level is None and base.oxipng_level is not None:
        flags.append("--no-oxipng")
    else:
        add("oxipng_level", "oxipng-level")
    add_bool("frontier", "frontier")
    add("frontier_levels", "frontier-levels")
    add("frontier_ceiling", "frontier-ceiling")
    add("deep_step", "deep-step")
    add_bool("ssim", "ssim")
    add_bool("compare_subsampling", "compare-subsampling")
    add_bool("prefer_software", "prefer-software-codecs")
    return flags


@dataclass(frozen=True)
class ImageModeSettings:
    # each field is a CLI flag, explicit wins
    dithering: bool
    nearest: bool
    snap: bool
    realesrgan_model: str     # "photo" / "drawing"
    resample_filter: str      # -f/--filter value


# "photo" is the base, so image_mode_flags("photo") is []. pixel-art
# forbids neural upscalers, so its model value is unused.
IMAGE_MODE_PRESETS: dict[str, ImageModeSettings] = {
    "photo": ImageModeSettings(
        dithering=True, nearest=False, snap=False, realesrgan_model="photo",
        resample_filter="Lanczos",
    ),
    "drawing": ImageModeSettings(
        dithering=False, nearest=False, snap=False, realesrgan_model="drawing",
        resample_filter="Bicubic",
    ),
    "pixel_art": ImageModeSettings(
        dithering=False, nearest=True, snap=True, realesrgan_model="photo",
        resample_filter="Point",
    ),
}


def resolve_image_mode_settings(args: Args, mode: str | None = None) -> ImageModeSettings:
    # explicit -f wins, otherwise --nearest forces Point
    if mode is None:
        mode = resolve_image_mode(args)
    base = IMAGE_MODE_PRESETS[mode]
    nearest = base.nearest if args.nearest is None else args.nearest
    if args.filter is not None:
        resample_filter = args.filter
    elif nearest:
        resample_filter = "Point"
    else:
        resample_filter = base.resample_filter
    return ImageModeSettings(
        dithering=base.dithering if args.dithering is None else args.dithering,
        nearest=nearest,
        snap=base.snap if args.snap is None else args.snap,
        realesrgan_model=(base.realesrgan_model if args.realesrgan_model is None
                          else args.realesrgan_model),
        resample_filter=resample_filter,
    )


def image_mode_flags(mode: str) -> list[str]:
    base = IMAGE_MODE_PRESETS["photo"]
    s = IMAGE_MODE_PRESETS[mode]
    flags: list[str] = []
    if s.dithering != base.dithering:
        flags.append("--dithering" if s.dithering else "--no-dithering")
    if s.nearest != base.nearest:
        flags.append("--nearest" if s.nearest else "--no-nearest")
    if s.snap != base.snap:
        flags.append("--snap" if s.snap else "--no-snap")
    if s.resample_filter != base.resample_filter:
        flags.append(f"--filter={s.resample_filter}")
    if s.realesrgan_model != base.realesrgan_model:
        flags.append(f"--realesrgan-model={s.realesrgan_model}")
    return flags
