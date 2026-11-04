from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import converters, utils
from .encoders import (
    CodecEncoder,
    SpeedProfile,
    animated_gif_reduce,
    animated_quality_reduce,
    color_palette_reduce,
    jpeg_quality_reduce,
    make_lossy_encoders,
    oxipng_compress,
    pngquant_compress,
    speed_profile,
)
from .formats import ANIMATED_LOSSY_FORMATS
from .search import (
    _MAX_PARALLEL,
    _animated_reducer,
    _art_reducer,
    _cleanup_probes,
    _coarse_grid_search,
    _frontier_lossy_round,
    _new_probe,
    _parallel_fit_search,
    _probe_extent,
    _probe_width,
    _RoundResult,
    _run_search_loop,
    _SearchStrategy,
    _simple_search_round,
)
from .ssim import (
    _ENCODER_PREF,
    _animated_chained_reference,
    _animated_ssim_vs_reference,
    _chained_reference,
    _rank_animated_candidates,
    _rank_art_candidates,
    _rank_lossy_candidates,
    _ssim_vs_reference,
)

# frontier stops here, lower levels still allowed
_SLOWEST_FRONTIER_CEILING_QUALITY = 90

# defaults when -q/-Q absent, pipeline resolves per preset
_DEFAULT_MIN_QUALITY_SLOWEST = 30
_DEFAULT_MIN_QUALITY = 85
_DEFAULT_MIN_COLORS_SLOWEST = 2
_DEFAULT_MIN_COLORS = 128
_DEFAULT_BAIL_RATIO = 1.3


def _resolve_knob_floor(
    custom_min: int | None,
    custom_val: int | None,
    floor: int,
) -> int:
    # Explicit -Q wins, else preset floor, lowered to -q if below it.
    if custom_min is not None:
        return custom_min
    floor = int(floor)
    if custom_val is not None and custom_val < floor:
        return custom_val
    return floor


def calibrate_ssim_target(
    current: Path,
    encoder: CodecEncoder,
    *,
    threshold: float,
    subsampling: str,
    strip_metadata: bool,
    min_quality: int,
    max_quality: int,
) -> int | None:
    # Lowest quality meeting the threshold, plus its size. A failed measurement
    # counts as not fitting, so noise can only raise the budget.
    probes: list[Path] = []

    def probe(q: int) -> tuple[float | None, int]:
        path = _new_probe(encoder.suffix, probes)
        got = encoder.encode(current, path, q, subsampling=subsampling,
                             strip_metadata=strip_metadata)
        utils._abort_if_interrupted()
        size = converters.get_image_filesize(path) if got else 0
        if not got or size <= 0:
            return None, 0
        return _ssim_vs_reference(current, path, probes), size

    try:
        score, size = probe(max_quality)
        if score is None or score < threshold:
            return None
        best_size = size
        lo, hi = min_quality, max_quality
        while lo < hi:
            mid = (lo + hi) // 2
            score, size = probe(mid)
            if score is not None and score >= threshold:
                hi = mid
                best_size = size
            else:
                lo = mid + 1
        if lo != max_quality:
            # The converged bound may be an unprobed quality, so measure it.
            score, size = probe(lo)
            if score is not None and score >= threshold and size > 0:
                best_size = size
        return best_size
    finally:
        _cleanup_probes(probes)


def calibrate_palette_ssim_target(
    current: Path,
    *,
    threshold: float,
    strip_metadata: bool,
    min_colors: int,
    max_colors: int,
) -> tuple[int, int] | None:
    # Returns (budget, colors). Palette sizes step in cliffs, so the caller
    # seeds the search at the crossing probe.
    probes: list[Path] = []

    def probe(colors: int) -> tuple[float | None, int]:
        path = _new_probe(".png", probes)
        got = color_palette_reduce(current, path, colors,
                                   strip_metadata=strip_metadata)
        utils._abort_if_interrupted()
        size = converters.get_image_filesize(path) if got else 0
        if not got or size <= 0:
            return None, 0
        return _ssim_vs_reference(current, path, probes), size

    try:
        score, size = probe(max_colors)
        if score is None or score < threshold:
            return None
        best_size, best_colors = size, max_colors
        lo, hi = min_colors, max_colors
        while lo < hi:
            mid = (lo + hi) // 2
            score, size = probe(mid)
            if score is not None and score >= threshold:
                hi = mid
                best_size, best_colors = size, mid
            else:
                lo = mid + 1
        if lo != max_colors:
            # The converged bound may be an unprobed color count, so measure it.
            score, size = probe(lo)
            if score is not None and score >= threshold and size > 0:
                best_size, best_colors = size, lo
        return best_size, best_colors
    finally:
        _cleanup_probes(probes)

# Public API plus the internals tests import from here. mypy --strict has
# no_implicit_reexport, hence the explicit list.
__all__ = [
    "_ENCODER_PREF",
    "_MAX_PARALLEL",
    "CodecEncoder",
    "SpeedProfile",
    "_RoundResult",
    "_SearchStrategy",
    "_animated_chained_reference",
    "_animated_reducer",
    "_animated_ssim_vs_reference",
    "_art_reducer",
    "_chained_reference",
    "_cleanup_probes",
    "_coarse_grid_search",
    "_frontier_lossy_round",
    "_new_probe",
    "_parallel_fit_search",
    "_probe_extent",
    "_probe_width",
    "_rank_animated_candidates",
    "_rank_art_candidates",
    "_rank_lossy_candidates",
    "_run_search_loop",
    "_simple_search_round",
    "_ssim_vs_reference",
    "animated_gif_optimize",
    "animated_gif_reduce",
    "animated_lossy_optimize",
    "animated_quality_reduce",
    "art_optimize",
    "calibrate_palette_ssim_target",
    "calibrate_ssim_target",
    "color_palette_reduce",
    "iterative_animated_gif_optimize",
    "iterative_animated_lossy_optimize",
    "iterative_art_optimize",
    "iterative_lossy_optimize",
    "jpeg_quality_reduce",
    "lossy_optimize",
    "make_lossy_encoders",
    "oxipng_compress",
    "pngquant_compress",
    "slowest_animated_gif_optimize",
    "slowest_animated_lossy_optimize",
    "slowest_art_optimize",
    "slowest_lossy_optimize",
    "speed_profile",
]


def _lossy_strategy(
    encoder_list: list[CodecEncoder],
    subsampling: str,
    strip_metadata: bool,
    max_size_bytes: int,
    use_extent_fast_path: bool,
    search_step: int | None,
    frontier: bool,
    frontier_ceiling: int | None,
    deep_step: int,
    ssim: bool,
    compare_subsampling: bool,
    ssim_floor: float | None = None,
    knob_hi: int = 100,
    knob_lo: int = 1,
) -> _SearchStrategy:
    extent_ok = use_extent_fast_path and encoder_list[0].extent is not None
    ceiling = _SLOWEST_FRONTIER_CEILING_QUALITY if frontier_ceiling is None else frontier_ceiling

    def encode_probe(src: Path, out: Path, quality: int) -> bool:
        return encoder_list[0].encode(src, out, quality, subsampling=subsampling,
                                      strip_metadata=strip_metadata)

    if frontier:
        if search_step is not None:
            search_round = _simple_search_round(
                probe_encode=encode_probe, probe_suffix=encoder_list[0].suffix,
                probe_message="   Testing quality {}%...",
                probe_skip_message="   Skipping quality {}% (fit at {}% already found).",
                search_step=search_step, max_size_bytes=max_size_bytes,
                round_collect_candidates=True, round_frontier_ceiling=ceiling,
                round_candidate_knob="quality",
            )
        else:
            search_round = _frontier_lossy_round(
                encoder_list, subsampling, strip_metadata,
                max_size_bytes, round_frontier_ceiling=ceiling,
                compare_subsampling=compare_subsampling,
            )
    else:
        search_round = _simple_search_round(
            probe_encode=encode_probe, probe_suffix=encoder_list[0].suffix,
            probe_message="   Testing quality {}%...",
            probe_skip_message="   Skipping quality {}% (fit at {}% already found).",
            search_step=search_step, max_size_bytes=max_size_bytes,
            round_collect_candidates=False, round_frontier_ceiling=0,
        )

    return _SearchStrategy(
        search_round=search_round,
        knob_range=(1, 100),
        fit_message=None if frontier else "   Fits at quality {}%.",
        extent_first=not frontier and search_step is None and extent_ok,
        extent_compare=not frontier and search_step is None and extent_ok,
        extent_after=not frontier and search_step is not None and extent_ok,
        extent_probe=(lambda current, probes: _probe_extent(
            encoder_list[0], current, probes, max_size_bytes, subsampling, strip_metadata))
        if extent_ok else None,
        reducer=lambda src, out, pct, filt: converters.resize_pct(src, out, pct, filt),
        resize_message=(lambda lo, pct: f"   Exploring resolution: resizing to {pct}%...")
        if frontier else (lambda lo, pct: f"   Nothing fits down to {lo}%. "
                                          f"Resizing to {pct}% and resetting quality..."),
        deep_resize_from_original=frontier,
        deep_step=deep_step,
        collect_candidates=frontier,
        break_on_all_probe_failure=True,
        optimizer_label="Best" if frontier else "Iterative",
        bail_encode=encode_probe,
        bail_suffix=encoder_list[0].suffix,
        select_winner=(lambda cands, ref, reg: _rank_lossy_candidates(
            cands, ref, reg, ssim=ssim,
            # Rank at original resolution. Own-resolution scores reward
            # downscaling (a halved q98 scores ~0.97 vs itself, 0.69 vs the
            # original). ssim_floor picks the smallest that passes.
            at_original=True, floor=ssim_floor))
        if frontier else None,
        winner_message=(lambda w, size: (
            f"   Best: {w['encoder'].upper()} quality {w['quality']} "
            f"({w['subsampling']}) at {size} bytes."))
        if frontier else None,
    )


def lossy_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    encoder: CodecEncoder | dict[str, CodecEncoder] | Iterable[CodecEncoder],
    start_quality: int = 100,
    min_quality: int = _DEFAULT_MIN_QUALITY,
    min_width: int = 4,
    strip_metadata: bool = False,
    subsampling: str = "4:2:0",
    use_extent_fast_path: bool = True,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    ssim: bool = True,
    compare_subsampling: bool = False,
    ssim_floor: float | None = None,
) -> dict[str, int | bool]:
    # Every probe is a fresh encode from the lossless source. encoder may be
    # one codec or several. Grid rounds probe the first one only.
    if isinstance(encoder, CodecEncoder):
        encoder_list = [encoder]
    elif isinstance(encoder, dict):
        encoder_list = list(encoder.values())
    else:
        encoder_list = list(encoder)
    if not encoder_list:
        raise ValueError("lossy_optimize requires at least one encoder")
    strategy = _lossy_strategy(
        encoder_list, subsampling, strip_metadata, max_size_bytes,
        use_extent_fast_path, search_step, frontier, frontier_ceiling,
        deep_step, ssim, compare_subsampling,
        ssim_floor=ssim_floor,
    )
    return _run_search_loop(
        strategy, input_path, output_path, max_size_bytes,
        start_quality, min_quality, min_width=min_width, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        extra_fit_levels=extra_fit_levels,
    )


def iterative_lossy_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    encoder: CodecEncoder,
    start_quality: int = 100,
    min_quality: int = _DEFAULT_MIN_QUALITY,
    min_width: int = 4,
    strip_metadata: bool = False,
    subsampling: str = "4:2:0",
    use_extent_fast_path: bool = True,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    ssim: bool = True,
    compare_subsampling: bool = False,
) -> dict[str, int | bool]:
    return lossy_optimize(
        input_path, output_path, max_size_bytes, encoder,
        start_quality=start_quality, min_quality=min_quality,
        min_width=min_width, strip_metadata=strip_metadata,
        subsampling=subsampling,
        use_extent_fast_path=use_extent_fast_path, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        search_step=search_step,
        frontier=frontier, extra_fit_levels=extra_fit_levels,
        frontier_ceiling=frontier_ceiling, deep_step=deep_step,
        ssim=ssim, compare_subsampling=compare_subsampling,
    )


def slowest_lossy_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    encoders: dict[str, CodecEncoder] | Iterable[CodecEncoder],
    start_quality: int = 100,
    min_quality: int = _DEFAULT_MIN_QUALITY_SLOWEST,
    min_width: int = 4,
    strip_metadata: bool = False,
    subsampling: str = "4:2:0",
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 200,
    filter: str = "Lanczos",
    extra_fit_levels: int = 3,
    ssim: bool = True,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    compare_subsampling: bool = True,
    ssim_floor: float | None = None,
) -> dict[str, int | bool]:
    # ssim_floor: smallest candidate meeting the floor wins.
    return lossy_optimize(
        input_path, output_path, max_size_bytes, encoders,
        start_quality=start_quality, min_quality=min_quality,
        min_width=min_width, strip_metadata=strip_metadata,
        subsampling=subsampling,
        bail_ratio=bail_ratio, max_iterations=max_iterations,
        filter=filter,
        frontier=True, extra_fit_levels=extra_fit_levels,
        frontier_ceiling=frontier_ceiling, deep_step=deep_step,
        ssim=ssim, compare_subsampling=compare_subsampling,
        ssim_floor=ssim_floor,
    )


def _art_strategy(
    pixel_art_mode: bool,
    strip_metadata: bool,
    max_size_bytes: int,
    search_step: int | None,
    frontier: bool,
    frontier_ceiling: int | None,
    deep_step: int,
    ssim: bool,
    container_corrected: bool,
    rank_width_first: bool,
    start_colors: int,
    filter: str,
    pixel_art_snap: bool = False,
    ssim_floor: float | None = None,
) -> _SearchStrategy:
    ceiling = start_colors if frontier_ceiling is None else frontier_ceiling

    def probe_encode(src: Path, out: Path, colors: int) -> bool:
        return color_palette_reduce(src, out, colors, strip_metadata=strip_metadata)

    def ceiling_fit(input_path: Path, candidates: list[dict[str, Any]],
                    probes: list[Path], lo: int, hi: int, max_size: int) -> None:
        # One extra level at the largest resolution where the full color count
        # fits, that source gives the final output the most detail.
        ceiling = [c for c in candidates if c.get("colors", 0) >= hi]
        if not ceiling:
            return
        big = max(ceiling, key=lambda c: c.get("size", 0))
        big_size = big.get("size", 0)
        if big_size <= 0:
            return
        scale_up = math.sqrt(max_size * 0.95 / big_size)
        if scale_up <= 1.0:
            return
        big_w = converters.get_image_width(big["path"])
        in_w = converters.get_image_width(input_path)
        pct = max(1, round(big_w * scale_up * 100 / in_w)) if in_w > 0 else 0
        if not 0 < pct < 100:
            return
        up_res = _new_probe(".png", probes)
        converters.resize_pct(input_path, up_res, pct, filter)
        utils._abort_if_interrupted()

        def probe_fn(c: int, _current: Path = up_res) -> tuple[bool, int, Path]:
            probe = _new_probe(".png", probes)
            got = color_palette_reduce(_current, probe, c, strip_metadata)
            size = converters.get_image_filesize(probe) if got else 0
            return (got and 0 < size <= max_size, size, probe)

        best, _ = _parallel_fit_search(probe_fn, lo, hi)
        if best is not None:
            candidates.append({
                "path": best[2], "size": best[1], "colors": best[0],
            })
            print(f"   Ceiling-fit level: {best[0]} colors at {pct}% "
                  f"({converters.get_image_filesize(best[2])} bytes).")

    return _SearchStrategy(
        search_round=_simple_search_round(
            probe_encode=probe_encode, probe_suffix=".png",
            probe_message="   Testing max colors {}...",
            probe_skip_message="   Skipping max colors {} (fit at {} already found).",
            search_step=search_step, max_size_bytes=max_size_bytes,
            round_collect_candidates=frontier, round_frontier_ceiling=ceiling,
        ),
        knob_range=(2, 256),
        fit_message="   Fits at {} colors.",
        reducer=_art_reducer(pixel_art_mode),
        pixel_art_snap=pixel_art_snap,
        deep_step=deep_step,
        deep_resize_from_original=frontier,
        palette=True,
        palette_pass=probe_encode,
        entry_size_guard=True,
        collect_candidates=frontier,
        prefer_current_if_fits=True,
        optimizer_label="Best" if frontier else "Iterative",
        extra_candidates=ceiling_fit if (frontier and container_corrected) else None,
        select_winner=(lambda cands, ref, reg: _rank_art_candidates(
            cands, ref, reg, ssim=ssim and not container_corrected,
            by_colors=container_corrected, width_first=rank_width_first,
            at_original=True,
            # Floor picks the smallest passing candidate, container-corrected
            # targets rank without SSIM, so no floor.
            floor=ssim_floor if not container_corrected else None))
        if frontier else None,
        winner_message=(lambda w, size: f"   Best: {w['colors']} colors at {size} bytes.")
        if frontier else None,
    )


def art_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    pixel_art_mode: bool = False,
    start_colors: int = 256,
    min_colors: int = _DEFAULT_MIN_COLORS,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    ssim: bool = True,
    container_corrected: bool = False,
    rank_width_first: bool = False,
    pixel_art_snap: bool | None = None,
    ssim_floor: float | None = None,
) -> dict[str, int | bool]:
    # None keeps the historical coupling: snap iff pixel_art_mode.
    strategy = _art_strategy(
        pixel_art_mode, strip_metadata, max_size_bytes, search_step,
        frontier, frontier_ceiling, deep_step, ssim, container_corrected,
        rank_width_first, start_colors, filter,
        pixel_art_snap=(pixel_art_mode if pixel_art_snap is None
                        else pixel_art_snap),
        ssim_floor=ssim_floor,
    )
    return _run_search_loop(
        strategy, input_path, output_path, max_size_bytes,
        start_colors, min_colors, min_width=min_width, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        extra_fit_levels=extra_fit_levels,
    )


def iterative_art_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    pixel_art_mode: bool = False,
    start_colors: int = 256,
    min_colors: int = _DEFAULT_MIN_COLORS,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    ssim: bool = True,
    container_corrected: bool = False,
    pixel_art_snap: bool | None = None,
    ssim_floor: float | None = None,
) -> dict[str, int | bool]:
    return art_optimize(
        input_path, output_path, max_size_bytes,
        pixel_art_mode=pixel_art_mode, start_colors=start_colors,
        min_colors=min_colors, min_width=min_width,
        strip_metadata=strip_metadata, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        search_step=search_step,
        frontier=frontier, extra_fit_levels=extra_fit_levels,
        frontier_ceiling=frontier_ceiling, ssim=ssim,
        container_corrected=container_corrected,
        pixel_art_snap=pixel_art_snap, ssim_floor=ssim_floor,
    )


def slowest_art_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    pixel_art_mode: bool = False,
    start_colors: int = 256,
    min_colors: int = _DEFAULT_MIN_COLORS_SLOWEST,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 200,
    filter: str = "Lanczos",
    extra_fit_levels: int = 3,
    ssim: bool = True,
    container_corrected: bool = False,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    rank_width_first: bool = False,
    pixel_art_snap: bool | None = None,
    ssim_floor: float | None = None,
) -> dict[str, int | bool]:
    return art_optimize(
        input_path, output_path, max_size_bytes,
        pixel_art_mode=pixel_art_mode, start_colors=start_colors,
        min_colors=min_colors, min_width=min_width,
        strip_metadata=strip_metadata, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        frontier=True, extra_fit_levels=extra_fit_levels,
        frontier_ceiling=frontier_ceiling, ssim=ssim,
        container_corrected=container_corrected, deep_step=deep_step,
        rank_width_first=rank_width_first, pixel_art_snap=pixel_art_snap,
        ssim_floor=ssim_floor,
    )


def _animated_strategy(
    pixel_art_mode: bool,
    strip_metadata: bool,
    max_size_bytes: int,
    search_step: int | None,
    frontier: bool,
    frontier_ceiling: int | None,
    deep_step: int,
    ssim: bool,
    start_colors: int,
    pixel_art_snap: bool = False,
    entry_size_guard: bool = True,
) -> _SearchStrategy:
    # Disabled when the source fits but the converted GIF doesn't, the guard
    # measures the search input, not the final container.
    ceiling = start_colors if frontier_ceiling is None else frontier_ceiling

    def probe_encode(src: Path, out: Path, colors: int) -> bool:
        return animated_gif_reduce(src, out, colors, strip_metadata=strip_metadata)

    return _SearchStrategy(
        search_round=_simple_search_round(
            probe_encode=probe_encode, probe_suffix=".gif",
            probe_message="   Testing max colors {}...",
            probe_skip_message="   Skipping max colors {} (fit at {} already found).",
            search_step=search_step, max_size_bytes=max_size_bytes,
            round_collect_candidates=frontier, round_frontier_ceiling=ceiling,
        ),
        knob_range=(2, 256),
        fit_message="   Fits at {} colors.",
        reducer=_animated_reducer(pixel_art_mode),
        resize_suffix=".gif",
        pixel_art_snap=pixel_art_snap,
        deep_step=deep_step,
        deep_resize_from_original=frontier,
        palette=True,
        palette_pass=probe_encode,
        entry_size_guard=entry_size_guard,
        collect_candidates=frontier,
        prefer_current_if_fits=True,
        optimizer_label="Best" if frontier else "Iterative",
        select_winner=(lambda cands, ref, reg: _rank_animated_candidates(cands, ref, reg,
                                                                         ssim=ssim,
                                                                         at_original=True))
        if frontier else None,
        winner_message=(lambda w, size: f"   Best: {w['colors']} colors at {size} bytes.")
        if frontier else None,
    )


def animated_gif_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    pixel_art_mode: bool = False,
    start_colors: int = 256,
    min_colors: int = _DEFAULT_MIN_COLORS,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    ssim: bool = True,
    pixel_art_snap: bool | None = None,
    entry_size_guard: bool = True,
) -> dict[str, int | bool]:
    strategy = _animated_strategy(
        pixel_art_mode, strip_metadata, max_size_bytes, search_step,
        frontier, frontier_ceiling, deep_step, ssim, start_colors,
        pixel_art_snap=(pixel_art_mode if pixel_art_snap is None
                        else pixel_art_snap),
        entry_size_guard=entry_size_guard,
    )
    return _run_search_loop(
        strategy, input_path, output_path, max_size_bytes,
        start_colors, min_colors, min_width=min_width, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        extra_fit_levels=extra_fit_levels,
    )


def iterative_animated_gif_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    pixel_art_mode: bool = False,
    start_colors: int = 256,
    min_colors: int = _DEFAULT_MIN_COLORS,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    ssim: bool = True,
    pixel_art_snap: bool | None = None,
    entry_size_guard: bool = True,
) -> dict[str, int | bool]:
    return animated_gif_optimize(
        input_path, output_path, max_size_bytes,
        pixel_art_mode=pixel_art_mode, start_colors=start_colors,
        min_colors=min_colors, min_width=min_width,
        strip_metadata=strip_metadata, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        search_step=search_step,
        frontier=frontier, extra_fit_levels=extra_fit_levels,
        frontier_ceiling=frontier_ceiling, ssim=ssim,
        pixel_art_snap=pixel_art_snap,
        entry_size_guard=entry_size_guard,
    )


def slowest_animated_gif_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    pixel_art_mode: bool = False,
    start_colors: int = 256,
    min_colors: int = _DEFAULT_MIN_COLORS_SLOWEST,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 200,
    filter: str = "Lanczos",
    extra_fit_levels: int = 3,
    ssim: bool = True,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    pixel_art_snap: bool | None = None,
    entry_size_guard: bool = True,
) -> dict[str, int | bool]:
    return animated_gif_optimize(
        input_path, output_path, max_size_bytes,
        pixel_art_mode=pixel_art_mode, start_colors=start_colors,
        min_colors=min_colors, min_width=min_width,
        strip_metadata=strip_metadata, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        frontier=True, extra_fit_levels=extra_fit_levels,
        frontier_ceiling=frontier_ceiling, ssim=ssim, deep_step=deep_step,
        pixel_art_snap=pixel_art_snap,
        entry_size_guard=entry_size_guard,
    )


def _animated_lossy_strategy(
    encoder: CodecEncoder,
    strip_metadata: bool,
    max_size_bytes: int,
    search_step: int | None,
    frontier: bool,
    frontier_ceiling: int | None,
    deep_step: int,
    ssim: bool,
    pixel_art_mode: bool,
    pixel_art_snap: bool,
) -> _SearchStrategy:
    ceiling = _SLOWEST_FRONTIER_CEILING_QUALITY if frontier_ceiling is None else frontier_ceiling

    def probe_encode(src: Path, out: Path, quality: int) -> bool:
        return encoder.encode(src, out, quality,
                              strip_metadata=strip_metadata)

    return _SearchStrategy(
        search_round=_simple_search_round(
            probe_encode=probe_encode, probe_suffix=encoder.suffix,
            probe_message="   Testing quality {}%...",
            probe_skip_message="   Skipping quality {}% (fit at {}% already found).",
            search_step=search_step, max_size_bytes=max_size_bytes,
            round_collect_candidates=frontier,
            round_frontier_ceiling=ceiling,
            round_candidate_knob="quality",
        ),
        knob_range=(1, 100),
        fit_message="   Fits at quality {}%.",
        reducer=_animated_reducer(pixel_art_mode),
        resize_suffix=".apng",
        resize_message=(lambda lo, pct: f"   Exploring resolution: resizing to {pct}%...")
        if frontier else (lambda lo, pct: f"   Nothing fits down to {lo}%. "
                                          f"Resizing to {pct}% and resetting quality..."),
        pixel_art_snap=pixel_art_snap,
        deep_step=deep_step,
        deep_resize_from_original=frontier,
        collect_candidates=frontier,
        break_on_all_probe_failure=True,
        optimizer_label="Best" if frontier else "Iterative",
        bail_encode=probe_encode,
        bail_suffix=encoder.suffix,
        select_winner=(lambda cands, ref, reg: _rank_animated_candidates(
            cands, ref, reg, ssim=ssim, at_original=True))
        if frontier else None,
        winner_message=(lambda w, size: (
            f"   Best: quality {w['quality']} at {size} bytes."))
        if frontier else None,
    )


def animated_lossy_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    target_format: str,
    start_quality: int = 100,
    min_quality: int = _DEFAULT_MIN_QUALITY,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    ssim: bool = True,
    pixel_art_mode: bool = False,
    pixel_art_snap: bool = False,
) -> dict[str, int | bool]:
    target = converters.normalize_format(target_format)
    if target not in ANIMATED_LOSSY_FORMATS:
        raise ValueError(f"animated_lossy_optimize: unsupported target {target}")
    suffix = "." + target.lower()

    def encode(src: Path, out: Path, quality: int,
               subsampling: str = "4:2:0",
               strip_metadata: bool = False) -> bool:
        return animated_quality_reduce(src, out, quality,
                                       strip_metadata=strip_metadata)

    encoder = CodecEncoder(name=target.lower(), encode=encode, suffix=suffix)
    strategy = _animated_lossy_strategy(
        encoder, strip_metadata, max_size_bytes, search_step, frontier,
        frontier_ceiling, deep_step, ssim, pixel_art_mode, pixel_art_snap,
    )
    return _run_search_loop(
        strategy, input_path, output_path, max_size_bytes,
        start_quality, min_quality, min_width=min_width, bail_ratio=bail_ratio,
        max_iterations=max_iterations, filter=filter,
        extra_fit_levels=extra_fit_levels,
    )


def iterative_animated_lossy_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    target_format: str,
    start_quality: int = 100,
    min_quality: int = _DEFAULT_MIN_QUALITY,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    search_step: int | None = None,
    frontier: bool = False,
    extra_fit_levels: int = 3,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    ssim: bool = True,
    pixel_art_mode: bool = False,
    pixel_art_snap: bool = False,
) -> dict[str, int | bool]:
    return animated_lossy_optimize(
        input_path, output_path, max_size_bytes, target_format,
        start_quality=start_quality, min_quality=min_quality,
        min_width=min_width, strip_metadata=strip_metadata,
        bail_ratio=bail_ratio, max_iterations=max_iterations, filter=filter,
        search_step=search_step, frontier=frontier,
        extra_fit_levels=extra_fit_levels, frontier_ceiling=frontier_ceiling,
        deep_step=deep_step, ssim=ssim, pixel_art_mode=pixel_art_mode,
        pixel_art_snap=pixel_art_snap,
    )


def slowest_animated_lossy_optimize(
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    target_format: str,
    start_quality: int = 100,
    min_quality: int = _DEFAULT_MIN_QUALITY_SLOWEST,
    min_width: int = 4,
    strip_metadata: bool = False,
    bail_ratio: float = _DEFAULT_BAIL_RATIO,
    max_iterations: int = 200,
    filter: str = "Lanczos",
    extra_fit_levels: int = 3,
    ssim: bool = True,
    frontier_ceiling: int | None = None,
    deep_step: int = 75,
    pixel_art_mode: bool = False,
    pixel_art_snap: bool = False,
) -> dict[str, int | bool]:
    return animated_lossy_optimize(
        input_path, output_path, max_size_bytes, target_format,
        start_quality=start_quality, min_quality=min_quality,
        min_width=min_width, strip_metadata=strip_metadata,
        bail_ratio=bail_ratio, max_iterations=max_iterations, filter=filter,
        frontier=True, extra_fit_levels=extra_fit_levels,
        frontier_ceiling=frontier_ceiling, deep_step=deep_step, ssim=ssim,
        pixel_art_mode=pixel_art_mode, pixel_art_snap=pixel_art_snap,
    )
