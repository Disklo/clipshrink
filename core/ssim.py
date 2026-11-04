from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from . import converters, memgate, search, utils

# codec preference on full ties: jpeg > webp > avif. only matters with
# multi-encoder runs.
_ENCODER_PREF = {"jpeg": 0, "webp": 1, "avif": 2}

# within this band candidates tie, then resolution/budget decide. stops noise
# from handing a win to a downscaled candidate. animated uses 0.02.
_SSIM_EPS = 0.04
_ANIMATED_SSIM_EPS = 0.02

# ffmpeg ssim requires (w+128)*(h+128) <= 2^28
_FFMPEG_SSIM_MAX_PRODUCT = 2 ** 28

# peak bytes per measured pixel on the native path (~78 B/px magick + ffmpeg).
# turns the memory budget into the largest frame measurable in place.
_MEASUREMENT_BYTES_PER_PIXEL = 80


def _measurement_cap() -> int:
    # an 8 GB machine measures a proxy instead of OOMing
    budget = memgate.budget_bytes()
    if budget is not None:
        return max(budget // _MEASUREMENT_BYTES_PER_PIXEL, 1_000_000)
    return 64_000_000  # conservative default when no budget is known


def _ffmpeg_safe_dims(width: int, height: int) -> tuple[int, int]:
    # margins make the cap aspect-dependent, extreme ratios hit it sooner
    while (width + 128) * (height + 128) > _FFMPEG_SSIM_MAX_PRODUCT:
        scale = math.sqrt(_FFMPEG_SSIM_MAX_PRODUCT
                          / ((width + 128) * (height + 128)))
        width = max(1, math.floor(width * scale))
        height = max(1, math.floor(height * scale))
    return width, height


def _ssim_vs_reference(reference: Path, candidate: Path, registry: list[Path],
                       at_original: bool = False) -> float | None:
    # at_original upscales the candidate so a downscaled one can't pass by
    # scoring at its own size. ffmpeg ssim, not magick compare, some builds
    # emit RMSE outside [0,1]. None when measurement can't run.
    if at_original:
        try:
            width = converters.get_image_width(reference)
            height = converters.get_image_height(reference)
        except RuntimeError:
            return None
    else:
        width = search._probe_width(candidate)
        try:
            height = converters.get_image_height(candidate)
        except RuntimeError:
            return None
    if width <= 0 or height <= 0:
        return None
    # memory or ffmpeg's cap decides proxy mode, both sides scale together
    cap = _measurement_cap()
    probe_px = _probe_pixels(candidate)
    if width * height > cap or probe_px > cap:
        # proxy must never exceed the budget-derived cap
        scale = math.sqrt(cap / max(width * height, probe_px))
        width = max(1, math.floor(width * scale))
        height = max(1, math.floor(height * scale))
        proxy = True
    else:
        proxy = False
    safe_width, safe_height = _ffmpeg_safe_dims(width, height)
    if (safe_width, safe_height) != (width, height):
        width, height = safe_width, safe_height
        proxy = True
    ref_resized = search._new_probe(".png", registry)
    cand_png = search._new_probe(".png", registry)
    try:
        if proxy:
            # proxy uses vips-only resizes, magick exact at HDRI needs
            # ~10 GiB for a 658 MP source.
            if converters.resize_exact_vips(reference, ref_resized, width, height, "Lanczos") is False:
                converters.resize_exact(reference, ref_resized, width, height, "Lanczos")
            if converters.resize_exact_vips(candidate, cand_png, width, height, "Lanczos") is False:
                converters.resize_exact(candidate, cand_png, width, height, "Lanczos")
            if not cand_png.exists():
                return None
        else:
            # native keeps small-image scores identical to the old path
            converters.resize_exact(reference, ref_resized, width, height, "Lanczos")
            result = converters._run_magick([
                "magick", str(candidate), "-resize", f"{width}x{height}!", "-depth", "8",
                str(cand_png),
            ])
            if result.returncode != 0 or not cand_png.exists():
                return None
        result = utils.run_capture([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "info",
            "-i", str(ref_resized), "-i", str(cand_png), "-frames:v", "1",
            "-lavfi", "ssim", "-f", "null", "-",
        ])
        text = f"{result.stderr or ''} {result.stdout or ''}"
        m = re.search(r"All:([0-9.]+)", text)
        return float(m.group(1)) if m else None
    except RuntimeError:
        return None


def fidelity_pick(a: tuple[Path, int], b: tuple[Path, int], reference: Path,
                  registry: list[Path], eps: float = _SSIM_EPS) -> tuple[Path, int]:
    # tie within eps prefers higher resolution, then larger size. if neither
    # measures, a ships (the deterministic extent result).
    scores = {a[0]: _ssim_vs_reference(reference, a[0], registry, at_original=True),
              b[0]: _ssim_vs_reference(reference, b[0], registry, at_original=True)}
    if scores[a[0]] is None and scores[b[0]] is None:
        return a

    def key(item: tuple[Path, int]) -> tuple[Any, ...]:
        score = scores[item[0]]
        bucket = round(score / eps) if score is not None else -(2 ** 30)
        return (bucket, search._probe_width(item[0]), item[1])

    return max((a, b), key=key)


def _probe_pixels(image: Path) -> int:
    try:
        return (converters.get_image_width(image)
                * converters.get_image_height(image))
    except RuntimeError:
        return 0


def _chained_reference(reference: Path, candidates: list[dict[str, Any]],
                       registry: list[Path]) -> Path:
    # one reference downscale at the largest candidate, so resolution loss
    # counts against a candidate
    max_width = max((search._probe_width(c["path"]) for c in candidates), default=0)
    if max_width <= 0:
        return reference
    try:
        ref_w = converters.get_image_width(reference)
        ref_h = converters.get_image_height(reference)
    except RuntimeError:
        return reference
    if ref_w <= 0:
        return reference
    ref_max = search._new_probe(".png", registry)
    ref_h_at_max = max(1, round(max_width * ref_h / ref_w))
    try:
        converters.resize_exact(reference, ref_max, max_width, ref_h_at_max, "Lanczos")
    except RuntimeError:
        return reference
    return ref_max


def _rank_lossy_candidates(candidates: list[dict[str, Any]], reference: Path,
                           registry: list[Path], ssim: bool = True,
                           eps: float = _SSIM_EPS, at_original: bool = False,
                           floor: float | None = None) -> dict[str, Any] | None:
    # tie order: resolution, size, 4:4:4, quality, codec. with floor, the
    # smallest passing candidate wins, else the highest score.
    if not candidates:
        return None
    ref_for_ssim = (reference if (at_original or not ssim)
                    else _chained_reference(reference, candidates, registry))
    scored: list[tuple[Any, dict[str, Any]]] = [
        (_ssim_vs_reference(ref_for_ssim, c["path"], registry,
                            at_original=at_original) if ssim else None, c)
        for c in candidates
    ]

    if floor is not None:
        passing = [(s, c) for s, c in scored if s is not None and s >= floor]
        if passing:
            # smallest file meeting the floor. ties prefer resolution, 4:4:4,
            # quality, then codec (lowest _ENCODER_PREF wins).
            return min(passing, key=lambda item: (
                item[1].get("size", 0),
                -search._probe_width(item[1]["path"]),
                0 if item[1].get("subsampling") == "4:4:4" else 1,
                -item[1].get("quality", 0),
                _ENCODER_PREF.get(item[1].get("encoder", ""), 0),
            ))[1]
        # nothing reaches the floor, take the closest
        return max(scored, key=lambda item: (item[0] if item[0] is not None
                                             else -(2 ** 30),))[1]

    def sort_key(item: tuple[Any, dict[str, Any]]) -> tuple[Any, ...]:
        score, c = item
        bucket = round(score / eps) if score is not None else -(2 ** 30)
        width = search._probe_width(c["path"])
        ss_bonus = 1 if c.get("subsampling") == "4:4:4" else 0
        # negated so descending order prefers lowest pref (jpeg first)
        pref = -_ENCODER_PREF.get(c.get("encoder", ""), 0)
        return (bucket, width, c.get("size", 0), ss_bonus, c.get("quality", 0), pref)

    scored.sort(key=sort_key, reverse=True)
    return scored[0][1]


def _rank_art_candidates(candidates: list[dict[str, Any]], reference: Path,
                         registry: list[Path], ssim: bool = True,
                         eps: float = _SSIM_EPS, by_colors: bool = False,
                         width_first: bool = False,
                         at_original: bool = False,
                         floor: float | None = None) -> dict[str, Any] | None:
    # by_colors skips SSIM for container-corrected targets, conversion resizes
    # and re-palettes anyway. GIF derives colors from source, so width first.
    # BMP/VIPS resize the intermediate, so colors first.
    if not candidates:
        return None
    if by_colors:
        if width_first:
            return max(candidates, key=lambda c: (
                search._probe_width(c["path"]), c.get("colors", 0),
                c.get("size", 0)))
        return max(candidates, key=lambda c: (
            c.get("colors", 0), search._probe_width(c["path"]), c.get("size", 0)
        ))
    if at_original or not ssim:
        ref_for_ssim = reference
    else:
        ref_for_ssim = _chained_reference(reference, candidates, registry)
    scored: list[tuple[Any, dict[str, Any]]] = [
        (_ssim_vs_reference(ref_for_ssim, c["path"], registry,
                            at_original=at_original) if ssim else None, c)
        for c in candidates
    ]

    if floor is not None:
        passing = [(s, c) for s, c in scored if s is not None and s >= floor]
        if passing:
            # smallest file meeting the floor, ties prefer resolution, colors
            return min(passing, key=lambda item: (
                item[1].get("size", 0),
                -search._probe_width(item[1]["path"]),
                -item[1].get("colors", 0),
            ))[1]
        return max(scored, key=lambda item: (item[0] if item[0] is not None
                                             else -(2 ** 30),))[1]

    def sort_key(item: tuple[Any, dict[str, Any]]) -> tuple[Any, ...]:
        score, c = item
        bucket = round(score / eps) if score is not None else -(2 ** 30)
        width = search._probe_width(c["path"])
        return (bucket, width, c.get("size", 0), c.get("colors", 0))

    scored.sort(key=sort_key, reverse=True)
    return scored[0][1]


def _animated_ssim_vs_reference(reference: Path, candidate: Path,
                                registry: list[Path],
                                at_original: bool = False) -> float | None:
    # coalesce first, ffmpeg's GIF demuxer misaligns delta frames and reports
    # ~0.07 for identical animations. frame counts must match.
    try:
        if at_original:
            ref_w = converters.get_image_width(reference)
            ref_h = converters.get_image_height(reference)
        else:
            ref_w = search._probe_width(candidate)
            ref_h = converters.get_image_height(candidate)
        if ref_w <= 0 or ref_h <= 0:
            return None
        if at_original:
            # upscale candidate to reference resolution, reference keeps native
            # frames. APNG is the lossless intermediate.
            cand_at_ref = search._new_probe(".apng", registry)
            converters.gif_resize(candidate, cand_at_ref, f"{ref_w}x{ref_h}!")
            pair = (reference, cand_at_ref)
        else:
            # downscale reference to candidate size, or ffmpeg errors and
            # lower-res candidates get no score.
            ref_at_cand = search._new_probe(".apng", registry)
            converters.gif_resize(reference, ref_at_cand, f"{ref_w}x{ref_h}!")
            pair = (ref_at_cand, candidate)

        ref_seq = search._new_probe("_%03d.png", registry)
        cand_seq = search._new_probe("_%03d.png", registry)
        for animation, seq in ((pair[0], ref_seq), (pair[1], cand_seq)):
            result = converters._run_magick(
                ["magick", converters._magick_spec(animation), "-coalesce", str(seq)])
            if result.returncode != 0:
                return None
            # magick fills in the sequence, register frames for cleanup
            pattern = seq.name.replace("%03d", "*")
            registry.extend(seq.parent.glob(pattern))
        result = utils.run_capture([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "info",
            "-start_number", "0", "-i", str(ref_seq),
            "-start_number", "0", "-i", str(cand_seq),
            "-lavfi", "ssim", "-f", "null", "-",
        ])
        text = f"{result.stderr or ''} {result.stdout or ''}"
        m = re.search(r"All:([0-9.]+)", text)
        return float(m.group(1)) if m else None
    except (RuntimeError, OSError):
        return None


def _animated_chained_reference(reference: Path, candidates: list[dict[str, Any]],
                                registry: list[Path]) -> Path:
    max_width = max((search._probe_width(c["path"]) for c in candidates), default=0)
    if max_width <= 0:
        return reference
    try:
        ref_w = converters.get_image_width(reference)
        ref_h = converters.get_image_height(reference)
    except RuntimeError:
        return reference
    if ref_w <= 0:
        return reference
    ref_max = search._new_probe(".apng", registry)
    ref_h_at_max = max(1, round(max_width * ref_h / ref_w))
    try:
        converters.gif_resize(reference, ref_max, f"{max_width}x{ref_h_at_max}!")
    except RuntimeError:
        return reference
    return ref_max


def _rank_animated_candidates(candidates: list[dict[str, Any]], reference: Path,
                              registry: list[Path], ssim: bool = True,
                              eps: float = _ANIMATED_SSIM_EPS,
                              at_original: bool = False) -> dict[str, Any] | None:
    # tighter epsilon than stills, per-frame averages are smoother
    if not candidates:
        return None

    def knob(c: dict[str, Any]) -> int:
        return int(c.get("colors", c.get("quality", 0)))

    if not ssim:
        return max(candidates, key=lambda c: (
            knob(c), search._probe_width(c["path"]), c.get("size", 0)
        ))
    if at_original:
        ref_for_ssim = reference
    else:
        ref_for_ssim = _animated_chained_reference(reference, candidates, registry)
    scored = [
        (_animated_ssim_vs_reference(ref_for_ssim, c["path"], registry,
                                     at_original=at_original) if ssim else None, c)
        for c in candidates
    ]

    def sort_key(item: tuple[Any, dict[str, Any]]) -> tuple[Any, ...]:
        score, c = item
        bucket = round(score / eps) if score is not None else -(2 ** 30)
        return (bucket, search._probe_width(c["path"]), c.get("size", 0), knob(c))

    scored.sort(key=sort_key, reverse=True)
    return scored[0][1]
