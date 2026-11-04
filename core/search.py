from __future__ import annotations

import contextlib
import math
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import converters, memgate, utils
from .encoders import CodecEncoder
from .ssim import fidelity_pick

_MAX_PARALLEL = min(4, max(2, os.cpu_count() or 2))

# --slowest allows this many deep levels after enough fits, so knob cliffs
# can't hide the sweet spot at lower resolution
_FRONTIER_DEEP_LEVEL_ALLOWANCE = 2


def _thread_tag() -> str:
    return threading.current_thread().name


def _tagged(message: str) -> str:
    stripped = message.lstrip()
    indent = message[:len(message) - len(stripped)]
    return f"{indent}[{_thread_tag()}] {stripped}"


def _new_probe(suffix: str, registry: list[Path]) -> Path:
    # probes pile up across rounds, can fill a RAM-backed /tmp
    fd, name = tempfile.mkstemp(prefix="clipshrink_probe_", suffix=suffix,
                                dir=memgate.tempdir())
    os.close(fd)
    path = Path(name)
    registry.append(path)
    return path


def _probe_extent(encoder: CodecEncoder, current: Path, probes: list[Path],
                  max_size_bytes: int, subsampling: str,
                  strip_metadata: bool) -> tuple[Path, int] | None:
    if encoder.extent is None:
        return None
    probe = _new_probe(encoder.suffix, probes)
    got = encoder.extent(current, probe, max_size_bytes, subsampling, strip_metadata)
    utils._abort_if_interrupted()
    test_size = converters.get_image_filesize(probe) if got else 0
    if test_size > 0 and test_size <= max_size_bytes:
        return probe, test_size
    return None


def _cleanup_probes(registry: list[Path]) -> None:
    for path in registry:
        with contextlib.suppress(OSError):
            if path.exists():
                path.unlink()


def _probe_width(input_path: Path) -> int:
    try:
        return converters.get_image_width(input_path)
    except RuntimeError:
        return 0


def _probe_worker_cap(current: Path) -> int | None:
    # unreadable files stay ungated
    try:
        width = converters.get_image_width(current)
        height = converters.get_image_height(current)
        return memgate.probe_worker_cap(width, height, _MAX_PARALLEL)
    except Exception:
        return None


def _apply_worker_cap(workers: int, worker_cap: int | None) -> int:
    if worker_cap is not None:
        workers = min(workers, max(worker_cap, 1))
    return max(workers, 1)


def _publish(src: Path, dst: Path) -> None:
    # os.replace fails across filesystems, probe temp may be elsewhere
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copyfile(src, dst)
        with contextlib.suppress(OSError):
            os.unlink(src)


def _parallel_fit_search(
    probe_fn: Callable[[int], tuple[bool, int, Path] | None],
    lo: int, hi: int, k: int | None = None,
    worker_cap: int | None = None,
) -> tuple[tuple[int, int, Path] | None, list[tuple[int, int, Path]]]:
    # monotone fit predicate. None = a higher knob already fit (_ProbeRace)
    if k is None:
        k = _MAX_PARALLEL
    workers = _apply_worker_cap(min(k, max(1, os.cpu_count() or 1)),
                                worker_cap)
    best: tuple[int, int, Path] | None = None
    executed: list[tuple[int, int, Path]] = []
    while lo <= hi and not utils._aborted:
        if hi - lo + 1 <= k:
            qs = list(range(lo, hi + 1))
            done = True
        else:
            step = (hi - lo + 1) // (k + 1)
            qs = [lo + i * step for i in range(1, k + 1)]
            done = False
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="probe") as ex:
            results = list(ex.map(probe_fn, qs))
        utils._abort_if_interrupted()
        round_fits: list[int] = []
        for q, result in zip(qs, results, strict=True):
            if result is None:
                continue
            fits, size, path = result
            executed.append((q, size, path))
            if fits:
                round_fits.append(q)
                if best is None or q > best[0]:
                    best = (q, size, path)
        if done:
            break
        if round_fits:
            top = max(round_fits)
            lo = top + 1
            above = [q for q in qs if q > top]
            if above:
                hi = min(above) - 1
        else:
            hi = min(qs) - 1
    return best, executed


def _coarse_grid_search(
    probe_fn: Callable[[int], tuple[bool, int, Path] | None],
    lo: int, hi: int, step: int = 10,
    worker_cap: int | None = None,
) -> tuple[tuple[int, int, Path] | None, list[tuple[int, int, Path]]]:
    # lands on the grid, not the exact best point, on purpose
    if hi <= lo:
        qs = [lo]
    else:
        qs = list(range(hi, lo, -step))
        if qs[-1] != lo:
            qs.append(lo)
    workers = _apply_worker_cap(min(len(qs), max(1, os.cpu_count() or 1)),
                                worker_cap)
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix="probe") as ex:
        results = list(ex.map(probe_fn, qs))
    utils._abort_if_interrupted()
    best: tuple[int, int, Path] | None = None
    executed: list[tuple[int, int, Path]] = []
    for q, result in zip(qs, results, strict=True):
        if result is None:
            continue
        fits, size, path = result
        executed.append((q, size, path))
        if fits and (best is None or q > best[0]):
            best = (q, size, path)
    return best, executed


@dataclass(frozen=True)
class _RoundResult:
    fitted: bool
    level_best: tuple[Path, int, int] | None      # (probe, size, knob)
    last_probe: tuple[Path, int] | None           # (probe, size)
    probe_ok: bool
    candidates: tuple[dict[str, Any], ...]
    frontier_ceiling_met: bool


@dataclass(frozen=True)
class _SearchStrategy:
    search_round: Callable[[Path, list[Path], int, int], _RoundResult]
    knob_range: tuple[int, int]
    fit_message: str | None = None
    extent_first: bool = False
    extent_after: bool = False
    extent_probe: Callable[[Path, list[Path]], tuple[Path, int] | None] | None = None
    extent_compare: bool = False
    reducer: Callable[[Path, Path, int, str], bool] = converters.resize_pct
    resize_suffix: str = ".png"
    resize_message: Callable[[int, int], str] | None = None
    pixel_art_snap: bool = False
    deep_resize_from_original: bool = False
    deep_step: int = 50
    palette: bool = False
    palette_pass: Callable[[Path, Path, int], bool] | None = None
    entry_size_guard: bool = False
    collect_candidates: bool = False
    break_on_all_probe_failure: bool = False
    prefer_current_if_fits: bool = False
    optimizer_label: str = "Iterative"
    bail_encode: Callable[[Path, Path, int], bool] | None = None
    bail_suffix: str = ".jpg"
    extra_candidates: Callable[[Path, list[dict[str, Any]], list[Path], int, int, int],
                               None] | None = None
    select_winner: Callable[[list[dict[str, Any]], Path, list[Path]],
                            dict[str, Any] | None] | None = None
    winner_message: Callable[[dict[str, Any], int], str] | None = None

    def __post_init__(self) -> None:
        # validate here, deep in the loop these look like a confusing TypeError
        if self.collect_candidates and (self.select_winner is None
                                        or self.winner_message is None):
            raise ValueError(
                "_SearchStrategy: collect_candidates requires "
                "select_winner and winner_message"
            )
        if self.palette and self.palette_pass is None:
            raise ValueError("_SearchStrategy: palette requires palette_pass")
        if (self.extent_first or self.extent_after) and self.extent_probe is None:
            raise ValueError(
                "_SearchStrategy: extent_first/extent_after require extent_probe"
            )
        if self.break_on_all_probe_failure and self.bail_encode is None:
            raise ValueError(
                "_SearchStrategy: break_on_all_probe_failure requires bail_encode"
            )


class _ProbeRace:
    # once a probe fits at k, lower knobs can't win, queued probes skip
    # and running ones cancel. first-fit rounds only, frontier needs the
    # lowest-knob size for the sqrt model.
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: dict[int, utils.CancelToken] = {}
        self.best_fit: int | None = None

    def register(self, knob: int, token: utils.CancelToken) -> None:
        with self._lock:
            self._tokens[knob] = token

    def unregister(self, knob: int) -> None:
        with self._lock:
            self._tokens.pop(knob, None)

    def is_dominated(self, knob: int) -> bool:
        with self._lock:
            return self.best_fit is not None and knob < self.best_fit

    def report_fit(self, knob: int) -> None:
        with self._lock:
            if self.best_fit is None or knob > self.best_fit:
                self.best_fit = knob
            doomed = [token for q, token in self._tokens.items()
                      if q < self.best_fit]
        for token in doomed:
            token.cancel()


def _simple_search_round(
    probe_encode: Callable[[Path, Path, int], bool],
    probe_suffix: str,
    probe_message: str,
    search_step: int | None,
    max_size_bytes: int,
    round_collect_candidates: bool,
    round_frontier_ceiling: int,
    round_candidate_knob: str = "colors",
    probe_skip_message: str = "   Skipping probe at {} (fit at {} already found).",
) -> Callable[[Path, list[Path], int, int], _RoundResult]:
    def search_round(current: Path, probes: list[Path], lo: int, hi: int) -> _RoundResult:
        race = None if round_collect_candidates else _ProbeRace()

        def probe_fn(k: int, _current: Path = current
                     ) -> tuple[bool, int, Path] | None:
            # bound as default arg, workers keep this round's source
            # after current rebinds below
            if race is not None and race.is_dominated(k):
                print(_tagged(probe_skip_message.format(k, race.best_fit)))
                return None
            token = utils.CancelToken()
            if race is not None:
                race.register(k, token)
            try:
                print(_tagged(probe_message.format(k)))
                probe = _new_probe(probe_suffix, probes)
                with utils.cancellable(token):
                    got = probe_encode(_current, probe, k)
                utils._abort_if_interrupted()
                test_size = converters.get_image_filesize(probe) if got else 0
                fits = got and test_size > 0 and test_size <= max_size_bytes
                if fits and race is not None:
                    # report before returning so the next queued probe sees it
                    race.report_fit(k)
                if race is not None and token.cancelled:
                    return None  # killed as dominated
                return (fits, test_size, probe)
            finally:
                if race is not None:
                    race.unregister(k)

        if search_step is not None:
            best, executed = _coarse_grid_search(probe_fn, lo, hi, step=search_step,
                                                 worker_cap=_probe_worker_cap(current))
        else:
            best, executed = _parallel_fit_search(probe_fn, lo, hi,
                                                  worker_cap=_probe_worker_cap(current))
        level_best = (best[2], best[1], best[0]) if best is not None else None
        last_probe: tuple[Path, int] | None = None
        if executed:
            smallest = min(executed, key=lambda e: e[0])
            last_probe = (smallest[2], smallest[1])
        fitted = level_best is not None
        candidates: tuple[dict[str, Any], ...] = ()
        if fitted and round_collect_candidates:
            level = cast(tuple[Path, int, int], level_best)
            candidates = ({"path": level[0], "size": level[1],
                           round_candidate_knob: level[2]},)
        return _RoundResult(
            fitted=fitted, level_best=level_best, last_probe=last_probe,
            probe_ok=any(size > 0 for _, size, _ in executed),
            candidates=candidates,
            frontier_ceiling_met=fitted and level_best is not None
            and level_best[2] >= round_frontier_ceiling,
        )

    return search_round


def _frontier_lossy_round(
    encoder_list: list[CodecEncoder],
    subsampling: str,
    strip_metadata: bool,
    max_size_bytes: int,
    round_frontier_ceiling: int,
    compare_subsampling: bool = True,
) -> Callable[[Path, list[Path], int, int], _RoundResult]:
    # every codec's knob is binary-searched in parallel
    def search_round(current: Path, probes: list[Path], lo: int, hi: int) -> _RoundResult:
        level_fits: list[tuple[Path, int, int, str]] = []
        level_probe_ok = False
        results_by_job: dict[tuple[CodecEncoder, str],
                             tuple[tuple[Path, int, int] | None,
                                   tuple[Path, int] | None, bool]] = {}

        def run_job(job: tuple[CodecEncoder, str], _current: Path = current
                    ) -> tuple[tuple[CodecEncoder, str],
                               tuple[Path, int, int] | None,
                               tuple[Path, int] | None, bool]:
            enc, ss = job
            jlo, jhi = lo, hi
            enc_best: tuple[Path, int, int] | None = None
            job_last: tuple[Path, int] | None = None
            job_probe_ok = False
            while jlo <= jhi and not utils._aborted:
                q = (jlo + jhi) // 2
                print(f"   [{_thread_tag()}/{enc.name}/{ss}] "
                      f"Testing quality {q}%...")
                probe = _new_probe(enc.suffix, probes)
                got = enc.encode(_current, probe, q, subsampling=ss,
                                 strip_metadata=strip_metadata)
                utils._abort_if_interrupted()
                test_size = converters.get_image_filesize(probe) if got else 0
                job_last = (probe, test_size)
                if got and test_size > 0:
                    job_probe_ok = True
                if got and test_size <= max_size_bytes:
                    enc_best = (probe, test_size, q)
                    jlo = q + 1      # fits, try higher quality
                else:
                    jhi = q - 1      # too big, lower quality
            return job, enc_best, job_last, job_probe_ok

        jobs = [
            (enc, ss)
            for enc in encoder_list
            for ss in (["4:4:4", "4:2:0"] if (enc.subsampling_supported and compare_subsampling)
                       else [subsampling])
        ]
        with ThreadPoolExecutor(
                max_workers=_apply_worker_cap(
                    min(len(jobs), _MAX_PARALLEL), _probe_worker_cap(current)),
                thread_name_prefix="probe") as ex:
            for job, enc_best, job_last, job_probe_ok in ex.map(run_job, jobs):
                results_by_job[job] = (enc_best, job_last, job_probe_ok)
                level_probe_ok = level_probe_ok or job_probe_ok

        last_probe: tuple[Path, int] | None = None
        candidates: list[dict[str, Any]] = []
        for enc in encoder_list:
            subsamplings = (["4:4:4", "4:2:0"]
                            if (enc.subsampling_supported and compare_subsampling)
                            else [subsampling])
            best_for_enc: tuple[Path, int, int, str] | None = None
            for ss in subsamplings:
                enc_best, job_last, _ = results_by_job[(enc, ss)]
                if job_last is not None:
                    last_probe = job_last
                if enc_best is not None and (
                    best_for_enc is None
                    or enc_best[2] > best_for_enc[2]
                    or (enc_best[2] == best_for_enc[2]
                        and ss == "4:4:4" and best_for_enc[3] != "4:4:4")
                ):
                    best_for_enc = (enc_best[0], enc_best[1], enc_best[2], ss)
            if best_for_enc is not None:
                level_fits.append(best_for_enc)
                candidates.append({
                    "path": best_for_enc[0], "size": best_for_enc[1],
                    "quality": best_for_enc[2], "subsampling": best_for_enc[3],
                    "encoder": enc.name,
                })
        level_quality = max((f[2] for f in level_fits), default=0)
        return _RoundResult(
            fitted=bool(level_fits), level_best=None, last_probe=last_probe,
            probe_ok=level_probe_ok, candidates=tuple(candidates),
            frontier_ceiling_met=level_quality >= round_frontier_ceiling,
        )

    return search_round


def _art_reducer(pixel_art_mode: bool) -> Callable[[Path, Path, int, str], bool]:
    if pixel_art_mode:
        return lambda src, out, pct, filt: converters.sample_pct(src, out, pct)
    return lambda src, out, pct, filt: converters.resize_pct(src, out, pct, filt)


def _animated_reducer(pixel_art_mode: bool) -> Callable[[Path, Path, int, str], bool]:
    def reduce(src: Path, out: Path, pct: int, filt: str) -> bool:
        return converters.gif_resize(
            src, out, f"{pct}%",
            kernel="Point" if pixel_art_mode else filt,
            nearest=pixel_art_mode,
        )
    return reduce


# Golden-trace tests pin these names, keep them.
def _adaptive_scale_pct(scale_base: int, max_size_bytes: int,
                        strategy: _SearchStrategy) -> int:
    # a saturated sqrt step must not skip the --slowest sweet spot
    scale_pct = (round(math.sqrt(max_size_bytes / scale_base) * 100)
                 if scale_base > 0 else 50)
    if scale_pct >= 100:
        scale_pct = strategy.deep_step if strategy.collect_candidates else 50
    return max(scale_pct, 1)


def _resolution_step(strategy: _SearchStrategy, input_path: Path, current: Path,
                     probes: list[Path], filter: str, scale_pct: int,
                     cum_scale: float, fits_levels: int,
                     extra_fit_levels: int) -> tuple[Path, float, int]:
    # frontier resizes the original at cumulative scale to avoid generational
    # blur. shallow pixel-art levels keep the working copy so snapping applies.
    resized = _new_probe(strategy.resize_suffix, probes)
    cum_scale = cum_scale * scale_pct / 100.0
    if (strategy.deep_resize_from_original
            and (fits_levels >= extra_fit_levels
                 or not strategy.pixel_art_snap)):
        strategy.reducer(input_path, resized, round(cum_scale), filter)
    else:
        if strategy.pixel_art_snap:
            # Snap to 1/n so the pixel grid stays exact under nearest-neighbor.
            f = round(100 / scale_pct)
            f = max(f, 2)
            scale_pct = round(100 / f)
        strategy.reducer(current, resized, scale_pct, filter)
    utils._abort_if_interrupted()
    return resized, cum_scale, scale_pct


@dataclass(frozen=True)
class _ReductionState:
    current: Path
    hi_knob: int
    current_size: int
    last_probe: tuple[Path, int] | None
    best_probe: tuple[Path, int] | None
    bailed: bool


def _post_reduction_bail(
    strategy: _SearchStrategy, current: Path, probes: list[Path],
    knob_lo: int, knob_hi: int, start_knob: int, lo_knob: int,
    old_size: int, scale_pct: int, bail_ratio: float, label: str,
    last_probe: tuple[Path, int] | None, best_probe: tuple[Path, int] | None,
) -> _ReductionState:
    # palette is linear (edge ~ scale), lossy quadratic (pixels ~ scale^2)
    if strategy.palette:
        palette_pass = cast(Callable[[Path, Path, int], bool], strategy.palette_pass)
        hi_knob = max(knob_lo, min(knob_hi, start_knob))
        print(f"   Applying palette reset: Testing max colors {hi_knob}...")
        tmp = _new_probe(strategy.resize_suffix, probes)
        palette_pass(current, tmp, hi_knob)
        utils._abort_if_interrupted()
        current = tmp
        current_size = converters.get_image_filesize(current)
        expected_size = old_size * scale_pct / 100.0
        if current_size > expected_size * bail_ratio:
            print("Warning: Resolution reduction is not shrinking the file. "
                  f"Stopping {label.lower()} optimization.")
            return _ReductionState(current, hi_knob, current_size, last_probe,
                                   best_probe, bailed=True)
        return _ReductionState(current, hi_knob, current_size, last_probe,
                               best_probe, bailed=False)

    bail_encode = cast(Callable[[Path, Path, int], bool], strategy.bail_encode)
    probe = _new_probe(strategy.bail_suffix, probes)
    got = bail_encode(current, probe, lo_knob)
    utils._abort_if_interrupted()
    new_probe_size = converters.get_image_filesize(probe) if got else 0
    last_probe = (probe, new_probe_size)
    if new_probe_size > 0 and (best_probe is None or new_probe_size < best_probe[1]):
        best_probe = last_probe
    expected_size = old_size * (scale_pct / 100.0) ** 2
    hi_knob = max(knob_lo, min(knob_hi, start_knob))   # reset
    if old_size > 0 and new_probe_size > expected_size * bail_ratio:
        print("Warning: Resolution reduction is not shrinking the file. "
              f"Stopping {label.lower()} optimization.")
        return _ReductionState(current, hi_knob, 0, last_probe, best_probe,
                               bailed=True)
    return _ReductionState(current, hi_knob, 0, last_probe, best_probe,
                           bailed=False)


def _restore_best(
    strategy: _SearchStrategy, current: Path, best_probe: tuple[Path, int] | None,
    candidates: list[dict[str, Any]], input_path: Path, output_path: Path,
    max_size_bytes: int, probes: list[Path], lo_knob: int, hi_knob: int,
    iteration: int, label: str, extent_hold: tuple[Path, int] | None = None,
) -> dict[str, int | bool]:
    # --slowest ranks. otherwise restore the smallest probe or current if it fits.
    utils._abort_if_interrupted()
    if extent_hold is not None:
        # only candidate meeting the size limit, ship it
        _publish(extent_hold[0], output_path)
        print(f"   Shipping the one-shot encode ({extent_hold[1]} bytes): "
              "nothing else fit.")
        return {"success": True, "final_size": extent_hold[1],
                "iterations": iteration}
    if strategy.extra_candidates is not None and candidates:
        strategy.extra_candidates(input_path, candidates, probes,
                                  lo_knob, hi_knob, max_size_bytes)
    if (candidates and strategy.select_winner is not None
            and strategy.winner_message is not None):
        winner = cast(dict[str, Any], strategy.select_winner(candidates, input_path, probes))
        _publish(winner["path"], output_path)
        final_size = converters.get_image_filesize(output_path)
        print(strategy.winner_message(winner, final_size))
        return {"success": True, "final_size": final_size, "iterations": iteration}

    if strategy.prefer_current_if_fits and converters.get_image_filesize(current) <= max_size_bytes:
        shutil.copyfile(current, output_path)
    elif best_probe is not None:
        print(f"Warning: {label} optimization did not reach target. Restoring best result.")
        shutil.copyfile(best_probe[0], output_path)
    else:
        shutil.copyfile(current, output_path)
    final_size = converters.get_image_filesize(output_path)
    return {"success": final_size <= max_size_bytes, "final_size": final_size,
            "iterations": iteration}


def _run_search_loop(
    strategy: _SearchStrategy,
    input_path: Path,
    output_path: Path,
    max_size_bytes: int,
    start_knob: int,
    min_knob: int,
    min_width: int = 4,
    bail_ratio: float = 1.3,
    max_iterations: int = 100,
    filter: str = "Lanczos",
    extra_fit_levels: int = 3,
) -> dict[str, int | bool]:
    knob_lo, knob_hi = strategy.knob_range
    hi_knob = max(knob_lo, min(knob_hi, start_knob))
    lo_knob = max(knob_lo, min(hi_knob, min_knob))
    iteration = 0
    fits_levels = 0
    candidates: list[dict[str, Any]] = []
    best_probe: tuple[Path, int] | None = None
    last_probe: tuple[Path, int] | None = None
    current = input_path
    extent_hold: tuple[Path, int] | None = None
    width = _probe_width(current)
    current_size = converters.get_image_filesize(current) if strategy.palette else 0
    cum_scale = 100.0
    probes: list[Path] = []
    label = strategy.optimizer_label

    try:
        while True:
            if iteration >= max_iterations:
                print("Warning: Iteration limit reached. Outputting best result.")
                break
            if utils._aborted:
                break
            if width <= min_width:
                break
            if (strategy.entry_size_guard and current_size <= max_size_bytes
                    and (not strategy.collect_candidates or not candidates)):
                break
            iteration += 1

            # one-shot extent fast path, K-ary only. extent_compare holds it
            # as a fidelity baseline, both are measured at the first fit.
            if iteration == 1 and strategy.extent_first and strategy.extent_probe is not None:
                print("-> Trying one-shot extent encode...")
                extent_result = strategy.extent_probe(current, probes)
                if extent_result is not None:
                    if strategy.extent_compare:
                        print(f"   One-shot encode fits ({extent_result[1]} bytes). "
                              "Keeping it as a fidelity baseline while searching...")
                        extent_hold = extent_result
                    else:
                        print(f"   One-shot encode fits ({extent_result[1]} bytes). Done.")
                        _publish(extent_result[0], output_path)
                        return {"success": True, "final_size": extent_result[1],
                                "iterations": iteration}

            round_ = strategy.search_round(current, probes, lo_knob, hi_knob)
            if round_.last_probe is not None:
                last_probe = round_.last_probe
            if utils._aborted:
                break

            if round_.fitted:
                fits_levels += 1
                if round_.candidates:
                    candidates.extend(round_.candidates)
                if not strategy.collect_candidates:
                    level = cast(tuple[Path, int, int], round_.level_best)
                    if extent_hold is not None:
                        winner = fidelity_pick(extent_hold, (level[0], level[1]),
                                               input_path, probes)
                        if winner[0] == extent_hold[0]:
                            print("   Fidelity comparison: shipping the one-shot encode.")
                        else:
                            print("   Fidelity comparison: shipping the searched candidate.")
                        _publish(winner[0], output_path)
                        if strategy.fit_message is not None:
                            print(strategy.fit_message.format(level[2]))
                        return {"success": True, "final_size": winner[1],
                                "iterations": iteration}
                    _publish(level[0], output_path)
                    if strategy.fit_message is not None:
                        print(strategy.fit_message.format(level[2]))
                    return {"success": True, "final_size": level[1],
                            "iterations": iteration}
                if strategy.fit_message is not None:
                    level = cast(tuple[Path, int, int], round_.level_best)
                    print(strategy.fit_message.format(level[2]))

            # grid rescue: extent fits and the grid found nothing, beats resizing
            if iteration == 1 and strategy.extent_after and strategy.extent_probe is not None:
                extent_result = strategy.extent_probe(current, probes)
                if extent_result is not None:
                    print(f"   One-shot extent encode fits ({extent_result[1]} bytes). Done.")
                    _publish(extent_result[0], output_path)
                    return {"success": True, "final_size": extent_result[1],
                            "iterations": iteration}

            if strategy.break_on_all_probe_failure and not round_.probe_ok:
                print(f"Warning: All encode probes failed. Stopping {label.lower()} optimization.")
                break

            # stop after enough fits or at the ceiling. while quality is low,
            # allow extra levels so cliffs can't hide the sweet spot.
            if (strategy.collect_candidates and fits_levels >= extra_fit_levels
                    and (round_.frontier_ceiling_met
                         or fits_levels >= extra_fit_levels + _FRONTIER_DEEP_LEVEL_ALLOWANCE)):
                break

            if (not round_.fitted and last_probe is not None
                    and (best_probe is None or last_probe[1] < best_probe[1])):
                best_probe = last_probe

            scale_base = last_probe[1] if last_probe is not None and last_probe[1] > 0 else 0
            old_size = current_size if strategy.palette else scale_base
            scale_pct = _adaptive_scale_pct(scale_base, max_size_bytes, strategy)
            if strategy.resize_message is not None:
                print(strategy.resize_message(lo_knob, scale_pct))
            current, cum_scale, scale_pct = _resolution_step(
                strategy, input_path, current, probes, filter, scale_pct,
                cum_scale, fits_levels, extra_fit_levels)
            width = _probe_width(current)

            state = _post_reduction_bail(
                strategy, current, probes, knob_lo, knob_hi, start_knob,
                lo_knob, old_size, scale_pct, bail_ratio, label,
                last_probe, best_probe)
            current = state.current
            hi_knob = state.hi_knob
            current_size = state.current_size
            last_probe = state.last_probe
            best_probe = state.best_probe
            if state.bailed:
                break

        return _restore_best(
            strategy, current, best_probe, candidates, input_path, output_path,
            max_size_bytes, probes, lo_knob, hi_knob, iteration, label,
            extent_hold=extent_hold)
    finally:
        _cleanup_probes(probes)

