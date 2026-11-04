from pathlib import Path

import pytest
from search_sim import MAX, W0, Sim, _spec, pixel_model, run_scenario

from core import search


def lossy_specs():
    return [
        _spec(name="il_fits_high", mode="iter_lossy",
             model=pixel_model(lossy=lambda q, w: round(105000 * q / 100 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_fits_after_resize", mode="iter_lossy",
             model=pixel_model(lossy=lambda q, w: round(157500 * q / 100 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_scale_guard_bail", mode="iter_lossy",
             model=pixel_model(lossy=lambda q, w: MAX + 1),
             max_size=MAX, src_size=4000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_pathological_bail", mode="iter_lossy",
             model=pixel_model(
                 lossy=lambda q, w: round(470588 * q / 100),
                 resized=lambda w, w2, s: round(s * 80 // 100)),
             max_size=MAX, src_size=400000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_min_width", mode="iter_lossy",
             model=pixel_model(lossy=lambda q, w: MAX + 1),
             max_size=MAX, src_size=4000, src_w=8,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "max_iterations": 100, "bail_ratio": 1000.0}),
        _spec(name="il_max_iterations", mode="iter_lossy",
             model=pixel_model(lossy=lambda q, w: MAX + 1),
             max_size=MAX, src_size=4000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "max_iterations": 2, "bail_ratio": 1000.0}),
        _spec(name="il_extent_first_kary", mode="iter_lossy", extent=True,
             model=pixel_model(lossy=lambda q, w: MAX + 50000, extent=lambda w: 90000),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": True, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_extent_rescue_fast", mode="iter_lossy", extent=True,
             model=pixel_model(lossy=lambda q, w: round(200000 * q / 100 * (w / W0) ** 2),
                               extent=lambda w: 90000),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": True, "search_step": 10,
                         "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_extent_overridden_by_grid", mode="iter_lossy", extent=True,
             model=pixel_model(lossy=lambda q, w: round(100000 * q / 100),
                               extent=lambda w: 90000),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": True, "search_step": 10,
                         "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_grid_fit", mode="iter_lossy",
             model=pixel_model(lossy=lambda q, w: round(105000 * q / 100)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "search_step": 3,
                         "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_extent_unavailable", mode="iter_lossy", extent=False,
             model=pixel_model(lossy=lambda q, w: round(105000 * q / 100 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": True, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="il_all_probes_fail", mode="iter_lossy_fail",
             model=pixel_model(),
             max_size=1000, src_size=5000, src_w=W0,
             params={"start_quality": 100, "min_quality": 85,
                         "use_extent_fast_path": False, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="sl_frontier_ceiling", mode="slowest_lossy",
             encoders=[("jpeg", False)],
             model=pixel_model(lossy=lambda q, w: round(95000 * q / 100 * (w / W0) ** 2)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"start_quality": 100, "min_quality": 30,
                         "max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sl_frontier_below_ceiling_deep", mode="slowest_lossy",
             encoders=[("jpeg", False)],
             model=pixel_model(
                 lossy=lambda q, w: round(95000 * q / 100 * (w / W0) ** 2)
                 if q <= 50 else MAX * 2),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"start_quality": 100, "min_quality": 30,
                         "max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sl_subsampling_444", mode="slowest_lossy",
             encoders=[("jpeg", True), ("webp", False)],
             model=pixel_model(lossy=lambda q, w: round(95000 * q / 100 * (w / W0) ** 2)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"start_quality": 100, "min_quality": 30,
                         "max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sl_no_fit_bail_restore", mode="slowest_lossy",
             encoders=[("jpeg", False)],
             model=pixel_model(lossy=lambda q, w: MAX * 2),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"start_quality": 100, "min_quality": 30,
                         "max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sl_no_fit_max_iterations", mode="slowest_lossy",
             encoders=[("jpeg", False)],
             model=pixel_model(lossy=lambda q, w: MAX * 2),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"start_quality": 100, "min_quality": 30,
                         "max_iterations": 2, "bail_ratio": 1000.0, "extra_fit_levels": 2}),
    ]


def palette_specs():
    return [
        _spec(name="ia_fits_high", mode="iter_art",
             model=pixel_model(palette=lambda c, w: round(128000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="ia_fits_after_resize", mode="iter_art",
             model=pixel_model(palette=lambda c, w: round(250000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="ia_pixel_art_snap", mode="iter_art", pixel_art=True,
             model=pixel_model(palette=lambda c, w: round(250000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="ia_pathological_bail", mode="iter_art",
             model=pixel_model(
                 palette=lambda c, w: round(300000 * c / 256),
                 resized=lambda w, w2, s: round(s * 80 // 100)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="ia_entry_guard_fits", mode="iter_art",
             model=pixel_model(palette=lambda c, w: MAX * 2),
             max_size=MAX, src_size=50000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="ia_current_fits_restore", mode="iter_art",
             model=pixel_model(palette=lambda c, w: round(200100 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="ia_all_probes_fail", mode="iter_art", palette_fail=True,
             model=pixel_model(),
             max_size=MAX, src_size=50000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="ia_grid_fit", mode="iter_art",
             model=pixel_model(palette=lambda c, w: round(105000 * c / 256)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"search_step": 32, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="sa_frontier_ceiling", mode="slowest_art",
             model=pixel_model(palette=lambda c, w: round(98000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sa_frontier_below_ceiling", mode="slowest_art",
             model=pixel_model(
                 palette=lambda c, w: round(98000 * c / 256 * (w / W0) ** 2)
                 if c <= 128 else MAX * 2),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sa_container_corrected_ceiling_fit", mode="slowest_art",
             model=pixel_model(palette=lambda c, w: round(98000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2,
                         "container_corrected": True}),
        _spec(name="sa_no_fit_restore", mode="slowest_art",
             model=pixel_model(palette=lambda c, w: MAX * 2,
                               resized=lambda w, w2, s: s),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 3, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sa_frontier_clamp_deep_step", mode="slowest_art",
             model=pixel_model(palette=lambda c, w: round(250000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="an_fits_high", mode="iter_anim",
             model=pixel_model(palette=lambda c, w: round(128000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="an_pixel_art_snap", mode="iter_anim", pixel_art=True,
             model=pixel_model(palette=lambda c, w: round(250000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="an_grid_fit", mode="iter_anim",
             model=pixel_model(palette=lambda c, w: round(105000 * c / 256)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"search_step": 32, "max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="an_resizes_all_frames_fits", mode="iter_anim",
             model=pixel_model(palette=lambda c, w: round(160000 * c / 256)),
             max_size=MAX, src_size=150000, src_w=W0,
             params={"max_iterations": 100, "bail_ratio": 1.3}),
        _spec(name="sa2_frontier_ceiling", mode="slowest_anim",
             model=pixel_model(palette=lambda c, w: round(98000 * c / 256 * (w / W0) ** 2)),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 200, "bail_ratio": 1.3, "extra_fit_levels": 2}),
        _spec(name="sa2_no_fit_restore", mode="slowest_anim",
             model=pixel_model(palette=lambda c, w: MAX * 2,
                               resized=lambda w, w2, s: s),
             max_size=MAX, src_size=5000, src_w=W0,
             params={"max_iterations": 3, "bail_ratio": 1.3, "extra_fit_levels": 2}),
    ]


def all_specs():
    return lossy_specs() + palette_specs()


EXPECTED = {'il_fits_high': {'result': {'success': True, 'final_size': 99750, 'iterations': 1}, 'flow': [['replace', 'F1000x99750', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 88, 92400, True], ['encode', 'fake', '4:2:0', 91, 95550, True], ['encode', 'fake', '4:2:0', 94, 98700, True], ['encode', 'fake', '4:2:0', 95, 99750, True], ['encode', 'fake', '4:2:0', 96, 100800, False], ['encode', 'fake', '4:2:0', 97, 101850, False]], 'stdout': '   Fits at quality 95%.\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 95%...\n   [probe_*] Testing quality 96%...\n   [probe_*] Testing quality 97%...'}, 'il_fits_after_resize': {'result': {'success': True, 'final_size': 99014, 'iterations': 2}, 'flow': [['resize', 'resize_pct', 86, 860, 110940, 'SRC'], ['encode', 'fake', '4:2:0', 85, 99014, True], ['replace', 'F860x99014', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 85, 99014, True], ['encode', 'fake', '4:2:0', 85, 133875, False], ['encode', 'fake', '4:2:0', 86, 100179, False], ['encode', 'fake', '4:2:0', 86, 135450, False], ['encode', 'fake', '4:2:0', 87, 101344, False], ['encode', 'fake', '4:2:0', 87, 137025, False], ['encode', 'fake', '4:2:0', 88, 102509, False], ['encode', 'fake', '4:2:0', 88, 138600, False], ['encode', 'fake', '4:2:0', 91, 106003, False], ['encode', 'fake', '4:2:0', 91, 143325, False], ['encode', 'fake', '4:2:0', 94, 109498, False], ['encode', 'fake', '4:2:0', 94, 148050, False], ['encode', 'fake', '4:2:0', 97, 112992, False], ['encode', 'fake', '4:2:0', 97, 152775, False]], 'stdout': '   Nothing fits down to 85%. Resizing to 86% and resetting quality...\n   Fits at quality 85%.\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...\n   [probe_*] Testing quality 97%...'}, 'il_scale_guard_bail': {'result': {'success': False, 'final_size': 100001, 'iterations': 1}, 'flow': [['resize', 'resize_pct', 50, 500, 1000, 'SRC'], ['encode', 'fake', '4:2:0', 85, 100001, False], ['copy', 'F1000x100001', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 85, 100001, False], ['encode', 'fake', '4:2:0', 86, 100001, False], ['encode', 'fake', '4:2:0', 87, 100001, False], ['encode', 'fake', '4:2:0', 88, 100001, False], ['encode', 'fake', '4:2:0', 91, 100001, False], ['encode', 'fake', '4:2:0', 94, 100001, False], ['encode', 'fake', '4:2:0', 97, 100001, False]], 'stdout': '   Nothing fits down to 85%. Resizing to 50% and resetting quality...\nWarning: Resolution reduction is not shrinking the file. Stopping iterative optimization.\nWarning: Iterative optimization did not reach target. Restoring best result.\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...'}, 'il_pathological_bail': {'result': {'success': False, 'final_size': 400000, 'iterations': 1}, 'flow': [['resize', 'resize_pct', 50, 500, 320000, 'SRC'], ['encode', 'fake', '4:2:0', 85, 400000, False], ['copy', 'F1000x400000', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 85, 400000, False], ['encode', 'fake', '4:2:0', 86, 404706, False], ['encode', 'fake', '4:2:0', 87, 409412, False], ['encode', 'fake', '4:2:0', 88, 414117, False], ['encode', 'fake', '4:2:0', 91, 428235, False], ['encode', 'fake', '4:2:0', 94, 442353, False], ['encode', 'fake', '4:2:0', 97, 456470, False]], 'stdout': '   Nothing fits down to 85%. Resizing to 50% and resetting quality...\nWarning: Resolution reduction is not shrinking the file. Stopping iterative optimization.\nWarning: Iterative optimization did not reach target. Restoring best result.\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...'}, 'il_min_width': {'result': {'success': False, 'final_size': 100001, 'iterations': 1}, 'flow': [['resize', 'resize_pct', 50, 4, 1000, 'SRC'], ['encode', 'fake', '4:2:0', 85, 100001, False], ['copy', 'F8x100001', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 85, 100001, False], ['encode', 'fake', '4:2:0', 86, 100001, False], ['encode', 'fake', '4:2:0', 87, 100001, False], ['encode', 'fake', '4:2:0', 88, 100001, False], ['encode', 'fake', '4:2:0', 91, 100001, False], ['encode', 'fake', '4:2:0', 94, 100001, False], ['encode', 'fake', '4:2:0', 97, 100001, False]], 'stdout': '   Nothing fits down to 85%. Resizing to 50% and resetting quality...\nWarning: Iterative optimization did not reach target. Restoring best result.\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...'}, 'il_max_iterations': {'result': {'success': False, 'final_size': 100001, 'iterations': 2}, 'flow': [['resize', 'resize_pct', 50, 500, 1000, 'SRC'], ['encode', 'fake', '4:2:0', 85, 100001, False], ['resize', 'resize_pct', 50, 250, 250, 'F500x1000'], ['encode', 'fake', '4:2:0', 85, 100001, False], ['copy', 'F1000x100001', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 85, 100001, False], ['encode', 'fake', '4:2:0', 85, 100001, False], ['encode', 'fake', '4:2:0', 86, 100001, False], ['encode', 'fake', '4:2:0', 86, 100001, False], ['encode', 'fake', '4:2:0', 87, 100001, False], ['encode', 'fake', '4:2:0', 87, 100001, False], ['encode', 'fake', '4:2:0', 88, 100001, False], ['encode', 'fake', '4:2:0', 88, 100001, False], ['encode', 'fake', '4:2:0', 91, 100001, False], ['encode', 'fake', '4:2:0', 91, 100001, False], ['encode', 'fake', '4:2:0', 94, 100001, False], ['encode', 'fake', '4:2:0', 94, 100001, False], ['encode', 'fake', '4:2:0', 97, 100001, False], ['encode', 'fake', '4:2:0', 97, 100001, False]], 'stdout': '   Nothing fits down to 85%. Resizing to 50% and resetting quality...\n   Nothing fits down to 85%. Resizing to 50% and resetting quality...\nWarning: Iteration limit reached. Outputting best result.\nWarning: Iterative optimization did not reach target. Restoring best result.\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...\n   [probe_*] Testing quality 97%...'}, 'il_extent_first_kary': {'result': {'success': True, 'final_size': 90000, 'iterations': 1}, 'flow': [['extent', 90000], ['resize', 'resize_pct', 82, 820, 100860, 'SRC'], ['encode', 'fake', '4:2:0', 85, 150000, False], ['replace', 'F1000x90000', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 85, 150000, False], ['encode', 'fake', '4:2:0', 86, 150000, False], ['encode', 'fake', '4:2:0', 87, 150000, False], ['encode', 'fake', '4:2:0', 88, 150000, False], ['encode', 'fake', '4:2:0', 91, 150000, False], ['encode', 'fake', '4:2:0', 94, 150000, False], ['encode', 'fake', '4:2:0', 97, 150000, False]], 'stdout': '-> Trying one-shot extent encode...\n   One-shot encode fits (90000 bytes). Keeping it as a fidelity baseline while searching...\n   Nothing fits down to 85%. Resizing to 82% and resetting quality...\nWarning: Resolution reduction is not shrinking the file. Stopping iterative optimization.\n   Shipping the one-shot encode (90000 bytes): nothing else fit.\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...'}, 'il_extent_rescue_fast': {'result': {'success': True, 'final_size': 90000, 'iterations': 1}, 'flow': [['extent', 90000], ['replace', 'F1000x90000', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 85, 170000, False], ['encode', 'fake', '4:2:0', 90, 180000, False], ['encode', 'fake', '4:2:0', 100, 200000, False]], 'stdout': '   One-shot extent encode fits (90000 bytes). Done.\n   [probe_*] Testing quality 100%...\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 90%...'}, 'il_extent_overridden_by_grid': {'result': {'success': True, 'final_size': 100000, 'iterations': 1}, 'flow': [['replace', 'F1000x100000', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 100, 100000, True]], 'stdout': '   Fits at quality 100%.\n   [probe_*] Skipping quality 85% (fit at 100% already found).\n   [probe_*] Skipping quality 90% (fit at 100% already found).\n   [probe_*] Testing quality 100%...'}, 'il_grid_fit': {'result': {'success': True, 'final_size': 98700, 'iterations': 1}, 'flow': [['replace', 'F1000x98700', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 94, 98700, True], ['encode', 'fake', '4:2:0', 97, 101850, False], ['encode', 'fake', '4:2:0', 100, 105000, False]], 'stdout': '   Fits at quality 94%.\n   [probe_*] Skipping quality 85% (fit at 94% already found).\n   [probe_*] Skipping quality 88% (fit at 94% already found).\n   [probe_*] Skipping quality 91% (fit at 94% already found).\n   [probe_*] Testing quality 100%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...'}, 'il_extent_unavailable': {'result': {'success': True, 'final_size': 99750, 'iterations': 1}, 'flow': [['replace', 'F1000x99750', 'OUT']], 'probes': [['encode', 'fake', '4:2:0', 88, 92400, True], ['encode', 'fake', '4:2:0', 91, 95550, True], ['encode', 'fake', '4:2:0', 94, 98700, True], ['encode', 'fake', '4:2:0', 95, 99750, True], ['encode', 'fake', '4:2:0', 96, 100800, False], ['encode', 'fake', '4:2:0', 97, 101850, False]], 'stdout': '   Fits at quality 95%.\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 95%...\n   [probe_*] Testing quality 96%...\n   [probe_*] Testing quality 97%...'}, 'il_all_probes_fail': {'result': {'success': False, 'final_size': 5000, 'iterations': 1}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [['encode', 'dead', '4:2:0', 85, 0, False], ['encode', 'dead', '4:2:0', 86, 0, False], ['encode', 'dead', '4:2:0', 87, 0, False], ['encode', 'dead', '4:2:0', 88, 0, False], ['encode', 'dead', '4:2:0', 91, 0, False], ['encode', 'dead', '4:2:0', 94, 0, False], ['encode', 'dead', '4:2:0', 97, 0, False]], 'stdout': 'Warning: All encode probes failed. Stopping iterative optimization.\n   [probe_*] Testing quality 85%...\n   [probe_*] Testing quality 86%...\n   [probe_*] Testing quality 87%...\n   [probe_*] Testing quality 88%...\n   [probe_*] Testing quality 91%...\n   [probe_*] Testing quality 94%...\n   [probe_*] Testing quality 97%...'}, 'sl_frontier_ceiling': {'result': {'success': True, 'final_size': 53438, 'iterations': 2}, 'flow': [['resize', 'resize_pct', 75, 750, 2812, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 16031, True], ['replace', 'F750x53438', 'OUT']], 'probes': [['encode', 'jpeg', '4:2:0', 65, 34734, True], ['encode', 'jpeg', '4:2:0', 65, 61750, True], ['encode', 'jpeg', '4:2:0', 83, 44353, True], ['encode', 'jpeg', '4:2:0', 83, 78850, True], ['encode', 'jpeg', '4:2:0', 92, 49162, True], ['encode', 'jpeg', '4:2:0', 92, 87400, True], ['encode', 'jpeg', '4:2:0', 96, 51300, True], ['encode', 'jpeg', '4:2:0', 96, 91200, True], ['encode', 'jpeg', '4:2:0', 98, 52369, True], ['encode', 'jpeg', '4:2:0', 98, 93100, True], ['encode', 'jpeg', '4:2:0', 99, 52903, True], ['encode', 'jpeg', '4:2:0', 99, 94050, True], ['encode', 'jpeg', '4:2:0', 100, 53438, True], ['encode', 'jpeg', '4:2:0', 100, 95000, True]], 'stdout': '   Exploring resolution: resizing to 75%...\n   Best: JPEG quality 100 (4:2:0) at 53438 bytes.\n   [probe_*/jpeg/4:2:0] Testing quality 100%...\n   [probe_*/jpeg/4:2:0] Testing quality 100%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 83%...\n   [probe_*/jpeg/4:2:0] Testing quality 83%...\n   [probe_*/jpeg/4:2:0] Testing quality 92%...\n   [probe_*/jpeg/4:2:0] Testing quality 92%...\n   [probe_*/jpeg/4:2:0] Testing quality 96%...\n   [probe_*/jpeg/4:2:0] Testing quality 96%...\n   [probe_*/jpeg/4:2:0] Testing quality 98%...\n   [probe_*/jpeg/4:2:0] Testing quality 98%...\n   [probe_*/jpeg/4:2:0] Testing quality 99%...\n   [probe_*/jpeg/4:2:0] Testing quality 99%...'}, 'sl_frontier_below_ceiling_deep': {'result': {'success': True, 'final_size': 26719, 'iterations': 4}, 'flow': [['resize', 'resize_pct', 75, 750, 2812, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 16031, True], ['resize', 'resize_pct', 56, 560, 1568, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 8938, True], ['resize', 'resize_pct', 42, 420, 882, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 5027, True], ['replace', 'F750x26719', 'OUT']], 'probes': [['encode', 'jpeg', '4:2:0', 47, 7876, True], ['encode', 'jpeg', '4:2:0', 47, 14002, True], ['encode', 'jpeg', '4:2:0', 47, 25116, True], ['encode', 'jpeg', '4:2:0', 47, 44650, True], ['encode', 'jpeg', '4:2:0', 49, 8211, True], ['encode', 'jpeg', '4:2:0', 49, 14598, True], ['encode', 'jpeg', '4:2:0', 49, 26184, True], ['encode', 'jpeg', '4:2:0', 49, 46550, True], ['encode', 'jpeg', '4:2:0', 50, 8379, True], ['encode', 'jpeg', '4:2:0', 50, 14896, True], ['encode', 'jpeg', '4:2:0', 50, 26719, True], ['encode', 'jpeg', '4:2:0', 50, 47500, True], ['encode', 'jpeg', '4:2:0', 51, 200000, False], ['encode', 'jpeg', '4:2:0', 51, 200000, False], ['encode', 'jpeg', '4:2:0', 51, 200000, False], ['encode', 'jpeg', '4:2:0', 51, 200000, False], ['encode', 'jpeg', '4:2:0', 56, 200000, False], ['encode', 'jpeg', '4:2:0', 56, 200000, False], ['encode', 'jpeg', '4:2:0', 56, 200000, False], ['encode', 'jpeg', '4:2:0', 56, 200000, False], ['encode', 'jpeg', '4:2:0', 65, 200000, False], ['encode', 'jpeg', '4:2:0', 65, 200000, False], ['encode', 'jpeg', '4:2:0', 65, 200000, False], ['encode', 'jpeg', '4:2:0', 65, 200000, False]], 'stdout': '   Exploring resolution: resizing to 75%...\n   Exploring resolution: resizing to 75%...\n   Exploring resolution: resizing to 75%...\n   Best: JPEG quality 50 (4:2:0) at 26719 bytes.\n   [probe_*/jpeg/4:2:0] Testing quality 47%...\n   [probe_*/jpeg/4:2:0] Testing quality 47%...\n   [probe_*/jpeg/4:2:0] Testing quality 47%...\n   [probe_*/jpeg/4:2:0] Testing quality 47%...\n   [probe_*/jpeg/4:2:0] Testing quality 49%...\n   [probe_*/jpeg/4:2:0] Testing quality 49%...\n   [probe_*/jpeg/4:2:0] Testing quality 49%...\n   [probe_*/jpeg/4:2:0] Testing quality 49%...\n   [probe_*/jpeg/4:2:0] Testing quality 50%...\n   [probe_*/jpeg/4:2:0] Testing quality 50%...\n   [probe_*/jpeg/4:2:0] Testing quality 50%...\n   [probe_*/jpeg/4:2:0] Testing quality 50%...\n   [probe_*/jpeg/4:2:0] Testing quality 51%...\n   [probe_*/jpeg/4:2:0] Testing quality 51%...\n   [probe_*/jpeg/4:2:0] Testing quality 51%...\n   [probe_*/jpeg/4:2:0] Testing quality 51%...\n   [probe_*/jpeg/4:2:0] Testing quality 56%...\n   [probe_*/jpeg/4:2:0] Testing quality 56%...\n   [probe_*/jpeg/4:2:0] Testing quality 56%...\n   [probe_*/jpeg/4:2:0] Testing quality 56%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...'}, 'sl_subsampling_444': {'result': {'success': True, 'final_size': 53438, 'iterations': 2}, 'flow': [['resize', 'resize_pct', 75, 750, 2812, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 16031, True], ['replace', 'F750x53438', 'OUT']], 'probes': [['encode', 'jpeg', '4:2:0', 65, 34734, True], ['encode', 'jpeg', '4:2:0', 65, 61750, True], ['encode', 'jpeg', '4:2:0', 83, 44353, True], ['encode', 'jpeg', '4:2:0', 83, 78850, True], ['encode', 'jpeg', '4:2:0', 92, 49162, True], ['encode', 'jpeg', '4:2:0', 92, 87400, True], ['encode', 'jpeg', '4:2:0', 96, 51300, True], ['encode', 'jpeg', '4:2:0', 96, 91200, True], ['encode', 'jpeg', '4:2:0', 98, 52369, True], ['encode', 'jpeg', '4:2:0', 98, 93100, True], ['encode', 'jpeg', '4:2:0', 99, 52903, True], ['encode', 'jpeg', '4:2:0', 99, 94050, True], ['encode', 'jpeg', '4:2:0', 100, 53438, True], ['encode', 'jpeg', '4:2:0', 100, 95000, True], ['encode', 'jpeg', '4:4:4', 65, 34734, True], ['encode', 'jpeg', '4:4:4', 65, 61750, True], ['encode', 'jpeg', '4:4:4', 83, 44353, True], ['encode', 'jpeg', '4:4:4', 83, 78850, True], ['encode', 'jpeg', '4:4:4', 92, 49162, True], ['encode', 'jpeg', '4:4:4', 92, 87400, True], ['encode', 'jpeg', '4:4:4', 96, 51300, True], ['encode', 'jpeg', '4:4:4', 96, 91200, True], ['encode', 'jpeg', '4:4:4', 98, 52369, True], ['encode', 'jpeg', '4:4:4', 98, 93100, True], ['encode', 'jpeg', '4:4:4', 99, 52903, True], ['encode', 'jpeg', '4:4:4', 99, 94050, True], ['encode', 'jpeg', '4:4:4', 100, 53438, True], ['encode', 'jpeg', '4:4:4', 100, 95000, True], ['encode', 'webp', '4:2:0', 65, 34734, True], ['encode', 'webp', '4:2:0', 65, 61750, True], ['encode', 'webp', '4:2:0', 83, 44353, True], ['encode', 'webp', '4:2:0', 83, 78850, True], ['encode', 'webp', '4:2:0', 92, 49162, True], ['encode', 'webp', '4:2:0', 92, 87400, True], ['encode', 'webp', '4:2:0', 96, 51300, True], ['encode', 'webp', '4:2:0', 96, 91200, True], ['encode', 'webp', '4:2:0', 98, 52369, True], ['encode', 'webp', '4:2:0', 98, 93100, True], ['encode', 'webp', '4:2:0', 99, 52903, True], ['encode', 'webp', '4:2:0', 99, 94050, True], ['encode', 'webp', '4:2:0', 100, 53438, True], ['encode', 'webp', '4:2:0', 100, 95000, True]], 'stdout': '   Exploring resolution: resizing to 75%...\n   Best: JPEG quality 100 (4:4:4) at 53438 bytes.\n   [probe_*/jpeg/4:2:0] Testing quality 100%...\n   [probe_*/jpeg/4:2:0] Testing quality 100%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 83%...\n   [probe_*/jpeg/4:2:0] Testing quality 83%...\n   [probe_*/jpeg/4:2:0] Testing quality 92%...\n   [probe_*/jpeg/4:2:0] Testing quality 92%...\n   [probe_*/jpeg/4:2:0] Testing quality 96%...\n   [probe_*/jpeg/4:2:0] Testing quality 96%...\n   [probe_*/jpeg/4:2:0] Testing quality 98%...\n   [probe_*/jpeg/4:2:0] Testing quality 98%...\n   [probe_*/jpeg/4:2:0] Testing quality 99%...\n   [probe_*/jpeg/4:2:0] Testing quality 99%...\n   [probe_*/jpeg/4:4:4] Testing quality 100%...\n   [probe_*/jpeg/4:4:4] Testing quality 100%...\n   [probe_*/jpeg/4:4:4] Testing quality 65%...\n   [probe_*/jpeg/4:4:4] Testing quality 65%...\n   [probe_*/jpeg/4:4:4] Testing quality 83%...\n   [probe_*/jpeg/4:4:4] Testing quality 83%...\n   [probe_*/jpeg/4:4:4] Testing quality 92%...\n   [probe_*/jpeg/4:4:4] Testing quality 92%...\n   [probe_*/jpeg/4:4:4] Testing quality 96%...\n   [probe_*/jpeg/4:4:4] Testing quality 96%...\n   [probe_*/jpeg/4:4:4] Testing quality 98%...\n   [probe_*/jpeg/4:4:4] Testing quality 98%...\n   [probe_*/jpeg/4:4:4] Testing quality 99%...\n   [probe_*/jpeg/4:4:4] Testing quality 99%...\n   [probe_*/webp/4:2:0] Testing quality 100%...\n   [probe_*/webp/4:2:0] Testing quality 100%...\n   [probe_*/webp/4:2:0] Testing quality 65%...\n   [probe_*/webp/4:2:0] Testing quality 65%...\n   [probe_*/webp/4:2:0] Testing quality 83%...\n   [probe_*/webp/4:2:0] Testing quality 83%...\n   [probe_*/webp/4:2:0] Testing quality 92%...\n   [probe_*/webp/4:2:0] Testing quality 92%...\n   [probe_*/webp/4:2:0] Testing quality 96%...\n   [probe_*/webp/4:2:0] Testing quality 96%...\n   [probe_*/webp/4:2:0] Testing quality 98%...\n   [probe_*/webp/4:2:0] Testing quality 98%...\n   [probe_*/webp/4:2:0] Testing quality 99%...\n   [probe_*/webp/4:2:0] Testing quality 99%...'}, 'sl_no_fit_bail_restore': {'result': {'success': False, 'final_size': 200000, 'iterations': 1}, 'flow': [['resize', 'resize_pct', 71, 710, 2520, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 200000, False], ['copy', 'F1000x200000', 'OUT']], 'probes': [['encode', 'jpeg', '4:2:0', 30, 200000, False], ['encode', 'jpeg', '4:2:0', 31, 200000, False], ['encode', 'jpeg', '4:2:0', 33, 200000, False], ['encode', 'jpeg', '4:2:0', 38, 200000, False], ['encode', 'jpeg', '4:2:0', 47, 200000, False], ['encode', 'jpeg', '4:2:0', 65, 200000, False]], 'stdout': '   Exploring resolution: resizing to 71%...\nWarning: Resolution reduction is not shrinking the file. Stopping best optimization.\nWarning: Best optimization did not reach target. Restoring best result.\n   [probe_*/jpeg/4:2:0] Testing quality 30%...\n   [probe_*/jpeg/4:2:0] Testing quality 31%...\n   [probe_*/jpeg/4:2:0] Testing quality 33%...\n   [probe_*/jpeg/4:2:0] Testing quality 38%...\n   [probe_*/jpeg/4:2:0] Testing quality 47%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...'}, 'sl_no_fit_max_iterations': {'result': {'success': False, 'final_size': 200000, 'iterations': 2}, 'flow': [['resize', 'resize_pct', 71, 710, 2520, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 200000, False], ['resize', 'resize_pct', 50, 500, 1250, 'SRC'], ['encode', 'jpeg', '4:2:0', 30, 200000, False], ['copy', 'F1000x200000', 'OUT']], 'probes': [['encode', 'jpeg', '4:2:0', 30, 200000, False], ['encode', 'jpeg', '4:2:0', 30, 200000, False], ['encode', 'jpeg', '4:2:0', 31, 200000, False], ['encode', 'jpeg', '4:2:0', 31, 200000, False], ['encode', 'jpeg', '4:2:0', 33, 200000, False], ['encode', 'jpeg', '4:2:0', 33, 200000, False], ['encode', 'jpeg', '4:2:0', 38, 200000, False], ['encode', 'jpeg', '4:2:0', 38, 200000, False], ['encode', 'jpeg', '4:2:0', 47, 200000, False], ['encode', 'jpeg', '4:2:0', 47, 200000, False], ['encode', 'jpeg', '4:2:0', 65, 200000, False], ['encode', 'jpeg', '4:2:0', 65, 200000, False]], 'stdout': '   Exploring resolution: resizing to 71%...\n   Exploring resolution: resizing to 71%...\nWarning: Iteration limit reached. Outputting best result.\nWarning: Best optimization did not reach target. Restoring best result.\n   [probe_*/jpeg/4:2:0] Testing quality 30%...\n   [probe_*/jpeg/4:2:0] Testing quality 30%...\n   [probe_*/jpeg/4:2:0] Testing quality 31%...\n   [probe_*/jpeg/4:2:0] Testing quality 31%...\n   [probe_*/jpeg/4:2:0] Testing quality 33%...\n   [probe_*/jpeg/4:2:0] Testing quality 33%...\n   [probe_*/jpeg/4:2:0] Testing quality 38%...\n   [probe_*/jpeg/4:2:0] Testing quality 38%...\n   [probe_*/jpeg/4:2:0] Testing quality 47%...\n   [probe_*/jpeg/4:2:0] Testing quality 47%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...\n   [probe_*/jpeg/4:2:0] Testing quality 65%...'}, 'ia_fits_high': {'result': {'success': True, 'final_size': 100000, 'iterations': 1}, 'flow': [['replace', 'F1000x100000', 'OUT']], 'probes': [['palette', 153, 76500, True], ['palette', 178, 89000, True], ['palette', 183, 91500, True], ['palette', 187, 93500, True], ['palette', 191, 95500, True], ['palette', 195, 97500, True], ['palette', 197, 98500, True], ['palette', 198, 99000, True], ['palette', 199, 99500, True], ['palette', 200, 100000, True], ['palette', 201, 100500, False], ['palette', 202, 101000, False], ['palette', 203, 101500, False], ['palette', 228, 114000, False]], 'stdout': '   Fits at 200 colors.\n   [probe_*] Testing max colors 153...\n   [probe_*] Testing max colors 178...\n   [probe_*] Testing max colors 183...\n   [probe_*] Testing max colors 187...\n   [probe_*] Testing max colors 191...\n   [probe_*] Testing max colors 195...\n   [probe_*] Testing max colors 197...\n   [probe_*] Testing max colors 198...\n   [probe_*] Testing max colors 199...\n   [probe_*] Testing max colors 200...\n   [probe_*] Testing max colors 201...\n   [probe_*] Testing max colors 202...\n   [probe_*] Testing max colors 203...\n   [probe_*] Testing max colors 228...'}, 'ia_fits_after_resize': {'result': {'success': False, 'final_size': 125000, 'iterations': 1}, 'flow': [['resize', 'resize_pct', 89, 890, 118815, 'SRC'], ['palette', 256, 198025, False], ['copy', 'F1000x125000', 'OUT']], 'probes': [['palette', 128, 125000, False], ['palette', 129, 125977, False], ['palette', 130, 126953, False], ['palette', 131, 127930, False], ['palette', 132, 128906, False], ['palette', 133, 129883, False], ['palette', 138, 134766, False], ['palette', 143, 139648, False], ['palette', 148, 144531, False], ['palette', 153, 149414, False], ['palette', 178, 173828, False], ['palette', 203, 198242, False], ['palette', 228, 222656, False]], 'stdout': '   Applying palette reset: Testing max colors 256...\nWarning: Resolution reduction is not shrinking the file. Stopping iterative optimization.\nWarning: Iterative optimization did not reach target. Restoring best result.\n   [probe_*] Testing max colors 128...\n   [probe_*] Testing max colors 129...\n   [probe_*] Testing max colors 130...\n   [probe_*] Testing max colors 131...\n   [probe_*] Testing max colors 132...\n   [probe_*] Testing max colors 133...\n   [probe_*] Testing max colors 138...\n   [probe_*] Testing max colors 143...\n   [probe_*] Testing max colors 148...\n   [probe_*] Testing max colors 153...\n   [probe_*] Testing max colors 178...\n   [probe_*] Testing max colors 203...\n   [probe_*] Testing max colors 228...'}, 'ia_pixel_art_snap': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'ia_pathological_bail': {'result': {'success': False, 'final_size': 150000, 'iterations': 1}, 'flow': [['resize', 'resize_pct', 82, 820, 120000, 'SRC'], ['palette', 256, 300000, False], ['copy', 'F1000x150000', 'OUT']], 'probes': [['palette', 128, 150000, False], ['palette', 129, 151172, False], ['palette', 130, 152344, False], ['palette', 131, 153516, False], ['palette', 132, 154688, False], ['palette', 133, 155859, False], ['palette', 138, 161719, False], ['palette', 143, 167578, False], ['palette', 148, 173438, False], ['palette', 153, 179297, False], ['palette', 178, 208594, False], ['palette', 203, 237891, False], ['palette', 228, 267188, False]], 'stdout': '   Applying palette reset: Testing max colors 256...\nWarning: Resolution reduction is not shrinking the file. Stopping iterative optimization.\nWarning: Iterative optimization did not reach target. Restoring best result.\n   [probe_*] Testing max colors 128...\n   [probe_*] Testing max colors 129...\n   [probe_*] Testing max colors 130...\n   [probe_*] Testing max colors 131...\n   [probe_*] Testing max colors 132...\n   [probe_*] Testing max colors 133...\n   [probe_*] Testing max colors 138...\n   [probe_*] Testing max colors 143...\n   [probe_*] Testing max colors 148...\n   [probe_*] Testing max colors 153...\n   [probe_*] Testing max colors 178...\n   [probe_*] Testing max colors 203...\n   [probe_*] Testing max colors 228...'}, 'ia_entry_guard_fits': {'result': {'success': True, 'final_size': 50000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'ia_current_fits_restore': {'result': {'success': True, 'final_size': 50025, 'iterations': 1}, 'flow': [['resize', 'resize_pct', 50, 500, 37500, 'SRC'], ['palette', 256, 50025, True], ['copy', 'F500x50025', 'OUT']], 'probes': [['palette', 128, 100050, False], ['palette', 129, 100832, False], ['palette', 130, 101613, False], ['palette', 131, 102395, False], ['palette', 132, 103177, False], ['palette', 133, 103958, False], ['palette', 138, 107866, False], ['palette', 143, 111775, False], ['palette', 148, 115683, False], ['palette', 153, 119591, False], ['palette', 178, 139132, False], ['palette', 203, 158673, False], ['palette', 228, 178214, False]], 'stdout': '   Applying palette reset: Testing max colors 256...\n   [probe_*] Testing max colors 128...\n   [probe_*] Testing max colors 129...\n   [probe_*] Testing max colors 130...\n   [probe_*] Testing max colors 131...\n   [probe_*] Testing max colors 132...\n   [probe_*] Testing max colors 133...\n   [probe_*] Testing max colors 138...\n   [probe_*] Testing max colors 143...\n   [probe_*] Testing max colors 148...\n   [probe_*] Testing max colors 153...\n   [probe_*] Testing max colors 178...\n   [probe_*] Testing max colors 203...\n   [probe_*] Testing max colors 228...'}, 'ia_all_probes_fail': {'result': {'success': True, 'final_size': 50000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'ia_grid_fit': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'sa_frontier_ceiling': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'sa_frontier_below_ceiling': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'sa_container_corrected_ceiling_fit': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'sa_no_fit_restore': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''},
'sa_frontier_clamp_deep_step': {'result': {'success': True, 'final_size': 78400, 'iterations': 3}, 'flow': [['resize', 'resize_pct', 75, 750, 84375, 'SRC'], ['palette', 256, 140625, False], ['resize', 'resize_pct', 56, 560, 47040, 'SRC'], ['palette', 256, 78400, True], ['replace', 'F560x78400', 'OUT']], 'probes': [['palette', 53, 16231, True], ['palette', 53, 29114, True], ['palette', 53, 51758, True], ['palette', 64, 62500, True], ['palette', 74, 72266, True], ['palette', 84, 82031, True], ['palette', 94, 91797, True], ['palette', 96, 93750, True], ['palette', 97, 94727, True], ['palette', 98, 95703, True], ['palette', 99, 96680, True], ['palette', 100, 97656, True], ['palette', 101, 98633, True], ['palette', 102, 99609, True], ['palette', 103, 100586, False], ['palette', 104, 31850, True], ['palette', 104, 57129, True], ['palette', 104, 101562, False], ['palette', 155, 47469, True], ['palette', 155, 85144, True], ['palette', 155, 151367, False], ['palette', 166, 91187, True], ['palette', 176, 96680, True], ['palette', 178, 97778, True], ['palette', 179, 98328, True], ['palette', 180, 98877, True], ['palette', 181, 99426, True], ['palette', 182, 99976, True], ['palette', 183, 100525, False], ['palette', 184, 101074, False], ['palette', 185, 101624, False], ['palette', 186, 102173, False], ['palette', 196, 107666, False], ['palette', 206, 63088, True], ['palette', 206, 113159, False], ['palette', 206, 201172, False], ['palette', 217, 66456, True], ['palette', 227, 69519, True], ['palette', 237, 72581, True], ['palette', 247, 75644, True], ['palette', 249, 76256, True], ['palette', 250, 76563, True], ['palette', 251, 76869, True], ['palette', 252, 77175, True], ['palette', 253, 77481, True], ['palette', 254, 77788, True], ['palette', 255, 78094, True], ['palette', 256, 78400, True]], 'stdout': '   Fits at 102 colors.\n   Applying palette reset: Testing max colors 256...\n   Fits at 182 colors.\n   Applying palette reset: Testing max colors 256...\n   Fits at 256 colors.\n   Best: 256 colors at 78400 bytes.\n   [probe_*] Testing max colors 100...\n   [probe_*] Testing max colors 101...\n   [probe_*] Testing max colors 102...\n   [probe_*] Testing max colors 103...\n   [probe_*] Testing max colors 104...\n   [probe_*] Testing max colors 104...\n   [probe_*] Testing max colors 104...\n   [probe_*] Testing max colors 155...\n   [probe_*] Testing max colors 155...\n   [probe_*] Testing max colors 155...\n   [probe_*] Testing max colors 166...\n   [probe_*] Testing max colors 176...\n   [probe_*] Testing max colors 178...\n   [probe_*] Testing max colors 179...\n   [probe_*] Testing max colors 180...\n   [probe_*] Testing max colors 181...\n   [probe_*] Testing max colors 182...\n   [probe_*] Testing max colors 183...\n   [probe_*] Testing max colors 184...\n   [probe_*] Testing max colors 185...\n   [probe_*] Testing max colors 186...\n   [probe_*] Testing max colors 196...\n   [probe_*] Testing max colors 206...\n   [probe_*] Testing max colors 206...\n   [probe_*] Testing max colors 206...\n   [probe_*] Testing max colors 217...\n   [probe_*] Testing max colors 227...\n   [probe_*] Testing max colors 237...\n   [probe_*] Testing max colors 247...\n   [probe_*] Testing max colors 249...\n   [probe_*] Testing max colors 250...\n   [probe_*] Testing max colors 251...\n   [probe_*] Testing max colors 252...\n   [probe_*] Testing max colors 253...\n   [probe_*] Testing max colors 254...\n   [probe_*] Testing max colors 255...\n   [probe_*] Testing max colors 256...\n   [probe_*] Testing max colors 53...\n   [probe_*] Testing max colors 53...\n   [probe_*] Testing max colors 53...\n   [probe_*] Testing max colors 64...\n   [probe_*] Testing max colors 74...\n   [probe_*] Testing max colors 84...\n   [probe_*] Testing max colors 94...\n   [probe_*] Testing max colors 96...\n   [probe_*] Testing max colors 97...\n   [probe_*] Testing max colors 98...\n   [probe_*] Testing max colors 99...'},
'an_fits_high': {'result': {'success': True, 'final_size': 100000, 'iterations': 1}, 'flow': [['replace', 'F1000x100000', 'OUT']], 'probes': [['anim', 153, 76500, True], ['anim', 178, 89000, True], ['anim', 183, 91500, True], ['anim', 187, 93500, True], ['anim', 191, 95500, True], ['anim', 195, 97500, True], ['anim', 197, 98500, True], ['anim', 198, 99000, True], ['anim', 199, 99500, True], ['anim', 200, 100000, True], ['anim', 201, 100500, False], ['anim', 202, 101000, False], ['anim', 203, 101500, False], ['anim', 228, 114000, False]], 'stdout': '   Fits at 200 colors.\n   [probe_*] Testing max colors 153...\n   [probe_*] Testing max colors 178...\n   [probe_*] Testing max colors 183...\n   [probe_*] Testing max colors 187...\n   [probe_*] Testing max colors 191...\n   [probe_*] Testing max colors 195...\n   [probe_*] Testing max colors 197...\n   [probe_*] Testing max colors 198...\n   [probe_*] Testing max colors 199...\n   [probe_*] Testing max colors 200...\n   [probe_*] Testing max colors 201...\n   [probe_*] Testing max colors 202...\n   [probe_*] Testing max colors 203...\n   [probe_*] Testing max colors 228...'}, 'an_pixel_art_snap': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'an_grid_fit': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'an_resizes_all_frames_fits': {'result': {'success': True, 'final_size': 100000, 'iterations': 1}, 'flow': [['replace', 'F1000x100000', 'OUT']], 'probes': [['anim', 153, 95625, True], ['anim', 158, 98750, True], ['anim', 159, 99375, True], ['anim', 160, 100000, True], ['anim', 161, 100625, False], ['anim', 162, 101250, False], ['anim', 166, 103750, False], ['anim', 170, 106250, False], ['anim', 178, 111250, False], ['anim', 203, 126875, False], ['anim', 228, 142500, False]], 'stdout': '   Fits at 160 colors.\n   [probe_*] Testing max colors 153...\n   [probe_*] Testing max colors 158...\n   [probe_*] Testing max colors 159...\n   [probe_*] Testing max colors 160...\n   [probe_*] Testing max colors 161...\n   [probe_*] Testing max colors 162...\n   [probe_*] Testing max colors 166...\n   [probe_*] Testing max colors 170...\n   [probe_*] Testing max colors 178...\n   [probe_*] Testing max colors 203...\n   [probe_*] Testing max colors 228...'}, 'sa2_frontier_ceiling': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}, 'sa2_no_fit_restore': {'result': {'success': True, 'final_size': 5000, 'iterations': 0}, 'flow': [['copy', 'SRC', 'OUT']], 'probes': [], 'stdout': ''}}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_search_loop_trace_matches_golden(name, tmp_path, monkeypatch, capsys):
    spec = next(s for s in all_specs() if s["name"] == name)
    spec = dict(spec)
    spec["src"] = str(tmp_path / f"src_{name}")
    spec["out"] = str(tmp_path / f"out_{name}")
    sim = Sim(spec)
    trace = run_scenario(sim, spec, monkeypatch, capsys)
    assert trace == EXPECTED[name], f"scenario {name} drifted from the golden trace"


FRONTIER_EQUIVALENCE = {
    "sl_frontier_ceiling": "iter_lossy_frontier",
    "sl_frontier_below_ceiling_deep": "iter_lossy_frontier",
    "sl_subsampling_444": "iter_lossy_frontier",
    "sl_no_fit_bail_restore": "iter_lossy_frontier",
    "sl_no_fit_max_iterations": "iter_lossy_frontier",
    "sa_frontier_ceiling": "iter_art_frontier",
    "sa_frontier_below_ceiling": "iter_art_frontier",
    "sa_container_corrected_ceiling_fit": "iter_art_frontier",
    "sa_no_fit_restore": "iter_art_frontier",
    "sa2_frontier_ceiling": "iter_anim_frontier",
    "sa2_no_fit_restore": "iter_anim_frontier",
}


@pytest.mark.parametrize("name", sorted(FRONTIER_EQUIVALENCE))
def test_iterative_alias_frontier_equals_slowest_wrapper(name, tmp_path, monkeypatch, capsys):
    spec = next(s for s in all_specs() if s["name"] == name)
    slowest_spec = dict(spec)
    slowest_spec["src"] = str(tmp_path / f"src_{name}")
    slowest_spec["out"] = str(tmp_path / f"out_{name}")
    slowest_trace = run_scenario(Sim(slowest_spec), slowest_spec, monkeypatch, capsys)

    iter_spec = dict(slowest_spec)
    iter_spec["mode"] = FRONTIER_EQUIVALENCE[name]
    iter_spec["src"] = str(tmp_path / f"isrc_{name}")
    iter_spec["out"] = str(tmp_path / f"iout_{name}")
    iter_trace = run_scenario(Sim(iter_spec), iter_spec, monkeypatch, capsys)
    assert iter_trace == slowest_trace, (
        f"scenario {name}: iterative alias with frontier=True differs from "
        f"the slowest wrapper"
    )


def _strategy(**overrides):
    base = {
        "search_round": lambda current, probes, lo, hi: search._RoundResult(
            fitted=False, level_best=None, last_probe=None, probe_ok=True,
            candidates=(), frontier_ceiling_met=False),
        "knob_range": (1, 100),
    }
    base.update(overrides)
    return search._SearchStrategy(**base)


@pytest.mark.parametrize("overrides", [
    {"collect_candidates": True},
    {"collect_candidates": True,
     "select_winner": lambda cands, ref, reg: None},
    {"palette": True},
    {"extent_first": True},
    {"extent_after": True},
    {"break_on_all_probe_failure": True},
])
def test_search_strategy_rejects_incomplete_combinations(overrides):
    with pytest.raises(ValueError):
        _strategy(**overrides)


def test_search_strategy_accepts_valid_combinations():
    _strategy()
    _strategy(collect_candidates=True,
              select_winner=lambda cands, ref, reg: None,
              winner_message=lambda winner, size: "")
    _strategy(palette=True, palette_pass=lambda src, out, colors: True)
    _strategy(extent_first=True, extent_probe=lambda current, probes: None)
    _strategy(break_on_all_probe_failure=True,
              bail_encode=lambda src, out, quality: True)


def test_probe_race_cancels_dominated_tokens():
    from core import utils

    race = search._ProbeRace()
    low, mid, high = utils.CancelToken(), utils.CancelToken(), utils.CancelToken()
    race.register(85, low)
    race.register(90, mid)
    race.register(95, high)
    assert race.is_dominated(85) is False

    race.report_fit(90)
    assert race.is_dominated(85) is True
    assert race.is_dominated(90) is False
    assert race.is_dominated(95) is False
    assert low.cancelled is True
    assert mid.cancelled is False and high.cancelled is False

    race.report_fit(95)
    assert mid.cancelled is True
    assert high.cancelled is False
    race.unregister(85)
    race.unregister(85)


def _fake_round(monkeypatch, tmp_path, size_of, *, collect, step=3):
    monkeypatch.setattr(search, "_probe_worker_cap", lambda current: 1)
    probed = []

    def encode(src, out, knob):
        probed.append(knob)
        out.write_bytes(b"x" * size_of(knob))
        return True

    round_ = search._simple_search_round(
        probe_encode=encode, probe_suffix=".bin",
        probe_message="   Testing quality {}%...",
        probe_skip_message="   Skipping quality {}% (fit at {}% already found).",
        search_step=step, max_size_bytes=9400,
        round_collect_candidates=collect, round_frontier_ceiling=256,
    )
    probes: list[Path] = []
    result = round_(tmp_path / "src", probes, 85, 100)
    return result, probed, probes


def test_grid_round_abandons_dominated_probes(monkeypatch, tmp_path, capsys):
    result, probed, probes = _fake_round(
        monkeypatch, tmp_path, lambda knob: knob * 100, collect=False)
    try:
        assert probed == [100, 97, 94]
        assert result.fitted is True
        assert result.level_best is not None and result.level_best[2] == 94
    finally:
        for p in probes:
            p.unlink(missing_ok=True)
    out = capsys.readouterr().out
    assert "Skipping quality 91% (fit at 94% already found)." in out
    assert "Skipping quality 85% (fit at 94% already found)." in out


def test_frontier_round_keeps_every_probe(monkeypatch, tmp_path):
    result, probed, probes = _fake_round(
        monkeypatch, tmp_path, lambda knob: knob * 100, collect=True)
    try:
        assert probed == [100, 97, 94, 91, 88, 85]
        assert result.level_best is not None and result.level_best[2] == 94
        assert result.last_probe is not None
    finally:
        for p in probes:
            p.unlink(missing_ok=True)


def test_parallel_fit_search_ignores_abandoned_probe_results(tmp_path):
    ran = []

    def probe_fn(q):
        if q < 95:
            return None
        ran.append(q)
        p = tmp_path / f"p{q}"
        p.write_bytes(b"x" * q)
        return (True, q, p)

    best, executed = search._parallel_fit_search(probe_fn, 85, 100,
                                                 k=4, worker_cap=1)
    assert ran == [97, 98, 99, 100]
    assert [e[0] for e in executed] == [97, 98, 99, 100]
    assert best is not None and best[0] == 100


def test_adaptive_scale_pct_frontier_clamps_to_deep_step_at_every_level():
    frontier = _strategy(collect_candidates=True,
                         select_winner=lambda cands, ref, reg: None,
                         winner_message=lambda winner, size: "",
                         deep_step=75)
    iterative = _strategy()
    for _fits in range(0, 6):
        assert search._adaptive_scale_pct(1, MAX, frontier) == 75
    assert search._adaptive_scale_pct(1, MAX, iterative) == 50
    assert search._adaptive_scale_pct(4 * MAX, MAX, frontier) == 50
    assert search._adaptive_scale_pct(4 * MAX, MAX, iterative) == 50
