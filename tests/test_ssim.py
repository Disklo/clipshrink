from core import ssim


def _cand(name, size, width=1000, quality=80, subsampling="4:2:0", encoder="fake"):
    return {"path": name, "size": size, "width": width, "quality": quality,
            "subsampling": subsampling, "encoder": encoder}


def _patch_scores(monkeypatch, scores, captured=None):
    def fake_ssim(reference, candidate, registry, at_original=False):
        if captured is not None:
            captured.append((candidate, at_original))
        return scores.get(candidate)
    monkeypatch.setattr(ssim, "_ssim_vs_reference", fake_ssim)
    monkeypatch.setattr(ssim.search, "_probe_width",
                        lambda p: {"full": 3840, "half": 1920}.get(str(p), 1000))
    monkeypatch.setattr(ssim.converters, "get_image_width", lambda p: 0)
    monkeypatch.setattr(ssim.converters, "get_image_height", lambda p: 1000)


def test_floor_prefers_smallest_meeting_candidate(monkeypatch):
    cands = [_cand("full", 2500), _cand("half", 2000), _cand("tiny", 900)]
    _patch_scores(monkeypatch, {"full": 0.96, "half": 0.951, "tiny": 0.70})
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True, floor=0.95)
    assert winner["path"] == "half"


def test_floor_all_below_falls_back_to_highest_score(monkeypatch):
    cands = [_cand("a", 2500), _cand("b", 2000)]
    _patch_scores(monkeypatch, {"a": 0.90, "b": 0.93})
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True, floor=0.95)
    assert winner["path"] == "b"


def test_floor_tie_prefers_higher_resolution(monkeypatch):
    cands = [_cand("half", 2000, width=1920), _cand("full", 2000, width=3840)]
    _patch_scores(monkeypatch, {"half": 0.96, "full": 0.955})
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True, floor=0.95)
    assert winner["path"] == "full"


def test_at_original_skips_chained_reference_and_flags_measurement(monkeypatch):
    captured = []
    cands = [_cand("full", 2500), _cand("half", 2000)]
    _patch_scores(monkeypatch, {"full": 0.95, "half": 0.94}, captured)
    chained = []
    monkeypatch.setattr(ssim, "_chained_reference",
                        lambda ref, cands, reg: chained.append(ref) or ref)
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True,
                                         at_original=True)
    assert chained == []
    assert all(at for _, at in captured)
    assert winner["path"] == "full"


def test_default_ranking_unchanged(monkeypatch):
    cands = [_cand("full", 2500), _cand("half", 2000)]
    _patch_scores(monkeypatch, {"full": 0.951, "half": 0.986})
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True)
    assert winner["path"] == "half"


def _capture_ssim_measurement(monkeypatch, tmp_path, width, height):
    from pathlib import Path
    captured = {}
    monkeypatch.setattr(ssim.converters, "get_image_width", lambda p: width)
    monkeypatch.setattr(ssim.converters, "get_image_height", lambda p: height)
    monkeypatch.setattr(ssim.search, "_probe_width", lambda p: width)
    monkeypatch.setattr(ssim.search, "_new_probe",
                        lambda suffix, reg: tmp_path / f"probe{suffix}")
    probe_files = (tmp_path / "probe_ref.png", tmp_path / "probe_cand.png")
    for p in probe_files:
        p.write_bytes(b"x")

    def fake_resize(src, out, w, h, _kernel):
        captured.setdefault("resizes", []).append((str(src), w, h))
        Path(out).write_bytes(b"x")

    def fake_magick(cmd):
        captured["magick_resize"] = cmd[cmd.index("-resize") + 1]
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_ffmpeg(cmd):
        import subprocess
        return subprocess.CompletedProcess(cmd, 0, "", "All:0.990000")

    monkeypatch.setattr(ssim.converters, "resize_exact_vips", fake_resize)
    monkeypatch.setattr(ssim.converters, "resize_exact", fake_resize)
    monkeypatch.setattr(ssim.converters, "_run_magick", fake_magick)
    monkeypatch.setattr(ssim.utils, "run_capture", fake_ffmpeg)
    score = ssim._ssim_vs_reference(tmp_path / "ref.png", tmp_path / "cand.jpg",
                                    [], at_original=True)
    return score, captured


def test_ssim_small_image_measured_at_native_resolution(monkeypatch, tmp_path):
    monkeypatch.setattr(ssim.memgate, "budget_bytes", lambda: 100 * 1024 ** 3)
    score, captured = _capture_ssim_measurement(monkeypatch, tmp_path, 8000, 6000)
    assert captured["resizes"] == [(str(tmp_path / "ref.png"), 8000, 6000)]
    assert captured["magick_resize"] == "8000x6000!"
    assert score == 0.99


def test_ssim_giant_image_measured_on_proxy(monkeypatch, tmp_path):
    monkeypatch.setattr(ssim.memgate, "budget_bytes", lambda: 8 * 1024 ** 3)
    score, captured = _capture_ssim_measurement(monkeypatch, tmp_path, 30000, 21952)
    assert "magick_resize" not in captured
    assert len(captured["resizes"]) == 2
    for _, w, h in captured["resizes"]:
        assert w * h <= ssim._measurement_cap()
        assert (w + 128) * (h + 128) <= ssim._FFMPEG_SSIM_MAX_PRODUCT
    assert score == 0.99


def test_ssim_proxy_falls_back_when_vips_unavailable(monkeypatch, tmp_path):
    _capture_ssim_measurement(monkeypatch, tmp_path, 30000, 21952)
    monkeypatch.setattr(ssim.converters, "resize_exact_vips", lambda *a, **k: False)
    score2, captured2 = _capture_ssim_measurement(monkeypatch, tmp_path, 30000, 21952)
    assert score2 == 0.99
    assert len(captured2["resizes"]) == 2


def test_ssim_proxy_aspect_ratio_preserved(monkeypatch, tmp_path):
    _, captured = _capture_ssim_measurement(monkeypatch, tmp_path, 30000, 21952)
    _, w, h = captured["resizes"][0]
    assert abs(w / h - 30000 / 21952) < 0.01


def test_ssim_cap_shrinks_with_small_memory_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(ssim.memgate, "budget_bytes", lambda: 3 * 1024 ** 3)
    _, captured = _capture_ssim_measurement(monkeypatch, tmp_path, 30000, 21952)
    _, w, h = captured["resizes"][0]
    assert w * h <= ssim._measurement_cap()
    assert w * h <= 3 * 1024 ** 3 // ssim._MEASUREMENT_BYTES_PER_PIXEL


def test_ssim_cap_scales_up_with_big_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(ssim.memgate, "budget_bytes", lambda: 32 * 1024 ** 3)
    assert ssim._measurement_cap() == 32 * 1024 ** 3 // ssim._MEASUREMENT_BYTES_PER_PIXEL


def test_ssim_cap_floor(monkeypatch):
    monkeypatch.setattr(ssim.memgate, "budget_bytes", lambda: 1024)
    assert ssim._measurement_cap() == 1_000_000


def test_ssim_uses_effective_cap_for_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(ssim.memgate, "budget_bytes", lambda: 3 * 1024 ** 3)
    _, captured = _capture_ssim_measurement(monkeypatch, tmp_path, 9000, 7000)
    _, w, h = captured["resizes"][0]
    assert w * h <= ssim._measurement_cap()
    assert (w, h) != (9000, 7000)


def test_ssim_ffmpeg_bound_proxies_even_with_big_budget(monkeypatch, tmp_path):
    monkeypatch.setattr(ssim.memgate, "budget_bytes", lambda: 100 * 1024 ** 3)
    score, captured = _capture_ssim_measurement(monkeypatch, tmp_path, 16384, 16384)
    assert "magick_resize" not in captured
    _, w, h = captured["resizes"][0]
    assert (w + 128) * (h + 128) <= ssim._FFMPEG_SSIM_MAX_PRODUCT
    assert score == 0.99


def test_ssim_ffmpeg_bound_is_aspect_aware():
    assert ssim._ffmpeg_safe_dims(32768, 5024) == (32768, 5024)
    assert ssim._ffmpeg_safe_dims(15000, 15000) == (15000, 15000)
    w, h = ssim._ffmpeg_safe_dims(16384, 16384)
    assert (w, h) != (16384, 16384)
    assert (w + 128) * (h + 128) <= ssim._FFMPEG_SSIM_MAX_PRODUCT


def test_fidelity_pick_prefers_higher_at_original_fidelity(monkeypatch, tmp_path):
    from pathlib import Path
    a = (tmp_path / "extent.jpg", 1500)
    b = (tmp_path / "rival.jpg", 2000)
    for p, _ in (a, b):
        Path(p).write_bytes(b"x")
    scores = {Path(a[0]): 0.6029, Path(b[0]): 0.6207}
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: scores.get(Path(cand)))
    monkeypatch.setattr(ssim.search, "_probe_width", lambda p: 3915 if "rival" in str(p) else 4500)
    assert ssim.fidelity_pick(a, b, tmp_path / "ref.png", []) == b


def test_fidelity_pick_tie_within_eps_prefers_higher_resolution(monkeypatch, tmp_path):
    from pathlib import Path
    a = (tmp_path / "extent.jpg", 1500)
    b = (tmp_path / "rival.jpg", 2000)
    for p, _ in (a, b):
        Path(p).write_bytes(b"x")
    scores = {Path(a[0]): 0.6029, Path(b[0]): 0.6120}
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: scores.get(Path(cand)))
    monkeypatch.setattr(ssim.search, "_probe_width", lambda p: 4500 if "extent" in str(p) else 3915)
    assert ssim.fidelity_pick(a, b, tmp_path / "ref.png", []) == a


def test_fidelity_pick_unmeasurable_candidates_ship_extent(monkeypatch, tmp_path):
    from pathlib import Path
    a = (tmp_path / "extent.jpg", 1500)
    b = (tmp_path / "rival.jpg", 2000)
    for p, _ in (a, b):
        Path(p).write_bytes(b"x")
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: None)
    assert ssim.fidelity_pick(a, b, tmp_path / "ref.png", []) == a


def test_tie_codec_preference_jpeg_over_webp_over_avif(monkeypatch):
    cands = [_cand("avif", 2000, encoder="avif"),
             _cand("jpeg", 2000, encoder="jpeg"),
             _cand("webp", 2000, encoder="webp")]
    _patch_scores(monkeypatch, {c["path"]: 0.95 for c in cands})
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True)
    assert winner["encoder"] == "jpeg"


def test_floor_tie_codec_preference_jpeg_over_avif(monkeypatch):
    cands = [_cand("avif", 2000, encoder="avif"),
             _cand("jpeg", 2000, encoder="jpeg")]
    _patch_scores(monkeypatch, {c["path"]: 0.96 for c in cands})
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True, floor=0.95)
    assert winner["encoder"] == "jpeg"


def test_tie_unknown_encoder_loses_to_jpeg(monkeypatch):
    cands = [_cand("webp", 2000, encoder="webp"),
             _cand("jpeg", 2000, encoder="jpeg")]
    _patch_scores(monkeypatch, {c["path"]: 0.95 for c in cands})
    winner = ssim._rank_lossy_candidates(cands, "ref", [], ssim=True)
    assert winner["encoder"] == "jpeg"
