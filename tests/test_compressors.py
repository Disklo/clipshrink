from pathlib import Path

from core import compressors, encoders, search, ssim
from core.backend import BackendInfo


def _backend(**overrides):
    fields = {"os_type": "X11", "vips": False, "magick": True, "pngquant": False,
                  "realesrgan": False, "waifu2x": False}
    fields.update(overrides)
    return BackendInfo(**fields)


def _write(src: Path, size: int) -> None:
    src.write_bytes(b"x" * size)


def test_resolve_knob_floor_explicit_min_wins():
    assert compressors._resolve_knob_floor(70, 90, 85) == 70
    assert compressors._resolve_knob_floor(20, 90, 30) == 20


def test_resolve_knob_floor_defaults_by_mode():
    assert compressors._resolve_knob_floor(None, None, 85) == 85
    assert compressors._resolve_knob_floor(None, None, 30) == 30


def test_resolve_knob_floor_start_below_default_lowers_floor():
    assert compressors._resolve_knob_floor(None, 60, 85) == 60
    assert compressors._resolve_knob_floor(None, 10, 30) == 10
    assert compressors._resolve_knob_floor(None, 95, 85) == 85
    assert compressors._resolve_knob_floor(None, 95, 30) == 30


def test_resolve_knob_floor_color_defaults():
    assert compressors._resolve_knob_floor(None, None, 128) == 128
    assert compressors._resolve_knob_floor(None, None, 2) == 2
    assert compressors._resolve_knob_floor(None, 64, 128) == 64


def test_pngquant_compress_missing_binary_returns_false(monkeypatch, tmp_path):
    def no_binary(cmd, **kwargs):
        raise FileNotFoundError("pngquant")

    monkeypatch.setattr(compressors.utils, "run_capture", no_binary)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.png"
    assert compressors.pngquant_compress(src, out) is False
    assert compressors.utils.get_last_error()


def test_jpeg_quality_reduce_delegates(monkeypatch, tmp_path):
    calls = []

    def fake_jpeg_save(inp, outp, quality, **kwargs):
        calls.append((inp, outp, quality, kwargs))
        outp.write_bytes(b"j" * 100)
        return True

    backend = _backend()
    monkeypatch.setattr(compressors.converters, "jpeg_save", fake_jpeg_save)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.jpg"
    assert compressors.jpeg_quality_reduce(src, out, 85, strip_metadata=True,
                                           subsampling="4:4:4", backend=backend) is True
    assert calls[0][2] == 85
    assert calls[0][3]["strip_metadata"] is True
    assert calls[0][3]["subsampling"] == "4:4:4"
    assert calls[0][3]["progressive"] is True


def test_jpeg_quality_reduce_prefers_jpegli(monkeypatch, tmp_path):
    calls = []

    def fake_jpegli_save(inp, outp, quality, **kwargs):
        calls.append((inp, outp, quality, kwargs))
        outp.write_bytes(b"j" * 100)
        return True

    backend = _backend(jpegli=True)
    monkeypatch.setattr(compressors.converters, "jpegli_save", fake_jpegli_save)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.jpg"
    assert compressors.jpeg_quality_reduce(src, out, 90, backend=backend) is True
    assert calls[0][2] == 90
    assert calls[0][3]["subsampling"] == "4:2:0"


def test_jpeg_quality_reduce_prefers_mozjpeg_over_turbo(monkeypatch, tmp_path):
    calls = []

    def fake_mozjpeg_save(inp, outp, quality, **kwargs):
        calls.append((inp, outp, quality, kwargs))
        outp.write_bytes(b"j" * 100)
        return True

    backend = _backend(mozjpeg=True)
    monkeypatch.setattr(compressors.converters, "mozjpeg_save", fake_mozjpeg_save)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.jpg"
    assert compressors.jpeg_quality_reduce(src, out, 90, backend=backend) is True
    assert calls[0][2] == 90


def test_oxipng_compress_missing_binary_returns_false(monkeypatch, tmp_path):
    def no_binary(cmd, **kwargs):
        raise FileNotFoundError("oxipng")

    monkeypatch.setattr(compressors.utils, "run_capture", no_binary)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.png"
    assert compressors.oxipng_compress(src, out) is False
    assert compressors.utils.get_last_error()


def test_oxipng_compress_builds_command(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(compressors.utils, "run_capture", fake_run)
    src = tmp_path / "in.png"
    src.write_bytes(b"x")
    out = tmp_path / "out.png"
    assert compressors.oxipng_compress(src, out, level=4, strip_metadata=True) is True
    assert captured["cmd"][:4] == ["oxipng", "-o", "4", "--strip"]
    assert captured["cmd"][4] == "all"
    assert captured["cmd"][-2:] == ["--", str(src)]


def test_make_lossy_encoders_extent_wired_only_for_stock_jpeg():
    plain = _backend()
    encoders = compressors.make_lossy_encoders(plain)
    assert encoders["JPEG"].extent is not None
    assert encoders["JPEG"].subsampling_supported is True
    assert encoders["WEBP"].suffix == ".webp"
    assert encoders["AVIF"].suffix == ".avif"

    better = _backend(jpegli=True)
    encoders = compressors.make_lossy_encoders(better, speed="slow")
    assert encoders["JPEG"].extent is None


def test_make_lossy_encoders_heic_prefers_software_only_in_slowest_mode(monkeypatch):
    calls = []

    def fake_heic_save(*args, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(compressors.converters, "heic_save", fake_heic_save)
    backend = _backend()
    from pathlib import Path

    encoders = compressors.make_lossy_encoders(backend, slowest=True)
    assert encoders["HEIC"].encode(Path("in"), Path("out"), 80) is True
    assert calls[-1]["prefer_software"] is True

    encoders = compressors.make_lossy_encoders(backend, slowest=False)
    assert encoders["HEIC"].encode(Path("in"), Path("out"), 80) is True
    assert calls[-1]["prefer_software"] is False


def test_make_lossy_encoders_avif_prefers_software_only_in_slowest_mode(monkeypatch):
    calls = []

    def fake_avif_save(*args, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(compressors.converters, "avif_save", fake_avif_save)
    backend = _backend()
    from pathlib import Path

    encoders = compressors.make_lossy_encoders(backend, slowest=True)
    assert encoders["AVIF"].encode(Path("in"), Path("out"), 80) is True
    assert calls[-1]["prefer_software"] is True
    assert calls[-1]["speed"] == 1

    encoders = compressors.make_lossy_encoders(backend, slowest=False)
    assert encoders["AVIF"].encode(Path("in"), Path("out"), 80) is True
    assert calls[-1]["prefer_software"] is False
    assert calls[-1]["speed"] == 8


def test_iterative_art_optimize_fits_at_high_colors(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 2000)
    out = tmp_path / "out.png"
    probes = []

    def fake_palette(inp, outp, colors, strip_metadata=False, **kwargs):
        size = 500 + colors
        outp.write_bytes(b"p" * size)
        probes.append(colors)
        return True

    def fake_width(path):
        return 1000

    monkeypatch.setattr(compressors, "color_palette_reduce", fake_palette)
    monkeypatch.setattr(compressors.converters, "get_image_width", fake_width)
    monkeypatch.setattr(search, "_MAX_PARALLEL", 4)
    monkeypatch.setattr(search, "_probe_worker_cap", lambda current: 1)

    result = compressors.iterative_art_optimize(
        src, out, max_size_bytes=700,
        pixel_art_mode=True, start_colors=256, min_colors=128,
    )
    assert result["success"] is True
    assert result["final_size"] == 700
    assert result["iterations"] == 1
    assert sorted(probes) == [153, 178, 183, 187, 191, 195,
                              197, 198, 199, 200, 201, 202, 203, 228]
    assert out.read_bytes() == b"p" * 700


def test_iterative_art_optimize_pixel_art_snaps_scale(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 5000)
    out = tmp_path / "out.png"
    resizes = []

    def fake_palette(inp, outp, colors, strip_metadata=False, **kwargs):
        outp.write_bytes(b"p" * 5000)
        return True

    def fake_sample(inp, outp, percentage, **kwargs):
        size = inp.stat().st_size * 80 // 100
        outp.write_bytes(b"p" * size)
        resizes.append(percentage)
        return True

    def fake_width(path):
        return 800

    monkeypatch.setattr(compressors, "color_palette_reduce", fake_palette)
    monkeypatch.setattr(compressors.converters, "sample_pct", fake_sample)
    monkeypatch.setattr(compressors.converters, "get_image_width", fake_width)

    result = compressors.iterative_art_optimize(
        src, out, max_size_bytes=1000,
        pixel_art_mode=True, start_colors=256, min_colors=128,
        bail_ratio=1.3,
    )
    assert result["success"] is False
    assert result["iterations"] == 1
    assert resizes == [50]
    assert 4000 * 1.3 > 5000


def _fake_encoder(qualities=None, suffix=".jpg"):
    recorded = [] if qualities is None else qualities

    def encode(inp, outp, quality, subsampling="4:2:0", strip_metadata=False, backend=None):
        recorded.append(quality)
        outp.write_bytes(b"e" * (quality * 100))
        return True

    return compressors.CodecEncoder("fake", encode, suffix=suffix), recorded


def test_iterative_lossy_optimize_probes_encoder(tmp_path):
    src = tmp_path / "src.png"
    _write(src, 100)
    out = tmp_path / "out.bin"
    encoder, recorded = _fake_encoder()

    def fake_width(path):
        return 1000

    monkeypatch = __import__("pytest").MonkeyPatch()
    monkeypatch.setattr(compressors.converters, "get_image_width", fake_width)
    monkeypatch.setattr(search, "_MAX_PARALLEL", 4)
    monkeypatch.setattr(search, "_probe_worker_cap", lambda current: 1)
    try:
        result = compressors.iterative_lossy_optimize(
            src, out, max_size_bytes=9500, encoder=encoder,
            start_quality=100, min_quality=85, use_extent_fast_path=False,
        )
    finally:
        monkeypatch.undo()
    assert result["success"] is True
    assert result["final_size"] == 9500
    assert sorted(recorded) == [88, 91, 94, 95, 96, 97]


def test_iterative_lossy_optimize_stops_when_all_probes_fail(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 5000)
    out = tmp_path / "out.bin"

    def encode(inp, outp, quality, subsampling="4:2:0", strip_metadata=False, backend=None):
        return False

    encoder = compressors.CodecEncoder("dead", encode)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    result = compressors.iterative_lossy_optimize(
        src, out, max_size_bytes=1000, encoder=encoder,
        start_quality=100, min_quality=85, use_extent_fast_path=False,
    )
    assert result["success"] is False
    assert result["iterations"] == 1
    assert out.read_bytes() == b"x" * 5000


def test_iterative_lossy_optimize_scale_guard_above_limit(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 4000)
    out = tmp_path / "out.bin"
    resizes = []

    def fake_encode(inp, outp, quality, subsampling="4:2:0",
                    strip_metadata=False, backend=None):
        outp.write_bytes(b"e" * 1001)
        return True

    def fake_resize(inp, outp, percentage, kernel="Lanczos", **kwargs):
        size = inp.stat().st_size * percentage * percentage // (100 * 100)
        outp.write_bytes(b"e" * size)
        resizes.append(percentage)
        return True

    def fake_width(path):
        return 1000 - len(resizes) * 100

    encoder = compressors.CodecEncoder("fake", fake_encode)
    monkeypatch.setattr(compressors.converters, "resize_pct", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", fake_width)

    result = compressors.iterative_lossy_optimize(
        src, out, max_size_bytes=1000, encoder=encoder,
        start_quality=100, min_quality=85, use_extent_fast_path=False,
    )
    assert resizes == [50]
    assert result["success"] is False
    assert result["final_size"] == 1001
    assert result["iterations"] == 1
    assert out.read_bytes() == b"e" * 1001


def test_rank_lossy_candidates_prefers_ssim_then_resolution(monkeypatch, tmp_path):
    candidates = [
        {"path": tmp_path / "a.bin", "size": 1000, "quality": 90,
         "subsampling": "4:2:0", "encoder": "jpeg"},
        {"path": tmp_path / "b.bin", "size": 2000, "quality": 85,
         "subsampling": "4:2:0", "encoder": "jpeg"},
    ]
    candidates[0]["path"].write_bytes(b"a")
    candidates[1]["path"].write_bytes(b"b")
    monkeypatch.setattr(ssim, "_chained_reference", lambda ref, cands, reg: ref)
    monkeypatch.setattr(ssim.search, "_probe_width", lambda p: 0)

    ssims = {str(candidates[0]["path"]): 0.90, str(candidates[1]["path"]): 0.95}
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: ssims.get(str(cand)))
    winner = compressors._rank_lossy_candidates(candidates, tmp_path / "src.png", [])
    assert winner is candidates[1]

    ssims = {str(candidates[0]["path"]): 0.90, str(candidates[1]["path"]): 0.90}
    widths = {str(candidates[0]["path"]): 2000, str(candidates[1]["path"]): 1000}
    monkeypatch.setattr(search, "_probe_width",
                        lambda p: widths.get(str(p), 0))
    winner = compressors._rank_lossy_candidates(candidates, tmp_path / "src.png", [])
    assert winner is candidates[0]

    ssims = {str(candidates[0]["path"]): 0.95, str(candidates[1]["path"]): 0.966}
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: ssims.get(str(cand)))
    winner = compressors._rank_lossy_candidates(candidates, tmp_path / "src.png", [])
    assert winner is candidates[0]


def test_chained_reference_downscales_to_largest_width(monkeypatch, tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"r")
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 1000, "quality": 90, "subsampling": "4:2:0", "encoder": "jpeg"},
        {"path": b, "size": 2000, "quality": 85, "subsampling": "4:2:0", "encoder": "jpeg"},
    ]
    recorded = {}
    monkeypatch.setattr(search, "_probe_width",
                        lambda p: {str(a): 500, str(b): 1000}.get(str(p), 0))
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 4000)
    monkeypatch.setattr(compressors.converters, "get_image_height", lambda p: 3000)
    monkeypatch.setattr(compressors.converters, "resize_exact",
                        lambda inp, outp, w, h, kernel: recorded.update(
                            width=w, height=h) or True)
    out = compressors._chained_reference(ref, candidates, [])
    assert recorded["width"] == 1000
    assert recorded["height"] == 750
    assert out != ref


def test_rank_lossy_candidates_prefers_larger_size_on_tie(monkeypatch, tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 500, "quality": 80, "subsampling": "4:2:0", "encoder": "jpeg"},
        {"path": b, "size": 1900, "quality": 95, "subsampling": "4:2:0", "encoder": "jpeg"},
    ]
    monkeypatch.setattr(ssim, "_chained_reference", lambda ref, cands, reg: ref)
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: 0.90)
    monkeypatch.setattr(search, "_probe_width", lambda p: 1000)
    winner = compressors._rank_lossy_candidates(candidates, tmp_path / "src.png", [])
    assert winner is candidates[1]


def test_rank_lossy_candidates_prefers_444_on_tie(monkeypatch, tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 1000, "quality": 90, "subsampling": "4:2:0", "encoder": "jpeg"},
        {"path": b, "size": 1000, "quality": 90, "subsampling": "4:4:4", "encoder": "jpeg"},
    ]
    monkeypatch.setattr(ssim, "_chained_reference", lambda ref, cands, reg: ref)
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: None)
    monkeypatch.setattr(search, "_probe_width", lambda p: 1000)
    winner = compressors._rank_lossy_candidates(candidates, tmp_path / "src.png", [])
    assert winner is candidates[1]


def test_slowest_lossy_optimize_selects_best_fit_across_levels(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 5000)
    out = tmp_path / "out.bin"
    probes = []
    encoder, _recorded = _fake_encoder(probes)

    def fake_resize(inp, outp, percentage, kernel="Lanczos", **kwargs):
        size = inp.stat().st_size * percentage * percentage // (100 * 100)
        outp.write_bytes(b"e" * size)
        return True

    monkeypatch.setattr(compressors.converters, "resize_pct", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    monkeypatch.setattr(ssim, "_chained_reference", lambda ref, cands, reg: ref)
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False:
                        0.99 if cand.stat().st_size == 3000 else 0.95)

    result = compressors.slowest_lossy_optimize(
        src, out, max_size_bytes=3000, encoders={"FAKE": encoder},
        start_quality=100, min_quality=30, extra_fit_levels=2,
    )
    assert result["success"] is True
    assert result["final_size"] == 3000
    assert out.read_bytes() == b"e" * 3000


def test_slowest_lossy_optimize_continues_below_quality_ceiling(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 5000)
    out = tmp_path / "out.bin"

    def fake_encode(inp, outp, quality, **kwargs):
        base = inp.stat().st_size
        if quality > 50:
            outp.write_bytes(b"x" * max(6000, base * 2))
            return True
        size = max(10, base // 2)
        outp.write_bytes(b"e" * size)
        return True

    encoder = compressors.CodecEncoder("FAKE", fake_encode, suffix=".bin")

    def fake_resize(inp, outp, percentage, kernel="Lanczos", **kwargs):
        size = inp.stat().st_size * percentage * percentage // (100 * 100)
        outp.write_bytes(b"e" * size)
        return True

    monkeypatch.setattr(compressors.converters, "resize_pct", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    monkeypatch.setattr(ssim, "_chained_reference", lambda ref, cands, reg: ref)
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: None)

    result = compressors.slowest_lossy_optimize(
        src, out, max_size_bytes=3000, encoders={"FAKE": encoder},
        start_quality=100, min_quality=30, extra_fit_levels=2,
    )
    assert result["success"] is True
    assert result["iterations"] >= 3


def test_rank_art_candidates_by_colors_ignores_ssim(monkeypatch, tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 500, "colors": 256},
        {"path": b, "size": 1900, "colors": 256},
    ]
    monkeypatch.setattr(search, "_probe_width",
                        lambda p: {str(a): 1200, str(b): 2400}.get(str(p), 0))
    winner = compressors._rank_art_candidates(candidates, tmp_path / "src.png", [],
                                              by_colors=True)
    assert winner is candidates[1]


def test_rank_art_candidates_width_first_vs_colors_first(monkeypatch, tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 1900, "colors": 200},
        {"path": b, "size": 1900, "colors": 256},
    ]
    monkeypatch.setattr(search, "_probe_width",
                        lambda p: {str(a): 2531, str(b): 1898}.get(str(p), 0))
    winner = compressors._rank_art_candidates(candidates, tmp_path / "src.png", [],
                                              by_colors=True, width_first=True)
    assert winner is candidates[0], "width-first must prefer the wider intermediate"
    winner = compressors._rank_art_candidates(candidates, tmp_path / "src.png", [],
                                              by_colors=True)
    assert winner is candidates[1], "colors-first must prefer the more colors"


def test_slowest_art_optimize_picks_more_colors_on_tie(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 5000)
    out = tmp_path / "out.png"

    def fake_palette(inp, outp, colors, strip_metadata=False, **kwargs):
        size = 400 + colors
        outp.write_bytes(b"p" * size)
        return True

    def fake_resize(inp, outp, percentage, kernel="Lanczos", **kwargs):
        size = inp.stat().st_size * percentage * percentage // (100 * 100)
        outp.write_bytes(b"p" * size)
        return True

    monkeypatch.setattr(compressors, "color_palette_reduce", fake_palette)
    monkeypatch.setattr(compressors.converters, "resize_pct", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 800)
    monkeypatch.setattr(ssim, "_chained_reference", lambda ref, cands, reg: ref)
    monkeypatch.setattr(ssim, "_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: None)

    result = compressors.slowest_art_optimize(
        src, out, max_size_bytes=1000, pixel_art_mode=False,
        start_colors=256, min_colors=2, extra_fit_levels=2,
    )
    assert result["success"] is True
    assert result["final_size"] == 656
    assert result["iterations"] == 2


def test_speed_profile_values():
    assert compressors.speed_profile("slow") == {
        "avif_speed": 4, "pngquant_speed": 1, "oxipng_level": 2,
        "search_step": None, "color_step": None,
        "max_iterations": 100,
    }
    default = compressors.speed_profile("default")
    assert default["avif_speed"] == 8
    assert default["pngquant_speed"] == 3
    assert default["oxipng_level"] == 0
    assert default["search_step"] == 3
    assert default["color_step"] == 16
    assert default["max_iterations"] == 40
    fast = compressors.speed_profile("fast")
    assert fast["avif_speed"] == 10
    assert fast["pngquant_speed"] == 5
    assert fast["oxipng_level"] is None
    assert fast["search_step"] == 10
    assert fast["color_step"] == 32
    assert fast["max_iterations"] == 10


def test_make_lossy_encoders_extent_wired_in_fast_modes():
    better = _backend(jpegli=True)
    assert compressors.make_lossy_encoders(better, speed="slow")["JPEG"].extent is None
    assert compressors.make_lossy_encoders(better)["JPEG"].extent is not None
    assert compressors.make_lossy_encoders(better, speed="fast")["JPEG"].extent is not None


def test_make_lossy_encoders_avif_speed_per_profile(monkeypatch):
    calls = []

    def fake_avif_save(*args, **kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(compressors.converters, "avif_save", fake_avif_save)
    backend = _backend()
    for speed, expected in (("slow", 4), ("default", 8), ("fast", 10)):
        calls.clear()
        encoders = compressors.make_lossy_encoders(backend, speed=speed)
        assert encoders["AVIF"].encode(Path("in"), Path("out"), 80) is True
        assert calls[-1]["speed"] == expected


def test_coarse_grid_search_returns_highest_grid_fit(tmp_path):
    def probe_fn(q):
        path = tmp_path / f"p{q}.bin"
        path.write_bytes(b"x")
        return q <= 93, q * 10, path

    best, executed = compressors._coarse_grid_search(probe_fn, 85, 100, step=10)
    assert [q for q, _, _ in executed] == [100, 90, 85]
    assert best[0] == 90
    assert best[1] == 900


def test_coarse_grid_search_handles_partial_and_degenerate_ranges():
    probed = []

    def probe_fn(q):
        probed.append(q)
        return q <= 97, q, None

    best, executed = compressors._coarse_grid_search(probe_fn, 96, 100, step=3)
    assert [q for q, _, _ in executed] == [100, 97, 96]
    assert best[0] == 97

    probed.clear()
    best, executed = compressors._coarse_grid_search(probe_fn, 90, 90)
    assert [q for q, _, _ in executed] == [90]
    assert best[0] == 90


def test_coarse_grid_search_all_fail_returns_none(tmp_path):
    def probe_fn(q):
        path = tmp_path / f"p{q}.bin"
        path.write_bytes(b"x")
        return False, q, path

    best, executed = compressors._coarse_grid_search(probe_fn, 85, 100, step=10)
    assert best is None
    assert [q for q, _, _ in executed] == [100, 90, 85]


def test_iterative_art_optimize_coarse_search_one_round(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 5000)
    out = tmp_path / "out.png"
    probed = []

    def fake_palette(inp, outp, colors, strip_metadata=False, **kwargs):
        probed.append(colors)
        outp.write_bytes(b"p" * (colors * 10))
        return True

    monkeypatch.setattr(compressors, "color_palette_reduce", fake_palette)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 800)
    monkeypatch.setattr(search, "_probe_worker_cap", lambda current: 1)

    result = compressors.iterative_art_optimize(
        src, out, max_size_bytes=2300,
        start_colors=256, min_colors=128, search_step=32,
    )
    assert result["success"] is True
    assert result["final_size"] == 2240
    assert sorted(probed) == [224, 256]
    assert out.read_bytes() == b"p" * (224 * 10)


def test_extent_fit_kept_when_grid_finds_nothing(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 100)
    out = tmp_path / "out.jpg"
    extent_hits = []
    grid_probes = []

    def fake_extent(inp, outp, max_bytes, subsampling, strip_metadata):
        outp.write_bytes(b"x" * 1900)
        extent_hits.append(1)
        return True

    def fake_jpeg_save(inp, outp, quality, **kwargs):
        grid_probes.append(quality)
        outp.write_bytes(b"j" * 3000)
        return True

    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    monkeypatch.setattr(encoders, "jpeg_quality_reduce", fake_jpeg_save)

    encoder = compressors.CodecEncoder(
        "jpeg", fake_jpeg_save, extent=fake_extent, subsampling_supported=True,
    )
    result = compressors.iterative_lossy_optimize(
        src, out, max_size_bytes=2000, encoder=encoder,
        start_quality=100, min_quality=85,
        use_extent_fast_path=True, search_step=10,
    )
    assert result["success"] is True
    assert result["final_size"] == 1900
    assert extent_hits == [1]
    assert grid_probes
    assert out.read_bytes() == b"x" * 1900


def test_extent_fit_overridden_by_higher_grid_fit(tmp_path, monkeypatch):
    src = tmp_path / "src.png"
    _write(src, 100)
    out = tmp_path / "out.jpg"
    extent_hits = []

    def fake_extent(inp, outp, max_bytes, subsampling, strip_metadata):
        outp.write_bytes(b"x" * 1800)
        extent_hits.append(1)
        return True

    def fake_jpeg_save(inp, outp, quality, **kwargs):
        outp.write_bytes(b"j" * (quality * 20))
        return True

    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    monkeypatch.setattr(encoders, "jpeg_quality_reduce", fake_jpeg_save)

    encoder = compressors.CodecEncoder(
        "jpeg", fake_jpeg_save, extent=fake_extent, subsampling_supported=True,
    )
    result = compressors.iterative_lossy_optimize(
        src, out, max_size_bytes=2000, encoder=encoder,
        start_quality=100, min_quality=85,
        use_extent_fast_path=True, search_step=10,
    )
    assert result["success"] is True
    assert result["final_size"] == 2000
    assert extent_hits == []
    assert out.read_bytes() == b"j" * 2000


def test_animated_gif_reduce_builds_global_palette_command(monkeypatch, tmp_path):
    captured = {}
    src = tmp_path / "in.gif"
    src.write_bytes(b"x")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(compressors.utils, "run_capture", fake_run)
    out = tmp_path / "out.gif"
    assert compressors.animated_gif_reduce(src, out, 64, strip_metadata=True) is True
    cmd = captured["cmd"]
    assert cmd[0] == "magick"
    assert "-coalesce" in cmd
    assert cmd[cmd.index("-define") + 1] == "quantize:global-colormap=true"
    assert cmd[cmd.index("-colors") + 1] == "64"
    assert "-strip" in cmd
    assert cmd[-3:] == ["-layers", "Optimize", str(out)]


def test_animated_quality_reduce_builds_quality_command(monkeypatch, tmp_path):
    captured = {}
    src = tmp_path / "in.gif"
    src.write_bytes(b"x")
    out = tmp_path / "out.avif"

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(compressors.utils, "run_capture", fake_run)
    assert compressors.animated_quality_reduce(src, out, 64,
                                               strip_metadata=True) is True
    cmd = captured["cmd"]
    assert cmd[0] == "magick"
    assert "-coalesce" in cmd
    assert cmd[cmd.index("-quality") + 1] == "64"
    assert "-strip" in cmd
    assert cmd[-3:] == ["-layers", "Optimize", str(out)]


def test_animated_quality_reduce_tiff_uses_jpeg_compression(monkeypatch, tmp_path):
    captured = {}
    src = tmp_path / "in.gif"
    src.write_bytes(b"x")
    out = tmp_path / "out.tiff"

    monkeypatch.setattr(
        compressors.utils, "run_capture",
        lambda cmd, **kwargs: captured.update(cmd=cmd)
        or type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    compressors.animated_quality_reduce(src, out, 50)
    assert captured["cmd"][captured["cmd"].index("-compress") + 1] == "JPEG"


def test_animated_gif_reduce_failure_returns_false(monkeypatch, tmp_path):
    def fail(cmd, **kwargs):
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()

    monkeypatch.setattr(compressors.utils, "run_capture", fail)
    src = tmp_path / "in.gif"
    src.write_bytes(b"x")
    assert compressors.animated_gif_reduce(src, tmp_path / "out.gif", 64) is False
    assert compressors.utils.get_last_error() == "boom"


def test_iterative_animated_gif_optimize_fits_at_high_colors(tmp_path, monkeypatch):
    src = tmp_path / "src.gif"
    _write(src, 2000)
    out = tmp_path / "out.gif"
    probes = []

    def fake_reduce(inp, outp, colors, strip_metadata=False):
        size = 500 + colors
        outp.write_bytes(b"p" * size)
        probes.append(colors)
        return True

    monkeypatch.setattr(compressors, "animated_gif_reduce", fake_reduce)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    monkeypatch.setattr(search, "_MAX_PARALLEL", 4)
    monkeypatch.setattr(search, "_probe_worker_cap", lambda current: 1)

    result = compressors.iterative_animated_gif_optimize(
        src, out, max_size_bytes=700,
        pixel_art_mode=True, start_colors=256, min_colors=128,
    )
    assert result["success"] is True
    assert result["final_size"] == 700
    assert result["iterations"] == 1
    assert sorted(probes) == [153, 178, 183, 187, 191, 195,
                              197, 198, 199, 200, 201, 202, 203, 228]
    assert out.read_bytes() == b"p" * 700


def test_iterative_animated_gif_optimize_coarse_grid_one_round(tmp_path, monkeypatch):
    src = tmp_path / "src.gif"
    _write(src, 5000)
    out = tmp_path / "out.gif"
    probed = []

    def fake_reduce(inp, outp, colors, strip_metadata=False):
        probed.append(colors)
        outp.write_bytes(b"p" * (colors * 10))
        return True

    monkeypatch.setattr(compressors, "animated_gif_reduce", fake_reduce)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 800)
    monkeypatch.setattr(search, "_probe_worker_cap", lambda current: 1)

    result = compressors.iterative_animated_gif_optimize(
        src, out, max_size_bytes=2300,
        start_colors=256, min_colors=128, search_step=32,
    )
    assert result["success"] is True
    assert result["final_size"] == 2240
    assert sorted(probed) == [224, 256]
    assert out.read_bytes() == b"p" * (224 * 10)


def test_iterative_animated_gif_optimize_resizes_all_frames_and_fits(tmp_path, monkeypatch):
    src = tmp_path / "src.gif"
    _write(src, 4000)
    out = tmp_path / "out.gif"
    resizes = []

    def fake_reduce(inp, outp, colors, strip_metadata=False):
        outp.write_bytes(b"p" * inp.stat().st_size)
        return True

    def fake_resize(inp, outp, geometry, kernel="Lanczos", nearest=False, strip_metadata=False):
        size = inp.stat().st_size * 50 * 50 // (100 * 100)
        outp.write_bytes(b"p" * size)
        resizes.append((geometry, nearest))
        return True

    monkeypatch.setattr(compressors, "animated_gif_reduce", fake_reduce)
    monkeypatch.setattr(compressors.converters, "gif_resize", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)

    result = compressors.iterative_animated_gif_optimize(
        src, out, max_size_bytes=1000,
        pixel_art_mode=False, start_colors=256, min_colors=128,
    )
    assert result["success"] is True
    assert result["final_size"] == 1000
    assert result["iterations"] == 1
    assert resizes == [("50%", False)]
    assert out.read_bytes() == b"p" * 1000


def test_iterative_animated_gif_optimize_pixel_art_snaps_scale(tmp_path, monkeypatch):
    src = tmp_path / "src.gif"
    _write(src, 5000)
    out = tmp_path / "out.gif"
    resizes = []

    def fake_reduce(inp, outp, colors, strip_metadata=False):
        outp.write_bytes(b"p" * 5000)
        return True

    def fake_resize(inp, outp, geometry, kernel="Lanczos", nearest=False, strip_metadata=False):
        size = inp.stat().st_size * 80 // 100
        outp.write_bytes(b"p" * size)
        resizes.append((geometry, nearest))
        return True

    monkeypatch.setattr(compressors, "animated_gif_reduce", fake_reduce)
    monkeypatch.setattr(compressors.converters, "gif_resize", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 800)

    result = compressors.iterative_animated_gif_optimize(
        src, out, max_size_bytes=1000,
        pixel_art_mode=True, start_colors=256, min_colors=128,
        bail_ratio=1.3,
    )
    assert result["success"] is False
    assert result["iterations"] == 1
    assert resizes == [("50%", True)]
    assert 4000 * 1.3 > 5000


def test_iterative_animated_lossy_optimize_fits_at_high_quality(tmp_path, monkeypatch):
    src = tmp_path / "src.gif"
    _write(src, 2000)
    out = tmp_path / "out.avif"
    probes = []

    def fake_reduce(inp, outp, quality, strip_metadata=False):
        size = 500 + quality * 4
        outp.write_bytes(b"p" * size)
        probes.append(quality)
        return True

    monkeypatch.setattr(compressors, "animated_quality_reduce", fake_reduce)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    result = compressors.iterative_animated_lossy_optimize(
        src, out, max_size_bytes=1000, target_format="AVIF",
        start_quality=100, min_quality=85,
    )
    assert result["success"] is True
    assert result["final_size"] == 900
    assert probes and max(probes) == 100
    assert out.stat().st_size == 900


def test_animated_ssim_parses_ffmpeg_output(monkeypatch, tmp_path):
    registry = []
    calls = []

    class R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_gif_resize(inp, outp, geometry, kernel="Lanczos", nearest=False,
                        strip_metadata=False):
        calls.append(("resize", geometry))
        outp.write_bytes(b"g")
        return True

    def fake_magick(cmd):
        calls.append(("magick", cmd))
        for arg in list(cmd):
            if "_%03d" in str(arg):
                base = str(arg).replace("%03d", "")
                for i in range(3):
                    Path(base + f"{i:03d}.png").write_bytes(b"f")
        return R()

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "",
                              "stderr": "All:0.902363"})()

    def fake_width(p):
        return 600 if p.read_bytes() == b"y" else 1200

    def fake_height(p):
        return 400 if p.read_bytes() == b"y" else 800

    monkeypatch.setattr(search, "_probe_width", fake_width)
    monkeypatch.setattr(compressors.converters, "get_image_height", fake_height)
    monkeypatch.setattr(compressors.converters, "gif_resize", fake_gif_resize)
    monkeypatch.setattr(compressors.converters, "_run_magick", fake_magick)
    monkeypatch.setattr(compressors.utils, "run_capture", fake_run)
    ref = tmp_path / "a.gif"
    cand = tmp_path / "b.gif"
    ref.write_bytes(b"x")
    cand.write_bytes(b"y")
    assert compressors._animated_ssim_vs_reference(ref, cand, registry) == 0.902363
    assert ("resize", "600x400!") in calls
    assert sum(1 for c in calls if c[0] == "magick") == 2
    assert all("-coalesce" in c[1] for c in calls if c[0] == "magick")
    assert registry


def test_animated_ssim_returns_none_on_failure(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "error"})()

    def fail_magick(cmd):
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "boom"})()

    ref = tmp_path / "a.gif"
    cand = tmp_path / "b.gif"
    ref.write_bytes(b"x")
    cand.write_bytes(b"y")

    def fake_resize(inp, outp, geometry, **kwargs):
        outp.write_bytes(b"g")
        return True

    monkeypatch.setattr(search, "_probe_width", lambda p: 600)
    monkeypatch.setattr(compressors.converters, "get_image_height", lambda p: 400)
    monkeypatch.setattr(compressors.converters, "gif_resize", fake_resize)
    monkeypatch.setattr(compressors.converters, "_run_magick", fail_magick)
    assert compressors._animated_ssim_vs_reference(ref, cand, []) is None
    class Ok:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(compressors.converters, "_run_magick", lambda cmd: Ok())
    monkeypatch.setattr(compressors.utils, "run_capture", fake_run)
    assert compressors._animated_ssim_vs_reference(ref, cand, []) is None


def test_rank_animated_candidates_prefers_ssim_then_resolution(monkeypatch, tmp_path):
    a = tmp_path / "a.gif"
    b = tmp_path / "b.gif"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 1000, "colors": 128},
        {"path": b, "size": 1900, "colors": 256},
    ]
    monkeypatch.setattr(ssim, "_animated_chained_reference",
                        lambda ref, cands, reg: ref)

    ssims = {str(a): 0.90, str(b): 0.97}
    monkeypatch.setattr(ssim, "_animated_ssim_vs_reference",
                        lambda ref, cand, reg, at_original=False: ssims.get(str(cand)))
    monkeypatch.setattr(search, "_probe_width", lambda p: 1000)
    winner = compressors._rank_animated_candidates(candidates, tmp_path / "ref.gif", [])
    assert winner is candidates[1]

    ssims = {str(a): 0.90, str(b): 0.90}
    widths = {str(a): 2000, str(b): 1000}
    monkeypatch.setattr(search, "_probe_width",
                        lambda p: widths.get(str(p), 0))
    winner = compressors._rank_animated_candidates(candidates, tmp_path / "ref.gif", [])
    assert winner is candidates[0]


def test_rank_animated_candidates_falls_back_to_colors_without_ssim(monkeypatch, tmp_path):
    a = tmp_path / "a.gif"
    b = tmp_path / "b.gif"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 500, "colors": 128},
        {"path": b, "size": 1900, "colors": 256},
    ]
    monkeypatch.setattr(search, "_probe_width",
                        lambda p: {str(a): 1200, str(b): 2400}.get(str(p), 0))
    winner = compressors._rank_animated_candidates(candidates, Path("ref.gif"), [],
                                                   ssim=False)
    assert winner is candidates[1]


def test_rank_animated_quality_candidates_without_ssim(monkeypatch, tmp_path):
    a = tmp_path / "a.avif"
    b = tmp_path / "b.avif"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    candidates = [
        {"path": a, "size": 900, "quality": 90},
        {"path": b, "size": 800, "quality": 95},
    ]
    monkeypatch.setattr(search, "_probe_width", lambda p: 1000)
    winner = compressors._rank_animated_candidates(candidates, Path("ref.avif"), [],
                                                   ssim=False)
    assert winner is candidates[1]


def test_slowest_animated_gif_optimize_explores_frontier(tmp_path, monkeypatch):
    src = tmp_path / "src.gif"
    _write(src, 5000)
    out = tmp_path / "out.gif"
    picked = []

    def fake_reduce(inp, outp, colors, strip_metadata=False):
        size = 400 + colors
        outp.write_bytes(b"p" * size)
        return True

    def fake_resize(inp, outp, geometry, kernel="Lanczos", nearest=False, strip_metadata=False):
        size = inp.stat().st_size * 50 * 50 // (100 * 100)
        outp.write_bytes(b"p" * size)
        return True

    monkeypatch.setattr(compressors, "animated_gif_reduce", fake_reduce)
    monkeypatch.setattr(compressors.converters, "gif_resize", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 800)
    monkeypatch.setattr(compressors, "_rank_animated_candidates",
                        lambda cands, ref, reg, ssim=True, at_original=False: picked.append(cands) or cands[0])

    result = compressors.slowest_animated_gif_optimize(
        src, out, max_size_bytes=1000, pixel_art_mode=False,
        start_colors=256, min_colors=2, extra_fit_levels=2,
    )
    assert result["success"] is True
    assert result["final_size"] == 656
    assert result["iterations"] == 2
    assert picked and len(picked[0]) == 2


def test_slowest_animated_gif_optimize_restores_best_when_no_fit(tmp_path, monkeypatch):
    src = tmp_path / "src.gif"
    _write(src, 5000)
    out = tmp_path / "out.gif"

    def fake_reduce(inp, outp, colors, strip_metadata=False):
        outp.write_bytes(b"p" * 9000)
        return True

    def fake_resize(inp, outp, geometry, kernel="Lanczos", nearest=False, strip_metadata=False):
        outp.write_bytes(b"p" * 5000)
        return True

    monkeypatch.setattr(compressors, "animated_gif_reduce", fake_reduce)
    monkeypatch.setattr(compressors.converters, "gif_resize", fake_resize)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 800)

    result = compressors.slowest_animated_gif_optimize(
        src, out, max_size_bytes=1000, pixel_art_mode=False,
        start_colors=256, min_colors=2, extra_fit_levels=2,
        bail_ratio=1.3,
    )
    assert result["success"] is False
    assert out.read_bytes() == b"p" * 9000


def _quality_encoder():
    def encode(inp, outp, quality, **kwargs):
        outp.write_bytes(b"x" * max(1, quality))
        return True
    return compressors.CodecEncoder("fake", encode, suffix=".jpg")


def _patch_size_and_ssim(monkeypatch, scores=None):
    monkeypatch.setattr(compressors.converters, "get_image_filesize",
                        lambda p: p.stat().st_size)
    def fake_ssim(reference, candidate, registry):
        if scores is not None:
            return scores.get(candidate.stat().st_size)
        return candidate.stat().st_size / 100
    monkeypatch.setattr(compressors, "_ssim_vs_reference", fake_ssim)


def test_calibrate_ssim_target_finds_lowest_fitting_quality(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    _patch_size_and_ssim(monkeypatch)
    budget = compressors.calibrate_ssim_target(
        src, _quality_encoder(), threshold=0.9, subsampling="4:2:0",
        strip_metadata=False, min_quality=1, max_quality=100)
    assert budget == 90


def test_calibrate_ssim_target_unreachable_returns_none(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    _patch_size_and_ssim(monkeypatch)
    budget = compressors.calibrate_ssim_target(
        src, _quality_encoder(), threshold=1.01, subsampling="4:2:0",
        strip_metadata=False, min_quality=1, max_quality=100)
    assert budget is None


def test_calibrate_ssim_target_encoder_failure_returns_none(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    _patch_size_and_ssim(monkeypatch)
    enc = compressors.CodecEncoder("dead", lambda *a, **k: False, suffix=".jpg")
    budget = compressors.calibrate_ssim_target(
        src, enc, threshold=0.9, subsampling="4:2:0",
        strip_metadata=False, min_quality=1, max_quality=100)
    assert budget is None


def test_calibrate_ssim_target_respects_min_quality_floor(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    _patch_size_and_ssim(monkeypatch)
    budget = compressors.calibrate_ssim_target(
        src, _quality_encoder(), threshold=0.9, subsampling="4:2:0",
        strip_metadata=False, min_quality=95, max_quality=100)
    assert budget == 95


def test_calibrate_ssim_target_noise_pushes_budget_up(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    _patch_size_and_ssim(monkeypatch, scores={100: 1.0})
    budget = compressors.calibrate_ssim_target(
        src, _quality_encoder(), threshold=0.9, subsampling="4:2:0",
        strip_metadata=False, min_quality=1, max_quality=100)
    assert budget == 100


def test_calibrate_ssim_target_cleans_up_probes(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    _patch_size_and_ssim(monkeypatch)
    created = []
    real_new_probe = compressors._new_probe
    def tracking_new_probe(suffix, registry):
        p = real_new_probe(suffix, registry)
        created.append(p)
        return p
    monkeypatch.setattr(compressors, "_new_probe", tracking_new_probe)
    compressors.calibrate_ssim_target(
        src, _quality_encoder(), threshold=0.9, subsampling="4:2:0",
        strip_metadata=False, min_quality=1, max_quality=100)
    assert created and all(not p.exists() for p in created)


def test_lossy_strategy_forwards_ssim_target_ranking_knobs(monkeypatch):
    captured = {}
    def fake_rank(cands, ref, reg, ssim=True, at_original=False, floor=None):
        captured["at_original"] = at_original
        captured["floor"] = floor
        return cands[0]
    monkeypatch.setattr(compressors, "_rank_lossy_candidates", fake_rank)
    enc = compressors.CodecEncoder("fake", lambda *a, **k: True, suffix=".jpg")
    strategy = compressors._lossy_strategy(
        [enc], "4:2:0", False, 1000, False, None, True, 90, 75, True, True,
        ssim_floor=0.95)
    assert strategy.select_winner is not None
    strategy.select_winner([{"path": "p", "size": 1}], "ref", [])
    assert captured == {"at_original": True, "floor": 0.95}


def test_slowest_frontier_ranks_at_original_resolution(monkeypatch, tmp_path):
    captured = {}

    def fake_rank(cands, ref, reg, ssim=True, at_original=False, floor=None):
        captured["at_original"] = at_original
        return cands[0]

    monkeypatch.setattr(compressors, "_rank_lossy_candidates", fake_rank)
    encoder = compressors.CodecEncoder(
        "jpeg", encode=lambda i, o, q, subsampling="4:2:0",
        strip_metadata=False: True, suffix=".jpg", subsampling_supported=True)
    strategy = compressors._lossy_strategy(
        [encoder], "4:2:0", False, 1000, use_extent_fast_path=False,
        search_step=None, frontier=True, frontier_ceiling=None, deep_step=75,
        ssim=True, compare_subsampling=True)
    cands = [{"path": tmp_path / "a.jpg", "size": 900, "quality": 90,
              "subsampling": "4:2:0", "encoder": "jpeg"}]
    (tmp_path / "a.jpg").write_bytes(b"a")
    strategy.select_winner(cands, tmp_path / "src.png", [])
    assert captured["at_original"] is True


def test_rank_art_candidates_at_original_skips_chained_reference(monkeypatch, tmp_path):
    chained = {"called": False}

    def fake_chain(ref, cands, reg):
        chained["called"] = True
        return ref

    scores_own = {}
    scores_orig = {}
    monkeypatch.setattr(ssim, "_chained_reference", fake_chain)
    monkeypatch.setattr(ssim.search, "_probe_width", lambda p: 0)

    def fake_ssim(ref, cand, reg, at_original=False):
        return (scores_own if not at_original else scores_orig).get(str(cand))

    monkeypatch.setattr(ssim, "_ssim_vs_reference", fake_ssim)
    candidates = [
        {"path": tmp_path / "a.bin", "size": 1000, "colors": 256},
        {"path": tmp_path / "b.bin", "size": 1000, "colors": 256},
    ]
    for c in candidates:
        c["path"].write_bytes(b"x")
    scores_own[str(candidates[0]["path"])] = 0.80
    scores_own[str(candidates[1]["path"])] = 0.95
    scores_orig[str(candidates[0]["path"])] = 0.80
    scores_orig[str(candidates[1]["path"])] = 0.69
    winner = compressors._rank_art_candidates(candidates, tmp_path / "src.png", [],
                                              at_original=True)
    assert winner is candidates[0]
    assert chained["called"] is False
    winner_own = compressors._rank_art_candidates(candidates, tmp_path / "src.png", [])
    assert winner_own is candidates[1]
    assert chained["called"] is True


def test_slow_only_strategy_wires_extent_rival(monkeypatch, tmp_path):
    encoder = compressors.CodecEncoder(
        "jpeg", encode=lambda i, o, q, subsampling="4:2:0",
        strip_metadata=False: True, suffix=".jpg", subsampling_supported=True,
        extent=lambda i, o, m, s, st: True)
    slow = compressors._lossy_strategy(
        [encoder], "4:2:0", False, 1000, use_extent_fast_path=True,
        search_step=None, frontier=False, frontier_ceiling=None, deep_step=75,
        ssim=True, compare_subsampling=False)
    assert slow.extent_compare is True
    grid = compressors._lossy_strategy(
        [encoder], "4:2:0", False, 1000, use_extent_fast_path=True,
        search_step=10, frontier=False, frontier_ceiling=None, deep_step=75,
        ssim=True, compare_subsampling=False)
    assert grid.extent_compare is False
    no_extent = compressors._lossy_strategy(
        [encoder], "4:2:0", False, 1000, use_extent_fast_path=False,
        search_step=None, frontier=False, frontier_ceiling=None, deep_step=75,
        ssim=True, compare_subsampling=False)
    assert no_extent.extent_compare is False


def test_extent_compare_ships_fidelity_winner_at_first_fit(monkeypatch, tmp_path):
    picked = {}

    (tmp_path / "src.bin").write_bytes(b"s" * 5000)

    def fake_pick(a, b, ref, reg):
        picked["winner"] = b
        return b

    monkeypatch.setattr(search, "fidelity_pick", fake_pick)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    monkeypatch.setattr(compressors.converters, "get_image_height", lambda p: 700)
    result = compressors.iterative_lossy_optimize(
        tmp_path / "src.bin", tmp_path / "out.bin", max_size_bytes=1000,
        encoder=compressors.CodecEncoder(
            "jpeg",
            encode=lambda i, o, q, subsampling="4:2:0",
            strip_metadata=False: (Path(o).write_bytes(b"f" * 1000), True)[1],
            suffix=".jpg", subsampling_supported=True,
            extent=lambda i, o, m, s, st: (Path(o).write_bytes(b"e"), (Path(o), 900))[1]),
        start_quality=100, min_quality=85,
        use_extent_fast_path=True, max_iterations=10,
        filter="Lanczos")
    assert picked["winner"][1] == 1000
    assert result["success"] is True
    assert (tmp_path / "out.bin").read_bytes() == b"f" * 1000


def test_extent_compare_ships_extent_when_search_fails(monkeypatch, tmp_path):
    picked = {}

    (tmp_path / "src.bin").write_bytes(b"s" * 5000)

    def fake_pick(a, b, ref, reg):
        picked["called"] = True
        return a

    monkeypatch.setattr(search, "fidelity_pick", fake_pick)
    monkeypatch.setattr(compressors.converters, "get_image_width", lambda p: 1000)
    monkeypatch.setattr(compressors.converters, "get_image_height", lambda p: 700)

    monkeypatch.setattr(search, "fidelity_pick", fake_pick)
    result = compressors.iterative_lossy_optimize(
        tmp_path / "src.bin", tmp_path / "out.bin", max_size_bytes=1000,
        encoder=compressors.CodecEncoder(
            "jpeg",
            encode=lambda i, o, q, subsampling="4:2:0", strip_metadata=False: False,
            suffix=".jpg", subsampling_supported=True,
            extent=lambda i, o, m, s, st: (Path(o).write_bytes(b"e" * 900), (Path(o), 900))[1]),
        start_quality=100, min_quality=85,
        use_extent_fast_path=True, max_iterations=10,
        filter="Lanczos")
    assert "called" not in picked
    assert result["success"] is True
    assert (tmp_path / "out.bin").read_bytes() == b"e" * 900


def _colors_reducer(sizes_by_colors=None):
    def reduce_(inp, outp, colors, strip_metadata=False):
        if sizes_by_colors is not None and colors not in sizes_by_colors:
            return False
        outp.write_bytes(b"x" * (colors * 10))
        return True
    return reduce_


def _patch_palette_ssim(monkeypatch, scores=None):
    monkeypatch.setattr(compressors.converters, "get_image_filesize",
                        lambda p: p.stat().st_size)
    def fake_ssim(reference, candidate, registry):
        if scores is not None:
            return scores.get(candidate.stat().st_size)
        return candidate.stat().st_size / 2560
    monkeypatch.setattr(compressors, "_ssim_vs_reference", fake_ssim)


def test_calibrate_palette_ssim_target_finds_lowest_fitting_colors(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    monkeypatch.setattr(compressors, "color_palette_reduce", _colors_reducer())
    _patch_palette_ssim(monkeypatch)
    calibrated = compressors.calibrate_palette_ssim_target(
        src, threshold=0.9, strip_metadata=False, min_colors=2, max_colors=256)
    assert calibrated == (2310, 231)


def test_calibrate_palette_ssim_target_unreachable_returns_none(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    monkeypatch.setattr(compressors, "color_palette_reduce", _colors_reducer())
    _patch_palette_ssim(monkeypatch)
    budget = compressors.calibrate_palette_ssim_target(
        src, threshold=1.01, strip_metadata=False, min_colors=2, max_colors=256)
    assert budget is None


def test_calibrate_palette_ssim_target_reducer_failure_returns_none(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    monkeypatch.setattr(compressors, "color_palette_reduce",
                        lambda *a, **k: False)
    _patch_palette_ssim(monkeypatch)
    budget = compressors.calibrate_palette_ssim_target(
        src, threshold=0.9, strip_metadata=False, min_colors=2, max_colors=256)
    assert budget is None


def test_calibrate_palette_ssim_target_respects_min_colors(monkeypatch, tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(b"r")
    monkeypatch.setattr(compressors, "color_palette_reduce", _colors_reducer())
    _patch_palette_ssim(monkeypatch)
    calibrated = compressors.calibrate_palette_ssim_target(
        src, threshold=0.5, strip_metadata=False, min_colors=240, max_colors=256)
    assert calibrated == (2400, 240)


def test_art_strategy_forwards_ssim_floor(monkeypatch):
    captured = {}
    def fake_rank(cands, ref, reg, ssim=True, by_colors=False,
                  width_first=False, at_original=False, floor=None):
        captured["floor"] = floor
        captured["by_colors"] = by_colors
        return cands[0]
    monkeypatch.setattr(compressors, "_rank_art_candidates", fake_rank)
    strategy = compressors._art_strategy(
        False, False, 1000, None, True, None, 75, True, False, False, 256,
        "Lanczos", ssim_floor=0.95)
    assert strategy.select_winner is not None
    strategy.select_winner([{"path": "p", "size": 1}], "ref", [])
    assert captured == {"floor": 0.95, "by_colors": False}
    strategy = compressors._art_strategy(
        False, False, 1000, None, True, None, 75, True, True, False, 256,
        "Lanczos", ssim_floor=0.95)
    strategy.select_winner([{"path": "p", "size": 1}], "ref", [])
    assert captured == {"floor": None, "by_colors": True}


def test_rank_art_candidates_floor_picks_smallest_passing(monkeypatch, tmp_path):
    registry: list[Path] = []
    pa = tmp_path / "a.png"
    pa.write_bytes(b"a" * 100)
    pb = tmp_path / "b.png"
    pb.write_bytes(b"b" * 200)
    pc = tmp_path / "c.png"
    pc.write_bytes(b"c" * 300)
    cands = [
        {"path": pa, "size": 100, "colors": 32},
        {"path": pb, "size": 200, "colors": 64},
        {"path": pc, "size": 300, "colors": 128},
    ]
    scores = {100: 0.94, 200: 0.96, 300: 0.98}
    def fake_ssim(reference, candidate, registry, at_original=False):
        return scores[candidate.stat().st_size]
    monkeypatch.setattr(ssim, "_ssim_vs_reference", fake_ssim)
    monkeypatch.setattr(ssim.search, "_probe_width", lambda p: 0)
    winner = compressors._rank_art_candidates(
        cands, tmp_path / "ref.png", registry, ssim=True, at_original=True,
        floor=0.95)
    assert winner["size"] == 200


def test_rank_art_candidates_floor_unreachable_falls_back_to_best(monkeypatch, tmp_path):
    registry: list[Path] = []
    pa = tmp_path / "a.png"
    pa.write_bytes(b"a" * 100)
    pb = tmp_path / "b.png"
    pb.write_bytes(b"b" * 200)
    cands = [
        {"path": pa, "size": 100, "colors": 32},
        {"path": pb, "size": 200, "colors": 64},
    ]
    scores = {100: 0.90, 200: 0.93}
    def fake_ssim(reference, candidate, registry, at_original=False):
        return scores[candidate.stat().st_size]
    monkeypatch.setattr(ssim, "_ssim_vs_reference", fake_ssim)
    winner = compressors._rank_art_candidates(
        cands, tmp_path / "ref.png", registry, ssim=True, at_original=True,
        floor=0.95)
    assert winner["size"] == 200
