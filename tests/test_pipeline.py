import os
import tempfile
from pathlib import Path

import pytest

from core import pipeline
from core.cli import create_parser


def parse(*argv):
    return create_parser().parse_args(list(argv))


@pytest.fixture(autouse=True)
def _cleanup_pipeline_temps():
    yield
    pipeline._cleanup_temps()


def test_resolve_image_mode():
    assert pipeline.resolve_image_mode(parse()) == "photo"
    assert pipeline.resolve_image_mode(parse("-p")) == "photo"
    assert pipeline.resolve_image_mode(parse("-d")) == "drawing"
    assert pipeline.resolve_image_mode(parse("-x")) == "pixel_art"
    assert pipeline.resolve_image_mode(parse("-d", "-x")) == "pixel_art"


def test_validate_flags_integer_scaling_conflicts_with_dimensions():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("-I", "50", "-w", "100"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("-I", "50", "-H", "100"))


def test_validate_flags_quality_below_min_quality():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("-q", "50", "-Q", "80"))


def test_validate_flags_quality_out_of_range():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("-q", "0"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("-q", "300"))


def test_validate_flags_conversion_only_requires_format():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--conversion-only"))


def test_validate_flags_accepts_valid_combinations():
    pipeline.validate_flags(parse())
    pipeline.validate_flags(parse("-q", "90", "-Q", "80", "-s", "2m"))
    pipeline.validate_flags(parse("-I", "50"))
    pipeline.validate_flags(parse("--conversion-only", "-F", "jpeg"))
    pipeline.validate_flags(parse("--slow"))
    pipeline.validate_flags(parse("--fast"))
    pipeline.validate_flags(parse("--slowest"))


def test_validate_flags_rejects_zero_or_negative_size():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("-s", "0"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("-s", "0k"))


def test_validate_flags_speed_modes_conflict():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--slow", "--fast"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--fast", "--slowest"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--slow", "--slowest"))


def test_resolve_processing_toggles_defaults():
    assert pipeline.resolve_processing_toggles(parse(), False) == pipeline._Toggles(
        resize=True, pngquant=True, iterative=True,
    )


def test_resolve_processing_toggles_only_flags():
    assert pipeline.resolve_processing_toggles(parse("--only-resize"), False) == \
        pipeline._Toggles(resize=True, pngquant=False, iterative=False)
    assert pipeline.resolve_processing_toggles(parse("--only-pngquant"), False) == \
        pipeline._Toggles(resize=False, pngquant=True, iterative=False)
    assert pipeline.resolve_processing_toggles(parse("--only-iterative"), False) == \
        pipeline._Toggles(resize=False, pngquant=False, iterative=True)


def test_resolve_processing_toggles_skip_flags():
    assert pipeline.resolve_processing_toggles(parse("--skip-resize"), False) == \
        pipeline._Toggles(resize=False, pngquant=True, iterative=True)
    assert pipeline.resolve_processing_toggles(parse("--skip-iterative"), False) == \
        pipeline._Toggles(resize=True, pngquant=True, iterative=False)


def test_resolve_processing_toggles_raw_and_conversion_only():
    assert pipeline.resolve_processing_toggles(parse("--raw"), False) == \
        pipeline._Toggles(resize=False, pngquant=False, iterative=False)
    assert pipeline.resolve_processing_toggles(parse("--conversion-only", "-F", "jpeg"), False) == \
        pipeline._Toggles(resize=False, pngquant=False, iterative=False)


def test_resolve_processing_toggles_upscale_disables_compression():
    assert pipeline.resolve_processing_toggles(parse("-u"), True) == \
        pipeline._Toggles(resize=True, pngquant=False, iterative=False)


def test_resolve_processing_toggles_upscale_with_manual_skipping():
    assert pipeline.resolve_processing_toggles(parse("-u", "-m"), True) == \
        pipeline._Toggles(resize=True, pngquant=True, iterative=True)


@pytest.fixture
def measurements(monkeypatch):
    def fake_measure(image_path):
        return {"width": 100, "height": 100, "format": "PNG", "size": 100}

    monkeypatch.setattr(pipeline, "measure_image", fake_measure)
    return fake_measure


def test_check_bypass_raw_returns_true_without_measuring(monkeypatch):
    def should_not_be_called(path):
        raise AssertionError("measure_image should not be called for --raw")

    monkeypatch.setattr(pipeline, "measure_image", should_not_be_called)
    assert pipeline.check_bypass(Path("/nonexistent"), pipeline._build_config(parse("-r"))) is True


def test_check_bypass_under_size_same_format(tmp_path, measurements):
    img = tmp_path / "x.png"
    img.write_bytes(b"x")
    assert pipeline.check_bypass(img, pipeline._build_config(parse("-s", "1m"))) is True


def test_check_bypass_over_size(measurements):
    def big_measure(image_path):
        return {"width": 100, "height": 100, "format": "PNG", "size": 3 * 1024 * 1024}

    pipeline.measure_image = big_measure
    try:
        assert pipeline.check_bypass(Path("x.png"), pipeline._build_config(parse("-s", "1m"))) is False
    finally:
        pipeline.measure_image = measurements


def test_check_bypass_different_target_format(measurements):
    assert pipeline.check_bypass(Path("x.png"), pipeline._build_config(parse("-s", "1", "-F", "jpeg"))) is False


def test_check_bypass_resize_requested(measurements):
    assert pipeline.check_bypass(Path("x.png"), pipeline._build_config(parse("-s", "1", "-w", "50"))) is False
    assert pipeline.check_bypass(Path("x.png"), pipeline._build_config(parse("-s", "1", "-I", "50"))) is False


def test_check_bypass_upscale_needed(measurements):
    def upscale_measure(image_path):
        return {"width": 100, "height": 100, "format": "PNG", "size": 100}

    pipeline.measure_image = upscale_measure
    try:
        assert pipeline.check_bypass(Path("x.png"), pipeline._build_config(parse("-s", "1", "-w", "500"))) is False
    finally:
        pipeline.measure_image = measurements


def test_build_config_target_format_normalization():
    cfg = pipeline._build_config(parse("-F", "jpg"))
    assert cfg.format == "JPEG"
    cfg = pipeline._build_config(parse("-F", "png"))
    assert cfg.format == "PNG"


def test_build_config_defaults():
    cfg = pipeline._build_config(parse())
    assert cfg.img_mode == "photo"
    assert cfg.max_size_bytes == 8 * 1024 * 1024
    assert cfg.filter == "Lanczos"
    assert cfg.subsampling == "4:2:0"
    assert cfg.nn_scaler is None


def test_build_config_relative_default_target():
    assert pipeline._build_config(parse(), measured_size=1_000_000).max_size_bytes \
        == 600_000
    assert pipeline._build_config(parse(), measured_size=2 * 1024 * 1024).max_size_bytes \
        == round(2 * 1024 * 1024 * 0.6)
    assert pipeline._build_config(parse(), measured_size=1).max_size_bytes == 1


def test_build_config_relative_default_target_per_format():
    assert pipeline._build_config(parse(), measured_size=1_000_000,
                                  target_format="JPEG").max_size_bytes == 420_000
    assert pipeline._build_config(parse(), measured_size=1_000_000,
                                  target_format="AVIF").max_size_bytes == 310_000
    assert pipeline._build_config(parse(), measured_size=1_000_000,
                                  target_format="TIFF").max_size_bytes == 50_000
    for fmt in ("PNG", "GIF", "BMP", "VIPS"):
        assert pipeline._build_config(parse(), measured_size=1_000_000,
                                      target_format=fmt).max_size_bytes \
            == 600_000
    assert pipeline._build_config(parse("-s", "1"), measured_size=1_000_000,
                                  target_format="AVIF").max_size_bytes == 1024


def test_build_config_explicit_size_overrides_relative():
    assert pipeline._build_config(parse("-s", "1"), measured_size=1_000_000).max_size_bytes \
        == 1024
    assert pipeline._build_config(parse("-s", "2m"), measured_size=1_000_000).max_size_bytes \
        == 2 * 1024 * 1024


def test_check_bypass_relative_target_never_fits(measurements):
    img = Path("x.png")
    cfg = pipeline._build_config(parse(), measured_size=100)
    assert cfg.max_size_bytes == 60
    assert pipeline.check_bypass(img, cfg) is False


def test_build_config_size_units():
    assert pipeline._build_config(parse("-s", "512")).max_size_bytes == 512 * 1024
    assert pipeline._build_config(parse("-s", "512k")).max_size_bytes == 512 * 1024
    assert pipeline._build_config(parse("-s", "2m")).max_size_bytes == 2 * 1024 * 1024
    assert pipeline._build_config(parse("-s", "2M")).max_size_bytes == 2 * 1024 * 1024
    assert pipeline._build_config(parse("-s", "1")).max_size_bytes == 1024


def test_build_config_filter_and_nn_scaler():
    assert pipeline._build_config(parse("-x")).filter == "Point"
    assert pipeline._build_config(parse("-f", "Mitchell")).filter == "Mitchell"
    assert pipeline._build_config(parse("-d")).filter == "Bicubic"
    assert pipeline._build_config(parse("-d", "-f", "Mitchell")).filter == "Mitchell"
    assert pipeline._build_config(parse("--waifu2x")).nn_scaler == "waifu2x"
    assert pipeline._build_config(parse("-f", "realesrgan")).nn_scaler == "realesrgan"
    assert pipeline._build_config(parse("-x", "--waifu2x")).filter == "Lanczos"


def test_build_config_slowest_flag():
    assert pipeline._build_config(parse()).slowest is False
    assert pipeline._build_config(parse("--slow")).slowest is False
    assert pipeline._build_config(parse("--fast")).slowest is False
    assert pipeline._build_config(parse("--slowest")).slowest is True


def test_build_config_speed_mapping():
    assert pipeline._build_config(parse()).speed == "default"
    assert pipeline._build_config(parse("--slow")).speed == "slow"
    assert pipeline._build_config(parse("--fast")).speed == "fast"
    assert pipeline._build_config(parse("--slowest")).speed == "slow"


def test_build_config_search_settings():
    cfg = pipeline._build_config(parse())
    assert cfg.search.frontier is False
    assert cfg.search.search_step == 3 and cfg.search.color_step == 16
    assert cfg.search.max_iterations == 40
    assert cfg.search.avif_speed == 8 and cfg.search.pngquant_speed == 3
    assert cfg.search.oxipng_level == 0
    assert cfg.search.min_quality_floor == 85 and cfg.search.min_colors_floor == 128

    cfg = pipeline._build_config(parse("--slow"))
    assert cfg.search.search_step is None and cfg.search.color_step is None
    assert cfg.search.max_iterations == 100
    assert cfg.search.avif_speed == 4 and cfg.search.pngquant_speed == 1
    assert cfg.search.oxipng_level == 2

    cfg = pipeline._build_config(parse("--slowest"))
    assert cfg.search.frontier is True
    assert cfg.search.ssim is True
    assert cfg.search.compare_subsampling is True
    assert cfg.search.prefer_software is True
    assert cfg.search.min_quality_floor == 30 and cfg.search.min_colors_floor == 2
    assert cfg.search.max_iterations == 200

    cfg = pipeline._build_config(parse("--fast", "--search-step", "1"))
    assert cfg.search.search_step == 1
    cfg = pipeline._build_config(parse("--slowest", "--no-ssim"))
    assert cfg.search.ssim is False
    cfg = pipeline._build_config(parse("--slowest", "--no-frontier"))
    assert cfg.search.frontier is False


def test_build_config_image_mode_settings():
    cfg = pipeline._build_config(parse())
    assert (cfg.dithering, cfg.nearest, cfg.snap) == (True, False, False)
    assert cfg.realesrgan_model == "photo"
    cfg = pipeline._build_config(parse("-d"))
    assert (cfg.dithering, cfg.nearest, cfg.snap) == (False, False, False)
    assert cfg.realesrgan_model == "drawing"
    assert cfg.filter == "Bicubic"
    cfg = pipeline._build_config(parse("-x"))
    assert (cfg.dithering, cfg.nearest, cfg.snap) == (False, True, True)
    assert cfg.filter == "Point"


def test_build_config_image_mode_flag_overrides():
    cfg = pipeline._build_config(parse("-d", "--dithering"))
    assert cfg.dithering is True and cfg.realesrgan_model == "drawing"
    cfg = pipeline._build_config(parse("-x", "--no-nearest"))
    assert cfg.nearest is False and cfg.snap is True and cfg.filter == "Point"
    cfg = pipeline._build_config(parse("-x", "--no-snap"))
    assert cfg.snap is False and cfg.nearest is True and cfg.filter == "Point"
    cfg = pipeline._build_config(parse("--nearest"))
    assert cfg.nearest is True and cfg.filter == "Point"
    cfg = pipeline._build_config(parse("--realesrgan-model", "drawing"))
    assert cfg.realesrgan_model == "drawing"


def _backend_stub(**overrides):
    from core.backend import BackendInfo
    fields = {"os_type": "X11", "vips": False, "magick": True, "pngquant": False,
                  "realesrgan": False, "waifu2x": False, "jpegli": False, "mozjpeg": False,
                  "avifenc": False, "oxipng": False}
    fields.update(overrides)
    return BackendInfo(**fields)


def _iterative_setup(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * (5 * 1024 * 1024))
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 5 * 1024 * 1024 if p == img else 0)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    calls = {}
    return img, calls


def test_run_iterative_phase_lossy_target_uses_lossy_loop(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["encoder"] = encoder.name
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["encoder"] == "jpeg"


def test_run_iterative_phase_png_target_uses_palette_loop(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_art(inp, outp, max_size, **kwargs):
        calls["art"] = True
        outp.write_bytes(b"p")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_art_optimize", fake_art)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls.get("art") is True
    assert "iterative_lossy_optimize" not in calls


def test_run_iterative_phase_pixel_art_knobs_follow_flags(monkeypatch, tmp_path):
    img, _ = _iterative_setup(monkeypatch, tmp_path)
    calls = {}

    def fake_art(inp, outp, max_size, **kwargs):
        calls["art"] = kwargs
        outp.write_bytes(b"p")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_art_optimize", fake_art)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}

    def run(argv):
        calls.clear()
        pipeline.run_iterative_phase(
            img, pipeline._build_config(parse(*argv)), measurements,
            pipeline._Toggles(resize=True, pngquant=False, iterative=True),
        )
        return calls["art"]

    x_kwargs = run(("-x", "-s", "2m"))
    assert x_kwargs["pixel_art_mode"] is True
    assert x_kwargs["pixel_art_snap"] is True
    manual = run(("--no-dithering", "--nearest", "--snap", "-s", "2m"))
    assert (manual["pixel_art_mode"], manual["pixel_art_snap"]) == \
        (x_kwargs["pixel_art_mode"], x_kwargs["pixel_art_snap"])
    assert run(("-x", "--no-snap", "-s", "2m"))["pixel_art_snap"] is False
    assert run(("-s", "2m"))["pixel_art_mode"] is False


def test_run_iterative_phase_webp_target_uses_webp_encoder(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["encoder"] = encoder.name
        outp.write_bytes(b"w")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-F", "webp", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["encoder"] == "webp"


def test_run_iterative_phase_slowest_dispatches_slowest_search(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_best(inp, outp, max_size, encoders, **kwargs):
        calls["slowest"] = True
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "slowest_lossy_optimize", fake_best)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("--slowest", "-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls.get("slowest") is True


@pytest.mark.parametrize("fmt", ["jxl", "jp2", "heic", "tiff"])
def test_run_iterative_phase_quality_knob_targets_use_lossy_loop(monkeypatch, tmp_path, fmt):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["encoder"] = encoder.name
        outp.write_bytes(b"x")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-F", fmt, "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["encoder"] == fmt


@pytest.mark.parametrize("fmt", ["bmp", "gif", "vips"])
def test_run_iterative_phase_container_targets_use_palette_loop(monkeypatch, tmp_path, fmt):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_art(inp, outp, max_size, **kwargs):
        calls["art"] = True
        outp.write_bytes(b"p")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_art_optimize", fake_art)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-F", fmt, "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls.get("art") is True


def test_run_iterative_phase_skips_when_under_size(monkeypatch, tmp_path):
    img, _calls = _iterative_setup(monkeypatch, tmp_path)

    def should_not_be_called(*a, **k):
        raise AssertionError("optimizer should not run for an under-size image")

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", should_not_be_called)
    monkeypatch.setattr(pipeline, "iterative_art_optimize", should_not_be_called)
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 100)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 100}
    out = pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert out == img


def test_run_iterative_phase_png_target_runs_oxipng_post_pass(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(oxipng=True))

    def fake_art(inp, outp, max_size, **kwargs):
        outp.write_bytes(b"p")
        return {"success": True, "final_size": 100, "iterations": 1}

    def fake_oxipng(inp, outp, **kwargs):
        calls["oxipng"] = (inp, outp)
        outp.write_bytes(b"o")
        return True

    monkeypatch.setattr(pipeline, "iterative_art_optimize", fake_art)
    monkeypatch.setattr(pipeline, "oxipng_compress", fake_oxipng)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    result = pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert "oxipng" in calls
    assert result.read_bytes() == b"o"


def test_run_format_conversion_lossy_target_uses_unified_loop(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def big_convert(inp, outp, target, **kwargs):
        outp.write_bytes(b"c" * (3 * 1024 * 1024))
        return True

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        outp.write_bytes(b"o")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "convert_format", big_convert)
    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 3 * 1024 * 1024 if p.suffix == ".webp" else 100)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 100}
    result = pipeline.run_format_conversion(img, pipeline._build_config(parse("-F", "webp", "-s", "2m")), measurements)
    assert result.read_bytes() == b"o"


def test_run_format_conversion_lossless_target_no_loop(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def fake_convert(inp, outp, target, **kwargs):
        outp.write_bytes(b"t")
        return True

    def should_not_be_called(*a, **k):
        raise AssertionError("lossy loop must not run for a lossless target")

    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "convert_format", fake_convert)
    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", should_not_be_called)
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 100)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 100}
    result = pipeline.run_format_conversion(img, pipeline._build_config(parse("-F", "tiff", "-s", "2m")), measurements)
    assert result.read_bytes() == b"t"


def test_run_format_conversion_bmp_correction_uses_precise_resize(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 1000)
    calls = {}

    def fake_convert(inp, outp, target, **kwargs):
        calls.setdefault("convert", []).append(inp.stat().st_size)
        outp.write_bytes(b"c" * (3 * 1024 * 1024))
        return True

    def fake_resize(inp, outp, percentage, kernel="Lanczos", **kwargs):
        calls["scale_pct"] = percentage
        outp.write_bytes(b"r")
        return True

    def fake_palette(inp, outp, colors, strip_metadata=False, **kwargs):
        calls["colors"] = colors
        outp.write_bytes(b"p")
        return True

    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "convert_format", fake_convert)
    monkeypatch.setattr(pipeline, "resize_pct", fake_resize)
    monkeypatch.setattr(pipeline, "color_palette_reduce", fake_palette)
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 3 * 1024 * 1024 if p.suffix == ".bmp" else 1000)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 1000}
    result = pipeline.run_format_conversion(img, pipeline._build_config(parse("-F", "bmp", "-s", "2m")), measurements)
    assert calls["scale_pct"] == 80
    assert calls["colors"] == 256
    assert result.read_bytes() == b"c" * (3 * 1024 * 1024)


def test_run_format_conversion_jxl_target_uses_lossy_loop(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def big_convert(inp, outp, target, **kwargs):
        outp.write_bytes(b"c" * (3 * 1024 * 1024))
        return True

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        outp.write_bytes(b"o")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "convert_format", big_convert)
    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 3 * 1024 * 1024 if p.suffix == ".jxl" else 100)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 100}
    result = pipeline.run_format_conversion(img, pipeline._build_config(parse("-F", "jxl", "-s", "2m")), measurements)
    assert result.read_bytes() == b"o"


def test_run_pngquant_phase_oxipng_post_pass(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        outp.write_bytes(b"q")
        return True

    def fake_oxipng(inp, outp, **kwargs):
        outp.write_bytes(b"o")
        return True

    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(pngquant=True, oxipng=True))
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 100 if p.read_bytes() == b"o" else 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    monkeypatch.setattr(pipeline, "oxipng_compress", fake_oxipng)
    result = pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("-s", "2m")), pipeline._Toggles(resize=True, pngquant=True, iterative=True),
        orig_format="PNG", target_format="PNG",
    )
    assert result.read_bytes() == b"o"


def test_run_pngquant_phase_discards_non_fitting_prepass(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        outp.write_bytes(b"q")
        return True

    def fake_oxipng(inp, outp, **kwargs):
        outp.write_bytes(b"o")
        return True

    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(pngquant=True, oxipng=True))
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    monkeypatch.setattr(pipeline, "oxipng_compress", fake_oxipng)
    result = pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("-s", "2m")), pipeline._Toggles(resize=True, pngquant=True, iterative=True),
        orig_format="PNG", target_format="PNG",
    )
    assert result == img


def test_run_pngquant_phase_dithering_follows_image_mode_flags(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)
    seen = {}

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        seen["no_dithering"] = no_dithering
        outp.write_bytes(b"q")
        return True

    monkeypatch.setattr(pipeline, "_get_backend",
                        lambda: _backend_stub(pngquant=True, oxipng=False))
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    toggles = pipeline._Toggles(resize=False, pngquant=True, iterative=False)
    for argv, expected in (
        (("-s", "2m"), False),
        (("-d", "-s", "2m"), True),
        (("-x", "-s", "2m"), True),
        (("-d", "--dithering", "-s", "2m"), False),
        (("--no-dithering", "-s", "2m"), True),
    ):
        pipeline.run_pngquant_phase(
            img, pipeline._build_config(parse(*argv)), toggles,
            orig_format="PNG", target_format="PNG",
        )
        assert seen["no_dithering"] is expected, argv


def test_run_pngquant_phase_only_pngquant_keeps_prepass(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        outp.write_bytes(b"q")
        return True

    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(pngquant=True, oxipng=False))
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    result = pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("-s", "2m")), pipeline._Toggles(resize=False, pngquant=True, iterative=False),
        orig_format="PNG", target_format="PNG",
    )
    assert result.read_bytes() == b"q"


def test_run_pngquant_phase_skips_non_png_source(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        raise AssertionError("pngquant must not run for non-PNG sources")

    monkeypatch.setattr(pipeline, "_get_backend",
                        lambda: _backend_stub(pngquant=True, oxipng=False))
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    result = pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("-s", "2m")), pipeline._Toggles(resize=True, pngquant=True, iterative=True),
        orig_format="JPEG", target_format="PNG",
    )
    assert result == img
    result = pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("-s", "2m")), pipeline._Toggles(resize=True, pngquant=True, iterative=True),
        orig_format="TIFF", target_format="PNG",
    )
    assert result == img


def test_run_pngquant_phase_skips_lossy_target(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        raise AssertionError("pngquant must not run for PNG sources of lossy targets")

    monkeypatch.setattr(pipeline, "_get_backend",
                        lambda: _backend_stub(pngquant=True, oxipng=False))
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    for target in ("JPEG", "WEBP", "AVIF", "JP2", "HEIC", "TIFF", "JXL"):
        result = pipeline.run_pngquant_phase(
            img, pipeline._build_config(parse("-s", "2m")), pipeline._Toggles(resize=True, pngquant=True, iterative=True),
            orig_format="PNG", target_format=target,
        )
        assert result == img


def test_run_pngquant_phase_uses_profile_speed(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)
    calls = {}

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        calls["speed"] = speed
        outp.write_bytes(b"q")
        return True

    monkeypatch.setattr(pipeline, "_get_backend",
                        lambda: _backend_stub(pngquant=True, oxipng=False))
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 100 if p.read_bytes() != b"x" * 100 else 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("--fast", "-s", "2m")),
        pipeline._Toggles(resize=True, pngquant=True, iterative=True),
        orig_format="PNG", target_format="PNG",
    )
    assert calls["speed"] == 5
    calls.clear()
    pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("-s", "2m")),
        pipeline._Toggles(resize=True, pngquant=True, iterative=True),
        orig_format="PNG", target_format="PNG",
    )
    assert calls["speed"] == 3


def test_run_pngquant_phase_skips_oxipng_in_fast(monkeypatch, tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"x" * 100)

    def fake_pngquant(inp, outp, speed=1, no_dithering=False):
        outp.write_bytes(b"q")
        return True

    def fake_oxipng(inp, outp, **kwargs):
        raise AssertionError("oxipng must be skipped in fast mode")

    monkeypatch.setattr(pipeline, "_get_backend",
                        lambda: _backend_stub(pngquant=True, oxipng=True))
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 100 if p.read_bytes() != b"x" * 100 else 5 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "PNG")
    monkeypatch.setattr(pipeline, "pngquant_compress", fake_pngquant)
    monkeypatch.setattr(pipeline, "oxipng_compress", fake_oxipng)
    result = pipeline.run_pngquant_phase(
        img, pipeline._build_config(parse("--fast", "-s", "2m")),
        pipeline._Toggles(resize=True, pngquant=True, iterative=True),
        orig_format="PNG", target_format="PNG",
    )
    assert result.read_bytes() == b"q"


def test_run_iterative_phase_fast_uses_coarse_grid(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["kwargs"] = kwargs
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("--fast", "-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["kwargs"]["search_step"] == 10
    assert calls["kwargs"]["max_iterations"] == 10
    assert calls["kwargs"]["use_extent_fast_path"] is True


def test_run_iterative_phase_default_uses_fine_grid(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["kwargs"] = kwargs
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["kwargs"]["search_step"] == 3
    assert calls["kwargs"]["max_iterations"] == 40
    assert calls["kwargs"]["use_extent_fast_path"] is True


def test_run_iterative_phase_slow_extent_gated_on_better_jpeg(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(jpegli=True))

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["kwargs"] = kwargs
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("--slow", "-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["kwargs"]["use_extent_fast_path"] is False
    calls.clear()
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["kwargs"]["use_extent_fast_path"] is True


def test_run_iterative_phase_manual_grid_gets_extent_rescue(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(jpegli=True))

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["kwargs"] = kwargs
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("--search-step", "5", "-F", "jpeg", "-s", "2m")),
        measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["kwargs"]["use_extent_fast_path"] is True
    assert calls["kwargs"]["search_step"] == 5


def test_run_iterative_phase_fast_skips_oxipng_post_pass(monkeypatch, tmp_path):
    img, _calls = _iterative_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(oxipng=True))

    def fake_art(inp, outp, max_size, **kwargs):
        outp.write_bytes(b"p")
        return {"success": True, "final_size": 100, "iterations": 1}

    def fake_oxipng(inp, outp, **kwargs):
        raise AssertionError("oxipng post-pass must be skipped in fast mode")

    monkeypatch.setattr(pipeline, "iterative_art_optimize", fake_art)
    monkeypatch.setattr(pipeline, "oxipng_compress", fake_oxipng)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    result = pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("--fast", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert result.read_bytes() == b"p"


def test_run_iterative_phase_fast_runs_oxipng_level_0(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(oxipng=True))

    def fake_art(inp, outp, max_size, **kwargs):
        outp.write_bytes(b"p")
        return {"success": True, "final_size": 100, "iterations": 1}

    def fake_oxipng(inp, outp, **kwargs):
        calls["oxipng_level"] = kwargs.get("level")
        outp.write_bytes(b"o")
        return True

    monkeypatch.setattr(pipeline, "iterative_art_optimize", fake_art)
    monkeypatch.setattr(pipeline, "oxipng_compress", fake_oxipng)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    result = pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["oxipng_level"] == 0
    assert result.read_bytes() == b"o"


def test_is_animated_uses_frame_count(monkeypatch, tmp_path):
    p = tmp_path / "a.gif"
    p.write_bytes(b"x")
    monkeypatch.setattr(pipeline.converters, "get_frame_count", lambda path: 5)
    assert pipeline._is_animated(p) is True
    monkeypatch.setattr(pipeline.converters, "get_frame_count", lambda path: 1)
    assert pipeline._is_animated(p) is False


def test_is_animated_gif_is_alias(monkeypatch, tmp_path):
    p = tmp_path / "a.gif"
    p.write_bytes(b"x")
    monkeypatch.setattr(pipeline.converters, "get_frame_count", lambda path: 3)
    assert pipeline._is_animated_gif(p) is True


def test_is_animated_defensive_on_read_error(monkeypatch, tmp_path):
    p = tmp_path / "a.gif"
    p.write_bytes(b"x")

    def boom(path):
        raise RuntimeError("cannot read")

    monkeypatch.setattr(pipeline.converters, "get_frame_count", boom)
    assert pipeline._is_animated(p) is False


def test_run_animated_gif_phase_dispatch_iterative(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * (5 * 1024 * 1024))
    calls = {}

    def fake_iter(inp, outp, max_size, **kwargs):
        calls["iter"] = kwargs
        outp.write_bytes(b"z")
        return {"success": True, "final_size": 100, "iterations": 1}

    def no_slowest(*a, **k):
        raise AssertionError("slowest search must not run in default mode")

    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 5 * 1024 * 1024 if p == img else 1)
    monkeypatch.setattr(pipeline, "iterative_animated_gif_optimize", fake_iter)
    monkeypatch.setattr(pipeline, "slowest_animated_gif_optimize", no_slowest)
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    result = pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("--slow", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=True, iterative=True),
    )
    assert result.read_bytes() == b"z"
    assert calls["iter"]["search_step"] is None
    assert calls["iter"]["max_iterations"] == 100
    assert calls["iter"]["pixel_art_mode"] is False
    assert calls["iter"]["start_colors"] == 256
    assert calls["iter"]["min_colors"] == 128


def test_run_animated_gif_phase_dispatch_slowest(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * (5 * 1024 * 1024))
    calls = {}

    def fake_best(inp, outp, max_size, **kwargs):
        calls["slowest"] = kwargs
        outp.write_bytes(b"z")
        return {"success": True, "final_size": 100, "iterations": 1}

    def no_iter(*a, **k):
        raise AssertionError("iterative search must not run in --slowest mode")

    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 5 * 1024 * 1024 if p == img else 1)
    monkeypatch.setattr(pipeline, "iterative_animated_gif_optimize", no_iter)
    monkeypatch.setattr(pipeline, "slowest_animated_gif_optimize", fake_best)
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    result = pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("--slowest", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=True, iterative=True),
    )
    assert result.read_bytes() == b"z"
    assert calls["slowest"]["min_colors"] == 2


def test_run_animated_gif_phase_fast_uses_coarse_grid(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * (5 * 1024 * 1024))
    calls = {}

    def fake_iter(inp, outp, max_size, **kwargs):
        calls["iter"] = kwargs
        outp.write_bytes(b"z")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 5 * 1024 * 1024 if p == img else 1)
    monkeypatch.setattr(pipeline, "iterative_animated_gif_optimize", fake_iter)
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("--fast", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=True, iterative=True),
    )
    assert calls["iter"]["search_step"] == 32
    assert calls["iter"]["max_iterations"] == 10


def test_run_animated_gif_phase_pixel_art_knobs_follow_flags(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * (5 * 1024 * 1024))
    calls = {}

    def fake_iter(inp, outp, max_size, **kwargs):
        calls["iter"] = kwargs
        outp.write_bytes(b"z")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 5 * 1024 * 1024 if p == img else 1)
    monkeypatch.setattr(pipeline, "iterative_animated_gif_optimize", fake_iter)
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    toggles = pipeline._Toggles(resize=True, pngquant=True, iterative=True)

    pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("-x", "-s", "2m")), measurements, toggles)
    assert calls["iter"]["pixel_art_mode"] is True
    assert calls["iter"]["pixel_art_snap"] is True
    pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("--nearest", "--snap", "-s", "2m")),
        measurements, toggles)
    assert calls["iter"]["pixel_art_mode"] is True
    assert calls["iter"]["pixel_art_snap"] is True
    pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("-x", "--no-snap", "-s", "2m")),
        measurements, toggles)
    assert calls["iter"]["pixel_art_mode"] is True
    assert calls["iter"]["pixel_art_snap"] is False


def test_run_animated_gif_phase_skips_when_under_size(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * 100)

    def should_not_be_called(*a, **k):
        raise AssertionError("optimizer must not run for an under-size GIF")

    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 100)
    monkeypatch.setattr(pipeline, "iterative_animated_gif_optimize", should_not_be_called)
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 100}
    result = pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=True, iterative=True),
    )
    assert result == img


def test_run_animated_gif_phase_warns_pngquant_only(monkeypatch, tmp_path, capsys):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * 100)

    def should_not_be_called(*a, **k):
        raise AssertionError("pngquant cannot run on animated GIFs")

    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 100)
    monkeypatch.setattr(pipeline, "pngquant_compress", should_not_be_called)
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 100}
    result = pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("-s", "2m")), measurements,
        pipeline._Toggles(resize=False, pngquant=True, iterative=False),
    )
    assert result == img
    assert "pngquant" in capsys.readouterr().out


def test_run_animated_gif_phase_warns_nn_scaler(monkeypatch, tmp_path, capsys):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * 100)

    def should_not_be_called(*a, **k):
        raise AssertionError("Neural upscalers cannot run on animated GIFs")

    monkeypatch.setattr(pipeline, "get_image_filesize", lambda p: 100)
    monkeypatch.setattr(pipeline, "realesrgan_upscale", should_not_be_called)
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 100}
    result = pipeline.run_animated_gif_phase(
        img, pipeline._build_config(parse("--realesrgan", "-s", "2m")), measurements,
        pipeline._Toggles(resize=False, pngquant=False, iterative=True),
    )
    assert result == img
    assert "Neural upscalers" in capsys.readouterr().out


def test_animated_resize_builds_geometry(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g")
    calls = {}

    def fake_gif_resize(inp, outp, geometry, kernel="Lanczos", nearest=False,
                        strip_metadata=False):
        calls["geometry"] = geometry
        calls["nearest"] = nearest
        calls["kernel"] = kernel
        outp.write_bytes(b"r")
        return True

    monkeypatch.setattr(pipeline.converters, "gif_resize", fake_gif_resize)
    monkeypatch.setattr(pipeline.converters, "get_image_width", lambda p: 50)
    monkeypatch.setattr(pipeline.converters, "get_image_height", lambda p: 40)

    def meas():
        return {"format": "GIF", "width": 100, "height": 80, "size": 100}

    out = pipeline._animated_resize(img, pipeline._build_config(parse("-w", "50")), meas())
    assert calls["geometry"] == "50x"
    assert calls["nearest"] is False
    assert calls["kernel"] == "Lanczos"
    assert out.read_bytes() == b"r"

    calls.clear()
    out = pipeline._animated_resize(img, pipeline._build_config(parse("-w", "50", "-H", "40")), meas())
    assert calls["geometry"] == "50x40!"

    calls.clear()
    out = pipeline._animated_resize(img, pipeline._build_config(parse("-x", "-I", "50")), meas())
    assert calls["geometry"] == "50%"
    assert calls["nearest"] is True
    assert calls["kernel"] == "Point"

    calls.clear()
    out = pipeline._animated_resize(img, pipeline._build_config(parse("--nearest", "-I", "50")), meas())
    assert calls["geometry"] == "50%"
    assert calls["nearest"] is True
    assert calls["kernel"] == "Point"


def test_animated_resize_noop_without_target(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g")
    monkeypatch.setattr(pipeline.converters, "gif_resize",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no resize")))
    measurements = {"format": "GIF", "width": 100, "height": 80, "size": 100}
    out = pipeline._animated_resize(img, pipeline._build_config(parse()), measurements)
    assert out == img


def test_process_image_animated_gif_routes_to_animated_phase(monkeypatch, tmp_path):
    img = tmp_path / "anim.gif"
    img.write_bytes(b"g" * (5 * 1024 * 1024))
    out = tmp_path / "out.gif"
    calls = {}

    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "GIF",
                                   "size": 5 * 1024 * 1024})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: True)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())

    def fake_animated(inp, args, measurements, toggles, target_format):
        calls["animated"] = (inp, measurements["format"], target_format)
        measurements["size"] = 100
        proc = tmp_path / "proc.gif"
        proc.write_bytes(b"z")
        return proc

    monkeypatch.setattr(pipeline, "run_animated_gif_phase", fake_animated)
    for phase in ("run_resize_phase", "run_pngquant_phase",
                  "run_iterative_phase", "run_format_conversion"):
        monkeypatch.setattr(
            pipeline, phase,
            lambda *a, _phase=phase, **k: (_ for _ in ()).throw(
                AssertionError(f"{_phase} must not run")
            ),
        )

    result = pipeline.process_image(img, parse("-o", str(out), "-s", "2m"))
    assert result["success"] is True
    assert result["bypassed"] is False
    assert calls["animated"][0] == img
    assert calls["animated"][1] == "GIF"
    assert calls["animated"][2] == "GIF"
    assert out.read_bytes() == b"z"


def test_process_image_still_gif_uses_static_pipeline(monkeypatch, tmp_path, capsys):
    img = tmp_path / "still.gif"
    img.write_bytes(b"g" * 100)
    out = tmp_path / "out.gif"
    calls = []

    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "GIF",
                                   "size": 5 * 1024 * 1024})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: False)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())
    monkeypatch.setattr(pipeline, "to_png_lossless",
                        lambda i, o: (o.write_bytes(b"p") or True))
    monkeypatch.setattr(pipeline.converters, "_has_png_magic", lambda p: False)
    monkeypatch.setattr(pipeline, "run_animated_gif_phase",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("animated phase must not run")))

    monkeypatch.setattr(pipeline, "run_resize_phase",
                        lambda inp, args, m, t: calls.append("resize") or inp)
    monkeypatch.setattr(pipeline, "run_pngquant_phase",
                        lambda inp, args, t, **k: inp)
    monkeypatch.setattr(pipeline, "run_iterative_phase",
                        lambda inp, args, m, mode, t, target=None: inp)
    monkeypatch.setattr(pipeline, "run_format_conversion",
                        lambda inp, args, m, target=None: inp)

    result = pipeline.process_image(img, parse("-o", str(out), "-s", "2m"))
    assert result["success"] is True
    assert calls == ["resize"]
    assert "only the first frame" not in capsys.readouterr().out


def test_process_image_animated_avif_to_gif_routes_to_animated_phase(
        monkeypatch, tmp_path):
    img = tmp_path / "anim.avif"
    img.write_bytes(b"a" * 100)
    out = tmp_path / "out.gif"
    calls = {}

    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "AVIF",
                                   "size": 5 * 1024 * 1024})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: True)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())

    def fake_animated(inp, cfg, measurements, toggles, target_format):
        calls.update(inp=inp, target=target_format)
        measurements["size"] = 100
        proc = tmp_path / "proc.gif"
        proc.write_bytes(b"z")
        return proc

    monkeypatch.setattr(pipeline, "run_animated_gif_phase", fake_animated)
    for phase in ("run_resize_phase", "run_pngquant_phase",
                  "run_iterative_phase", "run_format_conversion"):
        monkeypatch.setattr(
            pipeline, phase,
            lambda *a, _phase=phase, **k: (_ for _ in ()).throw(
                AssertionError(f"{_phase} must not run")
            ),
        )

    result = pipeline.process_image(img, parse("-o", str(out), "-F", "gif", "-s", "2m"))
    assert result["success"] is True
    assert calls == {"inp": img, "target": "GIF"}
    assert out.read_bytes() == b"z"


def test_process_image_animated_gif_to_avif_routes_to_animated_phase(
        monkeypatch, tmp_path, capsys):
    img = tmp_path / "anim.gif"
    img.write_bytes(b"g" * 100)
    out = tmp_path / "out.avif"
    calls = {}

    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "GIF",
                                   "size": 5 * 1024 * 1024})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: True)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())

    def fake_animated(inp, cfg, measurements, toggles, target_format):
        calls.update(inp=inp, target=target_format)
        measurements["size"] = 100
        proc = tmp_path / "proc.avif"
        proc.write_bytes(b"z")
        return proc

    monkeypatch.setattr(pipeline, "run_animated_gif_phase", fake_animated)
    for phase in ("run_resize_phase", "run_pngquant_phase",
                  "run_iterative_phase", "run_format_conversion"):
        monkeypatch.setattr(
            pipeline, phase,
            lambda *a, _phase=phase, **k: (_ for _ in ()).throw(
                AssertionError(f"{_phase} must not run")
            ),
        )

    result = pipeline.process_image(img, parse("-o", str(out), "-F", "avif", "-s", "2m"))
    assert result["success"] is True
    assert calls == {"inp": img, "target": "AVIF"}
    assert out.read_bytes() == b"z"
    assert "only the first frame" not in capsys.readouterr().out


def test_process_image_animated_avif_to_webp_routes_to_animated_phase(
        monkeypatch, tmp_path):
    img = tmp_path / "anim.avif"
    img.write_bytes(b"a" * 100)
    out = tmp_path / "out.webp"
    calls = {}

    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "AVIF",
                                   "size": 5 * 1024 * 1024})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: True)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())

    def fake_animated(inp, cfg, measurements, toggles, target_format):
        calls.update(inp=inp, target=target_format)
        measurements["size"] = 100
        proc = tmp_path / "proc.webp"
        proc.write_bytes(b"z")
        return proc

    monkeypatch.setattr(pipeline, "run_animated_gif_phase", fake_animated)
    for phase in ("run_resize_phase", "run_pngquant_phase",
                  "run_iterative_phase", "run_format_conversion"):
        monkeypatch.setattr(
            pipeline, phase,
            lambda *a, _phase=phase, **k: (_ for _ in ()).throw(
                AssertionError(f"{_phase} must not run")
            ),
        )

    result = pipeline.process_image(img, parse("-o", str(out), "-F", "webp", "-s", "2m"))
    assert result["success"] is True
    assert calls == {"inp": img, "target": "WEBP"}
    assert out.read_bytes() == b"z"


def test_run_animated_phase_lossy_target_uses_lossy_search(monkeypatch, tmp_path):
    img = tmp_path / "in.gif"
    img.write_bytes(b"g" * 5000)
    baseline = tmp_path / "baseline.gif"
    baseline.write_bytes(b"b" * 5000)
    calls = {}

    def fake_baseline(source, target, cfg, source_format=None):
        calls["baseline"] = (source, source_format)
        return baseline

    def fake_lossy(inp, outp, max_size, target_format, **kwargs):
        calls["lossy"] = (inp, target_format, kwargs)
        outp.write_bytes(b"z" * 100)
        return {"success": True, "final_size": 100, "iterations": 1}

    def no_palette(*a, **k):
        raise AssertionError("GIF palette search must not run for a lossy target")

    monkeypatch.setattr(pipeline, "_animated_baseline", fake_baseline)
    monkeypatch.setattr(pipeline, "animated_lossy_optimize", fake_lossy)
    monkeypatch.setattr(pipeline, "iterative_animated_gif_optimize", no_palette)
    monkeypatch.setattr(pipeline, "slowest_animated_gif_optimize", no_palette)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "AVIF")
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 5000 if p == baseline else 100)
    cfg = pipeline._build_config(parse("--slow", "-s", "1k"))
    measurements = {"format": "GIF", "width": 100, "height": 100, "size": 5000}
    result = pipeline.run_animated_gif_phase(
        img, cfg, measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
        target_format="AVIF",
    )
    assert result.read_bytes() == b"z" * 100
    assert calls["baseline"] == (img, "GIF")
    assert calls["lossy"][0] == img
    assert calls["lossy"][1] == "AVIF"
    assert calls["lossy"][2]["start_quality"] == 100
    assert calls["lossy"][2]["min_quality"] == 85
    assert measurements["size"] == 100


def test_run_animated_phase_non_gif_source_gif_target_disables_entry_guard(
        monkeypatch, tmp_path):
    img = tmp_path / "in.avif"
    img.write_bytes(b"a" * 5000)
    baseline = tmp_path / "baseline.gif"
    baseline.write_bytes(b"b" * 5000)
    calls = {}

    monkeypatch.setattr(pipeline, "_animated_baseline",
                        lambda source, target, cfg, source_format=None: baseline)
    monkeypatch.setattr(pipeline, "get_image_format", lambda p: "GIF")

    def fake_palette(inp, outp, max_size, **kwargs):
        calls["palette"] = (inp, kwargs)
        outp.write_bytes(b"z" * 100)
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_animated_gif_optimize", fake_palette)
    monkeypatch.setattr(pipeline, "animated_lossy_optimize",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("lossy search must not run for a GIF target")))
    monkeypatch.setattr(pipeline, "get_image_filesize",
                        lambda p: 5000 if p == baseline else 100)
    cfg = pipeline._build_config(parse("--slow", "-s", "1k"))
    measurements = {"format": "AVIF", "width": 100, "height": 100, "size": 5000}
    result = pipeline.run_animated_gif_phase(
        img, cfg, measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
        target_format="GIF",
    )
    assert result.read_bytes() == b"z" * 100
    assert calls["palette"][0] == img
    assert calls["palette"][1]["entry_size_guard"] is False
    assert measurements["size"] == 100


def test_process_image_animated_gif_to_still_warns(monkeypatch, tmp_path, capsys):
    img = tmp_path / "anim.gif"
    img.write_bytes(b"g" * 100)
    out = tmp_path / "out.png"

    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "GIF",
                                   "size": 5 * 1024 * 1024})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: True)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())
    monkeypatch.setattr(pipeline, "to_png_lossless",
                        lambda i, o: (o.write_bytes(b"p") or True))
    monkeypatch.setattr(pipeline.converters, "_has_png_magic", lambda p: False)
    monkeypatch.setattr(pipeline, "run_animated_gif_phase",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("animated phase must not run")))
    monkeypatch.setattr(pipeline, "run_resize_phase", lambda inp, args, m, t: inp)
    monkeypatch.setattr(pipeline, "run_pngquant_phase",
                        lambda inp, args, t, **k: inp)
    monkeypatch.setattr(pipeline, "run_iterative_phase",
                        lambda inp, args, m, mode, t, target=None: inp)
    monkeypatch.setattr(pipeline, "run_format_conversion",
                        lambda inp, args, m, target=None: inp)

    result = pipeline.process_image(img, parse("-o", str(out), "-F", "jpeg", "-s", "2m"))
    assert result["success"] is True
    assert "only the first frame" in capsys.readouterr().out


def _clip_cfg(*argv):
    return pipeline._build_config(parse(*argv))


def test_clipboard_target_format_explicit_format_wins():
    cfg = _clip_cfg("-s", "2m", "-F", "WEBP")
    args = parse("-s", "2m", "-F", "WEBP", "-o", "out.jpeg")
    assert pipeline._clipboard_target_format("DIB", cfg, args) == "WEBP"


def test_clipboard_target_format_output_extension_is_intent():
    for ext, fmt in ((".jpeg", "JPEG"), (".jpg", "JPEG"), (".png", "PNG"),
                     (".webp", "WEBP"), (".gif", "GIF")):
        args = parse("-s", "2m", "-o", "out" + ext)
        assert pipeline._clipboard_target_format("DIB", _clip_cfg("-s", "2m"), args) == fmt


def test_clipboard_target_format_transport_falls_back_to_png():
    args = parse("-s", "2m", "-o", "out/")
    for measured in ("DIB", "BMP", "TIFF"):
        assert pipeline._clipboard_target_format(measured, _clip_cfg("-s", "2m"), args) == "PNG"


def test_clipboard_target_format_keeps_real_measured_format():
    args = parse("-s", "2m", "-o", "out/")
    assert pipeline._clipboard_target_format("GIF", _clip_cfg("-s", "2m"), args) == "GIF"
    assert pipeline._clipboard_target_format("PNG", _clip_cfg("-s", "2m"), args) == "PNG"


def test_clipboard_target_format_unknown_extension_falls_back():
    args = parse("-s", "2m", "-o", "out.xyz")
    assert pipeline._clipboard_target_format("DIB", _clip_cfg("-s", "2m"), args) == "PNG"


def test_iterative_phase_dispatches_by_override_not_measured_format(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "get_image_filesize", lambda path: 10_000_000)
    monkeypatch.setattr(pipeline, "_run_lossy_optimize",
                        lambda path, out, cfg, backend, fmt: calls.append(fmt))
    monkeypatch.setattr(pipeline, "oxipng_compress", lambda *a, **kw: False)

    fd, name = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    src = Path(name)
    try:
        cfg = _clip_cfg("-s", "2m")
        measurements = pipeline._Measurements(
            width=100, height=100, format="DIB", size=10_000_000)
        toggles = pipeline.resolve_processing_toggles(parse("-s", "2m"), False)
        pipeline.run_iterative_phase(src, cfg, measurements, toggles,
                                     target="JPEG")
        assert calls == ["JPEG"]
    finally:
        src.unlink(missing_ok=True)


def test_validate_flags_ssim_target_requires_kary_mode():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "0.95"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "0.95", "--fast"))
    assert pipeline.validate_flags(parse("--ssim-target", "0.95", "--slowest")) is None
    assert pipeline.validate_flags(parse("--ssim-target", "0.95", "--slow")) is None


def test_validate_flags_ssim_target_conflicts_with_max_size():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "0.95", "--slowest", "-s", "2m"))


def test_validate_flags_ssim_target_range():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "0", "--slowest"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "1", "--slowest"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "1.5", "--slowest"))


def test_validate_flags_ssim_target_rejects_bypass_and_phase_skips():
    for extra in (["-r"], ["--conversion-only", "-F", "jpeg"],
                  ["--skip-iterative"], ["--only-resize"], ["--only-pngquant"]):
        with pytest.raises(pipeline.ValidationError):
            pipeline.validate_flags(parse("--ssim-target", "0.95", "--slowest", *extra))


def test_validate_flags_ssim_target_requires_lossy_format():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "0.95", "--slowest", "-F", "gif"))
    assert pipeline.validate_flags(
        parse("--ssim-target", "0.95", "--slowest", "-F", "png")) is None
    assert pipeline.validate_flags(
        parse("--ssim-target", "0.95", "--slowest", "-F", "jpg")) is None


def _ssim_target_setup(monkeypatch, tmp_path, calibrate_result, src_size=5 * 1024 * 1024):
    img = tmp_path / "in.jpg"
    img.write_bytes(b"x" * 100)
    out = tmp_path / "out.jpg"
    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "JPEG",
                                   "size": src_size})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: False)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())
    monkeypatch.setattr(pipeline.converters, "_has_png_magic", lambda p: False)
    monkeypatch.setattr(pipeline, "to_png_lossless",
                        lambda i, o: (o.write_bytes(b"p") or True))
    monkeypatch.setattr(pipeline, "run_resize_phase", lambda inp, c, m, t: inp)
    monkeypatch.setattr(pipeline, "run_pngquant_phase", lambda inp, c, t, **k: inp)
    monkeypatch.setattr(pipeline, "run_format_conversion", lambda inp, c, m, target=None: inp)
    calls = {}

    def fake_calibrate(current, encoder, *, threshold, subsampling,
                       strip_metadata, min_quality, max_quality):
        calls["threshold"] = threshold
        calls["quality_bounds"] = (min_quality, max_quality)
        return calibrate_result

    monkeypatch.setattr(pipeline, "calibrate_ssim_target", fake_calibrate)

    def fake_iter(inp, cfg, m, mode, toggles, target=None):
        calls["budget"] = cfg.max_size_bytes
        calls["seed_colors"] = cfg.quality
        return inp

    monkeypatch.setattr(pipeline, "run_iterative_phase", fake_iter)
    return img, out, calls


def test_process_image_ssim_target_calibrates_budget(monkeypatch, tmp_path, capsys):
    img, out, calls = _ssim_target_setup(monkeypatch, tmp_path, calibrate_result=5000)
    result = pipeline.process_image(img, parse("--ssim-target", "0.95", "--slowest",
                                               "-o", str(out)))
    assert result["success"] is True
    assert calls["threshold"] == 0.95
    assert calls["quality_bounds"] == (1, 100)
    assert calls["budget"] == 5000
    assert "SSIM target calibrated" in capsys.readouterr().out


def test_ssim_target_defaults_to_095_when_value_omitted():
    assert parse("--ssim-target").ssim_target == 0.95
    assert parse("--ssim-target", "0.9").ssim_target == 0.9
    assert parse("--slowest").ssim_target is None
    assert pipeline.validate_flags(parse("--ssim-target", "--slowest")) is None
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "--fast"))
    cfg = pipeline._build_config(parse("--ssim-target", "--slowest"))
    assert cfg.ssim_target == 0.95


def test_process_image_ssim_target_calibrates_in_slow_mode(monkeypatch, tmp_path):
    img, out, calls = _ssim_target_setup(monkeypatch, tmp_path, calibrate_result=5000)
    result = pipeline.process_image(img, parse("--ssim-target", "0.95", "--slow",
                                               "-o", str(out)))
    assert result["success"] is True
    assert calls["threshold"] == 0.95
    assert calls["quality_bounds"] == (1, 100)
    assert calls["budget"] == 5000


def test_process_image_ssim_target_skipped_in_grid_mode(monkeypatch, tmp_path):
    img, out, calls = _ssim_target_setup(monkeypatch, tmp_path, calibrate_result=5000)
    args = parse("--ssim-target", "0.95", "--fast", "-o", str(out))
    monkeypatch.setattr(pipeline, "validate_flags", lambda a: None)
    result = pipeline.process_image(img, args)
    assert result["success"] is True
    assert "threshold" not in calls
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.42)


def test_process_image_ssim_target_never_exceeds_original(monkeypatch, tmp_path):
    img, out, calls = _ssim_target_setup(monkeypatch, tmp_path,
                                         calibrate_result=10 * 1024 * 1024)
    pipeline.process_image(img, parse("--ssim-target", "0.95", "--slowest",
                                      "-o", str(out)))
    assert calls["budget"] < 5 * 1024 * 1024


def test_process_image_ssim_target_unreachable_falls_back(monkeypatch, tmp_path, capsys):
    img, out, calls = _ssim_target_setup(monkeypatch, tmp_path, calibrate_result=None)
    result = pipeline.process_image(img, parse("--ssim-target", "0.95", "--slowest",
                                               "-o", str(out)))
    assert result["success"] is True
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.42)
    out_text = capsys.readouterr().out
    assert "unreachable" in out_text


def test_process_image_ssim_target_warns_on_non_lossy_target(monkeypatch, tmp_path, capsys):
    img, out, calls = _ssim_target_setup(monkeypatch, tmp_path, calibrate_result=5000)
    result = pipeline.process_image(img, parse("--ssim-target", "0.95", "--slowest",
                                               "-F", "GIF", "-o", str(out)))
    assert result["success"] is True
    out_text = capsys.readouterr().out
    assert "applies to lossy" in out_text
    assert "threshold" not in calls
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.6)


def test_process_image_default_target_keys_on_target_format(monkeypatch, tmp_path):
    img, out, calls = _ssim_target_setup(monkeypatch, tmp_path, calibrate_result=5000)
    result = pipeline.process_image(img, parse("-F", "tiff", "-o", str(out)))
    assert result["success"] is True
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.05)
    img2, out2, calls2 = _ssim_target_setup(monkeypatch, tmp_path, calibrate_result=5000)
    result = pipeline.process_image(img2, parse("-o", str(out2)))
    assert result["success"] is True
    assert calls2["budget"] == round(5 * 1024 * 1024 * 0.42)


def test_run_lossy_optimize_passes_ssim_target_ranking_knobs(monkeypatch, tmp_path):
    import core.compressors as compressors
    img = tmp_path / "in.png"
    img.write_bytes(b"x" * 1000)
    out = tmp_path / "out.jpg"
    calls = {}
    monkeypatch.setattr(pipeline, "make_lossy_encoders",
                        lambda backend, **kw: {"JPEG": compressors.CodecEncoder(
                            "fake", lambda *a, **k: True, suffix=".jpg")})

    def fake_slowest(inp, outp, max_size, encoders, **kwargs):
        calls.update(kwargs)
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 1, "iterations": 1}

    monkeypatch.setattr(pipeline, "slowest_lossy_optimize", fake_slowest)
    backend = _backend_stub()
    cfg = pipeline._build_config(parse("--ssim-target", "0.95", "--slowest"))
    pipeline._run_lossy_optimize(img, out, cfg, backend, "JPEG")
    assert calls["ssim_floor"] == 0.95
    assert calls["min_quality"] == 1

    calls.clear()
    cfg = pipeline._build_config(parse("--slowest"))
    pipeline._run_lossy_optimize(img, out, cfg, backend, "JPEG")
    assert calls["ssim_floor"] is None
    assert calls["min_quality"] == 30


def test_run_iterative_phase_slow_keeps_extent_for_stock_encoder(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(vips=True))

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["kwargs"] = kwargs
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img, pipeline._build_config(parse("--slow", "-F", "jpeg", "-s", "2m")), measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["kwargs"]["use_extent_fast_path"] is True


def test_run_iterative_phase_ssim_target_disables_extent(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub(vips=True))

    def fake_lossy(inp, outp, max_size, encoder, **kwargs):
        calls["kwargs"] = kwargs
        outp.write_bytes(b"j")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "iterative_lossy_optimize", fake_lossy)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    pipeline.run_iterative_phase(
        img,
        pipeline._build_config(parse("--slow", "--ssim-target", "0.95", "-F", "jpeg", "-s", "2m")),
        measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["kwargs"]["use_extent_fast_path"] is False


def test_default_target_fraction_table():
    from core.formats import DEFAULT_TARGET_FRACTIONS, default_target_fraction
    for fmt, frac in DEFAULT_TARGET_FRACTIONS.items():
        assert frac == round(frac, 2) and 0 < frac <= 1, fmt
        assert frac * 100 == int(frac * 100), fmt
    for fmt in ("JPEG", "WEBP", "AVIF", "JXL", "JP2", "HEIC", "TIFF"):
        assert fmt in DEFAULT_TARGET_FRACTIONS
    assert default_target_fraction(None) == default_target_fraction("PNG") == 0.6
    assert default_target_fraction("jpeg") == DEFAULT_TARGET_FRACTIONS["JPEG"]


def _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result,
                        src_size=5 * 1024 * 1024):
    img = tmp_path / "in.png"
    img.write_bytes(b"x" * 100)
    out = tmp_path / "out.png"
    monkeypatch.setattr(pipeline, "measure_image",
                        lambda p: {"width": 100, "height": 100, "format": "PNG",
                                   "size": src_size})
    monkeypatch.setattr(pipeline, "_is_animated", lambda p: False)
    monkeypatch.setattr(pipeline, "_get_backend", lambda: _backend_stub())
    monkeypatch.setattr(pipeline.converters, "_has_png_magic", lambda p: True)
    monkeypatch.setattr(pipeline, "run_resize_phase", lambda inp, c, m, t: inp)
    monkeypatch.setattr(pipeline, "run_pngquant_phase", lambda inp, c, t, **k: inp)
    monkeypatch.setattr(pipeline, "run_format_conversion", lambda inp, c, m, target=None: inp)
    calls = {}

    def fake_calibrate(current, *, threshold, strip_metadata, min_colors, max_colors):
        calls["threshold"] = threshold
        calls["colors"] = (min_colors, max_colors)
        if calibrate_result is None:
            return None
        return (calibrate_result, 42)

    monkeypatch.setattr(pipeline, "calibrate_palette_ssim_target", fake_calibrate)

    def fake_iter(inp, cfg, m, mode, toggles, target=None):
        calls["budget"] = cfg.max_size_bytes
        calls["seed_colors"] = cfg.quality
        return inp

    monkeypatch.setattr(pipeline, "run_iterative_phase", fake_iter)
    return img, out, calls


def test_process_image_ssim_target_png_seeds_crossing_colors(monkeypatch, tmp_path):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result=5000)
    monkeypatch.setattr(pipeline, "detect_palette_friendly", lambda p: True)
    result = pipeline.process_image(img, parse("-o", str(out)))
    assert result["success"] is True
    assert calls["budget"] == 5000
    assert calls["seed_colors"] == 42


def test_process_image_ssim_target_png_calibrates_palette_budget(monkeypatch, tmp_path, capsys):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result=5000)
    result = pipeline.process_image(img, parse("--ssim-target", "0.95", "--slowest",
                                               "-o", str(out)))
    assert result["success"] is True
    assert calls["threshold"] == 0.95
    assert calls["colors"] == (2, 256)
    assert calls["budget"] == 5000
    assert "Calibrating --ssim-target SSIM floor 0.95 (PNG)" in capsys.readouterr().out


def test_process_image_content_aware_auto_triggers(monkeypatch, tmp_path, capsys):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result=5000)
    monkeypatch.setattr(pipeline, "detect_palette_friendly", lambda p: True)
    result = pipeline.process_image(img, parse("-o", str(out)))
    assert result["success"] is True
    out_text = capsys.readouterr().out
    assert "Content-aware" in out_text
    assert calls["budget"] == 5000


def test_process_image_content_aware_never_loosens_default(monkeypatch, tmp_path):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path,
                                          calibrate_result=4 * 1024 * 1024)
    monkeypatch.setattr(pipeline, "detect_palette_friendly", lambda p: True)
    pipeline.process_image(img, parse("-o", str(out)))
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.6)


def test_process_image_content_aware_unreachable_keeps_fraction(monkeypatch, tmp_path, capsys):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result=None)
    monkeypatch.setattr(pipeline, "detect_palette_friendly", lambda p: True)
    result = pipeline.process_image(img, parse("-o", str(out)))
    assert result["success"] is True
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.6)
    assert "not reachable" in capsys.readouterr().out


def test_process_image_content_aware_detector_gate(monkeypatch, tmp_path):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result=5000)
    monkeypatch.setattr(pipeline, "detect_palette_friendly", lambda p: False)
    pipeline.process_image(img, parse("-o", str(out)))
    assert "threshold" not in calls
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.6)


def test_process_image_content_aware_skipped_in_fast(monkeypatch, tmp_path):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result=5000)
    def _no_scan(p):
        raise AssertionError("detector must not run in --fast")
    monkeypatch.setattr(pipeline, "detect_palette_friendly", _no_scan)
    pipeline.process_image(img, parse("--fast", "-o", str(out)))
    assert "threshold" not in calls
    assert calls["budget"] == round(5 * 1024 * 1024 * 0.6)


def test_process_image_content_aware_skipped_with_explicit_size(monkeypatch, tmp_path):
    img, out, calls = _palette_ssim_setup(monkeypatch, tmp_path, calibrate_result=5000)
    monkeypatch.setattr(pipeline, "detect_palette_friendly", lambda p: True)
    pipeline.process_image(img, parse("-s", "1m", "-o", str(out)))
    assert "threshold" not in calls
    assert calls["budget"] == 1024 * 1024


def test_run_iterative_phase_slowest_art_passes_ssim_floor(monkeypatch, tmp_path):
    img, calls = _iterative_setup(monkeypatch, tmp_path)

    def fake_best(inp, outp, max_size, **kwargs):
        calls["ssim_floor"] = kwargs.get("ssim_floor")
        calls["min_colors"] = kwargs.get("min_colors")
        outp.write_bytes(b"p")
        return {"success": True, "final_size": 100, "iterations": 1}

    monkeypatch.setattr(pipeline, "slowest_art_optimize", fake_best)
    measurements = {"format": "PNG", "width": 100, "height": 100, "size": 5 * 1024 * 1024}
    cfg = pipeline._build_config(parse("--slowest", "--ssim-target", "0.95", "-s", "2m"))
    pipeline.run_iterative_phase(
        img, cfg, measurements,
        pipeline._Toggles(resize=True, pngquant=False, iterative=True),
    )
    assert calls["ssim_floor"] == 0.95
    assert calls["min_colors"] == 2


def test_validate_flags_ssim_target_png_still_requires_kary():
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "0.95", "-F", "png"))
    with pytest.raises(pipeline.ValidationError):
        pipeline.validate_flags(parse("--ssim-target", "0.95", "--fast", "-F", "png"))
    assert pipeline.validate_flags(
        parse("--ssim-target", "0.95", "--slow", "-F", "png")) is None
