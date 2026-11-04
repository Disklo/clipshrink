from dataclasses import replace

import pytest

from core import compressors
from core.cli import create_parser
from core.pipeline import ValidationError, validate_flags
from core.presets import (
    IMAGE_MODE_PRESETS,
    PRESETS,
    ImageModeSettings,
    image_mode_flags,
    preset_flags,
    resolve_image_mode_settings,
    resolve_preset,
    resolve_search_settings,
)


def parse(*argv):
    return create_parser().parse_args(list(argv))


def settings(*argv):
    return resolve_search_settings(parse(*argv))


def _knobs(s):
    return replace(s, min_quality_floor=0, min_colors_floor=0)


def test_preset_encoder_knobs_match_speed_profile():
    for name in ("default", "slow", "fast"):
        prof = compressors.speed_profile(name)
        s = PRESETS[name]
        assert s.search_step == prof["search_step"]
        assert s.color_step == prof["color_step"]
        assert s.max_iterations == prof["max_iterations"]
        assert s.avif_speed == prof["avif_speed"]
        assert s.pngquant_speed == prof["pngquant_speed"]
        assert s.oxipng_level == prof["oxipng_level"]


def test_preset_floors_match_wrapper_defaults():
    assert PRESETS["default"].min_quality_floor == compressors._DEFAULT_MIN_QUALITY
    assert PRESETS["default"].min_colors_floor == compressors._DEFAULT_MIN_COLORS
    assert PRESETS["slowest"].min_quality_floor == compressors._DEFAULT_MIN_QUALITY_SLOWEST
    assert PRESETS["slowest"].min_colors_floor == compressors._DEFAULT_MIN_COLORS_SLOWEST
    assert PRESETS["slow"].min_quality_floor == PRESETS["default"].min_quality_floor
    assert PRESETS["fast"].min_quality_floor == PRESETS["default"].min_quality_floor


def test_slowest_preset_values():
    s = PRESETS["slowest"]
    assert s.frontier is True
    assert s.frontier_levels == 3
    assert s.frontier_ceiling is None
    assert s.deep_step == 75
    assert s.ssim is True
    assert s.compare_subsampling is True
    assert s.prefer_software is True
    assert s.avif_speed == 1
    assert s.max_iterations == 200
    assert s.search_step is None and s.color_step is None
    assert s.pngquant_speed == 1 and s.oxipng_level == 2


def test_resolve_preset_precedence():
    assert resolve_preset(parse()) == "default"
    assert resolve_preset(parse("--slow")) == "slow"
    assert resolve_preset(parse("--fast")) == "fast"
    assert resolve_preset(parse("--slowest")) == "slowest"
    with pytest.raises(ValidationError):
        validate_flags(parse("--slow", "--fast"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--slowest", "--slow"))


def test_default_defaults():
    s = settings()
    assert s.search_step == 3 and s.color_step == 16
    assert s.max_iterations == 40
    assert s.avif_speed == 8 and s.pngquant_speed == 3 and s.oxipng_level == 0
    assert s.frontier is False
    assert s.prefer_software is False
    assert s.compare_subsampling is False
    assert s.min_quality_floor == 85 and s.min_colors_floor == 128


def test_search_mode_encodes_the_step_sentinels():
    assert settings().search_mode == "grid"
    assert settings("--fast").search_mode == "grid"
    assert settings("--slow").search_mode == "kary"
    assert settings("--slowest").search_mode == "kary"
    assert settings("--slow", "--search-step", "5").search_mode == "grid"


def test_slow_defaults():
    s = settings("--slow")
    assert s.search_step is None and s.color_step is None
    assert s.max_iterations == 100
    assert s.avif_speed == 4 and s.pngquant_speed == 1 and s.oxipng_level == 2
    assert s.frontier is False
    assert s.prefer_software is False
    assert s.compare_subsampling is False
    assert s.min_quality_floor == 85 and s.min_colors_floor == 128


def test_fast_equals_explicit_flag_expansion():
    assert settings("--fast") == settings(*preset_flags("fast"))
    assert settings("--fast") == PRESETS["fast"]


def test_slow_equals_explicit_flag_expansion():
    expanded = settings(*preset_flags("slow"))
    kary = replace(expanded, search_step=None, color_step=None)
    assert _knobs(kary) == _knobs(PRESETS["slow"])
    assert _knobs(settings("--slow")) == _knobs(PRESETS["slow"])


def test_slowest_equals_explicit_flag_expansion():
    expanded = settings(*preset_flags("slowest"))
    kary = replace(expanded, search_step=None, color_step=None)
    assert _knobs(kary) == _knobs(PRESETS["slowest"])
    assert _knobs(settings("--slowest")) == _knobs(PRESETS["slowest"])
    assert settings("--slowest").min_quality_floor == 30
    assert settings("--slowest").min_colors_floor == 2


def test_preset_flags_roundtrip():
    expanded = settings(*preset_flags("fast"))
    assert _knobs(expanded) == _knobs(PRESETS["fast"])
    assert expanded.min_quality_floor == PRESETS["default"].min_quality_floor
    assert expanded.min_colors_floor == PRESETS["default"].min_colors_floor
    for name in ("slow", "slowest"):
        expanded = settings(*preset_flags(name))
        kary = replace(expanded, search_step=None, color_step=None)
        assert _knobs(kary) == _knobs(PRESETS[name])
    assert preset_flags("default") == []


def test_explicit_flag_overrides_preset():
    assert settings("--fast", "--search-step", "1").search_step == 1
    assert settings("--fast", "--color-step", "4").color_step == 4
    assert settings("--fast", "--oxipng-level", "4").oxipng_level == 4
    assert settings("--fast", "--no-oxipng").oxipng_level is None
    assert settings("--slowest", "--no-ssim").ssim is False
    assert settings("--slowest", "--frontier-levels", "5").frontier_levels == 5
    assert settings("--slowest", "--frontier-ceiling", "60").frontier_ceiling == 60
    assert settings("--slowest", "--deep-step", "50").deep_step == 50
    assert settings("--slowest", "--max-iterations", "50").max_iterations == 50
    assert settings("--slowest", "--avif-speed", "6").avif_speed == 6
    assert settings("--slowest", "--pngquant-speed", "9").pngquant_speed == 9
    assert settings("--slowest", "--no-frontier").frontier is False
    assert settings("--slowest", "--no-compare-subsampling").compare_subsampling is False
    assert settings("--slowest", "--no-prefer-software-codecs").prefer_software is False
    s = settings("--fast", "--search-step", "1")
    assert s.avif_speed == 10 and s.max_iterations == 10


def test_manual_frontier_is_default_based():
    s = settings("--frontier")
    assert s.frontier is True
    assert s.ssim is True
    assert s.compare_subsampling is False
    assert s.prefer_software is False
    assert s.frontier_ceiling is None
    assert s.frontier_levels == 3 and s.deep_step == 75
    assert s.min_quality_floor == 85 and s.min_colors_floor == 128
    assert s.avif_speed == 8 and s.max_iterations == 40


def test_no_oxipng_flag_roundtrip():
    assert settings("--no-oxipng").oxipng_level is None
    assert settings("--oxipng-level", "6").oxipng_level == 6
    assert settings("--slow", "--no-oxipng").oxipng_level is None


def test_validate_flags_new_checks():
    with pytest.raises(ValidationError):
        validate_flags(parse("--compare-subsampling"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--no-oxipng", "--oxipng-level", "3"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--search-step", "0"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--search-step", "101"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--color-step", "257"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--avif-speed", "11"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--pngquant-speed", "0"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--oxipng-level", "7"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--max-iterations", "0"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--frontier-levels", "0"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--frontier-ceiling", "0"))
    with pytest.raises(ValidationError):
        validate_flags(parse("--deep-step", "100"))
    validate_flags(parse("--frontier", "--compare-subsampling"))
    validate_flags(parse("--frontier", "--frontier-levels", "5", "--deep-step", "50"))
    validate_flags(parse("--fast", "--search-step", "1"))
    validate_flags(parse("--slowest", "--no-ssim"))


def imode(*argv):
    return resolve_image_mode_settings(parse(*argv))


def test_image_mode_preset_values():
    assert IMAGE_MODE_PRESETS["photo"] == ImageModeSettings(
        dithering=True, nearest=False, snap=False, realesrgan_model="photo",
        resample_filter="Lanczos")
    assert IMAGE_MODE_PRESETS["drawing"] == ImageModeSettings(
        dithering=False, nearest=False, snap=False, realesrgan_model="drawing",
        resample_filter="Bicubic")
    assert IMAGE_MODE_PRESETS["pixel_art"] == ImageModeSettings(
        dithering=False, nearest=True, snap=True, realesrgan_model="photo",
        resample_filter="Point")


def test_image_mode_defaults():
    assert imode() == IMAGE_MODE_PRESETS["photo"]
    assert imode("-p") == IMAGE_MODE_PRESETS["photo"]
    assert imode("-d") == IMAGE_MODE_PRESETS["drawing"]
    assert imode("-x") == IMAGE_MODE_PRESETS["pixel_art"]


def test_image_mode_equals_explicit_flag_expansion():
    mode_argv = {"photo": (), "drawing": ("-d",), "pixel_art": ("-x",)}
    for name in ("photo", "drawing", "pixel_art"):
        assert imode(*mode_argv[name]) == imode(*image_mode_flags(name)), name
    assert image_mode_flags("photo") == []
    assert image_mode_flags("drawing") == [
        "--no-dithering", "--filter=Bicubic", "--realesrgan-model=drawing"]
    assert image_mode_flags("pixel_art") == [
        "--no-dithering", "--nearest", "--snap", "--filter=Point"]


def test_explicit_flag_overrides_image_mode():
    assert imode("-d", "--dithering").dithering is True
    assert imode("-x", "--no-nearest").nearest is False
    assert imode("-x", "--no-snap").snap is False
    assert imode("-d", "--realesrgan-model", "photo").realesrgan_model == "photo"
    assert imode("--no-dithering").dithering is False
    assert imode("--nearest").nearest is True
    assert imode("--snap").snap is True
    assert imode("--realesrgan-model", "drawing").realesrgan_model == "drawing"
    s = imode("-x", "--no-dithering")
    assert s.nearest is True and s.snap is True


def test_resample_filter_resolution():
    assert imode().resample_filter == "Lanczos"
    assert imode("-d").resample_filter == "Bicubic"
    assert imode("-x").resample_filter == "Point"
    assert imode("-d", "--filter", "Mitchell").resample_filter == "Mitchell"
    assert imode("--nearest").resample_filter == "Point"
    assert imode("-d", "--nearest").resample_filter == "Point"
    assert imode("--nearest", "--filter", "Lanczos").resample_filter == "Lanczos"
    assert imode("-x", "--no-nearest").resample_filter == "Point"


def test_make_lossy_encoders_resolved_kwargs(monkeypatch):
    calls = []

    def fake_avif_save(*args, **kwargs):
        calls.append(kwargs)
        return True

    from core.backend import BackendInfo

    backend = BackendInfo(os_type="X11", vips=False, magick=True, pngquant=False,
                          realesrgan=False, waifu2x=False, jpegli=False, mozjpeg=False,
                          avifenc=True, oxipng=False)
    monkeypatch.setattr(compressors.converters, "avif_save", fake_avif_save)

    encoders = compressors.make_lossy_encoders(
        backend, avif_speed=PRESETS["slowest"].avif_speed,
        prefer_software=PRESETS["slowest"].prefer_software)
    from pathlib import Path
    assert encoders["AVIF"].encode(Path("in"), Path("out"), 80) is True
    assert calls[-1]["speed"] == 1 and calls[-1]["prefer_software"] is True

    encoders = compressors.make_lossy_encoders(
        backend, avif_speed=PRESETS["slow"].avif_speed,
        prefer_software=PRESETS["slow"].prefer_software)
    assert encoders["AVIF"].encode(Path("in"), Path("out"), 80) is True
    assert calls[-1]["speed"] == 4 and calls[-1]["prefer_software"] is False
