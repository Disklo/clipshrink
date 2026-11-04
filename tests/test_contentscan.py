from pathlib import Path

import pytest

from core import contentscan


def _write_flat(tmp_path: Path) -> Path:
    p = tmp_path / "flat.png"
    im = __import__("pyvips").Image.black(64, 64).add(200)
    im.write_to_file(str(p))
    return p


def _write_noisy(tmp_path: Path) -> Path:
    p = tmp_path / "noisy.png"
    pyvips = __import__("pyvips")
    im = pyvips.Image.gaussnoise(256, 256, mean=128, sigma=60, seed=1)
    im = im.bandjoin([pyvips.Image.gaussnoise(256, 256, mean=128, sigma=60, seed=2),
                      pyvips.Image.gaussnoise(256, 256, mean=128, sigma=60, seed=3)])
    im.write_to_file(str(p))
    return p


try:
    __import__("pyvips")
except (ImportError, OSError):
    pytest.skip("pyvips not available", allow_module_level=True)


def test_flat_content_is_palette_friendly(tmp_path):
    assert contentscan.detect_palette_friendly(_write_flat(tmp_path)) is True


def test_photo_like_noise_is_not_palette_friendly(tmp_path):
    assert contentscan.detect_palette_friendly(_write_noisy(tmp_path)) is False


def test_count_proxy_colors_thumbnails_large_images(tmp_path):
    pyvips = __import__("pyvips")
    p = tmp_path / "big.png"
    im = pyvips.Image.black(1024, 1024).add(7)
    im.write_to_file(str(p))
    assert contentscan.count_proxy_colors(p) == 1


def test_count_proxy_colors_none_without_pyvips(monkeypatch, tmp_path):
    monkeypatch.setattr(contentscan, "pyvips", None)
    assert contentscan.count_proxy_colors(_write_flat(tmp_path)) is None


def test_detect_fail_open_on_error(tmp_path):
    assert contentscan.detect_palette_friendly(tmp_path / "missing.png") is False


def test_threshold_constant_sanity():
    assert 0 < contentscan.COLOR_COUNT_THRESHOLD <= 256 ** 3
    assert contentscan.COLOR_COUNT_THRESHOLD >= 4096
