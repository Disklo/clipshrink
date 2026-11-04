from __future__ import annotations

import contextlib
from pathlib import Path
from types import ModuleType

try:
    import pyvips
except (ImportError, OSError):
    # cffi raises OSError when libvips is missing or broken
    pyvips = None


def set_pyvips(module: ModuleType) -> None:
    global pyvips
    pyvips = module


_PROXY_SIZE = 512

# Photo proxies run >100k colors, screenshots and graphics stay under ~10k.
COLOR_COUNT_THRESHOLD = 16384


def count_proxy_colors(image_path: Path) -> int | None:
    if pyvips is None:
        return None
    try:
        image = pyvips.Image.thumbnail(str(image_path), _PROXY_SIZE)
        bands = int(image.bands)
        if bands >= 3:
            rgb = image.extract_band(0).bandjoin(
                [image.extract_band(1), image.extract_band(2)])
        else:
            base = image.extract_band(0)
            rgb = base.bandjoin([base, base])
        data: bytes = rgb.cast("uchar").write_to_memory()
        stride = 3
        pixels = len(data) // stride
        # hist_find only handles uchar/ushort bins, too narrow for 24-bit RGB.
        return len({data[i * stride:(i + 1) * stride] for i in range(pixels)})
    except Exception:
        return None


def detect_palette_friendly(image_path: Path) -> bool:
    with contextlib.suppress(Exception):
        count = count_proxy_colors(image_path)
        if count is not None:
            return count <= COLOR_COUNT_THRESHOLD
    return False
