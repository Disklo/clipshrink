import contextlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

from .utils import run_killable, set_last_error

# Vulkan tools write nothing or a stub on failure
_MIN_OUTPUT_BYTES = 1024

# availability comes from core.backend, don't re-probe

REALESRGAN_MODELS = {
    "photo": "realesrgan-x4plus",
    "drawing": "realesrgan-x4plus-anime",
}


def realesrgan_upscale(
    input_path: Path,
    output_path: Path,
    mode: str = "photo",
) -> bool:
    model = REALESRGAN_MODELS.get(mode, REALESRGAN_MODELS["photo"])
    return _run_upscaler(
        ["realesrgan-ncnn-vulkan", "-i", str(input_path), "-o", str(output_path),
         "-s", "4", "-n", model],
        output_path,
    )


def waifu2x_upscale(
    input_path: Path,
    output_path: Path,
    scale: int = 2,
    noise_level: int = 1,
) -> bool:
    # waifu2x only accepts 2, 4, 8
    if scale == 3:
        scale = 4
    elif scale > 4:
        scale = 8
    elif scale < 2:
        scale = 2
    return _run_upscaler(
        ["waifu2x-ncnn-vulkan", "-i", str(input_path), "-o", str(output_path),
         "-s", str(scale), "-n", str(noise_level)],
        output_path,
    )


def _run_upscaler(cmd: list[str], output_path: Path) -> bool:
    # stderr to temp file, run_killable would deadlock on a full pipe
    fd, err_name = tempfile.mkstemp(prefix="clipshrink_ai_err_", suffix=".log")
    os.close(fd)
    try:
        with open(err_name, "w", encoding="utf-8", errors="replace") as errf:
            code = run_killable(cmd, stdout=subprocess.DEVNULL, stderr=errf)
        if code != 0:
            try:
                err = Path(err_name).read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                err = ""
            set_last_error(err)
            return False
        if not output_path.exists() or output_path.stat().st_size <= _MIN_OUTPUT_BYTES:
            set_last_error("Upscaler produced no valid output.")
            return False
        set_last_error("")
        return True
    finally:
        with contextlib.suppress(OSError):
            Path(err_name).unlink()


def detect_upscale_needed(
    orig_width: int,
    orig_height: int,
    orig_size: int,
    max_width: int | None,
    max_height: int | None,
    integer_scaling: int | None,
    upscale_flag: bool,
) -> bool:
    if upscale_flag:
        return True
    if max_width is not None and max_width > orig_width:
        return True
    if max_height is not None and max_height > orig_height:
        return True
    return bool(integer_scaling is not None and integer_scaling > 100)


def compute_nn_scale_factor(
    orig_width: int,
    orig_height: int,
    max_width: int | None,
    max_height: int | None,
    integer_scaling: int | None,
) -> int:
    if integer_scaling is not None:
        scale = math.ceil(integer_scaling / 100)
    elif max_width is not None:
        scale = math.ceil(max_width / orig_width) if orig_width > 0 else 2
    elif max_height is not None:
        scale = math.ceil(max_height / orig_height) if orig_height > 0 else 2
    else:
        scale = 2
    return max(2, scale)
