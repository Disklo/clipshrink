from __future__ import annotations

import argparse
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Args:
    photo: bool = False
    drawing: bool = False
    pixel_art: bool = False
    dithering: bool | None = None
    nearest: bool | None = None
    snap: bool | None = None
    realesrgan_model: str | None = None
    realesrgan: bool = False
    waifu2x: bool = False
    input: str | None = None
    output: str | None = None
    max_size: int | None = None
    ssim_target: float | None = None
    max_width: int | None = None
    max_height: int | None = None
    integer_scaling: int | None = None
    format: str | None = None
    quality: int | None = None
    min_quality: int | None = None
    upscale: bool = False
    manual_skipping: bool = False
    filter: str | None = None
    raw: bool = False
    recursive: bool = False
    jobs: int | None = None
    no_suffix: bool = False
    conversion_only: bool = False
    skip_resize: bool = False
    skip_pngquant: bool = False
    skip_iterative: bool = False
    only_resize: bool = False
    only_pngquant: bool = False
    only_iterative: bool = False
    dry_run: bool = False
    strip_metadata: bool = False
    subsampling: str = "4:2:0"
    slow: bool = False
    fast: bool = False
    slowest: bool = False
    search_step: int | None = None
    color_step: int | None = None
    max_iterations: int | None = None
    avif_speed: int | None = None
    pngquant_speed: int | None = None
    oxipng_level: int | None = None
    no_oxipng: bool | None = None
    frontier: bool | None = None
    frontier_levels: int | None = None
    frontier_ceiling: int | None = None
    deep_step: int | None = None
    ssim: bool | None = None
    compare_subsampling: bool | None = None
    prefer_software: bool | None = None
    install_deps: bool = False
    venv: str | None = None
    venv_skip: bool = False
    verbose: bool = False
    debug: bool = False
    log_file: str | None = None
    print_completions: str | None = None
    print_man: bool = False

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> Args:
        return cls(**{f.name: getattr(ns, f.name) for f in fields(cls)})
