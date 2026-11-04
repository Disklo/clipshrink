import argparse
import re

from . import __version__
from .deps import VENV_DIRNAME
from .presets import PRESETS, preset_flags

# bare number = KB, k/kb or m/mb picks unit
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM][bB]?)?\s*$")


def parse_size(value: str) -> int:
    match = _SIZE_RE.match(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}: use a number of kilobytes (e.g. 512) "
            "or a size with a unit suffix (e.g. 512k, 8m, 1.5M)"
        )
    number = float(match.group(1))
    unit = (match.group(2) or "k").lower()
    multiplier = 1024 * 1024 if unit.startswith("m") else 1024
    return round(number * multiplier)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipshrink",
        description=(
            "Shrink images below a target file size while preserving quality.\n"
            "Input comes from a file, a folder, or the system clipboard; output "
            "goes to a file, a folder, or the clipboard."
        ),
        epilog=(
            "examples:\n"
            "  clipshrink                                    shrink the clipboard image back onto the clipboard\n"
            "  clipshrink -i photo.png -o out/ -s 2m         shrink a file to 2 MB\n"
            "  clipshrink -i folder/ -o out/ -R -s 5m        batch process a folder recursively\n"
            "  clipshrink -s 512 -p                          shrink clipboard image to 512 KB (photo mode)\n"
            "  clipshrink -i img.png -u -w 4000 --realesrgan  upscale with a neural upscaler\n"
            "  clipshrink -i img.png -F webp --conversion-only  convert format only\n"
            "  clipshrink -i photo.png -s 2m --slowest       slowest, best-quality-per-byte search\n\n"
            "size: -s/--max-size takes kilobytes by default; suffix k/KB (kilobytes) or\n"
            "m/MB (megabytes), e.g. -s 512, -s 512k, -s 8m. Without -s the target is a\n"
            "per-format fraction of each image's own file size, keyed by the TARGET\n"
            "format (JPEG 42%, WEBP 41%, AVIF 31%, JXL 41%, JP2 15%, HEIC 19%, TIFF 5%;\n"
            "other formats 60%). Content-aware PNG: palette-friendly PNG stills\n"
            "(screenshots/graphics) automatically get an SSIM 0.95 floor instead\n"
            "(never in --fast; photographic PNGs are unaffected).\n"
            "modes: -p/--photo (default), -d/--drawing, -x/--pixel-art; like the speed\n"
            "modes they are presets, and every knob (--dithering, --nearest, --snap,\n"
            "--realesrgan-model) is an explicit flag that overrides the mode's value"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('-p', '--photo', action='store_true',
                        help='Optimize for photos (default): smooth scaling, default dithering')
    parser.add_argument('-d', '--drawing', action='store_true',
                        help='Optimize for drawings: disables dithering, uses color reduction')
    parser.add_argument('-x', '--pixel-art', action='store_true',
                        help='Optimize for pixel art: nearest-neighbor scaling, no dithering')
    # modes are presets, explicit knobs win
    parser.add_argument('--dithering', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Floyd-Steinberg dithering for palette reduction '
                             '(mode default: photo on, drawing/pixel-art off)')
    parser.add_argument('--nearest', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Nearest-neighbor sampling for resizing '
                             '(pixel-art default)')
    parser.add_argument('--snap', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Snap search resolution steps to integer reciprocals '
                             '(pixel-art default)')
    parser.add_argument('--realesrgan-model', type=str, dest='realesrgan_model',
                        choices=['photo', 'drawing'], default=None,
                        help='realesrgan model (mode default: photo -> x4plus, '
                             'drawing -> x4plus-anime)')
    parser.add_argument('--realesrgan', action='store_true',
                        help='Use realesrgan-ncnn-vulkan (Model adapts based on photo/drawing mode)')
    parser.add_argument('--waifu2x', action='store_true',
                        help='Use waifu2x-ncnn-vulkan (Model adapts based on photo/drawing mode)')
    parser.add_argument('-i', '--input', type=str,
                        help='Input image file path (if omitted, uses clipboard)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output image file path (if omitted, copies to clipboard)')
    parser.add_argument('-s', '--max-size', type=parse_size, default=None,
                        metavar='SIZE',
                        help='Target max file size in kilobytes (default: a per-format '
                             'fraction of the original file size, keyed by the target '
                             'format - e.g. JPEG 42%%, AVIF 31%%, other formats 60%%; a '
                             'unit suffix selects the unit: '
                             'k/KB = kilobytes, m/MB = megabytes - e.g. 512, 512k, 8m)')
    parser.add_argument('--ssim-target', type=float, nargs='?', const=0.95,
                        default=None, metavar='VALUE',
                        help='Instead of a fixed size target, shrink each image to the '
                             'smallest size whose SSIM vs the original stays >= VALUE '
                             '(0-1 exclusive; defaults to 0.95 when the value is '
                             'omitted; requires --slow or --slowest). '
                             'Mutually exclusive with -s/--max-size and with the '
                             'flags that skip the iterative optimization phase '
                             '(--skip-iterative, --only-resize, --only-pngquant); '
                             'lossy or PNG still-image targets only')
    parser.add_argument('-w', '--max-width', type=int,
                        help='Target max width in pixels')
    parser.add_argument('-H', '--max-height', type=int,
                        help='Target max height in pixels')
    parser.add_argument('-I', '--integer-scaling', type=int,
                        help='Scale by increments (e.g., 50 for 50%%). Used for iterative downscaling.')
    parser.add_argument('-F', '--format', type=str,
                        help='Target output format (e.g., PNG, JPEG, WEBP). Defaults to original.')
    parser.add_argument('-q', '--quality', type=int,
                        help='Starting quality (up to 100) or max colors (up to 256)')
    parser.add_argument('-Q', '--min-quality', type=int,
                        help='Minimum allowed quality/colors before halving resolution')
    parser.add_argument('-u', '--upscale', action='store_true',
                        help='Upscale mode: resizes larger (default 200%%) and skips compression')
    parser.add_argument('-m', '--manual-skipping', action='store_true',
                        help='Allow optimizations during upscaling (requires -u or implicit upscale)')
    parser.add_argument('-f', '--filter', type=str,
                        help="Resampling filter for ImageMagick (or 'realesrgan'/'waifu2x')")
    parser.add_argument('-r', '--raw', action='store_true',
                        help='Skip all processing; extract/copy image completely untouched')
    parser.add_argument('-R', '--recursive', action='store_true',
                        help='Process images in subdirectories recursively (folder input only)')
    parser.add_argument('-j', '--jobs', type=int, metavar='N',
                        help='Process up to N images in parallel (folder input only; '
                             'default: min(4, max(2, CPU count)), 1 = fully serial)')
    parser.add_argument('--no-suffix', action='store_true',
                        help="Do not append '_processed' suffix to output filenames")
    parser.add_argument('--conversion-only', action='store_true',
                        help='Only convert format (requires --format); skip all optimizations')
    parser.add_argument('--skip-resize', action='store_true',
                        help='Skip resize phase')
    parser.add_argument('--skip-pngquant', action='store_true',
                        help='Skip pngquant compression')
    parser.add_argument('--skip-iterative', action='store_true',
                        help='Skip iterative optimization')
    parser.add_argument('--only-resize', action='store_true',
                        help='Only run resize phase')
    parser.add_argument('--only-pngquant', action='store_true',
                        help='Only run pngquant compression')
    parser.add_argument('--only-iterative', action='store_true',
                        help='Only run iterative optimization')
    parser.add_argument('--dry-run', action='store_true', dest='dry_run',
                        help='Report what would be done (measurements, target, budget, '
                             'planned phases) and exit without writing any output')
    parser.add_argument('--strip-metadata', action='store_true',
                        help='Strip EXIF, ICC profiles, and other metadata')
    parser.add_argument('--subsampling', type=str,
                        choices=['4:4:4', '4:2:2', '4:2:0'], default='4:2:0',
                        help='JPEG chroma subsampling for photo-mode compression (default: 4:2:0)')
    # help comes from preset_flags so it can't drift
    parser.add_argument('--slow', action='store_true',
                        help='Slow preset (the old default): K-ary accept-then-refine search '
                             f'instead of the default quality grid. Expands to: {" ".join(preset_flags("slow"))} '
                             '(leaves --search-step/--color-step unset)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast preset. Expands to: '
                             f'{" ".join(preset_flags("fast"))}')
    parser.add_argument('--slowest', action='store_true',
                        help='Slowest preset (best quality). Expands to: '
                             f'{" ".join(preset_flags("slowest"))} '
                             '(quality floor 30, colors floor 2; leaves --search-step/'
                             '--color-step unset)')
    parser.add_argument('--search-step', type=int, metavar='N',
                        help='Lossy quality grid step (1-100); unset = K-ary accept-then-refine search')
    parser.add_argument('--color-step', type=int, metavar='N',
                        help='Palette color grid step (1-256); unset = K-ary search')
    parser.add_argument('--max-iterations', type=int, metavar='N',
                        help='Hard cap on search-loop iterations '
                             f'(default {PRESETS["default"].max_iterations}; '
                             f'{PRESETS["fast"].max_iterations} in --fast; '
                             f'{PRESETS["slow"].max_iterations} in --slow; '
                             f'{PRESETS["slowest"].max_iterations} in --slowest)')
    parser.add_argument('--avif-speed', type=int, metavar='N',
                        help='avifenc --speed (0-10; '
                             f'default {PRESETS["default"].avif_speed}; '
                             f'{PRESETS["fast"].avif_speed} in --fast; '
                             f'{PRESETS["slow"].avif_speed} in --slow; '
                             f'{PRESETS["slowest"].avif_speed} in --slowest)')
    parser.add_argument('--pngquant-speed', type=int, metavar='N',
                        help='pngquant --speed (1-10; '
                             f'default {PRESETS["default"].pngquant_speed}; '
                             f'{PRESETS["fast"].pngquant_speed} in --fast; '
                             f'{PRESETS["slow"].pngquant_speed} in --slow; '
                             f'{PRESETS["slowest"].pngquant_speed} in --slowest)')
    parser.add_argument('--oxipng-level', type=int, metavar='N',
                        help='oxipng -o level (0-6; '
                             f'default {PRESETS["default"].oxipng_level}; '
                             f'{PRESETS["slow"].oxipng_level} in --slow; '
                             f'{PRESETS["slowest"].oxipng_level} in --slowest; '
                             'skipped in --fast)')
    parser.add_argument('--no-oxipng', action='store_true', default=None,
                        help='Skip the lossless oxipng post-pass (oxipng level None)')
    parser.add_argument('--frontier', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Frontier search: explore several resolution fit-levels instead of '
                             'stopping at the first fit, then pick the best candidate '
                             '(--no-frontier turns it off)')
    parser.add_argument('--frontier-levels', type=int, metavar='N',
                        help='Fit levels explored before the frontier stops (with --frontier; default 3)')
    parser.add_argument('--frontier-ceiling', type=int, metavar='N',
                        help='Knob ceiling (quality 1-100, or colors 1-256) that stops the frontier '
                             'early (with --frontier; default 90 quality / full color count)')
    parser.add_argument('--deep-step', type=int, metavar='N',
                        help='Frontier resolution step (%%) applied when the sqrt model '
                             'saturates (with --frontier; frontier levels resize the '
                             'original source; default 75)')
    parser.add_argument('--ssim', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Rank frontier candidates by SSIM vs the lossless reference '
                             '(with --frontier; default: yes; --no-ssim disables)')
    parser.add_argument('--compare-subsampling', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='Probe both JPEG 4:4:4 and 4:2:0 at every frontier level '
                             '(requires --frontier; --no-compare-subsampling turns it off)')
    parser.add_argument('--prefer-software-codecs', action=argparse.BooleanOptionalAction,
                        default=None, dest='prefer_software',
                        help='Use software AVIF/HEIC encoders before hardware (better '
                             'quality-per-byte, slower; default in --slowest; '
                             '--no-prefer-software-codecs turns it off)')
    parser.add_argument('--install-deps', action='store_true', dest='install_deps',
                        help='Install missing backend binaries and Python packages '
                             'before the requested operation runs (Python packages '
                             'are only installed inside a venv; use with --venv to '
                             'create/target one first)')
    parser.add_argument('--venv', nargs='?', const=VENV_DIRNAME, default=None,
                        metavar='PATH',
                        help=f'Check/create a virtual environment (default path: {VENV_DIRNAME})')
    parser.add_argument('--venv-skip', action='store_true', dest='venv_skip',
                        help='Disable virtual environment detection and prompts')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                        help='Verbose diagnostics on the console (phase-level detail)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug diagnostics on the console: every external subprocess '
                             'command line, exit status, and captured stderr on failure '
                             '(implies --verbose)')
    parser.add_argument('--log-file', type=str, dest='log_file', metavar='PATH',
                        help='Append all verbose/debug diagnostics to a log file '
                             '(written regardless of -v/--debug)')
    parser.add_argument('--print-completions', type=str, dest='print_completions',
                        choices=['bash', 'zsh', 'fish'], metavar='SHELL',
                        help='Print a shell completion script (bash, zsh, or fish) '
                             'for this CLI to stdout and exit')
    parser.add_argument('--print-man', action='store_true', dest='print_man',
                        help='Print a roff man page for this CLI to stdout and exit')
    parser.add_argument('--version', action='version', version=f'clipshrink {__version__}')
    return parser


# sentinel for "not given"
_UNSET = object()


class _ProvidedNamespace(argparse.Namespace):
    # tracks which dests came from the CLI, not a default diff

    def __init__(self) -> None:
        super().__init__()
        self.provided: set[str] = set()
        self._tracking = False

    def __setattr__(self, name: str, value: object) -> None:
        if not name.startswith("_") and self.__dict__.get("_tracking"):
            self.provided.add(name)
        super().__setattr__(name, value)


def parse_cli_args(parser: argparse.ArgumentParser,
                   argv: list[str]) -> tuple[argparse.Namespace, set[str]]:
    # seed sentinels so argparse can't fill defaults, patch holes after
    ns = _ProvidedNamespace()
    for action in parser._actions:
        if (action.dest is not argparse.SUPPRESS
                and action.default is not argparse.SUPPRESS
                and not hasattr(ns, action.dest)):
            setattr(ns, action.dest, _UNSET)
    ns._tracking = True
    parser.parse_args(argv, ns)
    ns._tracking = False
    for action in parser._actions:
        if (action.dest is not argparse.SUPPRESS
                and action.default is not argparse.SUPPRESS
                and getattr(ns, action.dest, _UNSET) is _UNSET):
            setattr(ns, action.dest, parser.get_default(action.dest))
    return ns, set(ns.provided)
