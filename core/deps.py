from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import venv as venv_module
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

from . import utils
from .errors import ClipshrinkError

if TYPE_CHECKING:
    from .backend import BackendInfo


class DepsError(ClipshrinkError):
    pass


CONFIG_FILENAME = "config.json"
DEFAULT_CONFIG: dict[str, Any] = {
    "first_run_completed": False,
    "warn_on_local_binaries": True,
    "custom_venv_path": None,
    "skip_venv": False,
}

_BOOL_KEYS = ("first_run_completed", "warn_on_local_binaries", "skip_venv")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return project_root() / CONFIG_FILENAME


def load_config(path: Path | None = None) -> dict[str, Any]:
    # unknown keys survive save_config
    config_path = path if path is not None else default_config_path()
    config: dict[str, Any] = dict(DEFAULT_CONFIG)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return config
    if not isinstance(raw, dict):
        return config
    for key, value in raw.items():
        if key in _BOOL_KEYS:
            if isinstance(value, bool):
                config[key] = value
        elif key == "custom_venv_path":
            if value is None or isinstance(value, str):
                config[key] = value
        else:
            config[key] = value
    return config


def save_config(config: dict[str, Any], path: Path | None = None) -> None:
    # best-effort, read-only project root is fine
    config_path = path if path is not None else default_config_path()
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
    except OSError:
        pass


VENV_DIRNAME = ".venv"  # default path for --venv


def is_in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def venv_python(venv_dir: Path) -> Path:
    # prefer find_venv_python, some interpreters deviate from the layout
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


_VENV_PYTHON_CANDIDATES = (
    "Scripts/python.exe", "Scripts/python",
    "bin/python.exe", "bin/python", "bin/python3",
)


def find_venv_python(venv_dir: Path) -> Path | None:
    # some Windows builds (MSYS2) use bin/, os.name doesn't decide
    for rel in _VENV_PYTHON_CANDIDATES:
        candidate = venv_dir / rel
        if candidate.is_file():
            return candidate
    for candidate in sorted(venv_dir.rglob("python*.exe")):
        if candidate.is_file():
            return candidate
    return None


def check_venv_python(python: Path) -> None:
    try:
        result = utils.run_capture([str(python), "-c", "import sys"])
    except OSError as exc:
        raise DepsError(f"The venv interpreter {python} failed to run: {exc}") from exc
    if result.returncode != 0:
        raise DepsError(
            f"The venv interpreter {python} failed to run: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )


def create_venv(venv_dir: Path) -> Path:
    venv_module.create(str(venv_dir), with_pip=True)
    python = find_venv_python(venv_dir)
    if python is None:
        raise DepsError(
            f"A venv directory was created at {venv_dir} but no python "
            f"executable was found inside it (created using {sys.executable})."
        )
    check_venv_python(python)
    return python


def clipshrink_script() -> Path:
    invoked = Path(sys.argv[0])
    if invoked.suffix == ".py" and invoked.exists():
        return invoked.resolve()
    fallback = project_root() / "clipshrink.py"
    if fallback.exists():
        return fallback
    raise DepsError("cannot locate clipshrink.py to relaunch inside the venv")


def _replace_process(args: list[str]) -> NoReturn:
    # Windows execv is emulated, a detached child loses the exit code
    if os.name == "nt":
        result = subprocess.run(args)
        raise SystemExit(result.returncode)
    os.execv(args[0], args)


def reexec_with_python(python: Path, argv: list[str]) -> NoReturn:
    # script path must be in argv or python parses our flags
    script = clipshrink_script()
    _replace_process([str(python), str(script), *argv])


def strip_venv_args(argv: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--venv":
            nxt = argv[index + 1] if index + 1 < len(argv) else None
            if nxt is not None and not nxt.startswith("-"):
                skip_next = True
            continue
        if token.startswith("--venv="):
            continue
        out.append(token)
    return out


def prompt_yes_no(prompt: str, default: bool = True) -> bool:
    while True:
        try:
            raw = input(prompt)
        except EOFError:
            return default
        raw = raw.strip().lower()
        if raw in ("", "y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


BINARIES_FILENAME = "binaries.json"
VERSIONS_FILENAME = "versions.json"


@dataclass(frozen=True)
class BinaryLinks:
    # url/mirror are Windows links, empty = none
    url: str = ""
    mirror: str = ""
    url_linux: str = ""
    mirror_linux: str = ""
    # sha256 of the matching download, empty = unverified
    sha256: str = ""
    sha256_mirror: str = ""
    sha256_linux: str = ""
    sha256_mirror_linux: str = ""


def default_binaries_path() -> Path:
    return project_root() / BINARIES_FILENAME


def platform_links(links: BinaryLinks | None) -> tuple[str, str]:
    # never fetch another platform's binary, macOS gets no links
    if links is None:
        return ("", "")
    if is_windows():
        return (links.url, links.mirror)
    if is_macos():
        return ("", "")
    return (links.url_linux, links.mirror_linux)


def platform_checksums(links: BinaryLinks | None) -> tuple[str, str]:
    if links is None:
        return ("", "")
    if is_windows():
        return (links.sha256, links.sha256_mirror)
    if is_macos():
        return ("", "")
    return (links.sha256_linux, links.sha256_mirror_linux)


def load_binaries(path: Path | None = None) -> dict[str, BinaryLinks]:
    path = path if path is not None else default_binaries_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    links: dict[str, BinaryLinks] = {}
    for label, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        url = entry.get("url", "")
        mirror = entry.get("mirror", "")
        url_linux = entry.get("url_linux", "")
        mirror_linux = entry.get("mirror_linux", "")
        sha256 = entry.get("sha256", "")
        sha256_mirror = entry.get("sha256_mirror", "")
        sha256_linux = entry.get("sha256_linux", "")
        sha256_mirror_linux = entry.get("sha256_mirror_linux", "")
        if not all(isinstance(v, str) for v in (
                url, mirror, url_linux, mirror_linux,
                sha256, sha256_mirror, sha256_linux, sha256_mirror_linux)):
            continue
        links[label] = BinaryLinks(
            url=url, mirror=mirror, url_linux=url_linux,
            mirror_linux=mirror_linux, sha256=sha256,
            sha256_mirror=sha256_mirror, sha256_linux=sha256_linux,
            sha256_mirror_linux=sha256_mirror_linux,
        )
    return links


def installed_versions_path(bin_path: Path) -> Path:
    return bin_path / VERSIONS_FILENAME


def load_installed_versions(bin_path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(installed_versions_path(bin_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    versions: dict[str, dict[str, Any]] = {}
    for label, entry in raw.items():
        if not isinstance(label, str) or not isinstance(entry, dict):
            continue
        versions[label] = {
            "url": entry.get("url", "") if isinstance(entry.get("url", ""), str) else "",
            "files": [f for f in entry.get("files", []) if isinstance(f, str)],
        }
    return versions


def save_installed_versions(bin_path: Path,
                            versions: dict[str, dict[str, Any]]) -> None:
    try:
        bin_path.mkdir(parents=True, exist_ok=True)
        installed_versions_path(bin_path).write_text(
            json.dumps(versions, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


@dataclass(frozen=True)
class ToolSpec:
    # download links come from binaries.json, keyed by label
    label: str
    binaries: tuple[str, ...]
    # ImageMagick is the only hard dependency
    required: bool = False


BACKEND_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("ImageMagick", ("magick",), required=True),
    ToolSpec("libvips", ("vips",)),
    ToolSpec("waifu2x-ncnn-vulkan", ("waifu2x-ncnn-vulkan",)),
    ToolSpec("realesrgan-ncnn-vulkan", ("realesrgan-ncnn-vulkan",)),
    ToolSpec("pngquant", ("pngquant",)),
    ToolSpec("ffmpeg", ("ffmpeg", "ffprobe")),
)

_EXEC_SUFFIXES = (".exe", ".dll", ".bat", ".cmd")


def bin_dir() -> Path:
    return project_root() / "bin"


def is_windows() -> bool:
    return sys.platform.startswith("win")


def is_macos() -> bool:
    # binaries.json POSIX links are Linux-only
    return sys.platform == "darwin"


def interactive() -> bool:
    return sys.stdin.isatty()


def _is_executable_file(path: Path) -> bool:
    # extensionless executables count, so manually placed POSIX binaries work
    suffix = path.suffix.lower()
    if suffix in _EXEC_SUFFIXES:
        return True
    return suffix == "" and os.access(path, os.X_OK)


def find_bin_tool(bin_path: Path, names: tuple[str, ...]) -> Path | None:
    if not bin_path.is_dir():
        return None
    wanted = {name.lower() for name in names}
    for candidate in sorted(bin_path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.stem.lower() in wanted and _is_executable_file(candidate):
            return candidate
    return None


def bin_path_dirs(bin_path: Path) -> list[Path]:
    dirs: list[Path] = []
    if not bin_path.is_dir():
        return dirs
    if any(
        _is_executable_file(file)
        for file in bin_path.iterdir() if file.is_file()
    ):
        dirs.append(bin_path)
    for subdir in sorted(bin_path.rglob("*")):
        if not subdir.is_dir():
            continue
        if any(
            _is_executable_file(file)
            for file in subdir.rglob("*") if file.is_file()
        ):
            dirs.append(subdir)
    return dirs


def activate_local_bins(warn: bool, bin_path: Path | None = None) -> list[Path]:
    bin_path = bin_path if bin_path is not None else bin_dir()
    dirs = bin_path_dirs(bin_path)
    if dirs:
        current = os.environ.get("PATH", "")
        additions = [str(d) for d in dirs if str(d) not in current.split(os.pathsep)]
        if additions:
            os.environ["PATH"] = os.pathsep.join([*additions, current])
    # bin/vips keeps libs in bin/vips/lib, pyvips reads LD_LIBRARY_PATH
    if not is_windows():
        vips_lib = bin_path / "vips" / "lib"
        if vips_lib.is_dir():
            ld_current = os.environ.get("LD_LIBRARY_PATH", "")
            entries = [p for p in ld_current.split(os.pathsep) if p]
            if str(vips_lib) not in entries:
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                    [str(vips_lib), *entries])
    if dirs and warn:
        found = [spec.label for spec in BACKEND_TOOLS if find_bin_tool(bin_path, spec.binaries)]
        if found:
            print(
                "Warning: using backend tools from the local bin/ directory "
                f"instead of the system PATH: {', '.join(found)}"
            )
    return dirs


def missing_backends(backend: BackendInfo, bin_path: Path | None = None) -> list[ToolSpec]:
    # libvips is present when the pyvips binding works
    bin_path = bin_path if bin_path is not None else bin_dir()
    missing: list[ToolSpec] = []
    for spec in BACKEND_TOOLS:
        if spec.label == "libvips" and backend.vips:
            continue
        if find_bin_tool(bin_path, spec.binaries) is not None:
            continue
        if all(shutil.which(binary) is not None for binary in spec.binaries):
            continue
        missing.append(spec)
    return missing


def outdated_backends(bin_path: Path | None = None,
                      links: dict[str, BinaryLinks] | None = None,
                      versions: dict[str, dict[str, Any]] | None = None,
                      ) -> list[ToolSpec]:
    # manually placed binaries have no manifest entry, leave them alone
    bin_path = bin_path if bin_path is not None else bin_dir()
    links = links if links is not None else load_binaries()
    versions = versions if versions is not None else load_installed_versions(bin_path)
    outdated: list[ToolSpec] = []
    for spec in BACKEND_TOOLS:
        tool_links = links.get(spec.label)
        if tool_links is None:
            continue
        primary, mirror = platform_links(tool_links)
        if not primary:
            continue
        if find_bin_tool(bin_path, spec.binaries) is None:
            continue
        entry = versions.get(spec.label)
        if entry is None:
            continue
        recorded = entry.get("url", "")
        if recorded == primary:
            continue
        if mirror and recorded == mirror:
            continue
        outdated.append(spec)
    return outdated


def check_backends_windows(backend: BackendInfo, first_run: bool,
                           bin_path: Path | None = None) -> bool:
    missing = missing_backends(backend, bin_path)
    if not missing:
        return False
    names = ", ".join(spec.label for spec in missing)
    print(f"Warning: missing backend tools: {names}.")
    if first_run:
        print("Don't worry if you skip this for now. "
              "We will ask again once you try to process an image.")
    if not interactive():
        print("Run 'clipshrink --install-deps' to install them automatically.")
        return False
    if not prompt_yes_no("Install missing backend tools now? (Y/n): "):
        return False
    install_backends(missing, bin_path)
    return True


def check_backends_posix(backend: BackendInfo, first_run: bool,
                         bin_path: Path | None = None) -> bool:
    missing = missing_backends(backend, bin_path)
    if not missing:
        return False
    if is_macos():
        # POSIX links are Linux-only, macOS gets package manager hints
        warn_backends_posix(backend, bin_path)
        return False
    names = ", ".join(spec.label for spec in missing)
    print(f"Warning: missing backend tools: {names}.")
    if first_run:
        print("Don't worry if you skip this for now. "
              "We will ask again once you try to process an image.")
    if not interactive():
        print("Run 'clipshrink --install-deps' to install them automatically.")
        return False
    if not prompt_yes_no("Install missing backend tools now? (Y/n): "):
        warn_backends_posix(backend, bin_path)
        return False
    install_backends(missing, bin_path)
    return True


def warn_backends_posix(backend: BackendInfo, bin_path: Path | None = None) -> None:
    missing = missing_backends(backend, bin_path)
    if not missing:
        return
    names = ", ".join(spec.label for spec in missing)
    print(f"Warning: missing backend tools: {names}.")
    print("Install them with your system's package manager first, or "
          "from the official project websites.")


def run_dependency_check(backend: BackendInfo, config: dict[str, Any],
                         config_path: Path | None = None) -> bool:
    # only called for real image work, error runs shouldn't burn first-run
    # state. true = caller re-detects.
    first_run = not config.get("first_run_completed", False)
    changed = False
    if is_windows():
        changed = check_backends_windows(backend, first_run)
    else:
        changed = check_backends_posix(backend, first_run)
    if first_run:
        config["first_run_completed"] = True
        save_config(config, config_path)
    return changed


def requirements_path() -> Path:
    return project_root() / "requirements.txt"


def install_python_deps(python: str | None = None,
                        requirements: Path | None = None) -> bool:
    # must run before backend binaries, py7zr comes from requirements.txt
    req = requirements if requirements is not None else requirements_path()
    if not req.exists():
        print(f"No requirements file at {req}; skipping Python dependencies.")
        return True
    print(f"Installing Python dependencies from {req.name} ...")
    cmd = [python or sys.executable, "-m", "pip", "install", "-r", str(req)]
    try:
        result = _run_timed(cmd, f"pip install -r {req.name}")
    except OSError as exc:
        # broken interpreter, missing pip, non-executable python
        print(f"Error installing Python dependencies: {exc}")
        return False
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        print("Error installing Python dependencies:")
        print(err)
        if "No module named pip" in err:
            print(
                "Hint: this interpreter has no pip and is probably a "
                "bundled/embedded build (e.g. Inkscape ships one as "
                "python.exe). Use an official Python instead, e.g. "
                "'py -3.12 clipshrink.py --install-deps'."
            )
        return False
    return True


def missing_python_packages(python: str | None = None) -> list[str]:
    # probe with the given interpreter so --venv can check a new one
    modules = ["pyvips"]
    if sys.platform.startswith("win"):
        modules.append("win32clipboard")
    elif sys.platform == "darwin":
        modules.append("AppKit")
    code = (
        "import importlib\n"
        f"missing = []\n"
        f"for _m in {modules!r}:\n"
        "    try:\n"
        "        importlib.import_module(_m)\n"
        "    except Exception:\n"
        "        missing.append(_m)\n"
        "print(' '.join(missing))"
    )
    try:
        result = utils.run_capture([python or sys.executable, "-c", code])
    except OSError:
        return modules
    if result.returncode != 0:
        return modules
    return result.stdout.split()


def ensure_venv_packages(python: Path) -> bool:
    missing = missing_python_packages(str(python))
    if not missing:
        return True
    print(f"Installing missing Python packages into the venv: "
          f"{', '.join(missing)} ...")
    return install_python_deps(python=str(python))


def suggest_venv_for_python_deps() -> None:
    missing = missing_python_packages()
    if not missing:
        return
    print(f"Note: some Python packages are missing from this interpreter: "
          f"{', '.join(missing)}.")
    print("Run 'clipshrink --install-deps --venv' to install the Python "
          "dependencies inside a virtual environment (recommended when pip "
          "cannot install into the current interpreter).")


@contextmanager
def _urlopen(url: str) -> Iterator[Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "clipshrink/2.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        yield response


def format_duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_speed(bytes_per_second: float) -> str:
    if bytes_per_second >= 1024 * 1024:
        return f"{bytes_per_second / (1024 * 1024):.1f} MB/s"
    if bytes_per_second >= 1024:
        return f"{bytes_per_second / 1024:.0f} KB/s"
    return f"{bytes_per_second:.0f} B/s"


class _ProgressLine:
    # captured commands print nothing, repaint this line. end() only
    # newlines if it was drawn.
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._drawn = False

    def update(self, elapsed: float) -> None:
        self._drawn = True
        print(f"\r  {self._prefix} {format_duration(elapsed)}   ",
              end="", flush=True)

    def end(self) -> None:
        if self._drawn:
            print()
            self._drawn = False


def _run_timed(cmd: list[str], label: str,
               **run_kwargs: Any) -> subprocess.CompletedProcess[str]:
    line = _ProgressLine(label)
    start = time.monotonic()
    try:
        result = utils.run_capture(cmd, poll_callback=line.update, **run_kwargs)
    finally:
        line.end()
    print(f"  {label}: {format_duration(time.monotonic() - start)}")
    return result


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} ...")
    start = time.monotonic()
    with _urlopen(url) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                elapsed = time.monotonic() - start
                speed = done / elapsed if elapsed > 0 else 0.0
                eta = (max(total - done, 0)) / speed if speed > 0 else 0.0
                print(f"\r  {min(done / total, 1.0):6.1%}  {_format_speed(speed)}"
                      f"  eta {format_duration(eta)}   ",
                      end="", flush=True)
    if total:
        # elapsed can be 0 on coarse clocks (Windows)
        elapsed = time.monotonic() - start
        if elapsed > 0 and done > 0:
            print(f"  ({_format_speed(done / elapsed)} average)")
        else:
            print()
    else:
        print()


def verify_checksum(path: Path, expected: str) -> None:
    # reject a truncated or swapped archive before it gets extracted
    if not expected:
        raise DepsError(f"no checksum for {path.name}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise DepsError(
            f"checksum mismatch for {path.name}: expected {expected}, got {actual}")


def load_py7zr() -> Any:
    try:
        import py7zr
    except ImportError as exc:
        raise DepsError(
            "py7zr is required to extract .7z archives; "
            "run 'clipshrink --install-deps' or 'pip install py7zr'."
        ) from exc
    return py7zr


def extract_archive(archive: Path, dest: Path) -> None:
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif name.endswith(".7z"):
        py7zr = load_py7zr()
        with py7zr.SevenZipFile(archive, mode="r") as zf:
            zf.extractall(str(dest))
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
        # tar filter keeps executable bits, Python < 3.12 has no filter arg
        with tarfile.open(archive) as tf:
            try:
                tf.extractall(dest, filter="tar")
            except TypeError:
                tf.extractall(dest)
    else:
        raise DepsError(f"Unsupported archive format: {archive.name}")


def _make_executables(bin_path: Path, moved: list[str],
                      binaries: tuple[str, ...]) -> None:
    # zip drops permissions (tar doesn't), restore the bit
    if is_windows():
        return
    wanted = {name.lower() for name in binaries}
    for rel in moved:
        path = bin_path / rel
        if path.stem.lower() in wanted:
            with contextlib.suppress(OSError):
                path.chmod(path.stat().st_mode | 0o111)


def _write_appimage_wrapper(bin_path: Path, binaries: tuple[str, ...]) -> str:
    # extracted binary loses AppRun's env, this wrapper rebuilds
    # MAGICK_HOME, config globs and libs. HOME stays under bin/.
    name = binaries[0]
    wrapper = bin_path / name
    home = bin_path / f"{name}.home"
    prefix = bin_path / "usr"
    script = (
        "#!/bin/sh\n"
        f"# clipshrink: portable environment for the AppImage-extracted {name}.\n"
        f"# Portable home ({name}.home): fontconfig/magick caches and user\n"
        "# config stay inside bin/, never touching the real $HOME.\n"
        f'HOME="{home}"\n'
        "export HOME\n"
        'mkdir -p "$HOME" 2>/dev/null || true\n'
        f"# AppRun's environment, rebuilt for the merged tree (globs instead\n"
        f"# of baked-in ImageMagick version paths).\n"
        f'MAGICK_HOME="{prefix}"\n'
        "export MAGICK_HOME\n"
        'MAGICK_CONFIGURE_PATH=""\n'
        f'for d in "{prefix}"/lib/ImageMagick-*/config-* '
        f'"{prefix}"/share/ImageMagick-* "{prefix}"/etc/ImageMagick-*; do\n'
        '    [ -d "$d" ] && MAGICK_CONFIGURE_PATH='
        '"${MAGICK_CONFIGURE_PATH:+$MAGICK_CONFIGURE_PATH:}$d"\n'
        "done\n"
        "export MAGICK_CONFIGURE_PATH\n"
        f'LD_LIBRARY_PATH="{prefix}/lib"\n'
        f'for d in "{prefix}"/lib/ImageMagick-*/modules-*/coders; do\n'
        '    [ -d "$d" ] && LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$d"\n'
        "done\n"
        "export LD_LIBRARY_PATH\n"
        f'exec "{prefix}/bin/{name}" "$@"\n'
    )
    wrapper.write_text(script, encoding="utf-8")
    with contextlib.suppress(OSError):
        wrapper.chmod(0o755)
    return name


def _single_root(dest: Path) -> Path:
    # some archives wrap everything in <tool>-<version>/, others don't
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


def merge_tree(root: Path, bin_path: Path) -> list[str]:
    # manifest records these so updates only replace this tool's files
    moved: list[str] = []
    for src in sorted(root.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(root)
        target = bin_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(src), str(target))
        moved.append(rel.as_posix())
    return moved


def _prune_empty_dirs(bin_path: Path, files: list[str]) -> None:
    dirs = {bin_path / f for f in files}
    parents: set[Path] = set()
    for d in dirs:
        parents.update(d.parents)
    for d in sorted(parents | dirs, key=lambda p: len(p.parts), reverse=True):
        if d == bin_path or bin_path not in d.parents:
            continue
        with contextlib.suppress(OSError):
            d.rmdir()


def _remove_stale_files(bin_path: Path, entry: dict[str, Any]) -> list[str]:
    stale = [f for f in entry.get("files", []) if isinstance(f, str)]
    for rel in stale:
        target = bin_path / rel
        try:
            if target.is_file():
                target.unlink()
        except OSError:
            pass
    _prune_empty_dirs(bin_path, stale)
    return stale


_VIPS_PKG_HINT = ("libvips: the package name varies by distro, so search "
                  "your system's package manager for libvips.")
_VIPS_BUILD_HINT = ("Building libvips needs a C toolchain, meson, ninja and "
                    "pkg-config plus the dev headers it uses (e.g. on "
                    "Debian/Ubuntu: build-essential meson ninja-build "
                    "pkg-config libglib2.0-dev libexpat1-dev). Install them "
                    "with your package manager, or install libvips itself "
                    "from it instead.")


def _pyvips_works(python: str | None = None) -> bool:
    # subprocess, because this one may have cached a failed import.
    # the binding dlopens libvips lazily, so do a real encode.
    code = "import pyvips; pyvips.Image.black(1, 1)"
    try:
        result = utils.run_capture([python or sys.executable, "-c", code])
    except OSError:
        return False
    return result.returncode == 0


def _pip_install(requirement: str) -> bool:
    if not is_in_venv():
        print("Not inside a virtual environment; skipping the pip install "
              f"of {requirement} (Python packages are only installed into "
              "a venv). Run 'clipshrink --install-deps --venv' to set one up.")
        return False
    print(f"Installing {requirement} ...")
    try:
        result = _run_timed(
            [sys.executable, "-m", "pip", "install", requirement],
            f"pip install {requirement}")
    except OSError as exc:
        print(f"Error installing {requirement}: {exc}")
        return False
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[-800:]
        print(f"Error installing {requirement}:")
        print(err)
        return False
    return True


def _vips_link_is_source(url: str) -> bool:
    # a source tarball is not a prebuilt binary
    name = url.rsplit("/", 1)[-1].lower()
    return name.startswith("vips-") and name.endswith(
        (".tar.xz", ".tar.gz", ".tar.bz2"))


def _have_c_toolchain() -> bool:
    return (any(shutil.which(cc) for cc in ("cc", "gcc", "clang"))
            and shutil.which("pkg-config") is not None)


def _find_build_tool(name: str, pip_name: str) -> str | None:
    # meson and ninja are pip-installable console scripts
    found = shutil.which(name)
    if found:
        return found
    _pip_install(pip_name)
    return shutil.which(name)


def _install_vips(spec: ToolSpec, tool_links: BinaryLinks, bin_path: Path,
                  versions: dict[str, dict[str, Any]]) -> bool:
    # pyvips[binary] wheel, then prebuilt binary, then source build
    if _pyvips_works():
        print("libvips is already usable through the pyvips binding.")
        return True
    print("libvips: trying the pyvips[binary] wheel (bundles libvips) ...")
    if _pip_install("pyvips[binary]") and _pyvips_works():
        print("Installed pyvips[binary]; its bundled libvips will be used.")
        return True
    primary, mirror = platform_links(tool_links)
    sha256, sha256_mirror = platform_checksums(tool_links)
    links = BinaryLinks(url=primary, mirror=mirror,
                        sha256=sha256, sha256_mirror=sha256_mirror)
    if not (primary or mirror) or _vips_link_is_source(primary):
        return _vips_without_binary(spec, links, bin_path, versions)
    return _install_one(spec, links, bin_path, versions)


def _vips_without_binary(spec: ToolSpec, links: BinaryLinks, bin_path: Path,
                         versions: dict[str, dict[str, Any]]) -> bool:
    print("No prebuilt libvips binary is available to download for this "
          "platform.")
    print("You can either install it through your distro's package manager "
          "or compile it from source.")
    if not interactive():
        print(_VIPS_PKG_HINT)
        return False
    if not prompt_yes_no("Do you want to try to build it from source "
                         "automatically? (Y/n): "):
        print("Please install libvips through your distro's package manager "
              "instead.")
        print(_VIPS_PKG_HINT)
        return False
    return _build_vips_from_source(links, bin_path, versions)


def _build_vips_from_source(links: BinaryLinks, bin_path: Path,
                            versions: dict[str, dict[str, Any]]) -> bool:
    # best-effort, failure cleans the partial prefix. manifest records the
    # tarball URL so outdated_backends tracks source builds too.
    prefix = bin_path / "vips"
    errors: list[str] = []
    for url, sha256 in ((links.url, links.sha256),
                        (links.mirror, links.sha256_mirror)):
        if not url:
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="clipshrink-vips-") as tmp:
                tmpdir = Path(tmp)
                archive = tmpdir / url.rsplit("/", 1)[-1]
                download_file(url, archive)
                if sha256:
                    verify_checksum(archive, sha256)
                else:
                    print(f"Warning: no checksum for {url}. Skipping verification.")
                extract_dir = tmpdir / "src"
                extract_dir.mkdir()
                extract_archive(archive, extract_dir)
                src = _single_root(extract_dir)
                meson = _find_build_tool("meson", "meson")
                ninja = _find_build_tool("ninja", "ninja")
                if not _have_c_toolchain() or meson is None or ninja is None:
                    print(_VIPS_BUILD_HINT)
                    return False
                for cmd in (
                    [meson, "setup", "build", f"--prefix={prefix}",
                     # Feature options take enabled/disabled/auto, booleans
                     # take true/false. Meson >= 1 rejects false for features.
                     "-Dintrospection=disabled",
                     "-Ddocs=false", "-Dexamples=false"],
                    [ninja, "-C", "build"],
                    [ninja, "-C", "build", "install"],
                ):
                    phase = ("configuring (meson setup)" if cmd[1] == "setup"
                             else "installing (ninja install)"
                             if cmd[-1] == "install"
                             else "compiling (ninja)")
                    result = _run_timed(cmd, phase, cwd=str(src))
                    if result.returncode != 0:
                        raise DepsError(
                            f"'{cmd[0]} {' '.join(cmd[1:3])}' failed:\n"
                            + (result.stderr or result.stdout or "").strip()[-1500:])
                if not (prefix / "lib").is_dir():
                    raise DepsError("the build reported success but installed "
                                    "no library under bin/vips/lib")
        except (DepsError, OSError, urllib.error.URLError, tarfile.TarError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        files = sorted(path.relative_to(bin_path).as_posix()
                       for path in prefix.rglob("*") if path.is_file())
        versions["libvips"] = {"url": url, "files": files}
        _pip_install("pyvips")
        print(f"Built and installed libvips into {prefix}.")
        print("Restart clipshrink (re-run your command) so the new library "
              "is picked up.")
        return True
    if errors:
        print(f"Building libvips from source failed: {'; '.join(errors)}")
    print("Please install libvips through your distro's package manager "
          "instead.")
    print(_VIPS_PKG_HINT)
    shutil.rmtree(prefix, ignore_errors=True)  # no partial builds
    return False


def _install_one(spec: ToolSpec, tool_links: BinaryLinks, bin_path: Path,
                 versions: dict[str, dict[str, Any]]) -> bool:
    # AppImages self-extract without FUSE, try the mirror if primary fails
    updating = spec.label in versions
    if updating:
        print(f"Updating {spec.label}: the installed bin/ version does not "
              "match binaries.json ...")
    errors: list[str] = []
    for url, sha256 in ((tool_links.url, tool_links.sha256),
                        (tool_links.mirror, tool_links.sha256_mirror)):
        if not url:
            continue
        archive_name = url.rsplit("/", 1)[-1]
        try:
            with tempfile.TemporaryDirectory(prefix="clipshrink-deps-") as tmp:
                tmpdir = Path(tmp)
                archive = tmpdir / archive_name
                download_file(url, archive)
                if sha256:
                    verify_checksum(archive, sha256)
                else:
                    print(f"Warning: no checksum for {url}. Skipping verification.")
                if archive_name.lower().endswith(".appimage"):
                    archive.chmod(0o755)
                    result = utils.run_capture([str(archive), "--appimage-extract"],
                                               cwd=str(tmpdir))
                    if result.returncode != 0:
                        raise DepsError(
                            "AppImage self-extraction failed: "
                            + (result.stderr or result.stdout or "").strip()[-500:])
                    extracted = tmpdir / "squashfs-root"
                    if not extracted.is_dir():
                        raise DepsError("AppImage extraction produced no squashfs-root")
                    root_known = True  # squashfs-root is the tree root
                else:
                    extracted = tmpdir / "extracted"
                    extracted.mkdir()
                    extract_archive(archive, extracted)
                    root_known = False
                # remove old files only after the new archive extracted
                if updating:
                    _remove_stale_files(bin_path, versions[spec.label])
                moved = merge_tree(extracted if root_known
                                   else _single_root(extracted), bin_path)
                _make_executables(bin_path, moved, spec.binaries)
                if root_known:
                    moved.append(_write_appimage_wrapper(bin_path, spec.binaries))
        except (DepsError, OSError, urllib.error.URLError,
                zipfile.BadZipFile, tarfile.TarError) as exc:
            errors.append(f"{url}: {exc}")
            continue
        versions[spec.label] = {"url": url, "files": sorted(moved)}
        return True
    if errors:
        print(f"Error installing {spec.label}: " + "; ".join(errors))
    return False


def _is_pipx_install() -> bool:
    # pipx leaves its metadata in the venv root
    venv = Path(sys.prefix).resolve()
    if not (venv / "pipx_metadata.json").is_file():
        return False
    root = project_root().resolve()
    return root == venv or venv in root.parents


def _print_manual_hint(spec: ToolSpec) -> None:
    status = "required" if spec.required else "optional"
    names = " and ".join(spec.binaries)
    noun = "binary" if len(spec.binaries) == 1 else "binaries"
    pronoun = "it" if len(spec.binaries) == 1 else "them"
    tool = "tool" if len(spec.binaries) == 1 else "tools"
    print(f"{spec.label} is {status}. Install {pronoun} with your system's "
          f"package manager first, or download the {names} {noun} from the "
          f"official {spec.label} website and install {pronoun} manually.")
    print(f"If you don't wish to install the {tool} system-wide or add "
          f"{pronoun} to PATH, you may place the {noun} in the bin folder: "
          f"{bin_dir()}.")
    if _is_pipx_install():
        print(f"With pipx, the package folder is {project_root()}, inside the "
              f"pipx environment. Its bin folder is {bin_dir()}.")
        print("Upgrading clipshrink may replace files there.")
    if not spec.required:
        print("Continuing with the rest of the installation.")


def install_backends(missing: list[ToolSpec],
                     bin_path: Path | None = None,
                     links: dict[str, BinaryLinks] | None = None) -> list[str]:
    # tools already in the manifest are updates, old files go first
    bin_path = bin_path if bin_path is not None else bin_dir()
    links = links if links is not None else load_binaries()
    if not links:
        print(f"Note: {BINARIES_FILENAME} is missing or empty "
              f"({default_binaries_path()}); no automatic downloads are "
              "available.")
    try:
        bin_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error creating {bin_path}: {exc}")
        print("The project directory may be read-only (e.g. a system-wide "
              "pip install); install the tools manually instead.")
        return []
    versions = load_installed_versions(bin_path)
    installed: list[str] = []
    run_start = time.monotonic()
    for spec in missing:
        tool_links = links.get(spec.label)
        primary, mirror = platform_links(tool_links)
        sha256, sha256_mirror = platform_checksums(tool_links)
        tool_start = time.monotonic()
        if spec.label == "libvips":
            # pyvips-first: wheel, prebuilt binary, source build. Handles its
            # own manifest entries and messages.
            was_update = spec.label in versions
            if _install_vips(spec, tool_links or BinaryLinks(), bin_path,
                             versions):
                print(f"{'Updated' if was_update else 'Installed'} libvips "
                      f"({format_duration(time.monotonic() - tool_start)}).")
                installed.append(spec.label)
            else:
                _print_manual_hint(spec)
            continue
        if not (primary or mirror):
            print(f"Skipping {spec.label}: no automatic download is "
                  "available for this platform.")
            _print_manual_hint(spec)
            continue
        was_update = spec.label in versions
        downloads = BinaryLinks(url=primary, mirror=mirror,
                                sha256=sha256, sha256_mirror=sha256_mirror)
        if not _install_one(spec, downloads, bin_path, versions):
            _print_manual_hint(spec)
            continue
        if find_bin_tool(bin_path, spec.binaries) is not None:
            print(f"{'Updated' if was_update else 'Installed'} "
                  f"{spec.label} into {bin_path} "
                  f"({format_duration(time.monotonic() - tool_start)}).")
            installed.append(spec.label)
        else:
            print(f"Error installing {spec.label}: "
                  f"{spec.binaries[0]} not found in the archive.")
            versions.pop(spec.label, None)
            _print_manual_hint(spec)
    save_installed_versions(bin_path, versions)
    if installed:
        print(f"Install complete: {len(installed)} tool(s) in "
              f"{format_duration(time.monotonic() - run_start)}.")
    return installed
