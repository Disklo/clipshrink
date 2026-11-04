import copy
import os
import sys
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

from core import __version__, completions, deps, log, memgate, utils
from core.args import Args
from core.backend import BackendInfo, detect_backends, reload_pyvips, set_backend
from core.cli import create_parser, parse_cli_args
from core.deps import VENV_DIRNAME
from core.fileops import collect_images, compute_output_name, is_image_file
from core.pipeline import process_image
from core.validation import ValidationError, validate_flags

# flags that aren't image work for --install-deps
_INSTALL_ONLY_FLAGS = frozenset({
    "install_deps", "venv", "venv_skip",
    "verbose", "debug", "log_file",
    "print_completions", "print_man",
})


def _print_tool_warnings(backend: BackendInfo) -> None:
    if not backend.vips:
        print("Warning: vips is not available. Some operations may be slower or unavailable.")
    if not backend.pngquant:
        print("Warning: pngquant is not available. Operations requiring pngquant will be skipped.")
    if not backend.realesrgan:
        print("Warning: realesrgan-ncnn-vulkan is not available. Operations requiring realesrgan will be skipped.")
    if not backend.waifu2x:
        print("Warning: waifu2x-ncnn-vulkan is not available. Operations requiring waifu2x will be skipped.")


def _install_deps_only(provided: set[str]) -> bool:
    # provided = what the CLI actually set, not defaults
    return provided <= _INSTALL_ONLY_FLAGS


def _run_install_deps() -> bool:
    # python packages only go into a venv. PATH tools stay alone,
    # stale bin/ links get refreshed.
    start = time.monotonic()
    if deps.is_in_venv():
        ok = deps.install_python_deps()
    else:
        print("Not inside a virtual environment; skipping Python package "
              "installation (Python packages are only installed into a "
              "venv). Run 'clipshrink --install-deps --venv' to set one up.")
        ok = True
    backend = detect_backends()
    installed: list[str] = []
    # no downloads for macOS, just package manager hints
    if deps.is_macos():
        deps.warn_backends_posix(backend)
        return ok
    links = deps.load_binaries()
    missing = deps.missing_backends(backend)
    outdated = deps.outdated_backends(links=links)
    if missing:
        installed.extend(deps.install_backends(missing, links=links))
        # bin/ is populated, python packages may still be missing here
        deps.suggest_venv_for_python_deps()
    if outdated:
        installed.extend(deps.install_backends(outdated, links=links))
    if not missing and not outdated:
        print("All backend tools are already installed.")
    if installed:
        print(f"Dependency installation finished in "
              f"{deps.format_duration(time.monotonic() - start)} "
              f"({len(installed)} tool(s) installed or updated).")
    # pyvips binds at import, detection is cached
    vips_reloaded = ok and reload_pyvips()
    if installed or vips_reloaded:
        detect_backends.cache_clear()
    return ok


def _run_venv_flow(args: Args, config: dict[str, Any], argv: list[str]) -> int | None:
    # exit code or None to continue
    if deps.is_in_venv():
        print(f"Virtual environment detected: {sys.prefix}")
        return None
    venv_path = Path(args.venv or config.get("custom_venv_path") or VENV_DIRNAME)
    launch_args = deps.strip_venv_args(argv)
    python = deps.find_venv_python(venv_path)
    if python is not None:
        try:
            deps.check_venv_python(python)
        except deps.DepsError:
            python = None  # recreate broken/partial venv
    if python is not None:
        if Path(sys.executable).resolve() == python.resolve():
            print(f"Already running with the {venv_path} interpreter.")
            return None
        print(f"Virtual environment found at {venv_path}. Re-launching...")
        _install_venv_requirements(python)
        deps.reexec_with_python(python, launch_args)
        return 0  # unreachable, exec replaces us
    if not deps.interactive():
        print("Virtual environment not detected; run interactively to "
              "create one with --venv.")
        return None
    if not deps.prompt_yes_no("Create virtual environment automatically? (Y/n): "):
        return None
    try:
        python = deps.create_venv(venv_path)
    except Exception as exc:
        # OSError, DepsError, broken ensurepip
        print(f"Error: could not create the virtual environment: {exc}")
        return 1
    config["custom_venv_path"] = str(venv_path)
    deps.save_config(config)
    print(f"Virtual environment created at {venv_path}. Re-launching...")
    _install_venv_requirements(python)
    deps.reexec_with_python(python, launch_args)
    return 0  # unreachable, exec replaces us


def _install_venv_requirements(python: Path) -> None:
    # non-fatal, file processing works without these
    if not deps.ensure_venv_packages(python):
        print(
            "Warning: could not install Python packages into the venv; "
            "clipboard/vips features may be unavailable. This usually means "
            "the venv's base interpreter is a non-standard build that cannot "
            f"use PyPI wheels (base: {sys.base_prefix}). Recreate the venv "
            "with an official Python, e.g. 'py -3.12 clipshrink.py --venv'."
        )


def _count_subdirectories(directory: Path) -> int:
    count = 0
    for _root, dirs, _files in os.walk(directory):
        count += len(dirs)
    return count


def _per_image_args(args: Args, output: str | None) -> Args:
    # dataclass args are frozen, duck-typed ones get a copy
    if is_dataclass(args):
        return replace(args, output=output)
    task_args = copy.copy(args)
    task_args.output = output
    return task_args


def _batch_workers(args: Args) -> int:
    jobs = getattr(args, "jobs", None)
    if jobs:
        return max(1, int(jobs))
    return utils.MAX_CONCURRENT_JOBS


def _limit_tool_threads(workers: int) -> None:
    # N children x cpu_count threads would oversubscribe
    if workers <= 1:
        return
    per_tool = max(1, (os.cpu_count() or 1) // workers)
    for var in ("MAGICK_THREAD_LIMIT", "VIPS_CONCURRENCY"):
        if var not in os.environ:
            os.environ[var] = str(per_tool)


def _batch_exit_code(failed: int) -> int:
    if utils.is_aborted():
        return 130
    return 1 if failed else 0


def _run_batch_serial(tasks: list[tuple[Path, Args]], total: int) -> int:
    processed = 0
    failed = 0
    for index, (img, task_args) in enumerate(tasks, 1):
        utils.reset_abort()
        print(f"[{index}/{total}]")
        result = process_image(img, task_args)
        if result["success"]:
            processed += 1
        else:
            failed += 1
            print(f"[{index}/{total}] Failed to process: {img}")
            if utils.get_last_error():
                print(f"   Reason: {utils.get_last_error()}")

        if utils.is_aborted():
            print("Warning: Interrupted. Stopping batch processing.")
            break

    print(f"Batch complete: {processed} succeeded, {failed} failed.")
    return _batch_exit_code(failed)


def _run_batch_parallel(tasks: list[tuple[Path, Args]], workers: int, total: int) -> int:
    processed = 0
    failed = 0
    done = 0
    print_lock = threading.Lock()
    utils.reset_abort()

    def worker(task: tuple[Path, Args]) -> tuple[Path, dict[str, object], str]:
        img, task_args = task
        result = process_image(img, task_args)
        return img, result, utils.get_last_error()

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="clipshrink-batch"
    ) as ex:
        futures = [ex.submit(worker, task) for task in tasks]
        for fut in as_completed(futures):
            if utils.is_aborted():
                for pending in futures:
                    if not pending.done():
                        pending.cancel()
            done += 1
            with print_lock:
                try:
                    img, result, err = fut.result()
                except CancelledError:
                    continue  # aborted before it ran, not a failure
                except Exception as e:
                    failed += 1
                    print(f"[{done}/{total}] Failed to process: {e}")
                    continue
                if result["success"]:
                    processed += 1
                    print(f"[{done}/{total}] Processed: {img}")
                else:
                    failed += 1
                    print(f"[{done}/{total}] Failed to process: {img}")
                    if err:
                        print(f"   Reason: {err}")

    if utils.is_aborted():
        print("Warning: Interrupted. Stopping batch processing.")
    print(f"Batch complete: {processed} succeeded, {failed} failed.")
    return _batch_exit_code(failed)


def _run_folder_mode(args: Args, input_dir: Path | None = None,
                     image_list: list[Path] | None = None) -> int:
    if input_dir is None:
        # tests pass input via args
        input_dir = Path(getattr(args, "input", "")).resolve()
    if image_list is None:
        image_list = collect_images(input_dir, args.recursive)
    total = len(image_list)

    if total == 0:
        print(f"Error: No images found in '{args.input}'.")
        return 1

    subfolder_count = _count_subdirectories(input_dir) if args.recursive else 0

    if total > 500 or subfolder_count > 50:
        print(f"Warning: {total} images across {subfolder_count} subfolders.")
        try:
            response = input("Are you sure you want to continue? (y/N) ")
        except EOFError:
            response = ""
        if response.strip().lower() != "y":
            print("Aborted.")
            return 1

    single_image = total == 1
    output_dir = None
    if args.output:
        out = Path(args.output)
        if out.exists() and not out.is_dir():
            print(f"Error: Multiple images require output to be a directory. '{args.output}' is a file.")
            return 1
        output_dir = out
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not single_image:
        print("Error: Multiple images require an output directory (-o).")
        return 1

    print(f"Found {total} image(s). Processing...")

    # one args per image, workers must not share output paths
    tasks = []
    for img in image_list:
        out_path = None
        if output_dir is not None:
            if single_image and args.no_suffix:
                out_path = output_dir / img.name
            else:
                out_path = compute_output_name(
                    img, output_dir, input_dir, args.recursive,
                    args.no_suffix, single_image, args.format,
                )
            out_path.parent.mkdir(parents=True, exist_ok=True)
        task_args = _per_image_args(args, str(out_path) if out_path is not None else None)
        tasks.append((img, task_args))

    workers = _batch_workers(args)
    log.log(f"Batch: {total} image(s), {workers} worker(s), "
            f"{'recursive' if args.recursive else 'flat'}")
    if workers > 1 and total > 1:
        _limit_tool_threads(workers)
        return _run_batch_parallel(tasks, workers, total)
    return _run_batch_serial(tasks, total)


def main(*argv: str) -> int:
    # keep big temp files off tmpfs
    memgate.apply_process_tempdir()
    parser = create_parser()
    # no args = clipboard in, clipboard out
    ns, provided = parse_cli_args(parser, list(argv))
    args = Args.from_namespace(ns)

    # before logging, a bad --log-file must not block introspection
    if args.print_completions or args.print_man:
        kind = "man" if args.print_man else "completions"
        print(completions.print_output(parser, kind, args.print_completions))
        return 0

    try:
        log.configure(verbose=args.verbose, debug=args.debug, log_path=args.log_file)
    except OSError as exc:
        print(f"Error: cannot open log file {args.log_file!r}: {exc}")
        return 1

    config = deps.load_config()

    # bin/ wins over PATH, activate before venv/install checks
    deps.activate_local_bins(bool(config.get("warn_on_local_binaries", True)))

    # --venv re-execs first so --install-deps runs inside it. conflict check
    # lives here because validate_flags needs binaries.
    if args.venv and args.venv_skip:
        print("Error: --venv and --venv-skip are mutually exclusive")
        return 1
    if args.venv:
        code = _run_venv_flow(args, config, list(argv) if argv else sys.argv[1:])
        if code is not None:
            return code

    # validate after install so scaler checks see new binaries
    install_only = args.install_deps and _install_deps_only(provided)
    if args.install_deps:
        ok = _run_install_deps()
        deps.activate_local_bins(bool(config.get("warn_on_local_binaries", True)))
        if install_only:
            return 0 if ok else 1
        if not ok:
            print("Warning: dependency installation did not complete "
                  "successfully; continuing with the requested operation.")

    try:
        validate_flags(args)
    except ValidationError as e:
        print(f"Error: {e}")
        return 1

    if not args.install_deps:
        venv_notice = not args.venv_skip and not config.get("skip_venv", False)
        if venv_notice and not deps.is_in_venv() and deps.interactive():
            print("Virtual environment not detected. Run with --venv to set "
                  "one up, or --venv-skip to ignore.")

    # resolve input first so errors don't burn the first-run check
    image_list: list[Path] | None = None
    will_process = True
    if args.input:
        input_path = Path(args.input)
        if input_path.is_dir():
            image_list = collect_images(input_path, args.recursive)
            will_process = bool(image_list)
        elif input_path.is_file():
            will_process = is_image_file(input_path)
        else:
            print(f"Error: Input path '{args.input}' is not a file or directory.")
            return 1
    else:
        input_path = None

    # dependency check may install magick, --install-deps already ran it
    backend = detect_backends()
    if will_process and not args.install_deps and deps.run_dependency_check(backend, config):
        detect_backends.cache_clear()
        backend = detect_backends()

    if not backend.magick:
        print("Error: ImageMagick (magick) is required but not found.")
        return 1
    if backend.os_type == "unknown":
        print("Your operating system is not supported.")
        return 1

    log.log(f"clipshrink {__version__} | backend: primary={backend.primary_backend} "
            f"vips={backend.vips} magick={backend.magick} pngquant={backend.pngquant} "
            f"os={backend.os_type}")
    _print_tool_warnings(backend)
    set_backend(backend)
    utils.register_signal_handlers()

    if image_list is not None:
        return _run_folder_mode(args, input_path, image_list)

    if args.output:
        out = Path(args.output)
        if out.is_dir() and input_path is not None:
            args = _per_image_args(args, str(out / input_path.name))
        elif out.is_dir():
            print(f"Error: Output path '{args.output}' is a directory; clipboard input needs a file path.")
            return 1

    utils.reset_abort()
    result = process_image(input_path, args)
    return 0 if result["success"] else 1


def console_main() -> None:
    sys.exit(main(*sys.argv[1:]))


if __name__ == "__main__":
    console_main()
