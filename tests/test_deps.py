import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from clipshrink import main
from core import deps
from core.args import Args
from core.backend import BackendInfo
from core.cli import create_parser
from core.deps import VENV_DIRNAME
from core.validation import ValidationError, validate_flags


def _parse(argv: list[str]) -> argparse.Namespace:
    return create_parser().parse_args(argv)


@pytest.fixture(autouse=True)
def _linux_links(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "is_macos", lambda: False)


def _args(argv: list[str]) -> Args:
    return Args.from_namespace(_parse(argv))


def _backend(**overrides: object) -> BackendInfo:
    values: dict[str, object] = {
        "os_type": "Windows", "vips": False, "magick": False,
        "pngquant": False, "realesrgan": False, "waifu2x": False,
    }
    values.update(overrides)
    return BackendInfo(**values)


def test_venv_flag_defaults_to_venv_dirname():
    assert _parse(["--venv"]).venv == VENV_DIRNAME == ".venv"


def test_venv_flag_accepts_a_path():
    assert _parse(["--venv", "envs/custom"]).venv == "envs/custom"


def test_venv_flag_default_is_none():
    assert _args([]).venv is None


def test_venv_skip_and_install_deps_flags():
    ns = _parse(["--venv-skip", "--install-deps"])
    assert ns.venv_skip is True
    assert ns.install_deps is True


def test_args_dataclass_has_env_fields():
    args = _args(["--venv", "x"])
    assert args.venv == "x"
    assert args.venv_skip is False
    assert args.install_deps is False


def test_venv_and_venv_skip_conflict():
    with pytest.raises(ValidationError, match="--venv and --venv-skip"):
        validate_flags(_args(["--venv", "--venv-skip"]))


def test_get_clipboard_image_missing_binding_is_clean(monkeypatch: pytest.MonkeyPatch):
    from core.clipboard import ClipboardError, get_clipboard_image

    monkeypatch.setitem(sys.modules, "win32clipboard", None)
    with pytest.raises(ClipboardError, match="pywin32"):
        get_clipboard_image("Windows")


def test_set_clipboard_image_missing_binding_is_clean(monkeypatch: pytest.MonkeyPatch):
    from core.clipboard import ClipboardError, set_clipboard_image

    monkeypatch.setitem(sys.modules, "win32clipboard", None)
    with pytest.raises(ClipboardError, match="pywin32"):
        set_clipboard_image(b"x", "Windows")


def test_missing_python_packages_probe(monkeypatch: pytest.MonkeyPatch):
    class _Out:
        returncode = 0
        stdout = "pyvips win32clipboard\n"
        stderr = ""

    monkeypatch.setattr(deps.utils, "run_capture", lambda cmd, **kw: _Out())
    assert deps.missing_python_packages("/some/python") == \
        ["pyvips", "win32clipboard"]

    class _Broken:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(deps.utils, "run_capture", lambda cmd, **kw: _Broken())
    result = deps.missing_python_packages()
    assert "pyvips" in result


def test_ensure_venv_packages_installs_only_when_missing(monkeypatch: pytest.MonkeyPatch):
    installed: list[str] = []
    monkeypatch.setattr(deps, "missing_python_packages", lambda python=None: [])
    assert deps.ensure_venv_packages(Path("v")) is True

    monkeypatch.setattr(deps, "missing_python_packages", lambda python=None: ["pyvips"])
    monkeypatch.setattr(deps, "install_python_deps",
                        lambda python=None, requirements=None:
                        installed.append(python) or True)
    assert deps.ensure_venv_packages(Path("v")) is True
    assert installed == ["v"]


def test_install_deps_only_tracks_provided_destinations():
    from clipshrink import _install_deps_only
    from core.cli import parse_cli_args

    def provided(*argv: str) -> set[str]:
        return parse_cli_args(create_parser(), list(argv))[1]

    assert _install_deps_only(provided("--install-deps")) is True
    assert _install_deps_only(provided("--install-deps", "-v")) is True
    assert _install_deps_only(provided("--install-deps", "--log-file", "x")) is True
    assert _install_deps_only(provided("--install-deps", "--venv")) is True
    assert _install_deps_only(provided("--install-deps", "--venv-skip")) is True
    assert _install_deps_only(provided("--install-deps", "--subsampling", "4:2:0")) is False
    assert _install_deps_only(provided("--install-deps", "-i", "x.png")) is False
    assert _install_deps_only(provided("--install-deps", "--dry-run")) is False


def test_install_deps_solo_installs_and_exits(monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]):
    calls: list[str] = []
    monkeypatch.setattr(deps, "is_in_venv", lambda: True)
    monkeypatch.setattr(deps, "install_python_deps", lambda **kw: calls.append("py") or True)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    monkeypatch.setattr(deps, "missing_backends", lambda backend, bin_path=None: [])
    assert main("--install-deps") == 0
    assert calls == ["py"]
    assert "All backend tools are already installed." in capsys.readouterr().out
    assert "clipboard" not in capsys.readouterr().out.lower()


def test_install_deps_pip_skipped_outside_venv(monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]):
    calls: list[str] = []
    monkeypatch.setattr(deps, "is_in_venv", lambda: False)
    monkeypatch.setattr(deps, "install_python_deps",
                        lambda **kw: calls.append("py") or True)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    monkeypatch.setattr(deps, "missing_backends", lambda backend, bin_path=None: [])
    assert main("--install-deps") == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "skipping Python package" in out
    assert "--install-deps --venv" in out


def test_install_deps_combines_with_processing_flags(monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str],
                                                     tmp_path: Path):
    calls: list[str] = []
    monkeypatch.setattr(deps, "is_in_venv", lambda: True)
    monkeypatch.setattr(deps, "install_python_deps",
                        lambda **kw: calls.append("py") or True)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    monkeypatch.setattr(deps, "missing_backends", lambda backend, bin_path=None: [])
    missing = tmp_path / "photo.png"
    assert main("--install-deps", "-i", str(missing)) == 1
    assert calls == ["py"]
    out = capsys.readouterr().out
    assert "All backend tools are already installed." in out
    assert "must be used alone" not in out
    assert "is not a file or directory" in out


def test_install_deps_with_venv_reexecs_into_the_venv(monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path):
    import clipshrink as entry

    _existing_venv(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(deps, "is_in_venv", lambda: False)
    monkeypatch.setattr(deps, "check_venv_python", lambda python: None)
    monkeypatch.setattr(entry, "_install_venv_requirements", lambda python: None)
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(deps, "_replace_process",
                        lambda args: captured.update(path=args[0], args=args))
    monkeypatch.setattr(deps.sys, "argv",
                        [str(deps.project_root() / "clipshrink.py"),
                         "--install-deps", "--venv"])

    assert main("--install-deps", "--venv") == 0
    assert Path(captured["path"]).resolve() == \
        (tmp_path / ".venv" / "bin" / "python.exe").resolve()
    assert captured["args"][2:] == ["--install-deps"]


def test_install_deps_with_venv_creates_then_reexecs(monkeypatch: pytest.MonkeyPatch,
                                                     tmp_path: Path):
    import clipshrink as entry

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(deps, "is_in_venv", lambda: False)
    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: True)
    monkeypatch.setattr(deps, "save_config", lambda config, path=None: None)
    monkeypatch.setattr(entry, "_install_venv_requirements", lambda python: None)
    created: list[Path] = []
    monkeypatch.setattr(deps, "create_venv",
                        lambda path: created.append(path) or path / "Scripts" / "python.exe")
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(deps, "_replace_process",
                        lambda args: captured.update(path=args[0], args=args))

    assert main("--install-deps", "--venv") == 0
    assert [p.name for p in created] == [".venv"]
    assert captured["path"] == str(Path(".venv") / "Scripts" / "python.exe")


def test_install_deps_solo_still_runs_in_current_interpreter(monkeypatch: pytest.MonkeyPatch,
                                                             capsys: pytest.CaptureFixture[str]):
    calls: list[str] = []
    monkeypatch.setattr(deps, "is_in_venv", lambda: True)
    monkeypatch.setattr(deps, "install_python_deps", lambda **kw: calls.append("py") or True)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    monkeypatch.setattr(deps, "missing_backends", lambda backend, bin_path=None: [])
    assert main("--install-deps") == 0
    assert calls == ["py"]
    assert "All backend tools are already installed." in capsys.readouterr().out


def test_install_python_deps_pip_missing_hint(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("py7zr\n", encoding="utf-8")

    class _NoPip:
        returncode = 1
        stdout = ""
        stderr = "C:\\Program Files\\Inkscape\\bin\\python.exe: No module named pip"

    monkeypatch.setattr(deps.utils, "run_capture", lambda cmd, **kw: _NoPip())
    assert deps.install_python_deps(requirements=reqs) is False
    out = capsys.readouterr().out
    assert "No module named pip" in out
    assert "py -3.12" in out


def test_suggest_venv_after_binary_install(monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "missing_python_packages", lambda python=None: ["pyvips"])
    deps.suggest_venv_for_python_deps()
    out = capsys.readouterr().out
    assert "some Python packages are missing" in out
    assert "pyvips" in out
    assert "--install-deps --venv" in out

    monkeypatch.setattr(deps, "missing_python_packages", lambda python=None: [])
    deps.suggest_venv_for_python_deps()
    assert capsys.readouterr().out == ""


def test_install_deps_installs_missing_backends(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}
    monkeypatch.setattr(deps, "install_python_deps", lambda **kw: True)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    spec = deps.ToolSpec("pngquant", ("pngquant",))
    monkeypatch.setattr(deps, "missing_backends", lambda backend, bin_path=None: [spec])
    monkeypatch.setattr(deps, "outdated_backends",
                        lambda bin_path=None, links=None: [])
    monkeypatch.setattr(deps, "install_backends",
                        lambda missing, bin_path=None, links=None:
                        seen.setdefault("missing", missing))
    assert main("--install-deps") == 0
    assert seen["missing"] == [spec]


def test_install_deps_refreshes_cached_backend_detection(monkeypatch: pytest.MonkeyPatch):
    import clipshrink as entry
    from core import backend as backend_mod
    from core.backend import detect_backends

    detect_backends.cache_clear()
    try:
        magick_installed = [False]
        monkeypatch.setattr(backend_mod, "_check_cli",
                            lambda cmd: cmd == "magick" and magick_installed[0])
        monkeypatch.setattr(backend_mod, "_check_mozjpeg", lambda: False)

        monkeypatch.setattr(deps, "is_in_venv", lambda: True)
        monkeypatch.setattr(deps, "install_python_deps", lambda **kw: True)
        monkeypatch.setattr(deps, "is_windows", lambda: True)
        monkeypatch.setattr(deps, "load_binaries", lambda: {})
        spec = deps.ToolSpec("ImageMagick", ("magick",), required=True)
        monkeypatch.setattr(deps, "missing_backends",
                            lambda backend, bin_path=None: [spec])
        monkeypatch.setattr(deps, "outdated_backends",
                            lambda bin_path=None, links=None: [])
        monkeypatch.setattr(deps, "suggest_venv_for_python_deps", lambda: None)
        reload_calls: list[int] = []

        def fake_reload() -> bool:
            reload_calls.append(1)
            return True

        monkeypatch.setattr(entry, "reload_pyvips", fake_reload)

        def fake_install(missing: list[deps.ToolSpec], bin_path: Path | None = None,
                         links: dict[str, object] | None = None) -> list[str]:
            magick_installed[0] = True
            return [s.label for s in missing]

        monkeypatch.setattr(deps, "install_backends", fake_install)

        assert entry._run_install_deps() is True
        assert detect_backends().magick is True
        assert reload_calls == [1]
    finally:
        detect_backends.cache_clear()


def test_load_config_missing_file_returns_defaults(tmp_path: Path):
    config = deps.load_config(tmp_path / "config.json")
    assert config == deps.DEFAULT_CONFIG


def test_config_round_trip(tmp_path: Path):
    path = tmp_path / "config.json"
    config = deps.load_config(path)
    config["first_run_completed"] = True
    config["custom_venv_path"] = "envs/dev"
    deps.save_config(config, path)
    reloaded = deps.load_config(path)
    assert reloaded["first_run_completed"] is True
    assert reloaded["custom_venv_path"] == "envs/dev"
    assert reloaded["warn_on_local_binaries"] is True
    assert reloaded["skip_venv"] is False


def test_load_config_corrupt_file_returns_defaults(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    assert deps.load_config(path) == deps.DEFAULT_CONFIG


def test_load_config_rejects_wrong_types(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "first_run_completed": "yes", "warn_on_local_binaries": 1,
        "custom_venv_path": 42, "skip_venv": ["no"],
    }), encoding="utf-8")
    config = deps.load_config(path)
    for key in ("first_run_completed", "warn_on_local_binaries", "skip_venv"):
        assert config[key] == deps.DEFAULT_CONFIG[key]
    assert config["custom_venv_path"] is None


def test_save_config_preserves_unknown_keys(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"skip_venv": True, "user_note": "keep me"}),
                    encoding="utf-8")
    config = deps.load_config(path)
    deps.save_config(config, path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["user_note"] == "keep me"
    assert on_disk["skip_venv"] is True


def test_save_config_is_best_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "config.json"

    def _deny(self: object, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(path), "write_text", _deny)
    config = deps.load_config(path)
    config["first_run_completed"] = True
    deps.save_config(config, path)


def test_is_in_venv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.sys, "base_prefix", "/base")
    monkeypatch.setattr(deps.sys, "prefix", "/base")
    assert deps.is_in_venv() is False
    monkeypatch.setattr(deps.sys, "prefix", "/base/venv")
    assert deps.is_in_venv() is True


def test_venv_python_layout_per_os():
    venv_dir = Path("/somewhere/.venv")
    with mock.patch.object(deps.os, "name", "nt"):
        assert deps.venv_python(venv_dir) == venv_dir / "Scripts" / "python.exe"
    with mock.patch.object(deps.os, "name", "posix"):
        assert deps.venv_python(venv_dir) == venv_dir / "bin" / "python"


def test_find_venv_python_layouts(tmp_path: Path):
    scripts = tmp_path / "v1" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_bytes(b"")
    assert deps.find_venv_python(tmp_path / "v1") == scripts / "python.exe"
    posix = tmp_path / "v2" / "bin"
    posix.mkdir(parents=True)
    (posix / "python.exe").write_bytes(b"")
    assert deps.find_venv_python(tmp_path / "v2") == posix / "python.exe"
    (tmp_path / "v3").mkdir()
    assert deps.find_venv_python(tmp_path / "v3") is None


def _fake_venv_create(layout: str):
    def _create(path: str, with_pip: bool = False) -> None:
        python_dir = Path(path) / layout
        python_dir.mkdir(parents=True)
        (python_dir / "python.exe").write_bytes(b"")
    return _create


class _RunOk:
    returncode = 0
    stdout = ""
    stderr = ""


class _RunFail:
    returncode = 1
    stdout = ""
    stderr = "interpreter exploded"


def test_create_venv_discovers_noncanonical_layout(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.venv_module, "create", _fake_venv_create("bin"))
    monkeypatch.setattr(deps.utils, "run_capture", lambda cmd, **kw: _RunOk())
    assert deps.create_venv(tmp_path / "v") == tmp_path / "v" / "bin" / "python.exe"


def test_create_venv_fails_when_no_python_inside(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    def _create(path: str, with_pip: bool = False) -> None:
        (Path(path) / "lib").mkdir(parents=True)

    monkeypatch.setattr(deps.venv_module, "create", _create)
    with pytest.raises(deps.DepsError, match="no python executable"):
        deps.create_venv(tmp_path / "v")


def test_create_venv_sanity_checks_the_interpreter(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.venv_module, "create", _fake_venv_create("Scripts"))
    monkeypatch.setattr(deps.utils, "run_capture", lambda cmd, **kw: _RunFail())
    with pytest.raises(deps.DepsError, match="interpreter exploded"):
        deps.create_venv(tmp_path / "v")


def test_prompt_yes_no_defaults_and_reprompts(monkeypatch: pytest.MonkeyPatch):
    answers = iter(["bogus", "maybe", "y"])
    with mock.patch("builtins.input", lambda prompt: next(answers)):
        assert deps.prompt_yes_no("? ") is True
    answers = iter(["", "n"])
    with mock.patch("builtins.input", lambda prompt: next(answers)):
        assert deps.prompt_yes_no("? ") is True
        assert deps.prompt_yes_no("? ") is False
    with mock.patch("builtins.input", lambda prompt: (_ for _ in ()).throw(EOFError)):
        assert deps.prompt_yes_no("? ", default=False) is False


def test_strip_venv_args_variants():
    assert deps.strip_venv_args(["--venv", ".venv", "-i", "a.png"]) == ["-i", "a.png"]
    assert deps.strip_venv_args(["--venv=.venv", "-i", "a.png"]) == ["-i", "a.png"]
    assert deps.strip_venv_args(["--venv"]) == []
    assert deps.strip_venv_args(["-i", "a.png", "--venv", "--venv-skip"]) == \
        ["-i", "a.png", "--venv-skip"]
    assert deps.strip_venv_args(["-i", "a.png"]) == ["-i", "a.png"]


def test_reexec_includes_the_script(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_replace(args: list[str]) -> None:
        captured["args"] = args

    monkeypatch.setattr(deps, "_replace_process", _fake_replace)
    monkeypatch.setattr(deps.sys, "argv", [str(deps.project_root() / "clipshrink.py"),
                                           "--venv", "-o", "x"])
    deps.reexec_with_python(deps.Path("venv-python"), ["-o", "x"])
    args = captured["args"]
    assert args[0] == "venv-python"
    assert args[1].endswith("clipshrink.py")
    assert args[2:] == ["-o", "x"]


def test_replace_process_waits_and_forwards_exit_code_on_windows(
        monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class _Result:
        returncode = 7

    def _fake_run(args, **kwargs):
        captured["args"] = args
        return _Result()

    monkeypatch.setattr(deps.os, "name", "nt")
    monkeypatch.setattr(deps.subprocess, "run", _fake_run)
    with pytest.raises(SystemExit) as excinfo:
        deps._replace_process(["python", "clipshrink.py"])
    assert excinfo.value.code == 7
    assert captured["args"] == ["python", "clipshrink.py"]


def test_replace_process_execs_on_posix(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def _fake_execv(path: str, args: list[str]) -> None:
        captured["path"] = path
        captured["args"] = args

    monkeypatch.setattr(deps.os, "name", "posix")
    monkeypatch.setattr(deps.os, "execv", _fake_execv)
    deps._replace_process(["python", "clipshrink.py"])
    assert captured["path"] == "python"
    assert captured["args"] == ["python", "clipshrink.py"]


def test_reexec_runs_clipshrink_end_to_end():
    code = ("import sys; from core import deps; "
            "deps.reexec_with_python(deps.Path(sys.executable), ['--version'])")
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "clipshrink" in proc.stdout


def test_clipshrink_script_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.sys, "argv", ["pytest", "tests/"])
    assert deps.clipshrink_script() == deps.project_root() / "clipshrink.py"


def test_find_bin_tool_searches_subdirectories(tmp_path: Path):
    subdir = tmp_path / "tool-1.0" / "bin"
    subdir.mkdir(parents=True)
    tool = subdir / "magick.exe"
    tool.write_bytes(b"MZ")
    assert deps.find_bin_tool(tmp_path, ("magick",)) == tool
    assert deps.find_bin_tool(tmp_path, ("missing",)) is None


def test_extensionless_executable_in_bin_is_used(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "/system")
    tool = tmp_path / "pngquant"
    tool.write_bytes(b"#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)
    assert deps.find_bin_tool(tmp_path, ("pngquant",)) == tool
    deps.activate_local_bins(warn=False, bin_path=tmp_path)
    assert deps.os.environ["PATH"].startswith(str(tmp_path))


@pytest.mark.skipif(sys.platform == "win32",
                    reason="os.access(X_OK) is True for any existing file on Windows")
def test_extensionless_non_executable_not_found(tmp_path: Path):
    (tmp_path / "pngquant").write_bytes(b"plain data, no exec bit")
    assert deps.find_bin_tool(tmp_path, ("pngquant",)) is None


def test_bin_path_dirs_includes_dirs_with_executables(tmp_path: Path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "ffmpeg.exe").write_bytes(b"MZ")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "x.param").write_bytes(b"x")
    (tmp_path / "dlls").mkdir()
    (tmp_path / "dlls" / "libvips-42.dll").write_bytes(b"MZ")
    dirs = deps.bin_path_dirs(tmp_path)
    assert tmp_path / "bin" in dirs
    assert tmp_path / "dlls" in dirs
    assert tmp_path / "models" not in dirs


def test_activate_local_bins_prepends_path_and_warns(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("PATH", "/system")
    (tmp_path / "pngquant.exe").write_bytes(b"MZ")
    dirs = deps.activate_local_bins(warn=True, bin_path=tmp_path)
    assert dirs == [tmp_path]
    assert deps.os.environ["PATH"].startswith(str(tmp_path))
    out = capsys.readouterr().out
    assert "local bin/ directory" in out
    assert "pngquant" in out


def test_activate_local_bins_warning_suppressible(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch,
                                                  capsys: pytest.CaptureFixture[str]):
    monkeypatch.setenv("PATH", "/system")
    (tmp_path / "pngquant.exe").write_bytes(b"MZ")
    deps.activate_local_bins(warn=False, bin_path=tmp_path)
    assert capsys.readouterr().out == ""


def test_activate_local_bins_noop_without_bin_dir(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "/system")
    assert deps.activate_local_bins(warn=True, bin_path=tmp_path / "absent") == []
    assert deps.os.environ["PATH"] == "/system"


def test_missing_backends_lists_absent_tools(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: None)
    missing = deps.missing_backends(_backend(), tmp_path)
    labels = [spec.label for spec in missing]
    assert labels == ["ImageMagick", "libvips", "waifu2x-ncnn-vulkan",
                      "realesrgan-ncnn-vulkan", "pngquant", "ffmpeg"]


def test_missing_backends_vips_binding_counts(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: None)
    missing = deps.missing_backends(_backend(vips=True), tmp_path)
    assert "libvips" not in [spec.label for spec in missing]


def test_missing_backends_silent_when_all_present(tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    assert deps.missing_backends(_backend(), tmp_path) == []


def test_check_backends_windows_non_interactive_hint(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "interactive", lambda: False)
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: None)
    changed = deps.check_backends_windows(_backend(), first_run=True, bin_path=tmp_path)
    assert changed is False
    out = capsys.readouterr().out
    assert "Warning: missing backend tools:" in out
    assert "Don't worry if you skip this for now. We will ask again once you try to process an image." in out
    assert "--install-deps" in out


def test_check_backends_windows_prompt_installs(tmp_path: Path,
                                                monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: True)
    installed: list[str] = []
    monkeypatch.setattr(deps, "install_backends",
                        lambda *a, **kw: installed.append("ran"))
    assert deps.check_backends_windows(_backend(), first_run=False, bin_path=tmp_path) is True
    assert installed == ["ran"]


def test_check_backends_windows_declined(tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: False)
    called = []
    monkeypatch.setattr(deps, "install_backends", lambda *a, **kw: called.append(a))
    assert deps.check_backends_windows(_backend(), first_run=False, bin_path=tmp_path) is False
    assert called == []


def test_check_backends_windows_silent_when_complete(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    assert deps.check_backends_windows(_backend(), first_run=True, bin_path=tmp_path) is False
    assert capsys.readouterr().out == ""


def test_warn_backends_posix_lists_missing_and_hints(tmp_path: Path,
                                                     capsys: pytest.CaptureFixture[str],
                                                     monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: None)
    deps.warn_backends_posix(_backend(os_type="X11"), tmp_path)
    out = capsys.readouterr().out
    assert "Warning: missing backend tools:" in out
    assert "your system's package manager" in out
    assert "official project websites" in out
    assert out.index("your system's package manager") < out.index("official project websites")


def test_warn_backends_posix_silent_when_complete(tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str],
                                                  monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    deps.warn_backends_posix(_backend(), tmp_path)
    assert capsys.readouterr().out == ""


def test_run_dependency_check_marks_first_run(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: None)
    config_path = tmp_path / "config.json"
    config = deps.load_config(config_path)
    deps.run_dependency_check(_backend(), config, config_path)
    assert config["first_run_completed"] is True
    assert deps.load_config(config_path)["first_run_completed"] is True


def test_run_dependency_check_later_runs_are_not_first(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    monkeypatch.setattr(deps.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    config = {"first_run_completed": True}
    assert deps.run_dependency_check(_backend(), config, tmp_path / "config.json") is False
    assert not (tmp_path / "config.json").exists()


def test_install_python_deps_runs_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("py7zr\n", encoding="utf-8")
    recorded: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(deps.utils, "run_capture", lambda cmd, **kw: recorded.append(cmd) or _Result())
    assert deps.install_python_deps(requirements=reqs) is True
    assert recorded[0][:4] == [deps.sys.executable, "-m", "pip", "install"]


def test_install_python_deps_reports_failure(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]):
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("py7zr\n", encoding="utf-8")

    class _Result:
        returncode = 1
        stdout = ""
        stderr = "pip blew up"

    monkeypatch.setattr(deps.utils, "run_capture", lambda cmd, **kw: _Result())
    assert deps.install_python_deps(requirements=reqs) is False
    assert "pip blew up" in capsys.readouterr().out


def test_install_python_deps_without_requirements(tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]):
    assert deps.install_python_deps(requirements=tmp_path / "absent.txt") is True
    assert "No requirements file" in capsys.readouterr().out


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_download_file_writes_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = b"0123456789"
    monkeypatch.setattr(deps, "_urlopen",
                        lambda url: contextlib.nullcontext(_FakeResponse(payload)))
    dest = tmp_path / "tool.zip"
    deps.download_file("https://example/tool.zip", dest)
    assert dest.read_bytes() == payload


def test_verify_checksum_accepts_matching_digest(tmp_path: Path):
    path = tmp_path / "tool.zip"
    path.write_bytes(b"payload")
    expected = hashlib.sha256(b"payload").hexdigest()
    deps.verify_checksum(path, expected)
    deps.verify_checksum(path, expected.upper())


def test_verify_checksum_rejects_mismatch(tmp_path: Path):
    path = tmp_path / "tool.zip"
    path.write_bytes(b"payload")
    with pytest.raises(deps.DepsError, match="checksum mismatch"):
        deps.verify_checksum(path, "0" * 64)


def test_verify_checksum_rejects_missing_digest(tmp_path: Path):
    path = tmp_path / "tool.zip"
    path.write_bytes(b"payload")
    with pytest.raises(deps.DepsError, match="no checksum"):
        deps.verify_checksum(path, "")


def test_extract_archive_zip(tmp_path: Path):
    archive = tmp_path / "tool.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tool-1.0/bin/tool.exe", b"MZ")
        zf.writestr("tool-1.0/models/x.param", b"x")
    dest = tmp_path / "extracted"
    dest.mkdir()
    deps.extract_archive(archive, dest)
    assert (dest / "tool-1.0" / "bin" / "tool.exe").exists()


def test_extract_archive_flat_zip(tmp_path: Path):
    archive = tmp_path / "flat.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("tool.exe", b"MZ")
    dest = tmp_path / "extracted"
    dest.mkdir()
    deps.extract_archive(archive, dest)
    assert (dest / "tool.exe").exists()


def test_extract_archive_unsupported_format(tmp_path: Path):
    archive = tmp_path / "tool.rar"
    archive.write_bytes(b"whatever")
    with pytest.raises(deps.DepsError, match="Unsupported archive format"):
        deps.extract_archive(archive, tmp_path / "out")


def test_extract_archive_7z_requires_py7zr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = tmp_path / "tool.7z"
    archive.write_bytes(b"7z\xbc\xaf\x27\x1c")
    monkeypatch.setattr(deps, "load_py7zr",
                        lambda: (_ for _ in ()).throw(deps.DepsError("py7zr required")))
    with pytest.raises(deps.DepsError, match="py7zr required"):
        deps.extract_archive(archive, tmp_path / "out")


def test_single_root_unwraps_wrapper_directory(tmp_path: Path):
    wrapped = tmp_path / "ex" / "tool-1.0"
    wrapped.mkdir(parents=True)
    (wrapped / "tool.exe").write_bytes(b"MZ")
    assert deps._single_root(tmp_path / "ex") == wrapped
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "tool.exe").write_bytes(b"MZ")
    assert deps._single_root(flat) == flat


def test_merge_tree_preserves_subpaths(tmp_path: Path):
    root = tmp_path / "root"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "tool.exe").write_bytes(b"MZ")
    (root / "models").mkdir()
    (root / "models" / "x.param").write_text("model")
    bin_path = tmp_path / "dest"
    moved = deps.merge_tree(root, bin_path)
    assert moved == ["bin/tool.exe", "models/x.param"]
    assert (bin_path / "bin" / "tool.exe").read_bytes() == b"MZ"
    assert (bin_path / "models" / "x.param").read_text() == "model"


def _write_tool_archive(dest: Path, root: str = "tool-1.0") -> None:
    with zipfile.ZipFile(dest, "w") as zf:
        zf.writestr(f"{root}/bin/tool.exe", b"MZ")
        zf.writestr(f"{root}/lib/support.dll", b"MZ")


def test_install_backends_downloads_extracts_merges(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: _write_tool_archive(dest))
    bin_path = tmp_path / "bin"
    spec = deps.ToolSpec("TestTool", ("tool",))
    installed = deps.install_backends(
        [spec], bin_path, links=_links("TestTool", "https://example/tool-1.0.zip"))
    assert installed == ["TestTool"]
    assert deps.find_bin_tool(bin_path, ("tool",)) is not None


def test_install_backends_verifies_checksum(tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch):
    archive = tmp_path / "tool-1.0.zip"
    _write_tool_archive(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: dest.write_bytes(archive.read_bytes()))
    bin_path = tmp_path / "bin"
    spec = deps.ToolSpec("TestTool", ("tool",))
    links = {"TestTool": deps.BinaryLinks(
        url="https://example/tool-1.0.zip", sha256=digest,
        url_linux="https://example/tool-1.0.zip", sha256_linux=digest)}
    assert deps.install_backends([spec], bin_path, links=links) == ["TestTool"]
    assert deps.find_bin_tool(bin_path, ("tool",)) is not None


def test_install_backends_rejects_bad_checksum(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    archive = tmp_path / "tool-1.0.zip"
    _write_tool_archive(archive)
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: dest.write_bytes(archive.read_bytes()))
    bin_path = tmp_path / "bin"
    spec = deps.ToolSpec("TestTool", ("tool",))
    links = {"TestTool": deps.BinaryLinks(
        url="https://example/tool-1.0.zip", sha256="0" * 64,
        url_linux="https://example/tool-1.0.zip", sha256_linux="0" * 64)}
    assert deps.install_backends([spec], bin_path, links=links) == []
    assert "checksum mismatch" in capsys.readouterr().out
    assert deps.find_bin_tool(bin_path, ("tool",)) is None


def test_install_backends_warns_without_checksum(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: _write_tool_archive(dest))
    bin_path = tmp_path / "bin"
    spec = deps.ToolSpec("TestTool", ("tool",))
    assert deps.install_backends(
        [spec], bin_path,
        links=_links("TestTool", "https://example/tool-1.0.zip")) == ["TestTool"]
    assert "no checksum" in capsys.readouterr().out


def test_install_backends_hints_optional_failure_and_continues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    def fake_download(url: str, dest: Path) -> None:
        if "bad" in url:
            dest.write_bytes(b"not a zip")
        else:
            _write_tool_archive(dest)

    monkeypatch.setattr(deps, "download_file", fake_download)
    bad = deps.ToolSpec("BadTool", ("bad",))
    good = deps.ToolSpec("OtherTool", ("tool",))
    links = {
        **_links("BadTool", "https://example/bad.zip"),
        **_links("OtherTool", "https://example/other.zip"),
    }
    installed = deps.install_backends(
        [bad, good], tmp_path / "bin", links=links)
    assert installed == ["OtherTool"]
    out = capsys.readouterr().out
    assert "BadTool is optional" in out
    assert "system's package manager" in out
    assert "official BadTool website" in out
    assert out.index("system's package manager") < out.index("official BadTool website")
    assert "Continuing with the rest of the installation." in out
    assert "Installed OtherTool" in out


def test_install_backends_hints_required_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: dest.write_bytes(b"not a zip"))
    spec = deps.ToolSpec("ImageMagick", ("magick",), required=True)
    assert deps.install_backends(
        [spec], tmp_path / "bin",
        links=_links("ImageMagick", "https://example/magick.zip")) == []
    out = capsys.readouterr().out
    assert "ImageMagick is required" in out
    assert "magick binary" in out
    assert "system's package manager" in out
    assert "official ImageMagick website" in out
    assert out.index("system's package manager") < out.index("official ImageMagick website")
    assert "Continuing with the rest of the installation." not in out


def test_manual_hint_includes_bin_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "project_root", lambda: tmp_path)
    monkeypatch.setattr(deps.sys, "prefix", str(tmp_path))
    deps._print_manual_hint(deps.ToolSpec("BadTool", ("bad",)))
    out = capsys.readouterr().out
    assert "may place the binary in the bin folder" in out
    assert str(tmp_path / "bin") in out
    assert "With pipx" not in out


def test_manual_hint_teaches_pipx_bin_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    venv = tmp_path / "pipx-venv"
    site = venv / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (venv / "pipx_metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(deps.sys, "prefix", str(venv))
    monkeypatch.setattr(deps, "project_root", lambda: site)
    deps._print_manual_hint(deps.ToolSpec("BadTool", ("bad",)))
    out = capsys.readouterr().out
    assert "may place the binary in the bin folder" in out
    assert str(site / "bin") in out
    assert "With pipx, the package folder is" in out
    assert "Its bin folder is" in out
    assert "Upgrading clipshrink may replace files there." in out


def test_install_backends_reports_bad_archive(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: dest.write_bytes(b"not a zip"))
    bin_path = tmp_path / "bin"
    spec = deps.ToolSpec("BadTool", ("tool",))
    assert deps.install_backends(
        [spec], bin_path,
        links=_links("BadTool", "https://example/tool.zip")) == []
    assert "Error installing BadTool" in capsys.readouterr().out


def test_install_backends_skips_tools_without_url(tmp_path: Path,
                                                  capsys: pytest.CaptureFixture[str]):
    spec = deps.ToolSpec("ManualTool", ("manual",))
    assert deps.install_backends([spec], tmp_path / "bin", links={}) == []
    assert "no automatic download" in capsys.readouterr().out


def test_install_backends_unwritable_bin_dir(tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]):
    def _deny(self: object, *args: object, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(tmp_path), "mkdir", _deny)
    spec = deps.ToolSpec("X", ("x",))
    assert deps.install_backends([spec], tmp_path / "bin") == []
    assert "read-only" in capsys.readouterr().out


def test_load_binaries_reads_links(tmp_path: Path):
    path = tmp_path / "binaries.json"
    path.write_text(json.dumps({
        "pngquant": {"url": "https://example/a.zip",
                     "sha256": "a" * 64,
                     "mirror": "https://mirror.example/b.zip",
                     "sha256_mirror": "b" * 64,
                     "url_linux": "https://example/a-linux.tar.bz2",
                     "sha256_linux": "c" * 64},
        "ffmpeg": {"url": "https://example/f.zip"},
        "broken": "not a dict",
        "badtype": {"url": 3},
        "badlinux": {"url": "https://example/x.zip", "url_linux": 7},
        "badsha": {"url": "https://example/x.zip", "sha256": 7},
    }), encoding="utf-8")
    links = deps.load_binaries(path)
    assert links["pngquant"] == deps.BinaryLinks(
        url="https://example/a.zip", mirror="https://mirror.example/b.zip",
        url_linux="https://example/a-linux.tar.bz2", mirror_linux="",
        sha256="a" * 64, sha256_mirror="b" * 64,
        sha256_linux="c" * 64)
    assert links["ffmpeg"] == deps.BinaryLinks(url="https://example/f.zip",
                                               mirror="")
    assert "broken" not in links and "badtype" not in links
    assert "badlinux" not in links and "badsha" not in links


def test_load_binaries_missing_or_malformed_yields_empty(tmp_path: Path):
    assert deps.load_binaries(tmp_path / "absent.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", encoding="utf-8")
    assert deps.load_binaries(bad) == {}
    arr = tmp_path / "arr.json"
    arr.write_text("[]", encoding="utf-8")
    assert deps.load_binaries(arr) == {}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def test_repo_binaries_file_covers_every_tool():
    links = deps.load_binaries()
    for spec in deps.BACKEND_TOOLS:
        assert spec.label in links, spec.label
        entry = links[spec.label]
        assert entry.url, spec.label
        assert entry.url_linux, spec.label
        assert _is_sha256(entry.sha256), spec.label
        assert _is_sha256(entry.sha256_linux), spec.label
        if entry.mirror:
            assert _is_sha256(entry.sha256_mirror), spec.label
        if entry.mirror_linux:
            assert _is_sha256(entry.sha256_mirror_linux), spec.label


def test_only_imagemagick_is_required():
    required = [spec.label for spec in deps.BACKEND_TOOLS if spec.required]
    assert required == ["ImageMagick"]


def _links(label: str, url: str, mirror: str = "") -> dict[str, deps.BinaryLinks]:
    return {label: deps.BinaryLinks(url=url, mirror=mirror,
                                    url_linux=url, mirror_linux=mirror)}


def test_outdated_backends_matches_url_or_mirror(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    spec = deps.ToolSpec("tool", ("tool",))
    monkeypatch.setattr(deps, "BACKEND_TOOLS", (spec,))
    (tmp_path / "tool.exe").write_bytes(b"MZ")
    versions = {"tool": {"url": "https://example/v1.zip", "files": []}}

    assert deps.outdated_backends(
        tmp_path, _links("tool", "https://example/v1.zip"), versions) == []
    assert deps.outdated_backends(
        tmp_path, _links("tool", "https://example/v2.zip",
                         "https://example/v1.zip"), versions) == []
    assert deps.outdated_backends(
        tmp_path, _links("tool", "https://example/v2.zip"), versions) == [spec]
    assert deps.outdated_backends(
        tmp_path, _links("tool", "https://example/v2.zip", ""), versions) == [spec]


def test_outdated_backends_ignores_unrecorded_and_absent_tools(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = deps.ToolSpec("tool", ("tool",))
    monkeypatch.setattr(deps, "BACKEND_TOOLS", (spec,))
    (tmp_path / "tool.exe").write_bytes(b"MZ")
    links = _links("tool", "https://example/v2.zip")
    assert deps.outdated_backends(tmp_path, links, {}) == []
    empty = tmp_path / "empty"
    empty.mkdir()
    versions = {"tool": {"url": "https://example/v1.zip", "files": []}}
    assert deps.outdated_backends(empty, links, versions) == []
    assert deps.outdated_backends(tmp_path, {}, versions) == []


def test_install_backends_uses_mirror_when_primary_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    calls: list[str] = []

    def fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        if "example.com" in url:
            raise urllib.error.URLError("primary down")
        _write_tool_archive(dest)

    monkeypatch.setattr(deps, "download_file", fake_download)
    spec = deps.ToolSpec("TestTool", ("tool",))
    links = _links("TestTool", "https://example.com/tool-1.0.zip",
                   "https://mirror.example/tool-1.0.zip")["TestTool"]
    bin_path = tmp_path / "bin"
    installed = deps.install_backends([spec], bin_path, links={"TestTool": links})
    assert installed == ["TestTool"]
    assert calls == ["https://example.com/tool-1.0.zip",
                     "https://mirror.example/tool-1.0.zip"]
    versions = deps.load_installed_versions(bin_path)
    assert versions["TestTool"]["url"] == "https://mirror.example/tool-1.0.zip"
    assert "bin/tool.exe" in versions["TestTool"]["files"]


def test_install_backends_empty_mirror_never_tried(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    def fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        raise urllib.error.URLError("down")

    monkeypatch.setattr(deps, "download_file", fake_download)
    spec = deps.ToolSpec("TestTool", ("tool",))
    assert deps.install_backends(
        [spec], tmp_path / "bin",
        links=_links("TestTool", "https://example/t.zip")) == []
    assert calls == ["https://example/t.zip"]


def _write_files_zip(dest: Path, files: dict[str, bytes]) -> None:
    with zipfile.ZipFile(dest, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)


def test_install_backends_update_replaces_old_version(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    archives = {
        "https://example/v1.zip": {"t-1.0/bin/tool.exe": b"v1",
                                   "t-1.0/lib/old.dll": b"x"},
        "https://example/v2.zip": {"t-2.0/bin/tool.exe": b"v2",
                                   "t-2.0/lib/new.dll": b"y"},
    }
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: _write_files_zip(dest, archives[url]))
    bin_path = tmp_path / "bin"
    spec = deps.ToolSpec("TestTool", ("tool",))
    assert deps.install_backends(
        [spec], bin_path,
        _links("TestTool", "https://example/v1.zip")) == ["TestTool"]
    assert (bin_path / "bin" / "tool.exe").read_bytes() == b"v1"

    assert deps.install_backends(
        [spec], bin_path,
        _links("TestTool", "https://example/v2.zip")) == ["TestTool"]
    assert (bin_path / "bin" / "tool.exe").read_bytes() == b"v2"
    assert not (bin_path / "lib" / "old.dll").exists()
    assert (bin_path / "lib" / "new.dll").exists()
    versions = deps.load_installed_versions(bin_path)
    assert versions["TestTool"]["url"] == "https://example/v2.zip"
    assert versions["TestTool"]["files"] == ["bin/tool.exe", "lib/new.dll"]


def test_install_backends_update_keeps_other_tools(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    archives = {
        "https://example/other.zip": {"o-1/other.exe": b"o"},
        "https://example/v1.zip": {"t-1.0/bin/tool.exe": b"v1",
                                   "t-1.0/lib/old.dll": b"x"},
        "https://example/v2.zip": {"t-2.0/bin/tool.exe": b"v2"},
    }
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: _write_files_zip(dest, archives[url]))
    bin_path = tmp_path / "bin"
    deps.install_backends([deps.ToolSpec("OtherTool", ("other",))],
                          bin_path, _links("OtherTool", "https://example/other.zip"))
    versions = deps.load_installed_versions(bin_path)
    assert versions["OtherTool"]["files"] == ["other.exe"]
    assert deps.install_backends(
        [deps.ToolSpec("TestTool", ("tool",))], bin_path,
        _links("TestTool", "https://example/v1.zip")) == ["TestTool"]
    deps.install_backends([deps.ToolSpec("TestTool", ("tool",))],
                          bin_path, _links("TestTool", "https://example/v2.zip"))
    assert (bin_path / "other.exe").read_bytes() == b"o"


def test_install_deps_updates_outdated_backends(monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}
    monkeypatch.setattr(deps, "install_python_deps", lambda **kw: True)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    spec = deps.ToolSpec("pngquant", ("pngquant",))
    monkeypatch.setattr(deps, "missing_backends", lambda backend, bin_path=None: [])
    monkeypatch.setattr(deps, "outdated_backends",
                        lambda bin_path=None, links=None: [spec])
    monkeypatch.setattr(deps, "install_backends",
                        lambda missing, bin_path=None, links=None:
                        seen.setdefault("updated", missing))
    assert main("--install-deps") == 0
    assert seen["updated"] == [spec]


def test_install_deps_all_current_when_nothing_missing_or_outdated(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "install_python_deps", lambda **kw: True)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    monkeypatch.setattr(deps, "missing_backends", lambda backend, bin_path=None: [])
    monkeypatch.setattr(deps, "outdated_backends", lambda bin_path=None, links=None: [])
    assert main("--install-deps") == 0
    assert "All backend tools are already installed." in capsys.readouterr().out


def test_install_deps_optional_failure_does_not_fail_run(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    import clipshrink as entry

    monkeypatch.setattr(deps, "is_in_venv", lambda: True)
    monkeypatch.setattr(deps, "install_python_deps", lambda **kw: True)
    spec = deps.ToolSpec("pngquant", ("pngquant",))
    monkeypatch.setattr(deps, "missing_backends",
                        lambda backend, bin_path=None: [spec])
    monkeypatch.setattr(deps, "outdated_backends",
                        lambda bin_path=None, links=None: [])
    monkeypatch.setattr(deps, "install_backends", lambda *a, **kw: [])

    assert entry._run_install_deps() is True
    assert "Required tool(s) failed to install" not in capsys.readouterr().out


def test_install_deps_required_failure_marks_run_failed(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    import clipshrink as entry

    monkeypatch.setattr(deps, "is_in_venv", lambda: True)
    monkeypatch.setattr(deps, "install_python_deps", lambda **kw: True)
    spec = deps.ToolSpec("ImageMagick", ("magick",), required=True)
    monkeypatch.setattr(deps, "missing_backends",
                        lambda backend, bin_path=None: [spec])
    monkeypatch.setattr(deps, "outdated_backends",
                        lambda bin_path=None, links=None: [])
    monkeypatch.setattr(deps, "install_backends", lambda *a, **kw: [])

    assert entry._run_install_deps() is False
    assert "Required tool(s) failed to install: ImageMagick" in \
        capsys.readouterr().out


def test_venv_flow_creation_failure_is_clean(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path,
                                             capsys: pytest.CaptureFixture[str]):
    import clipshrink as entry

    monkeypatch.setattr(deps, "is_in_venv", lambda: False)
    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: True)

    def _boom(venv_dir: Path) -> Path:
        raise RuntimeError("ensurepip is disabled in Debian/Ubuntu")

    monkeypatch.setattr(deps, "create_venv", _boom)
    args = Args(venv=str(tmp_path / ".venv"))
    assert entry._run_venv_flow(args, {}, []) == 1
    assert "could not create the virtual environment" in capsys.readouterr().out


def _existing_venv(tmp_path: Path, layout: str = "bin") -> Path:
    venv_path = tmp_path / ".venv"
    python_dir = venv_path / layout
    python_dir.mkdir(parents=True)
    (python_dir / "python.exe").write_bytes(b"")
    return venv_path


def test_venv_flow_reuses_existing_venv_without_prompt(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: Path,
                                                       capsys: pytest.CaptureFixture[str]):
    import clipshrink as entry

    venv_path = _existing_venv(tmp_path)
    monkeypatch.setattr(deps, "is_in_venv", lambda: False)
    monkeypatch.setattr(deps, "check_venv_python", lambda python: None)
    monkeypatch.setattr(entry, "_install_venv_requirements", lambda python: None)
    captured: dict[str, list[str]] = {}
    monkeypatch.setattr(deps, "_replace_process",
                        lambda args: captured.update(path=args[0], args=args))
    prompted: list[str] = []
    monkeypatch.setattr(deps, "prompt_yes_no",
                        lambda prompt, default=True: prompted.append(prompt))
    monkeypatch.setattr(deps.sys, "argv",
                        [str(deps.project_root() / "clipshrink.py"),
                         "--venv", "-o", "x.jpeg"])

    args = Args(venv=str(venv_path))
    assert entry._run_venv_flow(args, {}, ["--venv", "-o", "x.jpeg"]) == 0
    assert prompted == []
    assert "Virtual environment found" in capsys.readouterr().out
    assert captured["path"] == str(venv_path / "bin" / "python.exe")
    assert captured["args"][1].endswith("clipshrink.py")
    assert captured["args"][2:] == ["-o", "x.jpeg"]


def test_venv_flow_recreates_broken_venv_after_prompt(monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path):
    import clipshrink as entry

    venv_path = _existing_venv(tmp_path)
    monkeypatch.setattr(deps, "is_in_venv", lambda: False)
    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: True)
    monkeypatch.setattr(deps, "save_config", lambda config, path=None: None)
    monkeypatch.setattr(entry, "_install_venv_requirements", lambda python: None)
    created: list[Path] = []
    monkeypatch.setattr(deps, "create_venv",
                        lambda path: created.append(path) or path / "bin" / "python.exe")
    monkeypatch.setattr(deps, "_replace_process", lambda args: None)

    args = Args(venv=str(venv_path))
    assert entry._run_venv_flow(args, {}, ["--venv"]) == 0
    assert created == [venv_path]


def test_venv_flow_no_reexec_when_already_using_the_venv(monkeypatch: pytest.MonkeyPatch,
                                                         tmp_path: Path,
                                                         capsys: pytest.CaptureFixture[str]):
    import clipshrink as entry

    venv_path = _existing_venv(tmp_path)
    monkeypatch.setattr(deps, "is_in_venv", lambda: False)
    monkeypatch.setattr(deps, "check_venv_python", lambda python: None)
    monkeypatch.setattr(sys, "executable", str(venv_path / "bin" / "python.exe"))
    replace_called: list[object] = []
    monkeypatch.setattr(deps, "_replace_process",
                        lambda args: replace_called.append(args))

    args = Args(venv=str(venv_path))
    assert entry._run_venv_flow(args, {}, ["--venv"]) is None
    assert replace_called == []
    assert "Already running" in capsys.readouterr().out


def test_check_venv_python_rejects_broken_interpreter(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch):
    garbage = tmp_path / "python.exe"
    garbage.write_bytes(b"not an executable")

    class _OSErrorPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise OSError("not a valid application")

    monkeypatch.setattr(deps.utils.subprocess, "Popen", _OSErrorPopen)
    with pytest.raises(deps.DepsError, match="failed to run"):
        deps.check_venv_python(garbage)


def test_dependency_check_deferred_until_processing(monkeypatch: pytest.MonkeyPatch,
                                                    tmp_path: Path):
    import clipshrink as entry

    monkeypatch.setattr(entry, "detect_backends",
                        lambda: BackendInfo(os_type="X11", vips=False,
                                            magick=True, pngquant=False,
                                            realesrgan=False, waifu2x=False))
    calls: list[str] = []
    monkeypatch.setattr(deps, "run_dependency_check",
                        lambda backend, config, path=None:
                        calls.append("check") or False)
    monkeypatch.setattr(entry, "process_image",
                        lambda path, args: {"success": True})
    png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

    assert main("-i", str(tmp_path / "absent.png"),
                "-o", str(tmp_path / "o.png")) == 1
    assert calls == []

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main("-i", str(empty), "-o", str(tmp_path / "out")) == 1
    assert calls == []

    note = tmp_path / "note.txt"
    note.write_text("not an image", encoding="utf-8")
    assert main("-i", str(note), "-o", str(tmp_path / "n.png")) == 0
    assert calls == []

    img = tmp_path / "a.png"
    img.write_bytes(png_magic)
    assert main("-i", str(img), "-o", str(tmp_path / "a_out.png")) == 0
    assert calls == ["check"]

    folder = tmp_path / "imgs"
    folder.mkdir()
    (folder / "b.png").write_bytes(png_magic)
    assert main("-i", str(folder), "-o", str(tmp_path / "f_out")) == 0
    assert calls == ["check", "check"]


def test_no_arguments_runs_clipboard_to_clipboard(monkeypatch: pytest.MonkeyPatch):
    import clipshrink as entry

    monkeypatch.setattr(entry, "detect_backends",
                        lambda: BackendInfo(os_type="X11", vips=False,
                                            magick=True, pngquant=False,
                                            realesrgan=False, waifu2x=False))
    monkeypatch.setattr(deps, "run_dependency_check",
                        lambda backend, config, path=None: False)
    seen: list[tuple[Path | None, Args]] = []

    def fake_process(path, args):
        seen.append((path, args))
        return {"success": True}

    monkeypatch.setattr(entry, "process_image", fake_process)

    assert main() == 0
    assert len(seen) == 1
    assert seen[0][0] is None
    assert seen[0][1].output is None


def test_platform_links_selection(monkeypatch: pytest.MonkeyPatch):
    links = deps.BinaryLinks(url="w.zip", mirror="wm.zip",
                             url_linux="l.tar.xz", mirror_linux="lm.tar.xz")
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    assert deps.platform_links(links) == ("w.zip", "wm.zip")
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    monkeypatch.setattr(deps, "is_macos", lambda: False)
    assert deps.platform_links(links) == ("l.tar.xz", "lm.tar.xz")
    monkeypatch.setattr(deps, "is_macos", lambda: True)
    assert deps.platform_links(links) == ("", "")
    assert deps.platform_links(None) == ("", "")
    monkeypatch.setattr(deps, "is_macos", lambda: False)
    assert deps.platform_links(deps.BinaryLinks(url="w.zip")) == ("", "")


def test_platform_checksums_selection(monkeypatch: pytest.MonkeyPatch):
    links = deps.BinaryLinks(sha256="w", sha256_mirror="wm",
                             sha256_linux="l", sha256_mirror_linux="lm")
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    assert deps.platform_checksums(links) == ("w", "wm")
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    monkeypatch.setattr(deps, "is_macos", lambda: False)
    assert deps.platform_checksums(links) == ("l", "lm")
    monkeypatch.setattr(deps, "is_macos", lambda: True)
    assert deps.platform_checksums(links) == ("", "")
    assert deps.platform_checksums(None) == ("", "")
    monkeypatch.setattr(deps, "is_macos", lambda: False)
    assert deps.platform_checksums(deps.BinaryLinks(url="w.zip")) == ("", "")


_NEEDS_POSIX = pytest.mark.skipif(
    os.name == "nt", reason="POSIX-only semantics (exec bits, LD_LIBRARY_PATH)")


@_NEEDS_POSIX
def test_extract_archive_tar_family_preserves_exec_bits(tmp_path: Path):
    import tarfile

    payload = tmp_path / "payload"
    payload.mkdir()
    binary = payload / "tool"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    for suffix in (".tar.xz", ".tar.bz2"):
        archive = tmp_path / f"tool-1.0{suffix}"
        with tarfile.open(archive, f"w:{suffix.split('.')[1]}") as tf:
            tf.add(binary, arcname="tool-1.0/bin/tool")
        dest = tmp_path / f"out-{suffix}"
        dest.mkdir()
        deps.extract_archive(archive, dest)
        extracted = dest / "tool-1.0" / "bin" / "tool"
        assert extracted.read_bytes() == b"#!/bin/sh\n"
        assert extracted.stat().st_mode & 0o111


@_NEEDS_POSIX
def test_make_executables_chmods_the_tool_binaries(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    (tmp_path / "tool").write_bytes(b"MZ")
    (tmp_path / "tool").chmod(0o644)
    (tmp_path / "readme.txt").write_text("hi")
    deps._make_executables(tmp_path, ["tool", "readme.txt"], ("tool",))
    assert (tmp_path / "tool").stat().st_mode & 0o111
    assert not ((tmp_path / "readme.txt").stat().st_mode & 0o111)


@_NEEDS_POSIX
def test_install_backends_appimage_self_extracts(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
    def fake_download(url: str, dest: Path) -> None:
        dest.write_bytes(b"AppImage payload")

    def fake_run_capture(cmd: list[str], **kwargs: object):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        if "--appimage-extract" in cmd:
            magick = Path(kwargs["cwd"]) / "squashfs-root" / "usr" / "bin" / "magick"
            magick.parent.mkdir(parents=True)
            magick.write_bytes(b"#!/bin/sh\n")
            magick.chmod(0o755)
        return R()

    monkeypatch.setattr(deps, "download_file", fake_download)
    monkeypatch.setattr(deps.utils, "run_capture", fake_run_capture)
    spec = deps.ToolSpec("ImageMagick", ("magick",))
    bin_path = tmp_path / "bin"
    installed = deps.install_backends(
        [spec], bin_path,
        _links("ImageMagick", "https://example/ImageMagick.AppImage"))
    assert installed == ["ImageMagick"]
    assert (bin_path / "usr" / "bin" / "magick").is_file()
    assert deps.find_bin_tool(bin_path, ("magick",)) is not None
    versions = deps.load_installed_versions(bin_path)
    assert "usr/bin/magick" in versions["ImageMagick"]["files"]
    wrapper = bin_path / "magick"
    assert wrapper.is_file() and wrapper.stat().st_mode & 0o111
    script = wrapper.read_text(encoding="utf-8")
    assert "MAGICK_HOME=" in script and str(bin_path / "usr") in script
    assert "HOME=" in script and str(bin_path / "magick.home") in script
    assert 'exec "' in script and 'usr/bin/magick" "$@"' in script
    assert "magick" in versions["ImageMagick"]["files"]


def test_install_vips_pyvips_already_works(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "_pyvips_works", lambda python=None: True)
    pip_calls: list[str] = []
    monkeypatch.setattr(deps, "_pip_install", pip_calls.append)
    assert deps._install_vips(deps.ToolSpec("libvips", ("vips",)),
                              deps.BinaryLinks(), Path("/tmp"), {})
    assert pip_calls == []


def test_install_vips_wheel_path(monkeypatch: pytest.MonkeyPatch):
    state = {"works": False}
    monkeypatch.setattr(deps, "_pyvips_works", lambda python=None: state["works"])
    pip_calls: list[str] = []
    monkeypatch.setattr(deps, "_pip_install",
                        lambda req: pip_calls.append(req) or state.__setitem__("works", True) or True)
    assert deps._install_vips(deps.ToolSpec("libvips", ("vips",)),
                              deps.BinaryLinks(), Path("/tmp"), {})
    assert pip_calls == ["pyvips[binary]"]


def test_install_vips_windows_falls_back_to_prebuilt_zip(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "_pyvips_works", lambda python=None: False)
    monkeypatch.setattr(deps, "_pip_install", lambda req: False)
    monkeypatch.setattr(deps, "is_windows", lambda: True)
    spec = deps.ToolSpec("libvips", ("vips",))
    links = _links("libvips", "https://example/vips-dev.zip")

    def fake_install_one(spec: deps.ToolSpec, tool_links: deps.BinaryLinks,
                         bin_path: Path, versions: dict) -> bool:
        assert tool_links.url == "https://example/vips-dev.zip"
        return True

    monkeypatch.setattr(deps, "_install_one", fake_install_one)
    assert deps._install_vips(spec, links["libvips"], Path("/tmp"), {})


def test_install_vips_posix_source_link_prompts(monkeypatch: pytest.MonkeyPatch,
                                                capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "_pyvips_works", lambda python=None: False)
    monkeypatch.setattr(deps, "_pip_install", lambda req: False)
    monkeypatch.setattr(deps, "is_windows", lambda: False)
    spec = deps.ToolSpec("libvips", ("vips",))
    links = _links("libvips", "https://example/vips-8.18.6.tar.xz")

    monkeypatch.setattr(deps, "interactive", lambda: False)
    assert not deps._install_vips(spec, links["libvips"], Path("/tmp"), {})
    assert "package manager" in capsys.readouterr().out

    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: False)
    assert not deps._install_vips(spec, links["libvips"], Path("/tmp"), {})
    assert "package manager" in capsys.readouterr().out

    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: True)
    build_calls: list[tuple[deps.BinaryLinks, Path, dict]] = []

    def fake_build(links: deps.BinaryLinks, bin_path: Path,
                   versions: dict) -> bool:
        build_calls.append((links, bin_path, versions))
        return True

    monkeypatch.setattr(deps, "_build_vips_from_source", fake_build)
    assert deps._install_vips(spec, links["libvips"], Path("/tmp"), {})
    assert build_calls[0][0].url == "https://example/vips-8.18.6.tar.xz"


def test_vips_source_link_detection():
    assert deps._vips_link_is_source("https://example/vips-8.18.6.tar.xz")
    assert deps._vips_link_is_source("https://example/vips-8.18.tar.gz")
    assert not deps._vips_link_is_source("https://example/vips-dev-x64.zip")
    assert not deps._vips_link_is_source("https://example/pngquant.tar.bz2")


def _write_vips_tarball(tmp_path: Path, dest: Path) -> None:
    import tarfile

    src = tmp_path / "src-tree"
    src.mkdir(exist_ok=True)
    (src / "meson.build").write_text("project('vips')\n")
    with tarfile.open(dest, "w:xz") as tf:
        tf.add(src / "meson.build", arcname="vips-8.18.6/meson.build")


def test_build_vips_from_source_success(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []
    state: dict = {}

    def fake_download(url: str, dest: Path) -> None:
        _write_vips_tarball(tmp_path, dest)

    def fake_run_capture(cmd: list[str], **kwargs: object):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        if any(a.startswith("--prefix=") for a in cmd):
            state["prefix"] = Path(
                next(a for a in cmd if a.startswith("--prefix="))[9:])
        if cmd[-1] == "install":
            lib = state["prefix"] / "lib"
            lib.mkdir(parents=True)
            (lib / "libvips.so.42").write_bytes(b"ELF")
        return R()

    pip_calls: list[str] = []
    monkeypatch.setattr(deps, "download_file", fake_download)
    monkeypatch.setattr(deps.utils, "run_capture", fake_run_capture)
    monkeypatch.setattr(deps, "_find_build_tool",
                        lambda name, pip_name: f"/usr/bin/{name}")
    monkeypatch.setattr(deps, "_have_c_toolchain", lambda: True)
    monkeypatch.setattr(deps, "_pip_install", pip_calls.append)

    bin_path = tmp_path / "bin"
    versions: dict = {}
    assert deps._build_vips_from_source(
        deps.BinaryLinks(url="https://example/vips-8.18.6.tar.xz"),
        bin_path, versions)
    assert [c[0] for c in calls] == ["/usr/bin/meson", "/usr/bin/ninja",
                                     "/usr/bin/ninja"]
    assert any("--prefix=" in a and str(bin_path / "vips") in a
               for a in calls[0])
    assert versions["libvips"]["url"] == "https://example/vips-8.18.6.tar.xz"
    assert "vips/lib/libvips.so.42" in versions["libvips"]["files"]
    assert pip_calls == ["pyvips"]


def test_build_vips_from_source_missing_toolchain(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: _write_vips_tarball(tmp_path, dest))
    monkeypatch.setattr(deps, "_find_build_tool",
                        lambda name, pip_name: f"/usr/bin/{name}")
    monkeypatch.setattr(deps, "_have_c_toolchain", lambda: False)
    bin_path = tmp_path / "bin"
    assert not deps._build_vips_from_source(
        deps.BinaryLinks(url="https://example/vips-8.18.6.tar.xz"),
        bin_path, {})
    assert "meson" in capsys.readouterr().out
    assert not (bin_path / "vips").exists()


def test_build_vips_from_source_failure_cleans_up_and_hints_package_manager(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    import tarfile

    def fake_download(url: str, dest: Path) -> None:
        src = tmp_path / "src-tree"
        src.mkdir(exist_ok=True)
        (src / "meson.build").write_text("project('vips')\n")
        with tarfile.open(dest, "w:xz") as tf:
            tf.add(src / "meson.build", arcname="vips-8.18.6/meson.build")

    class FailR:
        returncode = 1
        stdout = ""
        stderr = "meson: compiler not found"

    monkeypatch.setattr(deps, "download_file", fake_download)
    monkeypatch.setattr(deps.utils, "run_capture",
                        lambda cmd, **kw: FailR())
    monkeypatch.setattr(deps, "_find_build_tool",
                        lambda name, pip_name: f"/usr/bin/{name}")
    monkeypatch.setattr(deps, "_have_c_toolchain", lambda: True)

    bin_path = tmp_path / "bin"
    versions: dict = {}
    assert not deps._build_vips_from_source(
        deps.BinaryLinks(url="https://example/vips-8.18.6.tar.xz"),
        bin_path, versions)
    out = capsys.readouterr().out
    assert "failed" in out
    assert "package manager" in out
    assert versions == {}
    assert not (bin_path / "vips").exists()


def _posix_missing_specs() -> list[deps.ToolSpec]:
    return [deps.ToolSpec("libvips", ("vips",)),
            deps.ToolSpec("OtherTool", ("other",))]


def test_check_backends_posix_prompts_and_installs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deps, "is_macos", lambda: False)
    monkeypatch.setattr(deps, "missing_backends",
                        lambda backend, bin_path=None: _posix_missing_specs())
    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: True)
    installed_specs: list = []
    monkeypatch.setattr(deps, "install_backends",
                        lambda missing, bin_path=None: installed_specs.extend(missing) or [])
    backend = BackendInfo(os_type="X11", vips=False, magick=False,
                          pngquant=False, realesrgan=False, waifu2x=False)
    assert deps.check_backends_posix(backend, False, tmp_path) is True
    assert [spec.label for spec in installed_specs] == ["libvips", "OtherTool"]


def test_check_backends_posix_non_interactive_hints(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "is_macos", lambda: False)
    monkeypatch.setattr(deps, "missing_backends",
                        lambda backend, bin_path=None: _posix_missing_specs())
    monkeypatch.setattr(deps, "interactive", lambda: False)
    backend = BackendInfo(os_type="X11", vips=False, magick=False,
                          pngquant=False, realesrgan=False, waifu2x=False)
    assert deps.check_backends_posix(backend, False, tmp_path) is False
    assert "clipshrink --install-deps" in capsys.readouterr().out


def test_check_backends_posix_decline_falls_back_to_hints(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "is_macos", lambda: False)
    monkeypatch.setattr(deps, "missing_backends",
                        lambda backend, bin_path=None: _posix_missing_specs())
    monkeypatch.setattr(deps, "interactive", lambda: True)
    monkeypatch.setattr(deps, "prompt_yes_no", lambda prompt, default=True: False)
    backend = BackendInfo(os_type="X11", vips=False, magick=False,
                          pngquant=False, realesrgan=False, waifu2x=False)
    assert deps.check_backends_posix(backend, False, tmp_path) is False
    out = capsys.readouterr().out
    assert "your system's package manager" in out
    assert "official project websites" in out


@_NEEDS_POSIX
def test_activate_local_bins_sets_ld_library_path_for_vips_build(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    vips_lib = tmp_path / "vips" / "lib"
    vips_lib.mkdir(parents=True)
    (vips_lib / "libvips.so.42").write_bytes(b"ELF")
    deps.activate_local_bins(warn=False, bin_path=tmp_path)
    assert str(vips_lib) in deps.os.environ.get("LD_LIBRARY_PATH", "")


def test_format_duration():
    assert deps.format_duration(0) == "0:00"
    assert deps.format_duration(9.4) == "0:09"
    assert deps.format_duration(45) == "0:45"
    assert deps.format_duration(60) == "1:00"
    assert deps.format_duration(3723) == "1:02:03"


def test_download_file_shows_speed_and_eta(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    import contextlib

    total = 1000
    state = {"sent": 0}

    class FakeResponse:
        def __init__(self) -> None:
            self.headers = {"Content-Length": str(total)}

        def read(self, size: int) -> bytes:
            if state["sent"] >= total:
                return b""
            chunk = b"x" * min(size, total - state["sent"], 400)
            state["sent"] += len(chunk)
            return chunk

    @contextlib.contextmanager
    def fake_urlopen(url: str):
        yield FakeResponse()

    clock = iter([0.0, 0.2, 0.4, 0.6, 0.8])
    monkeypatch.setattr(deps.time, "monotonic",
                        lambda: next(clock, 99.0))

    monkeypatch.setattr(deps, "_urlopen", fake_urlopen)
    deps.download_file("https://example/file.bin", tmp_path / "f.bin")
    out = capsys.readouterr().out
    assert "100.0%" in out
    assert "eta" in out and "/s" in out
    assert "average" in out
    assert (tmp_path / "f.bin").stat().st_size == total


def test_install_backends_reports_durations(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(deps, "download_file",
                        lambda url, dest: _write_tool_archive(dest))
    spec = deps.ToolSpec("TestTool", ("tool",))
    installed = deps.install_backends(
        [spec], tmp_path / "bin",
        _links("TestTool", "https://example/tool-1.0.zip"))
    assert installed == ["TestTool"]
    out = capsys.readouterr().out
    assert "Installed TestTool into" in out and "0:00" in out
    assert "Install complete: 1 tool(s) in 0:00." in out
