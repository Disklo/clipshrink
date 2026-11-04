from __future__ import annotations

from pathlib import Path

import pytest

from core import log


def test_silent_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    log.configure(verbose=False, debug=False, log_path=None)
    log.log("verbose line")
    log.debug("debug line")
    assert capsys.readouterr().out == ""


def test_verbose_shows_log_but_not_debug(capsys: pytest.CaptureFixture[str]) -> None:
    log.configure(verbose=True, debug=False, log_path=None)
    log.log("verbose line")
    log.debug("debug line")
    out = capsys.readouterr().out
    assert "verbose line" in out
    assert "debug line" not in out


def test_debug_implies_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    log.configure(verbose=False, debug=True, log_path=None)
    log.log("verbose line")
    log.debug("debug line")
    out = capsys.readouterr().out
    assert "verbose line" in out
    assert "debug line" in out
    assert log.is_verbose() and log.is_debug()


def test_log_file_records_everything(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    log.configure(verbose=False, debug=False, log_path=str(path))
    log.log("a verbose line")
    log.debug("a debug line")
    log.close()
    content = path.read_text(encoding="utf-8")
    assert "a verbose line" in content
    assert "a debug line" in content
    for line in content.splitlines():
        assert line.startswith("[") and "]" in line


def test_log_file_appends(tmp_path: Path) -> None:
    path = tmp_path / "run.log"
    path.write_text("old\n", encoding="utf-8")
    log.configure(verbose=False, debug=False, log_path=str(path))
    log.log("new")
    log.close()
    assert path.read_text(encoding="utf-8").startswith("old\n")
    assert path.read_text(encoding="utf-8").endswith("new\n")


def test_log_file_open_failure(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        log.configure(verbose=False, debug=False,
                      log_path=str(tmp_path / "missing-dir" / "run.log"))


def test_log_command_traces_start_and_failure(
        capsys: pytest.CaptureFixture[str]) -> None:
    log.configure(verbose=True, debug=True, log_path=None)
    log.log_command(["magick", "input.png", "out.jpg"])
    log.log_command(["avifenc", "--speed", "4"], returncode=1, stderr="boom")
    out = capsys.readouterr().out
    assert "$ magick input.png out.jpg" in out
    assert "$ avifenc --speed 4 -> exit 1" in out
    assert "boom" in out


def test_log_command_truncates_stderr_tail(capsys: pytest.CaptureFixture[str]) -> None:
    log.configure(verbose=False, debug=True, log_path=None)
    log.log_command(["tool"], returncode=2, stderr="x" * 3000)
    out = capsys.readouterr().out
    assert "..." in out
    assert "x" * 2500 not in out


def test_log_command_silent_without_debug(capsys: pytest.CaptureFixture[str]) -> None:
    log.configure(verbose=True, debug=False, log_path=None)
    log.log_command(["magick", "x"])
    log.log_command(["magick", "x"], returncode=1, stderr="err")
    assert capsys.readouterr().out == ""


def test_close_is_idempotent(tmp_path: Path) -> None:
    log.configure(verbose=False, debug=False, log_path=str(tmp_path / "run.log"))
    log.close()
    log.close()
    assert log.is_verbose() is False and log.is_debug() is False


def test_cli_flags_parse() -> None:
    from core.args import Args
    from core.cli import create_parser

    ns = create_parser().parse_args(
        ["-v", "--debug", "--log-file", "run.log"])
    args = Args.from_namespace(ns)
    assert args.verbose is True
    assert args.debug is True
    assert args.log_file == "run.log"


def test_main_rejects_unwritable_log_file(
        capsys: pytest.CaptureFixture[str]) -> None:
    import clipshrink

    assert clipshrink.main("--log-file", "/nonexistent-dir/run.log") == 1
    assert "cannot open log file" in capsys.readouterr().out
