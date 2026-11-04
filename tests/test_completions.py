from __future__ import annotations

import pytest

from core import completions
from core.cli import create_parser


@pytest.fixture
def parser() -> object:
    return create_parser()


def test_bash_completions_covers_flags(parser: object) -> None:
    script = completions.bash_completions(parser)
    assert script.startswith("# bash completion for clipshrink")
    assert "_clipshrink()" in script
    assert "complete -F _clipshrink clipshrink" in script
    for flag in ("--photo", "--pixel-art", "--slowest", "--print-completions"):
        assert flag in script


def test_bash_completions_handle_choices(parser: object) -> None:
    script = completions.bash_completions(parser)
    assert "--subsampling)" in script
    assert "4:4:4 4:2:2 4:2:0" in script
    assert "--realesrgan-model)" in script
    assert "photo drawing" in script


def test_zsh_completions_reuse_the_bash_function(parser: object) -> None:
    script = completions.zsh_completions(parser)
    assert script.startswith("#compdef clipshrink")
    assert "bashcompinit" in script
    assert "_clipshrink()" in script


def test_fish_completions_covers_flags_values_and_files(
        parser: object) -> None:
    script = completions.fish_completions(parser)
    assert "complete -c clipshrink -l photo" in script
    assert "-s p" in script
    assert "-l photo -d " in script
    assert "-l subsampling -r -f -a '4:4:4 4:2:2 4:2:0'" in script
    assert "-l input -r -F" in script
    assert "-s i -r -F" in script


def test_fish_completions_never_mix_no_files_and_force_files(
        parser: object) -> None:
    script = completions.fish_completions(parser)
    for line in script.splitlines():
        if line.startswith("complete "):
            tokens = set(line.split())
            assert not ({"-f", "-F"} <= tokens), line


def test_man_page_structure(parser: object) -> None:
    page = completions.man_page(parser)
    assert page.startswith(".\\")
    assert ".TH CLIPSHRINK 1" in page
    assert ".SH NAME" in page
    assert ".SH SYNOPSIS" in page
    assert ".SH DESCRIPTION" in page
    assert ".SH OPTIONS" in page
    assert "Shrink images below a target file size" in page
    assert "\\-\\-photo" in page
    assert "\\-\\-max-size" in page


def test_print_output_dispatch(parser: object) -> None:
    p = parser
    assert completions.print_output(p, "man", None).startswith(".\\")
    assert "complete -F" in completions.print_output(p, "completions", "bash")
    assert "#compdef" in completions.print_output(p, "completions", "zsh")
    assert "complete -c clipshrink" in completions.print_output(p, "completions", "fish")
    with pytest.raises(ValueError):
        completions.print_output(p, "completions", "tcsh")


def test_main_print_completions(monkeypatch: pytest.MonkeyPatch,
                                capsys: pytest.CaptureFixture[str]) -> None:
    import clipshrink
    monkeypatch.setattr("core.deps.load_config", lambda: {})
    assert clipshrink.main("--print-completions", "fish") == 0
    out = capsys.readouterr().out
    assert out.startswith("# fish completion for clipshrink")
    assert "complete -c clipshrink" in out


def test_main_print_man(monkeypatch: pytest.MonkeyPatch,
                        capsys: pytest.CaptureFixture[str]) -> None:
    import clipshrink
    monkeypatch.setattr("core.deps.load_config", lambda: {})
    assert clipshrink.main("--print-man") == 0
    out = capsys.readouterr().out
    assert out.startswith(".\\")
    assert ".TH CLIPSHRINK" in out


def test_main_print_beats_a_bad_log_file(tmp_path: object,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    import clipshrink
    assert clipshrink.main("--print-completions", "bash",
                           "--log-file", "/nonexistent-dir/run.log") == 0
    assert "complete -F" in capsys.readouterr().out
