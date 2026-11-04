import argparse
from dataclasses import fields

import pytest

from core.args import Args
from core.cli import create_parser, parse_size


def test_parse_size_bare_number_is_kilobytes():
    assert parse_size("512") == 512 * 1024
    assert parse_size("8") == 8 * 1024
    assert parse_size("0") == 0


def test_parse_size_kb_suffixes():
    for value in ("512k", "512K", "512kb", "512KB", "512 kB", " 512k "):
        assert parse_size(value) == 512 * 1024


def test_parse_size_mb_suffixes():
    for value in ("8m", "8M", "8mb", "8MB", "8 MB"):
        assert parse_size(value) == 8 * 1024 * 1024


def test_parse_size_fractional_values():
    assert parse_size("1.5m") == round(1.5 * 1024 * 1024)
    assert parse_size("1.5k") == round(1.5 * 1024)


@pytest.mark.parametrize("bad", ["", "abc", "8b", "-1", "8m m", "8x", "m8", "1.2.3"])
def test_parse_size_rejects_invalid(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_size(bad)


def test_parser_default_is_relative_target():
    args = create_parser().parse_args([])
    assert args.max_size is None


def test_parser_accepts_units():
    assert create_parser().parse_args(["-s", "512"]).max_size == 512 * 1024
    assert create_parser().parse_args(["-s", "512k"]).max_size == 512 * 1024
    assert create_parser().parse_args(["-s", "2m"]).max_size == 2 * 1024 * 1024
    assert create_parser().parse_args(["--max-size", "2M"]).max_size == 2 * 1024 * 1024


def test_parser_rejects_invalid_size():
    with pytest.raises(SystemExit):
        create_parser().parse_args(["-s", "bogus"])


def test_help_mentions_kilobytes_and_suffixes():
    help_text = " ".join(create_parser().format_help().split())
    assert "kilobytes" in help_text
    assert "per-format fraction of each image's own file size" in help_text
    assert "JPEG 42%" in help_text
    assert "m/MB" in help_text


def test_parse_cli_args_tracks_explicitly_given_dests():
    from core.cli import parse_cli_args

    ns, provided = parse_cli_args(create_parser(), ["-s", "512", "-v"])
    assert provided == {"max_size", "verbose"}
    assert ns.max_size == 512 * 1024 and ns.verbose is True
    assert ns.subsampling == "4:2:0" and ns.format is None and ns.fast is False


def test_parse_cli_args_value_equal_to_default_still_counts_as_given():
    from core.cli import parse_cli_args

    _, provided = parse_cli_args(create_parser(), ["--subsampling", "4:2:0"])
    assert provided == {"subsampling"}
    _, provided = parse_cli_args(create_parser(), [])
    assert provided == set()


def test_args_fields_match_parser_destinations():
    dests = {a.dest for a in create_parser()._actions
             if a.dest not in ("help", "version")}
    assert dests == {f.name for f in fields(Args)}


def test_from_namespace_copies_every_field():
    ns = create_parser().parse_args(
        ["-i", "in.png", "-o", "out.jpg", "-s", "2m", "-F", "jpeg",
         "--fast", "--strip-metadata", "--no-oxipng", "--search-step", "7"]
    )
    args = Args.from_namespace(ns)
    for f in fields(Args):
        assert getattr(args, f.name) == getattr(ns, f.name)
    assert args.max_size == 2 * 1024 * 1024
    assert args.search_step == 7
    assert args.fast is True
    assert args.output == "out.jpg"
