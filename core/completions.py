from __future__ import annotations

import argparse

from . import __version__

_INPUT_FLAGS = frozenset({"-i", "--input", "-o", "--output", "--log-file"})


def _flag_actions(parser: argparse.ArgumentParser) -> list[argparse.Action]:
    return [a for a in parser._actions if a.option_strings]


def _takes_value(action: argparse.Action) -> bool:
    return action.nargs != 0


def _help_text(action: argparse.Action) -> str:
    return (action.help or "").replace("%%", "%").replace("\n", " ").strip()


def _all_flags(parser: argparse.ArgumentParser) -> list[str]:
    flags: list[str] = []
    for action in _flag_actions(parser):
        flags.extend(action.option_strings)
    return sorted(flags, key=lambda f: (not f.startswith("--"), f))


def _bash_function(parser: argparse.ArgumentParser) -> str:
    flags = " ".join(_all_flags(parser))
    choice_cases: list[str] = []
    for action in _flag_actions(parser):
        if action.choices and _takes_value(action):
            words = " ".join(str(c) for c in action.choices)
            choice_cases.extend(
                f"        {opt})\n"
                f'            COMPREPLY=( $(compgen -W "{words}" -- "${{cur}}") )\n'
                f"            return 0\n"
                f"            ;;"
                for opt in action.option_strings
            )
    choices_block = "\n".join(choice_cases)
    case_block = f"    case \"${{prev}}\" in\n{choices_block}\n    esac\n" if choice_cases else ""
    return f'''_clipshrink() {{
    local cur prev
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"
{case_block}    if [[ "${{cur}}" == -* ]]; then
        COMPREPLY=( $(compgen -W "{flags}" -- "${{cur}}") )
        return 0
    fi
    COMPREPLY=( $(compgen -f -- "${{cur}}") )
}}'''


def bash_completions(parser: argparse.ArgumentParser) -> str:
    return (
        "# bash completion for clipshrink (generated: clipshrink --print-completions bash)\n"
        f"{_bash_function(parser)}\n"
        "complete -F _clipshrink clipshrink\n"
    )


def zsh_completions(parser: argparse.ArgumentParser) -> str:
    return (
        "#compdef clipshrink\n"
        "# zsh completion for clipshrink (generated: clipshrink --print-completions zsh)\n"
        "# Reuses the bash function: zsh _arguments can't handle colon choices.\n"
        "autoload -U +X bashcompinit && bashcompinit\n"
        f"{_bash_function(parser)}\n"
        "complete -F _clipshrink clipshrink\n"
    )


def _fish_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def fish_completions(parser: argparse.ArgumentParser) -> str:
    lines = [
        "# fish completion for clipshrink "
        "(generated: clipshrink --print-completions fish)",
        "complete -c clipshrink -l help -d 'Show help'",
        "complete -c clipshrink -l version -d 'Show version'",
    ]
    for action in _flag_actions(parser):
        if action.dest in ("help", "version"):
            continue
        desc = _fish_escape(_help_text(action))
        for opt in action.option_strings:
            parts = ["complete -c clipshrink"]
            if opt.startswith("--"):
                parts.append(f"-l {opt[2:]}")
            else:
                parts.append(f"-s {opt[1:]}")
            if _takes_value(action):
                parts.append("-r")
                if action.choices:
                    # choices need -f, and -f/-F are exclusive in fish
                    parts.append("-f")
                    words = " ".join(str(c) for c in action.choices)
                    parts.append(f"-a '{_fish_escape(words)}'")
                elif opt in _INPUT_FLAGS:
                    parts.append("-F")
            if desc:
                parts.append(f"-d '{desc}'")
            lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _roff_escape(text: str) -> str:
    text = text.replace("\\", "\\e").replace("%%", "%")
    if text.startswith("-"):
        text = f"\\-{text[1:]}"
    return text.replace(" -", " \\-")


def man_page(parser: argparse.ArgumentParser) -> str:
    description = (parser.description or "").replace("%%", "%").strip()
    lines = [
        ".\\" " Man page for clipshrink (generated: clipshrink --print-man)",
        f'.TH CLIPSHRINK 1 "" "clipshrink {__version__}" "User Commands"',
        ".SH NAME",
        "clipshrink \\- shrink images below a target file size while preserving quality",
        ".SH SYNOPSIS",
        ".B clipshrink",
        ".RI [ options ]",
        ".SH DESCRIPTION",
        description.replace("\n", "\n.PP\n"),
        ".SH OPTIONS",
    ]
    for action in _flag_actions(parser):
        if action.dest in ("help", "version"):
            continue
        flags = ", ".join(
            "\\-" * opt.count("-") + opt.lstrip("-")
            for opt in action.option_strings
        )
        lines.append(".TP")
        lines.append(f".BR {flags}")
        help_text = _roff_escape(_help_text(action))
        if help_text:
            lines.append(help_text)
    lines.append(".SH VERSION")
    lines.append(f"clipshrink {__version__}")
    return "\n".join(lines) + "\n"


def print_output(parser: argparse.ArgumentParser, kind: str, shell: str | None) -> str:
    if kind == "man":
        return man_page(parser)
    if shell == "bash":
        return bash_completions(parser)
    if shell == "zsh":
        return zsh_completions(parser)
    if shell == "fish":
        return fish_completions(parser)
    raise ValueError(f"unsupported completion shell: {shell!r}")
