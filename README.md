# clipshrink

Shrink images below a target file size while preserving quality. Input
from a file, a folder, or the system clipboard and output to a file, a
folder, or the clipboard.

## Install

### Installing the package

The recommended way for a CLI is [pipx]:

```bash
pipx install .
```

Afterwards `clipshrink --version` should work from any directory.
If you don't wish to install (although it is recommended) you can run
in place instead with `python clipshrink.py`.

### Shell completions & man page

Both are generated from the CLI parser, so they always match the current
flag surface. Re-run the same commands after updating clipshrink:

```bash
# fish:
clipshrink --print-completions fish > ~/.config/fish/completions/clipshrink.fish

# bash:
clipshrink --print-completions bash > ~/.local/share/bash-completions/completions/clipshrink
#   (ensure bash-completion is installed and ~/.local/share/bash-completions is sourced)

# zsh:
mkdir -p ~/.zfunc && clipshrink --print-completions zsh > ~/.zfunc/_clipshrink
#   then add to ~/.zshrc BEFORE compinit:  fpath=(~/.zfunc $fpath)

# view manpage without installing:
clipshrink --print-man | man -l -

# man page, permanent (user-local):
mkdir -p ~/.local/share/man/man1
clipshrink --print-man | gzip > ~/.local/share/man/man1/clipshrink.1.gz
man clipshrink     # works if ~/.local/share/man is in MANPATH (default on most distros)
```

## Usage

Run `clipshrink --help` for the full flag list.
`clipshrink` is an all-default clipboard in and clipboard out run.