#!/bin/zsh
set -e
cd -- "${0:A:h}"
exec uv run --locked python scripts/viewer.py "$@"
