@echo off
setlocal
cd /d "%~dp0"
uv run --locked python scripts/viewer.py
pause
