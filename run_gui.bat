@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
  py -3 "tools\gui.py" %*
) else (
  python "tools\gui.py" %*
)

if errorlevel 1 pause
