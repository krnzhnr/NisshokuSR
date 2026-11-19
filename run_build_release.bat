@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
echo [*] NisshokuSR release build: portable + installer
python build_release.py
if errorlevel 1 (
  echo [!] Build failed.
  pause
  exit /b 1
)
echo.
echo [*] Done. See dist\ and setup_output\.
pause
