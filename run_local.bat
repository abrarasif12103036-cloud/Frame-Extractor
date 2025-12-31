@echo off
REM Run the full local setup using PowerShell (bypasses execution policy for this script)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1"
pause