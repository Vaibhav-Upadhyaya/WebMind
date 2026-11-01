@echo off
setlocal
cd /d "%~dp0"
title WebMind AI Bridge
if exist "runtime\python.exe" (
    runtime\python.exe launcher.py %*
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python launcher.py %*
    ) else (
        echo [ERROR] Python not found. Please install Python 3.10+ or place embedded Python in runtime\.
        pause
    )
)
