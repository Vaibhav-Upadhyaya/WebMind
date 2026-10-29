@echo off
setlocal
cd /d "%~dp0"
title WebMind Multi-AI Launcher
python launcher.py %*
