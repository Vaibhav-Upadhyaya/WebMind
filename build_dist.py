# ============================================================
# WebMind Standalone Distribution Builder
#
# Generates a production-grade, zero-setup portable distribution
# package for WebMind (dist/WebMind and dist/WebMind-Windows-Portable.zip).
# ============================================================

import os
import shutil
import subprocess
import sys
import zipfile

DIST_DIR = os.path.join(os.getcwd(), "dist", "WebMind")
ZIP_NAME_V101 = os.path.join(os.getcwd(), "dist", "WebMind-v1.0.1-Windows-Portable.zip")
ZIP_NAME_LEGACY = os.path.join(os.getcwd(), "dist", "WebMind-v1.0.0-Windows-Portable.zip")
ZIP_NAME_HOTPATCH = os.path.join(os.getcwd(), "dist", "WebMind-v1.0.1-HotPatch.zip")
ZIP_NAME = ZIP_NAME_V101

CORE_FILES = [
    "launcher.py",
    "launch.bat",
    "browser_manager.py",
    "profile_manager.py",
    "process_supervisor.py",
    "config_manager.py",
    "browser_guard.py",
    "logger.py",
    "diagnostics.py",
    "updater.py",
    "dom_patches.py",
    "agent_config.py",
    "service_manager.py",
    "dashboard.py",
    "tray_app.py",
    # ChatGPT
    "main.py",
    "protocol.py",
    "server.py",
    # DeepSeek
    "deepseek_browser.py",
    "deepseek_protocol.py",
    "deepseek_server.py",
    # Kimi
    "kimi_browser.py",
    "kimi_protocol.py",
    "kimi_server.py",
    # GLM
    "glm_browser.py",
    "glm_protocol.py",
    "glm_server.py",
    # Documentation
    "README.md",
]

README_DIST = """============================================================
WEBMIND - Production Local AI Bridge for Coding Agents
============================================================

WebMind connects your web AI accounts (ChatGPT, DeepSeek, Kimi, GLM)
directly into coding agents like OpenCode, Cursor, Continue.dev, and Aider
via standard OpenAI-compatible API endpoints (http://127.0.0.1:8001/v1).

QUICK START:
1. Double-click `WebMind.bat` (or `launch.bat`).
2. Choose your model:
     [1] ChatGPT   (Port 8001)
     [2] DeepSeek  (Port 8001)
     [3] Kimi      (Port 8001)
     [4] GLM       (Port 8001)
3. Connect your coding agent (OpenCode / Cursor / Continue):
     - Base URL: http://127.0.0.1:8001/v1
     - API Key:  sk-webmind
     - Press 'C' in the menu for 1-click automatic OpenCode configuration!

FEATURES:
- Universal Browser Support: Auto-detects Google Chrome and Microsoft Edge.
- Zero Conflict: Runs in dedicated, isolated browser profiles (%LOCALAPPDATA%\\WebMind).
  Your personal Chrome remains 100% untouched and can run side-by-side.
- Tool-Call Translation: DeepSeek, Kimi, GLM, and ChatGPT execute real workspace edits.
- Multi-Port Modes: Run any model on port 8001 or all 4 models concurrently (8001-8004).

============================================================
"""

WEBMIND_BAT = """@echo off
setlocal
cd /d "%~dp0"
title WebMind AI Bridge
if exist "runtime\\python.exe" (\r
    runtime\\python.exe launcher.py %*\r
) else (\r
    where python >nul 2>nul\r
    if %ERRORLEVEL% EQU 0 (\r
        python launcher.py %*\r
    ) else (\r
        echo [ERROR] Python not found. Please install Python 3.10+ or place embedded Python in runtime\\.\r
        pause\r
    )\r
)
"""

REQUIREMENTS_TXT = """fastapi>=0.100.0
uvicorn>=0.22.0
playwright>=1.40.0
httpx>=0.24.0
pydantic>=2.0.0
"""

UPDATE_WEBMIND_PS1 = """# ============================================================
# WebMind 1-Click Offline Updater (v1.0.1)
# Updates WebMind on this PC while preserving all profiles & logins
# ============================================================

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Target directory detection
$possibleDirs = @(
    (Join-Path $env:LOCALAPPDATA "Programs\\WebMind"),
    $scriptDir,
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "WebMind"),
    (Join-Path ([Environment]::GetFolderPath("MyDocuments")) "WebMind")
)

$targetDir = $null
foreach ($d in $possibleDirs) {
    if ((Test-Path (Join-Path $d "launcher.py")) -and ($d -ne $scriptDir -or -not (Test-Path (Join-Path $env:LOCALAPPDATA "Programs\\WebMind\\launcher.py")))) {
        $targetDir = $d
        break
    }
}

if (-not $targetDir -and (Test-Path (Join-Path $env:LOCALAPPDATA "Programs\\WebMind"))) {
    $targetDir = Join-Path $env:LOCALAPPDATA "Programs\\WebMind"
}

if (-not $targetDir) {
    $targetDir = $scriptDir
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       Updating WebMind to v1.0.1 (Resilient Engine)        " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Target Installation: $targetDir" -ForegroundColor Yellow

# Terminate running WebMind processes if any
Write-Host "Checking for running WebMind instances..." -ForegroundColor Green
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*launcher.py*" -or $_.CommandLine -like "*main.py*" -or $_.CommandLine -like "*server.py*"
} | Stop-Process -Force -ErrorAction SilentlyContinue

# Files to update
$files = Get-ChildItem -Path $scriptDir -File | Where-Object {
    $_.Extension -in @(".py", ".bat", ".vbs", ".txt", ".md", ".json") -and $_.Name -notlike "Update-WebMind*"
}

Write-Host "Copying updated modules..." -ForegroundColor Green
foreach ($f in $files) {
    Copy-Item -Path $f.FullName -Destination (Join-Path $targetDir $f.Name) -Force
}

# Update patches.json in AppData for immediate hotpatching
$appDataDir = Join-Path $env:LOCALAPPDATA "WebMind"
if (-not (Test-Path $appDataDir)) {
    New-Item -ItemType Directory -Path $appDataDir -Force | Out-Null
}
$patchesFile = Join-Path $appDataDir "patches.json"
if (Test-Path (Join-Path $scriptDir "patches.json")) {
    Copy-Item -Path (Join-Path $scriptDir "patches.json") -Destination $patchesFile -Force
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " [SUCCESS] WebMind updated to v1.0.1 successfully!           " -ForegroundColor Green
Write-Host " All browser profiles, logins, and settings preserved. " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
"""

UPDATE_WEBMIND_BAT = """@echo off
setlocal
cd /d "%~dp0"
title WebMind Updater v1.0.1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Update-WebMind.ps1"
echo.
pause
"""


def build():
    print("=" * 60)
    print("Building WebMind Standalone Distribution...")
    print("=" * 60)

    # 1. Clean previous build
    if os.path.exists(DIST_DIR):
        print(f"Cleaning existing directory: {DIST_DIR}")
        shutil.rmtree(DIST_DIR, ignore_errors=True)
    os.makedirs(DIST_DIR, exist_ok=True)

    # 2. Copy core scripts and bridges
    print("Copying core runtime modules...")
    for filename in CORE_FILES:
        src = os.path.join(os.getcwd(), filename)
        if os.path.exists(src):
            dst = os.path.join(DIST_DIR, filename)
            shutil.copy2(src, dst)
        else:
            print(f"[!] Warning: Missing source file {filename}")

    # 3. Create WebMind.bat & WebMind.vbs
    print("Generating one-click launchers & updater...")
    with open(os.path.join(DIST_DIR, "WebMind.bat"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(WEBMIND_BAT)

    # Silent background launcher (no CMD window)
    vbs_content = 'Set WshShell = CreateObject("WScript.Shell")\r\nWshShell.Run "cmd /c WebMind.bat -d", 0, False\r\n'
    with open(os.path.join(DIST_DIR, "WebMind-Silent.vbs"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(vbs_content)

    # 1-click Offline Updater scripts
    with open(os.path.join(DIST_DIR, "Update-WebMind.ps1"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(UPDATE_WEBMIND_PS1)

    with open(os.path.join(DIST_DIR, "Update-WebMind.bat"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(UPDATE_WEBMIND_BAT)

    # 4. Create requirements.txt and README
    with open(os.path.join(DIST_DIR, "requirements.txt"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(REQUIREMENTS_TXT)

    with open(os.path.join(DIST_DIR, "README_DIST.txt"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write(README_DIST)

    # 5. Create ZIP Archives
    print(f"Creating portable ZIP archive: {ZIP_NAME_V101}")
    os.makedirs(os.path.dirname(ZIP_NAME_V101), exist_ok=True)
    with zipfile.ZipFile(ZIP_NAME_V101, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(DIST_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(DIST_DIR))
                zipf.write(file_path, arcname)

    # Copy to legacy v1.0.0 filename for backward compatibility
    shutil.copy2(ZIP_NAME_V101, ZIP_NAME_LEGACY)
    print(f"Created backward-compatible alias: {ZIP_NAME_LEGACY}")

    # 6. Create lightweight HotPatch archive (< 200KB)
    print(f"Creating lightweight HotPatch archive: {ZIP_NAME_HOTPATCH}")
    with zipfile.ZipFile(ZIP_NAME_HOTPATCH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(DIST_DIR):
            file_path = os.path.join(DIST_DIR, file)
            if os.path.isfile(file_path) and not file.endswith(".zip"):
                zipf.write(file_path, file)

    print("\n[SUCCESS] WebMind distribution built successfully!")
    print(f"Directory:       {DIST_DIR}")
    print(f"Portable (v1.0.1): {ZIP_NAME_V101}")
    print(f"HotPatch Zip:    {ZIP_NAME_HOTPATCH}")
    print("=" * 60)


if __name__ == "__main__":
    build()
