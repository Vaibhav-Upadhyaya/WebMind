# ============================================================
# WebMind 1-Line Installer for Windows
# Usage:
#   irm https://raw.githubusercontent.com/Vaibhav-Upadhyaya/WebMind/main/install.ps1 | iex
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "       Installing WebMind AI Bridge (Production v1.0.1)      " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Target directory
$installDir = Join-Path $env:LOCALAPPDATA "Programs\WebMind"
if (-not (Test-Path $installDir)) {
    New-Item -ItemType Directory -Path $installDir -Force | Out-Null
}

# 2. Check for Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[!] Python 3.11+ is required but not found in PATH." -ForegroundColor Red
    Write-Host "    Please install Python from https://www.python.org/downloads/ and check 'Add to PATH'." -ForegroundColor Yellow
    Exit 1
}

# 3. Download latest release
$repo = "Vaibhav-Upadhyaya/WebMind"
$tempZip = Join-Path $env:TEMP "WebMind-Windows-Portable.zip"

Write-Host "[1/5] Checking latest WebMind package from GitHub..." -ForegroundColor Green
$downloadUrl = $null

try {
    $headers = @{ "User-Agent" = "WebMind-Installer" }
    $releaseInfo = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/releases/latest" -Headers $headers -TimeoutSec 5 -ErrorAction Stop
    $asset = $releaseInfo.assets | Where-Object { $_.name -like "*Portable.zip" } | Select-Object -First 1
    if ($asset -and $asset.browser_download_url) {
        $downloadUrl = $asset.browser_download_url
    }
} catch {
    # No published releases yet or rate-limited; fallback below
}

Write-Host "[2/5] Downloading package..." -ForegroundColor Green
$downloadSuccess = $false

if ($downloadUrl) {
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $tempZip -UserAgent "WebMind-Installer" -UseBasicParsing
        $downloadSuccess = $true
    } catch {
        # Fall back to repository archive
    }
}

if (-not $downloadSuccess) {
    $archiveUrl = "https://github.com/$repo/archive/refs/heads/main.zip"
    try {
        Invoke-WebRequest -Uri $archiveUrl -OutFile $tempZip -UserAgent "WebMind-Installer" -UseBasicParsing
        $downloadSuccess = $true
    } catch {
        Write-Host "[!] Download failed: $_" -ForegroundColor Red
        Write-Host "    Please ensure your internet connection is active or clone manually:" -ForegroundColor Yellow
        Write-Host "    git clone https://github.com/$repo.git" -ForegroundColor Yellow
        Exit 1
    }
}

Write-Host "[3/5] Extracting to $installDir..." -ForegroundColor Green
Expand-Archive -Path $tempZip -DestinationPath $env:TEMP -Force

# Locate extracted directory (supports both portable zip and GitHub archive)
$extractedFolder = Join-Path $env:TEMP "WebMind"
if (-not (Test-Path $extractedFolder)) {
    $extractedFolder = Join-Path $env:TEMP "WebMind-main"
}
if (Test-Path $extractedFolder) {
    Copy-Item -Path "$extractedFolder\*" -Destination $installDir -Recurse -Force
    Remove-Item -Recurse -Force $extractedFolder
}
Remove-Item -Force $tempZip

# Ensure WebMind.bat launcher is present
$batLauncher = Join-Path $installDir "WebMind.bat"
if (-not (Test-Path $batLauncher)) {
    $batContent = "@echo off`r`nsetlocal`r`ncd /d `"%~dp0`"`r`ntitle WebMind AI Bridge`r`npython launcher.py %*`r`n"
    Set-Content -Path $batLauncher -Value $batContent -Encoding Ascii
}

Write-Host "[4/5] Installing dependencies..." -ForegroundColor Green
& python -m pip install -r "$installDir\requirements.txt" --quiet --disable-pip-version-check

Write-Host "[5/5] Creating Desktop & Start Menu shortcuts..." -ForegroundColor Green
$wsh = New-Object -ComObject WScript.Shell

# Desktop shortcut
$desktopShortcut = $wsh.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "WebMind.lnk"))
$desktopShortcut.TargetPath = (Join-Path $installDir "WebMind.bat")
$desktopShortcut.WorkingDirectory = $installDir
$desktopShortcut.Description = "Launch WebMind Local AI Bridge"
$desktopShortcut.Save()

# Start menu shortcut
$startMenuDir = Join-Path ([Environment]::GetFolderPath("StartMenu")) "Programs\WebMind"
if (-not (Test-Path $startMenuDir)) {
    New-Item -ItemType Directory -Path $startMenuDir -Force | Out-Null
}
$startShortcut = $wsh.CreateShortcut((Join-Path $startMenuDir "WebMind.lnk"))
$startShortcut.TargetPath = (Join-Path $installDir "WebMind.bat")
$startShortcut.WorkingDirectory = $installDir
$startShortcut.Description = "Launch WebMind Local AI Bridge"
$startShortcut.Save()

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Installation Complete! Starting WebMind...                  " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

# Launch in background
Start-Process -FilePath (Join-Path $installDir "WebMind.bat") -WorkingDirectory $installDir

Start-Sleep -Seconds 2
Start-Process "http://localhost:8000/dashboard"
