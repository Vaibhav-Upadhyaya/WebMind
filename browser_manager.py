# ============================================================
# WebMind Browser Manager
#
# Universal Chromium browser detection and launch orchestrator.
# Automatically detects Google Chrome, Microsoft Edge, Brave,
# and Chromium, providing zero-barrier execution on any machine.
# ============================================================

import os
import shutil
import subprocess
import urllib.request
from typing import Tuple, List, Optional


def find_chromium_executable() -> Tuple[str, str]:
    """
    Locates the best available Chromium-based browser on the system.
    Returns a tuple of (executable_path, browser_name).

    Search priority:
      1. Explicit environment variable WEBMIND_BROWSER_PATH
      2. Google Chrome (Standard, Beta, Dev, Canary)
      3. Microsoft Edge (Pre-installed on all modern Windows systems)
      4. Brave Browser
      5. PATH lookups ('chrome', 'google-chrome', 'msedge', 'brave', 'chromium')
    """
    # 1. Custom override via environment variable
    custom_path = os.environ.get("WEBMIND_BROWSER_PATH")
    if custom_path and os.path.exists(custom_path):
        return custom_path, "Custom Browser"

    # Search candidates (path, display_name)
    candidates: List[Tuple[str, str]] = []

    # Windows standard paths
    if os.name == "nt":
        prog_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
        prog_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")

        candidates.extend([
            # Google Chrome
            (os.path.join(prog_files, "Google", "Chrome", "Application", "chrome.exe"), "Google Chrome"),
            (os.path.join(prog_files_x86, "Google", "Chrome", "Application", "chrome.exe"), "Google Chrome"),
            (os.path.join(local_app_data, "Google", "Chrome", "Application", "chrome.exe"), "Google Chrome (User)"),
            # Microsoft Edge (Available on 100% of modern Windows PCs)
            (os.path.join(prog_files_x86, "Microsoft", "Edge", "Application", "msedge.exe"), "Microsoft Edge"),
            (os.path.join(prog_files, "Microsoft", "Edge", "Application", "msedge.exe"), "Microsoft Edge"),
            (os.path.join(local_app_data, "Microsoft", "Edge", "Application", "msedge.exe"), "Microsoft Edge (User)"),
            # Brave Browser
            (os.path.join(prog_files, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"), "Brave Browser"),
            (os.path.join(local_app_data, "BraveSoftware", "Brave-Browser", "Application", "brave.exe"), "Brave Browser (User)"),
        ])
    else:
        # macOS & Linux candidate paths
        candidates.extend([
            # macOS
            ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "Google Chrome"),
            ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge", "Microsoft Edge"),
            ("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser", "Brave Browser"),
            # Linux
            ("/usr/bin/google-chrome", "Google Chrome"),
            ("/usr/bin/google-chrome-stable", "Google Chrome"),
            ("/usr/bin/microsoft-edge", "Microsoft Edge"),
            ("/usr/bin/microsoft-edge-stable", "Microsoft Edge"),
            ("/usr/bin/brave-browser", "Brave Browser"),
            ("/usr/bin/chromium", "Chromium"),
            ("/usr/bin/chromium-browser", "Chromium"),
        ])

    for path, name in candidates:
        if path and os.path.exists(path):
            return path, name

    # PATH lookups
    for bin_name, name in [
        ("chrome", "Google Chrome"),
        ("google-chrome", "Google Chrome"),
        ("google-chrome-stable", "Google Chrome"),
        ("msedge", "Microsoft Edge"),
        ("brave", "Brave Browser"),
        ("chromium", "Chromium"),
        ("chromium-browser", "Chromium"),
    ]:
        resolved = shutil.which(bin_name)
        if resolved:
            return resolved, name

    # Fallback to standard binary name
    fallback_bin = "chrome.exe" if os.name == "nt" else "google-chrome"
    return fallback_bin, "Chromium (Fallback)"


def get_chromium_launch_args(
    executable: str,
    cdp_port: int,
    profile_dir: str,
    target_url: str,
    extra_flags: Optional[List[str]] = None,
) -> List[str]:
    """
    Constructs the robust standard command-line argument list for launching
    a dedicated Chromium instance with its own profile and remote debugging.
    """
    args = [
        executable,
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-blink-features=AutomationControlled",
    ]
    if extra_flags:
        args.extend(extra_flags)
    args.append(target_url)
    return args


def is_cdp_available(host: str = "127.0.0.1", port: int = 9222, timeout: float = 1.5) -> bool:
    """Checks whether the Chrome DevTools Protocol endpoint is answering on host:port."""
    url = f"http://{host}:{port}/json/version"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False
