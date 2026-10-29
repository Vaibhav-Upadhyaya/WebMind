# ============================================================
# WebMind System Diagnostics & Health Reporting
#
# Generates sanitized diagnostic summaries and 1-click exportable
# diagnostic bundles for rapid, privacy-preserving troubleshooting.
# ============================================================

from datetime import datetime, timezone
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, Optional
import zipfile

import browser_manager
import config_manager
import logger
import profile_manager

log = logger.get_logger("diagnostics")


def get_diagnostics_summary() -> Dict[str, Any]:
    """Generates a complete, sanitized system and runtime health report."""
    cfg = config_manager.load_config()
    browser_exe, browser_name = browser_manager.find_chromium_executable()
    profiles_root = profile_manager.get_profiles_root()

    cdp_ports = cfg.get("cdp_ports", {
        "chatgpt": 9222,
        "deepseek": 9223,
        "kimi": 9224,
        "glm": 9225,
    })

    cdp_status = {}
    for name, port in cdp_ports.items():
        cdp_status[name] = {
            "port": port,
            "connected": browser_manager.is_cdp_available("127.0.0.1", int(port), timeout=0.3),
        }

    # Profile directory status
    profiles_status = {}
    for name in ["chatgpt", "deepseek", "kimi", "glm"]:
        pdir = profile_manager.get_profile_dir(name)
        exists = os.path.exists(pdir)
        size_mb = 0.0
        if exists:
            try:
                total_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(pdir)
                    for filename in filenames
                )
                size_mb = round(total_bytes / (1024 * 1024), 2)
            except Exception:
                pass
        profiles_status[name] = {
            "path": pdir,
            "exists": exists,
            "size_mb": size_mb,
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "system": {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": sys.version.split()[0],
        },
        "browser": {
            "detected_name": browser_name,
            "executable_path": browser_exe,
            "available": bool(browser_exe and os.path.exists(browser_exe)),
        },
        "config": {
            "mode": cfg.get("mode", "shared"),
            "shared_port": cfg.get("shared_port", 8001),
            "ports": cfg.get("ports", {}),
            "api_key_required": cfg.get("api_key_required", False),
            "auto_recover_browser": cfg.get("auto_recover_browser", True),
        },
        "cdp_reachability": cdp_status,
        "profiles": {
            "root": profiles_root,
            "items": profiles_status,
        },
        "recent_log_tail": logger.get_recent_logs(lines=50),
    }

    return report


def export_diagnostic_zip(output_dir: Optional[str] = None) -> str:
    """
    Bundles the diagnostic summary and active log files into a zip archive.
    Returns the path to the created zip file.
    """
    summary = get_diagnostics_summary()

    if output_dir:
        target_dir = os.path.abspath(output_dir)
    else:
        target_dir = os.path.join(profile_manager.get_app_data_dir(), "diagnostics")
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(target_dir, f"webmind_diagnostics_{timestamp}.zip")

    temp_dir = tempfile.mkdtemp()
    try:
        # Write summary.json
        summary_file = os.path.join(temp_dir, "diagnostics_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Write sanitized log file
        log_file = logger.get_log_file()
        bundled_log = os.path.join(temp_dir, "webmind.log")
        if os.path.exists(log_file):
            try:
                shutil.copy2(log_file, bundled_log)
            except Exception:
                with open(bundled_log, "w", encoding="utf-8") as f:
                    f.write(summary.get("recent_log_tail", ""))
        else:
            with open(bundled_log, "w", encoding="utf-8") as f:
                f.write(summary.get("recent_log_tail", ""))

        # Create zip
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(summary_file, "diagnostics_summary.json")
            zf.write(bundled_log, "webmind.log")

        log.info(f"Diagnostic bundle created at: {zip_path}")
        return zip_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
