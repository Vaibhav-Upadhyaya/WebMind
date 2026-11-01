# ============================================================
# WebMind Auto-Updater Engine
#
# Checks for updates against GitHub Releases using semantic
# versioning, downloads release manifests, and stages updates.
# ============================================================

import json
import os
import re
import urllib.request
from typing import Any, Dict, Optional, Tuple

import logger
import profile_manager

log = logger.get_logger("updater")

CURRENT_VERSION = "1.0.1"
DEFAULT_GITHUB_REPO = os.environ.get("WEBMIND_GITHUB_REPO", "Vaibhav-Upadhyaya/WebMind")


def parse_version(v: str) -> Tuple[int, ...]:
    """
    Parses a version string like 'v1.0.2' or '1.0.0-rc1' into a tuple of ints
    for reliable semantic comparisons.
    """
    if not v:
        return (0, 0, 0)
    clean = re.sub(r"^[vV]", "", v.strip())
    # Split on first hyphen if pre-release tag exists
    core = clean.split("-")[0]
    parts = []
    for piece in core.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def is_newer_version(candidate: str, baseline: str = CURRENT_VERSION) -> bool:
    """Returns True if candidate version is strictly newer than baseline."""
    return parse_version(candidate) > parse_version(baseline)


def get_update_dir() -> str:
    """Returns the persistent directory used for update staging."""
    update_dir = os.path.join(profile_manager.get_app_data_dir(), "updates")
    os.makedirs(update_dir, exist_ok=True)
    return update_dir


def check_for_updates(
    repo: str = DEFAULT_GITHUB_REPO,
    current_version: str = CURRENT_VERSION,
    timeout: float = 3.0,
) -> Dict[str, Any]:
    """
    Queries GitHub API for the latest published release.
    Returns a dict with update status, version, download URLs, and changelog.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"WebMind-Updater/{current_version}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        latest_tag = data.get("tag_name", "")
        changelog = data.get("body", "")
        assets = data.get("assets", [])

        portable_zip_url: Optional[str] = None
        hotpatch_zip_url: Optional[str] = None

        for asset in assets:
            name = asset.get("name", "").lower()
            download_url = asset.get("browser_download_url", "")
            if "hotpatch" in name:
                hotpatch_zip_url = download_url
            elif "portable" in name or name.endswith(".zip"):
                portable_zip_url = download_url

        has_update = is_newer_version(latest_tag, current_version)
        if has_update:
            log.info(f"Update available: {current_version} -> {latest_tag}")
        else:
            log.info(f"WebMind is up to date ({current_version}).")

        return {
            "has_update": has_update,
            "current_version": current_version,
            "latest_version": latest_tag,
            "changelog": changelog,
            "portable_zip_url": portable_zip_url,
            "hotpatch_zip_url": hotpatch_zip_url,
            "published_at": data.get("published_at", ""),
        }
    except Exception as e:
        log.warning(f"Update check skipped or failed ({e}). WebMind continues normally.")
        return {
            "has_update": False,
            "current_version": current_version,
            "error": str(e),
        }


def stage_update_payload(payload_bytes: bytes, target_filename: str) -> str:
    """
    Writes downloaded update package to the staging directory.
    Returns the absolute path to the staged file.
    """
    dest = os.path.join(get_update_dir(), target_filename)
    with open(dest, "wb") as f:
        f.write(payload_bytes)
    log.info(f"Update staged at: {dest}")
    return dest
