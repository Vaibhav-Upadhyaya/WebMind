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
DEFAULT_GITHUB_REPO = os.environ.get("WEBMIND_GITHUB_REPO", "Vaibhav-Upadhyaya-codes/WebMind")


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
    Queries GitHub Releases API for the latest release.
    Returns update status, release notes, and asset download URLs.
    """
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"WebMind/{current_version}",
            "Accept": "application/vnd.github.v3+json",
        },
    )

    result: Dict[str, Any] = {
        "update_available": False,
        "current_version": current_version,
        "latest_version": current_version,
        "release_name": "",
        "release_notes": "",
        "download_url": "",
        "assets": [],
        "published_at": "",
        "checked_at": "",
    }

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                tag_name = data.get("tag_name", "").lstrip("vV")
                result["latest_version"] = tag_name
                result["release_name"] = data.get("name", "")
                result["release_notes"] = data.get("body", "")
                result["published_at"] = data.get("published_at", "")
                result["download_url"] = data.get("html_url", "")

                assets = []
                for a in data.get("assets", []):
                    assets.append({
                        "name": a.get("name"),
                        "size": a.get("size"),
                        "download_url": a.get("browser_download_url"),
                    })
                result["assets"] = assets

                if is_newer_version(tag_name, current_version):
                    result["update_available"] = True
                    log.info(f"Update available: {current_version} -> {tag_name}")
                else:
                    log.info(f"WebMind is up to date ({current_version}).")
    except Exception as e:
        log.warning(f"Update check skipped or failed ({e}). WebMind continues normally.")

    return result


def stage_update_file(source_path: str, filename: Optional[str] = None) -> str:
    """Copies an update file into the staging directory for deployment."""
    dest_dir = get_update_dir()
    name = filename or os.path.basename(source_path)
    dest_path = os.path.join(dest_dir, name)

    with open(source_path, "rb") as sf, open(dest_path, "wb") as df:
        df.write(sf.read())

    log.info(f"Update staged at: {dest_path}")
    return dest_path
