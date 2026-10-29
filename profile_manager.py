# ============================================================
# WebMind Profile Manager
#
# Standardized application data and isolated browser profile manager.
# Stores profiles in standard system application directories
# (%LOCALAPPDATA%\WebMind\profiles) and provides zero-data-loss
# migration for existing user logins.
# ============================================================

import os
import shutil
import sys
from typing import Dict, Optional


PROVIDER_MAP: Dict[str, Dict[str, str]] = {
    "chatgpt": {
        "folder_name": "ChatGPT",
        "legacy_folder": "WebMindChrome",
    },
    "deepseek": {
        "folder_name": "DeepSeek",
        "legacy_folder": "WebMindChrome_DeepSeek",
    },
    "kimi": {
        "folder_name": "Kimi",
        "legacy_folder": "WebMindChrome_Kimi",
    },
    "glm": {
        "folder_name": "GLM",
        "legacy_folder": "WebMindChrome_GLM",
    },
}

# Alias lookups
ALIASES = {
    "1": "chatgpt",
    "2": "deepseek",
    "3": "kimi",
    "4": "glm",
    "chatglm": "glm",
}


def get_app_data_dir() -> str:
    """Returns the base WebMind application data directory (%LOCALAPPDATA%\WebMind)."""
    return os.path.dirname(get_profiles_root())


def get_profiles_root() -> str:
    """
    Returns the standard root directory for WebMind isolated profiles.
    Windows: %LOCALAPPDATA%\WebMind\profiles
    Other:   ~/.config/webmind/profiles
    """
    custom_root = os.environ.get("WEBMIND_PROFILES_DIR")
    if custom_root:
        return os.path.abspath(custom_root)

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "AppData", "Local")
        return os.path.join(base, "WebMind", "profiles")
    else:
        base = os.path.expanduser("~/.config")
        return os.path.join(base, "webmind", "profiles")


def _get_legacy_dir(legacy_folder_name: str) -> str:
    """Returns the legacy path where WebMind originally stored profiles (in ~)."""
    user_home = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    return os.path.join(user_home, legacy_folder_name)


def migrate_legacy_profiles(verbose: bool = True) -> int:
    """
    Checks for legacy profile directories in the user's home directory
    and migrates them into the standard %LOCALAPPDATA% location so that
    existing logins, cookies, and sessions are preserved with ZERO data loss.

    Returns the count of profiles migrated.
    """
    root = get_profiles_root()
    os.makedirs(root, exist_ok=True)
    migrated_count = 0

    for key, info in PROVIDER_MAP.items():
        legacy_path = _get_legacy_dir(info["legacy_folder"])
        target_path = os.path.join(root, info["folder_name"])

        # If legacy directory exists and contains files, but target does not exist yet:
        if os.path.exists(legacy_path) and not os.path.exists(target_path):
            try:
                if verbose:
                    print(f"[WebMind] Migrating {key.upper()} profile to standard location...")
                    print(f"          From: {legacy_path}")
                    print(f"          To:   {target_path}")
                shutil.move(legacy_path, target_path)
                migrated_count += 1
            except Exception as e:
                # If move fails (e.g. cross-volume or permission), attempt copytree
                try:
                    shutil.copytree(legacy_path, target_path)
                    migrated_count += 1
                except Exception as ce:
                    if verbose:
                        print(f"[!] Warning: Could not migrate {legacy_path} ({ce}). Using legacy path.")

    return migrated_count


def get_profile_dir(provider_key: str, auto_migrate: bool = True) -> str:
    """
    Retrieves the standard profile directory path for the specified AI provider.
    Ensures the directory exists. Automatically migrates legacy folders if found.
    """
    normalized_key = ALIASES.get(provider_key.lower().strip(), provider_key.lower().strip())
    info = PROVIDER_MAP.get(normalized_key)

    if not info:
        # Fallback for unrecognized provider name
        folder_name = provider_key.capitalize()
    else:
        folder_name = info["folder_name"]

    root = get_profiles_root()
    target_path = os.path.join(root, folder_name)

    if auto_migrate and info:
        legacy_path = _get_legacy_dir(info["legacy_folder"])
        if os.path.exists(legacy_path) and not os.path.exists(target_path):
            migrate_legacy_profiles(verbose=False)

    # If migration moved it, or if target already exists, return target
    if os.path.exists(target_path):
        return target_path

    # If legacy still exists (e.g. locked during migration), return legacy to keep login working
    if info:
        legacy_path = _get_legacy_dir(info["legacy_folder"])
        if os.path.exists(legacy_path):
            return legacy_path

    # Otherwise create standard target path
    os.makedirs(target_path, exist_ok=True)
    return target_path
