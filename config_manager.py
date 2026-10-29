# ============================================================
# WebMind Configuration Manager
#
# Persistent configuration storage for WebMind settings and custom ports.
# Stores settings in %LOCALAPPDATA%\WebMind\config.json, with automatic
# fallback to default ports (8001 shared, 8001-8004 discrete).
# ============================================================

import json
import os
from typing import Dict, Any, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "mode": "shared",
    "shared_port": 8001,
    "dashboard_port": 8000,
    "api_key_required": False,
    "api_key": "sk-webmind",
    "auto_recover_browser": True,
    "persistent_chat": True,
    "ports": {
        "chatgpt": 8001,
        "deepseek": 8002,
        "kimi": 8003,
        "glm": 8004,
    },
    "cdp_ports": {
        "chatgpt": 9222,
        "deepseek": 9223,
        "kimi": 9224,
        "glm": 9225,
    },
}


def get_config_path() -> str:
    """Returns the persistent path to config.json."""
    custom_path = os.environ.get("WEBMIND_CONFIG_PATH")
    if custom_path:
        return os.path.abspath(custom_path)

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = os.path.join(os.environ.get("USERPROFILE", os.path.expanduser("~")), "AppData", "Local")
        return os.path.join(base, "WebMind", "config.json")
    else:
        base = os.path.expanduser("~/.config")
        return os.path.join(base, "webmind", "config.json")


def load_config() -> Dict[str, Any]:
    """Loads config.json from disk, merging with defaults for any missing keys."""
    path = get_config_path()
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_data = json.load(f)
                if isinstance(user_data, dict):
                    # Merge top-level keys
                    for k in [
                        "mode",
                        "shared_port",
                        "dashboard_port",
                        "api_key_required",
                        "api_key",
                        "auto_recover_browser",
                        "persistent_chat",
                    ]:
                        if k in user_data:
                            config[k] = user_data[k]
                    # Merge ports dict
                    if "ports" in user_data and isinstance(user_data["ports"], dict):
                        for k, v in user_data["ports"].items():
                            config["ports"][k.lower()] = int(v)
                    # Merge cdp_ports dict
                    if "cdp_ports" in user_data and isinstance(user_data["cdp_ports"], dict):
                        for k, v in user_data["cdp_ports"].items():
                            config["cdp_ports"][k.lower()] = int(v)
        except Exception:
            pass

    return config


def save_config(config_data: Dict[str, Any]) -> str:
    """Saves configuration data to config.json."""
    path = get_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(config_data, f, indent=2)
    return path


def validate_port(port: Any) -> bool:
    """Validates that a port number is an integer in the legal unprivileged range (1024-65535)."""
    if not isinstance(port, int) or isinstance(port, bool):
        return False
    return 1024 <= port <= 65535


def get_service_port(service_name: str, mode: Optional[str] = None) -> int:
    """Retrieves the active port for a given AI bridge based on mode."""
    cfg = load_config()
    svc = service_name.lower().strip()
    active_mode = mode or cfg.get("mode", "shared")

    if active_mode == "shared":
        return int(cfg.get("shared_port", 8001))
    else:
        return int(cfg.get("ports", {}).get(svc, DEFAULT_CONFIG["ports"].get(svc, 8001)))


def set_service_port(service_name: str, port: int) -> bool:
    """Sets and persists a custom discrete port for an AI bridge."""
    if not validate_port(port):
        return False

    svc = service_name.lower().strip()
    cfg = load_config()
    cfg["ports"][svc] = int(port)
    save_config(cfg)
    return True


def set_shared_port(port: int) -> bool:
    """Sets and persists the Shared Mode port."""
    if not validate_port(port):
        return False

    cfg = load_config()
    cfg["shared_port"] = int(port)
    save_config(cfg)
    return True


def reset_to_defaults() -> Dict[str, Any]:
    """Resets all settings and ports to factory defaults."""
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


reset_defaults = reset_to_defaults


def set_api_key_required(enabled: bool) -> bool:
    """Enables or disables strict API key authentication."""
    cfg = load_config()
    cfg["api_key_required"] = bool(enabled)
    save_config(cfg)
    return True


def set_api_key(key: str) -> bool:
    """Sets the custom API key."""
    if not key or not isinstance(key, str):
        return False
    cfg = load_config()
    cfg["api_key"] = key.strip()
    save_config(cfg)
    return True


def set_auto_recover(enabled: bool) -> bool:
    """Enables or disables automatic browser crash recovery."""
    cfg = load_config()
    cfg["auto_recover_browser"] = bool(enabled)
    save_config(cfg)
    return True


def is_api_key_valid(token: Optional[str]) -> bool:
    """
    Validates a bearer token.
    If api_key_required is False, accepts any token (or absence).
    If api_key_required is True, verifies token matches the configured api_key.
    """
    cfg = load_config()
    if not cfg.get("api_key_required", False):
        return True
    if not token:
        return False
    clean_token = token.strip()
    if clean_token.lower().startswith("bearer "):
        clean_token = clean_token[7:].strip()
    configured_key = cfg.get("api_key", "sk-webmind").strip()
    return clean_token == configured_key
