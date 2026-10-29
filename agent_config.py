# ============================================================
# WebMind Coding Agent Config Generator
#
# Generates zero-setup, copy-pasteable or 1-click exportable
# configuration files for OpenCode, Cursor, Continue.dev, and Aider.
# ============================================================

import json
import os
from typing import Dict, Any, Optional

from config_manager import load_config


def get_default_opencode_config_path() -> str:
    """Finds the default opencode.json location on the current system."""
    home = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    candidates = [
        os.path.join(home, ".config", "opencode", "opencode.json"),
        os.path.join(home, ".opencode", "opencode.json"),
        os.path.join(os.getcwd(), "opencode.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(home, ".config", "opencode", "opencode.json")


def generate_opencode_config(mode: Optional[str] = None, shared_port: Optional[int] = None) -> Dict[str, Any]:
    """
    Generates a full opencode.json model configuration dictionary.
    In 'shared' mode (default), all models route to the active shared port.
    In 'discrete' mode, models route to their dedicated ports.
    Configuration is loaded from config_manager unless explicitly overridden.
    """
    cfg = load_config()
    active_mode = mode if mode is not None else cfg.get("mode", "shared")
    sp = shared_port if shared_port is not None else cfg.get("shared_port", 8001)

    if active_mode == "discrete":
        cfg_ports = cfg.get("ports", {})
        ports = {
            "chatgpt": cfg_ports.get("chatgpt", 8001),
            "deepseek": cfg_ports.get("deepseek", 8002),
            "kimi": cfg_ports.get("kimi", 8003),
            "glm": cfg_ports.get("glm", 8004),
        }
    else:
        ports = {"chatgpt": sp, "deepseek": sp, "kimi": sp, "glm": sp}

    config = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "webmind": {
                "name": "WebMind",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": f"http://127.0.0.1:{ports['chatgpt']}/v1",
                    "apiKey": "sk-webmind",
                },
                "models": {
                    "shared": {"name": "shared"},
                    "chatgpt": {"name": "ChatGPT"},
                },
            },
            "webmind-deepseek": {
                "name": "WebMind",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": f"http://127.0.0.1:{ports['deepseek']}/v1",
                    "apiKey": "sk-webmind",
                },
                "models": {"deepseek": {"name": "DeepSeek"}},
            },
            "webmind-kimi": {
                "name": "WebMind",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": f"http://127.0.0.1:{ports['kimi']}/v1",
                    "apiKey": "sk-webmind",
                },
                "models": {"kimi": {"name": "Kimi"}},
            },
            "webmind-glm": {
                "name": "WebMind",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "baseURL": f"http://127.0.0.1:{ports['glm']}/v1",
                    "apiKey": "sk-webmind",
                },
                "models": {"glm-4": {"name": "GLM-4"}},
            },
        },
        "models": {
            "webmind-shared": {
                "type": "openai",
                "name": "shared",
                "provider": "WebMind",
                "baseUrl": f"http://127.0.0.1:{sp}/v1",
                "apiKey": "sk-webmind",
                "model": "webmind",
            },
            "webmind-chatgpt": {
                "type": "openai",
                "name": "ChatGPT",
                "provider": "WebMind",
                "baseUrl": f"http://127.0.0.1:{ports['chatgpt']}/v1",
                "apiKey": "sk-webmind",
                "model": "webmind",
            },
            "webmind-deepseek": {
                "type": "openai",
                "name": "DeepSeek",
                "provider": "WebMind",
                "baseUrl": f"http://127.0.0.1:{ports['deepseek']}/v1",
                "apiKey": "sk-webmind",
                "model": "deepseek-chat",
            },
            "webmind-kimi": {
                "type": "openai",
                "name": "Kimi",
                "provider": "WebMind",
                "baseUrl": f"http://127.0.0.1:{ports['kimi']}/v1",
                "apiKey": "sk-webmind",
                "model": "kimi",
            },
            "webmind-glm": {
                "type": "openai",
                "name": "GLM-4",
                "provider": "WebMind",
                "baseUrl": f"http://127.0.0.1:{ports['glm']}/v1",
                "apiKey": "sk-webmind",
                "model": "glm-4",
            },
        },
    }
    return config


def write_opencode_config(target_path: Optional[str] = None, mode: Optional[str] = None) -> str:
    """Writes or merges WebMind models into the user's opencode.json file."""
    path = target_path or get_default_opencode_config_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    generated = generate_opencode_config(mode=mode)

    existing_data: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception:
            existing_data = {}

    # Merge provider
    if "provider" not in existing_data or not isinstance(existing_data["provider"], dict):
        existing_data["provider"] = {}
    for k, v in generated.get("provider", {}).items():
        existing_data["provider"][k] = v

    # Merge models
    if "models" not in existing_data or not isinstance(existing_data["models"], dict):
        existing_data["models"] = {}

    for k, v in generated["models"].items():
        existing_data["models"][k] = v

    if "$schema" not in existing_data:
        existing_data["$schema"] = generated["$schema"]

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(existing_data, f, indent=2)

    return path


def get_cursor_instructions(port: Optional[int] = None) -> str:
    """Returns exact configuration setup instructions for Cursor IDE."""
    if port is None:
        port = load_config().get("shared_port", 8001)
    return f"""Cursor Settings -> Models -> OpenAI API Key & Base URL:
  1. Base URL: http://127.0.0.1:{port}/v1
  2. API Key:  sk-webmind
  3. Model:    webmind  (or deepseek-chat, kimi, glm-4)"""


def get_continue_config_yaml(port: Optional[int] = None) -> str:
    """Returns YAML block for Continue.dev (~/.continue/config.yaml)."""
    if port is None:
        port = load_config().get("shared_port", 8001)
    return f"""models:
  - name: WebMind
    provider: openai
    model: webmind
    apiBase: http://127.0.0.1:{port}/v1
    apiKey: sk-webmind"""


def get_aider_command(port: Optional[int] = None) -> str:
    """Returns the one-line terminal command to launch Aider with WebMind."""
    if port is None:
        port = load_config().get("shared_port", 8001)
    return f"aider --openai-api-base http://127.0.0.1:{port}/v1 --openai-api-key sk-webmind --model webmind"
