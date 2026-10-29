# ============================================================
# WebMind Dynamic DOM Selector Hot-Patching Engine
#
# Provides resilient selector cascades for web AI interfaces and allows
# live hot-patching from %LOCALAPPDATA%\WebMind\patches.json when
# providers update their DOM elements without binary reinstallation.
# ============================================================

import json
import os
from typing import Any, Dict, List, Union

import logger
import profile_manager

log = logger.get_logger("dom_patches")

DEFAULT_SELECTORS: Dict[str, Dict[str, List[str]]] = {
    "chatgpt": {
        "input": [
            "#prompt-textarea",
            'div[contenteditable="true"]',
            "textarea",
        ],
        "send": [
            'button[data-testid="send-button"]:visible',
            'button[aria-label*="Send"]:visible',
            'button[aria-label*="\u53d1\u9001"]:visible',
            'button[aria-label="Dictate"]:visible',
            'button[aria-label="Start Voice"]:visible',
            'button:has(svg):visible',
        ],
        "stop": [
            'button[data-testid="stop-button"]:visible',
            'button[aria-label*="Stop"]:visible',
            'button[aria-label*="\u505c\u6b62"]:visible',
        ],
        "messages": [
            '[data-markdown-text-style="assistant-message"]',
            '[data-chatgpt-search-unit-key*="assistant"]',
            '[data-content-search-unit-key*="assistant"]',
            'div[class*="MarkdownRoot"]',
            '[data-conversation-role="assistant"]',
            '[data-message-author-role="assistant"]',
            'article [data-message-author-role="assistant"]',
            '.markdown',
        ],
    },
    "deepseek": {
        "input": [
            "textarea#chat-input",
            'textarea[placeholder*="DeepSeek"]',
            'textarea[placeholder*="\u53d1\u9001"]',
            'div[contenteditable="true"]',
            "textarea",
        ],
        "send": [
            'div[role="button"][aria-label*="Send"]:visible',
            'button[aria-label*="Send"]:visible',
            'button[aria-label*="\u53d1\u9001"]:visible',
            'button:has(svg):visible',
        ],
        "stop": [
            'div[role="button"][aria-label*="Stop"]:visible',
            'button[aria-label*="Stop"]:visible',
            'button[aria-label*="\u505c\u6b62"]:visible',
        ],
        "messages": [
            ".ds-markdown:visible",
            ".chat-message:visible",
            '[class*="message-content"]:visible',
        ],
    },
    "kimi": {
        "input": [
            'div.chat-input-editor[contenteditable="true"]',
            'div[contenteditable="true"]',
            'div[role="textbox"][contenteditable="true"]',
            'textarea[placeholder*="Kimi"]',
            'textarea[placeholder*="\u53d1\u9001"]',
            "textarea",
        ],
        "send": [
            'button[aria-label*="\u53d1\u9001"]:visible',
            'div[role="button"][aria-label*="Send"]:visible',
            'button[aria-label*="Send"]:visible',
            'button:has(svg):visible',
        ],
        "messages": [
            ".chat-content-container:visible",
            ".markdown-body:visible",
            '[class*="segment-content"]:visible',
        ],
    },
    "glm": {
        "input": [
            'textarea[placeholder*="GLM"]',
            'textarea[placeholder*="\u8bf7\u8f93\u5165"]',
            'div[contenteditable="true"]',
            "textarea",
        ],
        "send": [
            'button[aria-label*="\u53d1\u9001"]:visible',
            'div[role="button"][aria-label*="Send"]:visible',
            'button[aria-label*="Send"]:visible',
            'button:has(svg):visible',
        ],
        "messages": [
            ".conversation-item:visible",
            ".markdown-body:visible",
            '[class*="message-content"]:visible',
        ],
    },
}


def get_patches_file() -> str:
    """Returns the persistent patches.json path."""
    return os.path.join(profile_manager.get_app_data_dir(), "patches.json")


def load_patches() -> Dict[str, Any]:
    """Loads active hot-patches from disk."""
    path = get_patches_file()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        log.warning(f"Could not load patches.json: {e}")
        return {}


def save_patches(patches: Dict[str, Any]) -> str:
    """Saves updated hot-patches to disk."""
    path = get_patches_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(patches, f, indent=2)
    return path


def get_active_selectors(provider: str) -> Dict[str, List[str]]:
    """
    Returns active selector cascades for a given provider, merging
    built-in defaults with any hot-patches from disk.
    Patches take precedence (placed first), but default selectors are
    always preserved as fallback to ensure resilient degradation.
    """
    key = provider.lower().strip()
    defaults = DEFAULT_SELECTORS.get(key, {})
    patches = load_patches().get(key, {})

    merged: Dict[str, List[str]] = {}
    for role, default_list in defaults.items():
        if role in patches:
            patch_val = patches[role]
            if isinstance(patch_val, list):
                patched = [str(s).strip() for s in patch_val if str(s).strip()]
                merged[role] = patched + [s for s in default_list if s not in patched]
            elif isinstance(patch_val, str):
                patched = [patch_val.strip()] if patch_val.strip() else []
                merged[role] = patched + [s for s in default_list if s not in patched]
            else:
                merged[role] = list(default_list)
        else:
            merged[role] = list(default_list)

    # In case the hot-patch defines custom roles not in defaults:
    for role, patch_val in patches.items():
        if role not in merged:
            if isinstance(patch_val, list):
                merged[role] = [str(s).strip() for s in patch_val if str(s).strip()]
            elif isinstance(patch_val, str) and patch_val.strip():
                merged[role] = [patch_val.strip()]

    return merged


def apply_patch(provider: str, role: str, selectors: Union[List[str], str]) -> bool:
    """
    Applies and persists a selector hot-patch for a provider element role
    (e.g., 'input', 'send', 'messages').
    """
    p_key = provider.lower().strip()
    r_key = role.lower().strip()

    if isinstance(selectors, str):
        sel_list = [selectors.strip()]
    elif isinstance(selectors, list):
        sel_list = [str(s).strip() for s in selectors if str(s).strip()]
    else:
        return False

    patches = load_patches()
    if p_key not in patches or not isinstance(patches[p_key], dict):
        patches[p_key] = {}

    patches[p_key][r_key] = sel_list
    save_patches(patches)
    log.info(f"Applied DOM patch for {p_key}.{r_key}: {sel_list}")
    return True


def reset_patches() -> bool:
    """Clears all custom selector hot-patches."""
    path = get_patches_file()
    if os.path.exists(path):
        try:
            os.remove(path)
            log.info("Reset all DOM patches to factory defaults.")
            return True
        except Exception as e:
            log.warning(f"Failed to remove patches.json: {e}")
            return False
    return True
