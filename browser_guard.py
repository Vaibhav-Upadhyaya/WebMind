# ============================================================
# WebMind Browser Guard & Self-Healing Engine
#
# Monitors Chromium CDP session health, detects browser crashes
# or accidental window closures, and automatically recovers
# sessions with zero data loss so coding agent requests never fail.
# ============================================================

import subprocess
import time
from typing import Any, Callable, Optional

import browser_manager
import config_manager
import logger
import process_supervisor

log = logger.get_logger("browser_guard")

DISCONNECT_PATTERNS = [
    "target page, context or browser has been closed",
    "connection closed",
    "target closed",
    "browser has been closed",
    "protocol error",
    "cannot find context with specified id",
    "navigation failed because page was closed",
    "session closed",
]


def is_connection_error(exc: Exception) -> bool:
    """Detects if an exception originates from a closed browser, tab, or CDP session."""
    if exc is None:
        return False
    msg = str(exc).lower()
    err_type = type(exc).__name__.lower()

    if "targetclosed" in err_type or "connectionclosed" in err_type:
        return True

    return any(p in msg for p in DISCONNECT_PATTERNS)


def is_browser_healthy(browser: Any, page: Any, timeout_ms: int = 1500) -> bool:
    """Verifies that the Playwright browser and page are open and responsive."""
    if not browser:
        return False
    try:
        if hasattr(browser, "is_connected") and not browser.is_connected():
            return False
    except Exception:
        return False

    if not page:
        return False
    try:
        if hasattr(page, "is_closed") and page.is_closed():
            return False
        # Fast evaluation ping to check if Chromium is responding
        res = page.evaluate("() => 1 + 1")
        return res == 2
    except Exception:
        return False


def relaunch_and_reconnect(
    browser_instance: Any,
    ai_name: str,
    cdp_port: int,
    profile_dir: str,
    target_url: str,
    timeout_sec: int = 25,
) -> bool:
    """
    Relaunches the dedicated browser if dead, waits for CDP, and reconnects
    the Playwright browser and page context.
    """
    log.warning(f"Self-healing triggered for {ai_name}. Reconnecting browser session on port {cdp_port}...")

    # 1. Close existing dead browser handles gracefully
    try:
        if hasattr(browser_instance, "browser") and browser_instance.browser:
            browser_instance.browser.close()
    except Exception:
        pass

    # 2. Check if CDP port is open; if not, relaunch Chromium
    if not browser_manager.is_cdp_available("127.0.0.1", cdp_port):
        log.info(f"CDP port {cdp_port} unavailable. Launching fresh Chromium instance for {ai_name}...")
        chrome_bin, bname = browser_manager.find_chromium_executable()
        args = browser_manager.get_chromium_launch_args(
            chrome_bin,
            cdp_port,
            profile_dir,
            target_url,
        )
        proc = subprocess.Popen(args)
        process_supervisor.register_process(proc)

    # 3. Wait for CDP endpoint to become ready
    deadline = time.time() + timeout_sec
    cdp_ready = False
    while time.time() < deadline:
        if browser_manager.is_cdp_available("127.0.0.1", cdp_port):
            cdp_ready = True
            break
        time.sleep(0.3)

    if not cdp_ready:
        log.error(f"CDP port {cdp_port} failed to respond within {timeout_sec}s.")
        return False

    # 4. Re-attach Playwright over CDP
    try:
        endpoint = f"http://127.0.0.1:{cdp_port}"
        browser_instance.browser = browser_instance.playwright.chromium.connect_over_cdp(endpoint)

        if not browser_instance.browser.contexts:
            log.error("No browser context exposed on CDP.")
            return False

        browser_instance.context = browser_instance.browser.contexts[0]
        if browser_instance.context.pages:
            browser_instance.page = browser_instance.context.pages[0]
        else:
            browser_instance.page = browser_instance.context.new_page()

        # Re-navigate if blank or not on the target domain
        current_url = browser_instance.page.url or ""
        if "about:blank" in current_url or target_url.split("//")[-1].split("/")[0] not in current_url:
            browser_instance.page.goto(target_url, wait_until="domcontentloaded")
            browser_instance.page.wait_for_timeout(2000)

        log.info(f"Successfully re-attached and healed {ai_name} session.")
        return True
    except Exception as e:
        log.error(f"Failed to reconnect Playwright for {ai_name}: {e}")
        return False


def with_recovery(
    browser_instance: Any,
    ai_name: str,
    cdp_port: int,
    profile_dir: str,
    target_url: str,
    action_fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """
    Executes an action with automatic failure recovery. If the browser closed
    or the CDP connection drops, it heals the browser and retries the action once.
    """
    cfg = config_manager.load_config()
    auto_recover = cfg.get("auto_recover_browser", True)

    try:
        # Pre-check health if auto_recover is enabled
        if auto_recover:
            page = getattr(browser_instance, "page", None)
            browser = getattr(browser_instance, "browser", None)
            if not is_browser_healthy(browser, page):
                relaunch_and_reconnect(
                    browser_instance,
                    ai_name,
                    cdp_port,
                    profile_dir,
                    target_url,
                )

        return action_fn(*args, **kwargs)
    except Exception as exc:
        if auto_recover and is_connection_error(exc):
            log.warning(f"Connection dropped during {ai_name} operation ({exc}). Attempting auto-recovery...")
            healed = relaunch_and_reconnect(
                browser_instance,
                ai_name,
                cdp_port,
                profile_dir,
                target_url,
            )
            if healed:
                log.info(f"Retrying operation after successful recovery for {ai_name}...")
                return action_fn(*args, **kwargs)
        raise
