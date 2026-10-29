# ============================================================
# DeepSeek Browser Automation Engine for WebMind
#
# Connects to a dedicated Chrome instance over CDP with its own
# isolated profile (`WebMindChrome_DeepSeek`), interacts with
# https://chat.deepseek.com/, handles anti-hang sub-second
# settlement, and extracts multi-block markdown responses.
# ============================================================

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from playwright.sync_api import sync_playwright
import browser_manager
import profile_manager
import process_supervisor
import browser_guard
import dom_patches

DEEPSEEK_URL = "https://chat.deepseek.com/"

PROFILE_DIR = profile_manager.get_profile_dir("deepseek")

CDP_HOST = "127.0.0.1"
CDP_PORT = int(os.environ.get("DEEPSEEK_CDP_PORT", "9223"))
CDP_ENDPOINT = f"http://{CDP_HOST}:{CDP_PORT}"

RESPONSE_WAIT_TIMEOUT = float(os.environ.get("DEEPSEEK_RESPONSE_WAIT_TIMEOUT", "60.0"))
SETTLEMENT_SECONDS = float(os.environ.get("DEEPSEEK_SETTLEMENT_SECONDS", "0.7"))


def _cdp_available() -> bool:
    return browser_manager.is_cdp_available(CDP_HOST, CDP_PORT)


def _find_chrome_executable() -> str:
    exe, _ = browser_manager.find_chromium_executable()
    return exe


def _launch_chrome():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    chrome_bin, bname = browser_manager.find_chromium_executable()
    args = browser_manager.get_chromium_launch_args(
        chrome_bin,
        CDP_PORT,
        PROFILE_DIR,
        DEEPSEEK_URL,
    )
    print(f"Launching dedicated {bname} for DeepSeek on port {CDP_PORT}...")
    proc = subprocess.Popen(args)
    process_supervisor.register_process(proc)


def _wait_for_cdp(timeout: int = 30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _cdp_available():
            return
        time.sleep(0.5)
    raise TimeoutError(f"Chrome did not expose CDP endpoint on {CDP_ENDPOINT} within {timeout}s.")



def _clean_extracted_text(text: str) -> str:
    if not text:
        return ""
    # Strip leading UI chrome buttons (json, Copy, Download, 复制, 下载) before {
    cleaned = re.sub(
        r"^(?:json|python|bash|javascript|text|sh|cmd|powershell)?\s*(?:copy|download|code|复制|下载|\s)*(?=\{)",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


class DeepSeekResponse:
    def __init__(self, text: str):
        self.text = text


class DeepSeekBrowser:
    def __init__(self):
        print("Starting DeepSeek Browser session...")
        self.playwright = sync_playwright().start()

        if not _cdp_available():
            _launch_chrome()
            _wait_for_cdp()

        print(f"Connecting to Chrome over CDP on {CDP_ENDPOINT}...")
        self.browser = self.playwright.chromium.connect_over_cdp(CDP_ENDPOINT)

        if not self.browser.contexts:
            raise RuntimeError("Chrome exposed no browser context on CDP.")

        self.context = self.browser.contexts[0]
        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = self.context.new_page()

        self._conversation_id = None

        if "chat.deepseek.com" not in self.page.url:
            self.page.goto(DEEPSEEK_URL, wait_until="domcontentloaded")
            self.page.wait_for_timeout(3000)

        self._check_login()

    def _check_login(self):
        body = self.page.locator("body").inner_text()
        if "Log in" in body or "Sign in" in body or "登录" in body:
            print("\n" + "=" * 60)
            print("DEEPSEEK LOGIN REQUIRED")
            print("=" * 60)
            print("Log into DeepSeek in the opened Chrome window.")
            print(f"Profile directory: {PROFILE_DIR}")
            input("Press ENTER after you have logged in...")
            self.page.wait_for_timeout(3000)

    def _is_send_button_visible(self) -> bool:
        if not hasattr(self, "page") or not hasattr(self.page, "locator"):
            return False
        for sel in dom_patches.get_active_selectors("deepseek").get("send", []):
            try:
                if self.page.locator(sel).count() > 0:
                    return True
            except Exception:
                pass
        return False

    def _is_generating(self) -> bool:
        if not hasattr(self, "page") or not hasattr(self.page, "locator"):
            return False
        for sel in dom_patches.get_active_selectors("deepseek").get("stop", []):
            try:
                if self.page.locator(sel).count() > 0:
                    return True
            except Exception:
                pass
        return False

    def _locate_assistant_messages(self):
        if not hasattr(self, "page") or not hasattr(self.page, "locator"):
            return None
        selectors = dom_patches.get_active_selectors("deepseek").get("messages", [
            '.ds-markdown',
            'div[class*="ds-markdown"]',
            '.markdown',
        ])
        for sel in selectors:
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0:
                    return loc
            except Exception:
                pass
        return None

    def _extract_node_text(self, node) -> str:
        if node is None:
            return ""
        try:
            code = node.locator("pre code")
            if code.count() == 0:
                code = node.locator("pre")
            c_count = code.count() if hasattr(code, "count") else 0
            if c_count > 1:
                blocks = []
                for i in range(c_count):
                    c_text = code.nth(i).text_content()
                    if c_text and c_text.strip():
                        blocks.append(f"```json\n{_clean_extracted_text(c_text)}\n```")
                if blocks:
                    return "\n\n".join(blocks)
            elif c_count == 1:
                text = code.first.text_content()
                if text is not None and text.strip():
                    return _clean_extracted_text(text)
            return _clean_extracted_text(node.text_content() or "")
        except Exception:
            try:
                return _clean_extracted_text(node.text_content() or "")
            except Exception:
                return ""

    def _get_assistant_message(self, since_count: int | None = None) -> str:
        """Extract latest assistant message or multi-bubble turn if since_count provided."""
        messages = self._locate_assistant_messages()
        if messages is None:
            return ""
        total = messages.count() if hasattr(messages, "count") else 0
        if total == 0:
            return ""
        if since_count is not None and total > since_count:
            parts = []
            for i in range(since_count, total):
                t = self._extract_node_text(messages.nth(i))
                if t:
                    parts.append(t)
            if parts:
                return "\n\n".join(parts)
        return self._extract_node_text(messages.last)

    def _send_prompt(self, prompt: str):
        sels = dom_patches.get_active_selectors("deepseek")
        input_sel = None
        target_el = None
        candidate_selectors = list(sels.get("input", ["textarea#chat-input", "textarea"]))
        for fallback in ["textarea", 'div[contenteditable="true"]', 'div[role="textbox"]', 'input[type="text"]']:
            if fallback not in candidate_selectors:
                candidate_selectors.append(fallback)

        for sel in candidate_selectors:
            try:
                loc = self.page.locator(sel)
                cnt = loc.count()
                if cnt > 0:
                    for idx in range(cnt):
                        el = loc.nth(idx)
                        if el.is_visible():
                            input_sel = sel
                            target_el = el
                            break
                    if input_sel:
                        break
            except Exception:
                pass

        if not input_sel:
            for sel in candidate_selectors:
                try:
                    loc = self.page.locator(sel)
                    if loc.count() > 0:
                        input_sel = sel
                        target_el = loc.first
                        break
                except Exception:
                    pass

        if not input_sel or target_el is None:
            raise RuntimeError("Could not find DeepSeek chat input box.")

        target_el.click()
        try:
            target_el.fill(prompt)
        except Exception:
            self.page.keyboard.insert_text(prompt)

        # Dispatch standard input/change events to ensure Vue/React state updates
        try:
            self.page.evaluate(
                """el => {
                    if (!el) return;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }""",
                target_el,
            )
        except Exception:
            pass

        time.sleep(0.3)

        # Click send or press Enter
        sent = False
        send_candidates = list(sels.get("send", ['button[aria-label*="Send"]:visible']))
        for fb_send in ['div[role="button"][aria-label*="Send"]:visible', 'button[type="submit"]:visible', 'button:has(svg):visible']:
            if fb_send not in send_candidates:
                send_candidates.append(fb_send)

        for btn_sel in send_candidates:
            try:
                btn = self.page.locator(btn_sel).last
                if btn.count() > 0 and btn.is_visible():
                    is_disabled = False
                    if hasattr(btn, "is_disabled"):
                        try:
                            is_disabled = btn.is_disabled()
                        except Exception:
                            is_disabled = False
                    if not is_disabled:
                        btn.click()
                        sent = True
                        break
            except Exception:
                pass

        if not sent:
            self.page.keyboard.press("Enter")

    def _wait_for_response(self, old_text: str, deadline: float, since_count: int | None = None) -> str:
        last_text = ""
        settled_start = None
        was_generating = False

        while time.time() < deadline:
            current = self._get_assistant_message(since_count=since_count)
            is_gen = self._is_generating()
            send_visible = self._is_send_button_visible()

            if is_gen:
                was_generating = True

            if current and current != old_text:
                if current == last_text:
                    if settled_start is None:
                        settled_start = time.time()
                    elapsed_settled = time.time() - settled_start
                    if (send_visible or not is_gen or elapsed_settled >= 6.0) and elapsed_settled >= SETTLEMENT_SECONDS:
                        return current
                else:
                    last_text = current
                    settled_start = None
            elif was_generating and not is_gen and send_visible and current:
                return current

            time.sleep(0.2)

        return last_text or self._get_assistant_message(since_count=since_count)

    def ask(self, prompt: str, conversation_id: str | None = None, deadline: float | None = None) -> DeepSeekResponse:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        messages = self._locate_assistant_messages()
        since_count = messages.count() if messages is not None and hasattr(messages, "count") else 0
        old_text = self._get_assistant_message(since_count=since_count)
        self._send_prompt(prompt)

        wall_deadline = (time.time() + RESPONSE_WAIT_TIMEOUT) if deadline is None else deadline
        response_text = self._wait_for_response(old_text, wall_deadline, since_count=since_count)
        return DeepSeekResponse(response_text)

    def ensure_healthy(self) -> bool:
        """Verifies session health; reconnects if the browser closed."""
        if not browser_guard.is_browser_healthy(getattr(self, "browser", None), getattr(self, "page", None)):
            return self.reconnect()
        return True

    def reconnect(self) -> bool:
        """Relaunches and re-attaches to DeepSeek Chrome session."""
        return browser_guard.relaunch_and_reconnect(
            self,
            "DeepSeek",
            CDP_PORT,
            PROFILE_DIR,
            DEEPSEEK_URL,
        )

    def close(self):
        try:
            self.playwright.stop()
        except Exception:
            pass
        process_supervisor.cleanup_all()


class DeepSeekWorker:
    def __init__(self):
        self._queue = queue.Queue()
        self._thread = None
        self._browser = None
        self._ready = threading.Event()
        self._init_error = None

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._init_error = None
        self._thread = threading.Thread(target=self._run, name="DeepSeekBrowserWorker", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=35)
        if self._init_error:
            raise self._init_error

    def _run(self):
        try:
            self._browser = DeepSeekBrowser()
        except Exception as e:
            self._init_error = e
        finally:
            self._ready.set()

        while True:
            item = self._queue.get()
            if item is None:
                if self._browser:
                    try:
                        self._browser.close()
                    except Exception:
                        pass
                self._queue.task_done()
                break

            action, args, kwargs, result_q = item
            try:
                if action == "ask":
                    res = browser_guard.with_recovery(
                        self._browser,
                        "DeepSeek",
                        CDP_PORT,
                        PROFILE_DIR,
                        DEEPSEEK_URL,
                        self._browser.ask,
                        *args,
                        **kwargs,
                    )
                    result_q.put((True, res))
                elif action == "close":
                    if self._browser:
                        self._browser.close()
                    result_q.put((True, None))
            except Exception as e:
                result_q.put((False, e))
            finally:
                self._queue.task_done()

    def ask(self, *args, **kwargs):
        self.start()
        result_q = queue.Queue()
        self._queue.put(("ask", args, kwargs, result_q))
        ok, val = result_q.get()
        if not ok:
            raise val
        return val

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5)
            self._thread = None
            self._browser = None


_worker = DeepSeekWorker()


def start():
    _worker.start()
    return _worker


def ask(prompt: str, conversation_id: str | None = None, deadline: float | None = None) -> DeepSeekResponse:
    return _worker.ask(prompt, conversation_id=conversation_id, deadline=deadline)


def stop():
    _worker.stop()


if __name__ == "__main__":
    try:
        start()
        while True:
            p = input("\nDeepSeek Prompt (or 'exit'): ")
            if p.strip().lower() == "exit":
                break
            resp = ask(p)
            print("\n" + "=" * 60)
            print(resp.text)
            print("=" * 60)
    finally:
        stop()
