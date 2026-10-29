# pyrefly: ignore [missing-import]
from playwright.sync_api import sync_playwright
import hashlib
import os
import re
import uuid
import shutil
import subprocess
import sys
import time
import urllib.request
import profile_manager
import browser_manager
import process_supervisor
import browser_guard


class ChatGPTInputError(RuntimeError):
    """Raised when the prompt could NOT be submitted to ChatGPT.

    This is distinct from browser/response failures: it means the
    composer was never successfully sent, so no ChatGPT message was
    created.  The server's idempotency guard treats these as
    retryable (the next identical request may re-attempt safely).
    """


class PromptTooLargeError(RuntimeError):
    """Raised when the prompt would land as a pasted-text ATTACHMENT.

    A prompt pasted into ChatGPT's composer is only treated as the live
    conversation while it stays under ChatGPT's inline limit.  Past that
    limit the composer silently converts the paste into an uploaded
    "Pasted text" document, and the model then answers as an outside
    reader of a FILE rather than as the agent continuing its own loop --
    e.g. reporting that "the material available in this turn contains the
    transcript, not the repository".

    That failure is silent: the submission is reported as successful and a
    plausible-but-wrong answer comes back.  This error exists to convert it
    into an honest, client-visible failure instead.

    `kind` separates the two ways a prompt can be rejected, because they
    call for different handling upstream:

    * "over_ceiling" -- the pre-flight measured the prompt above
      MAX_INLINE_PROMPT_CHARS and refused BEFORE pasting.  Nothing reached
      ChatGPT, and a smaller request is the only fix.
    * "attachment" -- the prompt was within budget but ChatGPT still
      converted the paste to a file.  The HTTP layer can meaningfully
      re-serialize the same conversation more compactly and try again,
      because the content that triggered it may simply be smaller.

    So "attachment" is worth one smaller retry; "over_ceiling" is not, and
    an identical retry of either is pointless: it re-pastes an identical
    prompt and gets the same verdict.
    """

    def __init__(self, message: str, kind: str = "over_ceiling"):
        super().__init__(message)
        self.kind = kind

    @property
    def is_attachment(self) -> bool:
        return self.kind == "attachment"


import json as _json
from datetime import datetime as _dt, timezone as _tz


def _log_event(event: str, **kw) -> None:
    """Structured lifecycle log.  One JSON object per line to stdout AND to
    the runtime log file, so a server started as a background process can
    be diagnosed after the fact."""
    record = {"ts": _dt.now(_tz.utc).isoformat(), "event": event, **kw}
    line = _json.dumps(record, default=str)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(RUNTIME_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================
# SAFE CONSOLE ENCODING
# ============================================================
# When stdout/stderr are redirected to a file (e.g. Start-Process
# with -RedirectStandardOutput), Python falls back to the Windows
# ANSI code page (cp1252), which cannot encode emoji or non-Latin
# text. Any print() of such text then raises UnicodeEncodeError
# and turns into a 500 "WebMind error: 'charmap' codec...".
# Force UTF-8 and never let a log string crash the request.

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

CHATGPT_URL = "https://chatgpt.com/"

# Bounded persistent-chat growth (Phase 3, survival #7).
# ChatGPT keeps ONE live conversation in the browser until the user
# starts a fresh one, so an unbounded number of WebMind turns makes the
# accumulated scrollback grow forever.  That matters on the continuation
# path: build_prompt re-serialises the whole history into every next
# prompt, and a large enough prompt can be turned into a pasted
# ATTACHMENT instead of inline text, which is exactly the failure mode
# that has been chasing us.  (This used to claim a ~43K char threshold;
# that figure was never measured, and measurement showed the conversion
# is content-dependent rather than length-dependent -- see
# MAX_INLINE_PROMPT_CHARS below.)
#
# So, before a NEW (non-pinned) turn is dispatched, the live chat is
# rotated to a brand-new ChatGPT chat once it has accumulated more than
# MAX_TURNS_BEFORE_NEW_CHAT assistant turns.  Each new prompt still
# carries the ENTIRE conversation (build_prompt re-serialises history
# after rotation), so continuity is preserved at the nearest-fresh-chat
# granularity the product allows while prompt size stays comfortably
# bounded below the inline threshold forever.
#
# MUST be failure-safe: clearing/rotating only ever navigates to a new
# chat URL and NEVER pastes anything, so a rotation that fails leaves
# the prompt flow untouched.  MUST NOT run when the caller pinned a
# chat_url (that is an explicit "continue THIS conversation" signal) or
# when a stream is in flight (we never rotate mid-answer).
MAX_TURNS_BEFORE_NEW_CHAT = int(
    os.environ.get("WEBMIND_MAX_TURNS_BEFORE_NEW_CHAT", "24")
)

# ---------------------------------------------------------
# THE INLINE LIMIT -- an observed-failure backstop, NOT a
# measured character threshold
# ---------------------------------------------------------
# There is no character budget that reliably prevents the
# pasted-text ATTACHMENT conversion.  Verified live submissions
# on this account (mode read from the browser.submitted_content
# log event):
#
#      4600  INLINE       6587  INLINE
#     12042  ATTACHMENT  14124  ATTACHMENT  14285  ATTACHMENT
#
# The 12042-char ATTACHMENT is the row that matters: it sits well
# BELOW the ceiling configured below, so passing the pre-flight
# check does NOT mean the paste will stay inline.  That is why
# sizing cannot be the fix.
#
# (An earlier round of synthetic filler probes appeared to show a
# clean ~17K cutoff.  Those probes built large prompts but never
# submitted them, so the conclusion was wrong and nothing here
# relies on it.)
#
# Two things are done instead:
#
#   1. DETECTION (the actual fix).  _send_prompt returns the
#      submission mode, and an ATTACHMENT submission is refused
#      rather than answered, turning a silent wrong answer into a
#      clear, client-visible error.  The caller (the server) then
#      retries once with a materially shorter prompt, which is
#      what actually recovers the answer.
#
#   2. THIS BACKSTOP.  Refuse absurdly large prompts before the
#      paste so a runaway request cannot wedge the composer.  It
#      is deliberately generous: trimming down to where conversion
#      stops appearing would cost most of the conversation to fix
#      a rule that does not exist.
#
# Lowering it costs context without preventing conversion.
# Raising it risks opaque submit failures.
MAX_INLINE_PROMPT_CHARS = int(
    os.environ.get("WEBMIND_MAX_INLINE_PROMPT_CHARS", "1000000")
)

ALLOW_ATTACHMENT_PROMPT = (
    os.environ.get("WEBMIND_ALLOW_ATTACHMENT", "true").strip().lower()
    in ("true", "1", "yes")
)

ATTACHMENT_COMPANION_PROMPT = os.environ.get(
    "WEBMIND_ATTACHMENT_COMPANION_PROMPT",
    "You are WebMind, the live autonomous coding agent running directly in this workspace. "
    "The attached file contains your full context, tools, and the raw workspace files you requested. "
    "You have DIRECT live filesystem access via tools (read, grep, glob, edit, write, bash). "
    "Do NOT treat this attachment as a passive transcript or third-party document. "
    "Read the raw file contents from the attachment, decide the exact code changes needed, and emit your next action "
    "in ```json using the mandatory envelope (tool_call or message) to apply changes to the live files."
)

# Fraction of MAX_INLINE_PROMPT_CHARS at which a prompt starts emitting a
# browser.prompt_budget_exceeded warning.  Deliberately well under 1.0 so
# the warning arrives while there is still room to react.
PROMPT_BUDGET_WARN_RATIO = 0.8

# Dedicated Chrome profile (a real, persistent Chrome profile).
# WebMind never loads the user's normal Chrome profile and never
# touches authentication data manually; Chrome owns the session.
PROFILE_DIR = profile_manager.get_profile_dir("chatgpt")

# Remote debugging endpoint used to attach Playwright via CDP.
CDP_HOST = "127.0.0.1"
CDP_PORT = 9222
CDP_ENDPOINT = f"http://{CDP_HOST}:{CDP_PORT}"

# How long to wait for the dedicated Chrome to expose its CDP
# endpoint after launch.
CHROME_START_TIMEOUT = 30


# ============================================================
# PROMPT INSERTION STATE MACHINE
# ============================================================
# Large clipboard pastes are handled by ChatGPT asynchronously: it creates
# a "Pasted text(...).txt" document and the contenteditable can stay
# EMPTY while that document is processing.  A verification wait that is
# too short, followed by a blind clear + re-paste, corrupts that in-flight
# operation and duplicates/mangles the prompt.  The insertion supervisor
# therefore classifies every observed composer read:
#
#     INSERTING / PROCESSING  keep waiting (empty, or still growing)
#     VERIFIED_EXACT          normalized exact match seen 2x in a row
#     VERIFIED_ATTACHMENT_READY  a pasted-text attachment exists, is
#                            sendable (aria-disabled="false") and its tile
#                            label associates with the expected text.  This
#                            is NOT content equality; it is a separate,
#                            weaker verification state (invariant 7).
#     FAILED                  stable non-exact content (insertion done)
#     UNKNOWN                 deadline hit while still potentially processing
#
# Only a FAILED outcome (previous insertion no longer processing) may
# clear and retry.  UNKNOWN/PROCESSING never clears the composer.
# VERIFIED_ATTACHMENT_READY is sendable but never used to justify a
# clear or a second insertion.

VERIFY_STABLE_READS = 2              # consecutive exact reads required
VERIFY_POLL_INTERVAL = 0.25          # seconds between composer reads
PROMPT_INSERT_MAX_ATTEMPTS = 2       # bounded clear+retry after FAILED only

# Insertion deadline: base + a per-character allowance, capped.  The loop
# exits the moment content settles (exact or stable wrong), so this only
# bites while the pasted-text document is genuinely still processing; it
# must not be made enormous.
INSERTION_DEADLINE_BASE = 30.0       # seconds
INSERTION_DEADLINE_PER_CHAR = 0.0015 # +1.5s per 1000 chars
INSERTION_DEADLINE_CAP = 120.0       # absolute per-dispatch cap
INSERTION_DEADLINE_GRACE = 20.0      # extra budget shared across attempts

# How long to wait, while ChatGPT is NOT generating, for a new assistant
# message node before treating the turn as lost.  The stop-button and
# thinking indicators exempt active generation from this cap; this timeout
# only applies while ChatGPT is completely idle with no response started.
RESPONSE_WAIT_TIMEOUT = float(
    os.environ.get("WEBMIND_RESPONSE_WAIT_TIMEOUT", "45.0")
)        # seconds, idle with no new message

# Every lifecycle event is appended here (in addition to stdout) so a
# server started as a background process can be diagnosed after the fact.
RUNTIME_LOG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "webmind_runtime.log",
)


_INVISIBLE_CHARS_RE = re.compile(
    r"[\u2000-\u200f\u2028-\u202f\u205f-\u206f\ufeff\u00ad\u034f\u180e\u3000\xa0\u200b-\u200d\u2060\ufeff]+"
)


def _clean_composer_text(text):
    """Strip zero-width characters and normalize quotes and non-breaking spaces."""
    if not text:
        return ""
    s = str(text)
    s = _INVISIBLE_CHARS_RE.sub(" ", s)
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return s


def _normalize_prompt(text):
    """Collapse all whitespace and strip invisible unicode characters for prompt comparison."""
    if text is None:
        return ""
    cleaned = _clean_composer_text(text)
    return " ".join(cleaned.split())


def _is_prompt_match(actual, expected):
    """Check if composer text matches expected prompt, with ProseMirror artifact tolerance.

    1. Exact equality of normalized text.
    2. Minor ProseMirror DOM artifact tolerance: if expected is large (>50 chars),
       length differs by at most max(10, 5% of length), and prefix or suffix matches >=95%.
    """
    clean_actual = _normalize_prompt(actual)
    clean_expected = _normalize_prompt(expected)
    if clean_actual == clean_expected:
        return True
    if not clean_actual or not clean_expected:
        return False
    diff_len = abs(len(clean_actual) - len(clean_expected))
    allowed_diff = max(10, int(len(clean_expected) * 0.05))
    if len(clean_expected) > 50 and diff_len <= allowed_diff:
        check_len = min(len(clean_actual), len(clean_expected)) - diff_len
        if check_len > 40:
            if clean_actual[:check_len] == clean_expected[:check_len]:
                return True
            if clean_actual[-check_len:] == clean_expected[-check_len:]:
                return True
    return False


def _content_hash(text):
    """Short stable hash of the normalized text for insertion logs."""
    return hashlib.sha256(
        _normalize_prompt(text).encode("utf-8", "replace")
    ).hexdigest()[:16]


# ============================================================
# CDP CHROME LAUNCH HELPERS
# ============================================================
# WebMind connects to an already-running Chrome over CDP instead of
# launching an automated browser context. If the dedicated Chrome
# (profile + port) is not running, WebMind starts it once.

def _find_chrome_executable():
    """Locate an installed Chrome/Chromium executable (Windows first)."""
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return shutil.which("chrome") or shutil.which("google-chrome")


def _cdp_available(timeout=2.0):
    """True if a Chrome CDP endpoint is already reachable on the port."""
    try:
        with urllib.request.urlopen(
            f"{CDP_ENDPOINT}/json/version",
            timeout=timeout,
        ):
            return True
    except Exception:
        return False


def _launch_chrome():
    """Start the dedicated Chrome with remote debugging and its own profile."""

    executable = _find_chrome_executable()

    if not executable:
        raise RuntimeError(
            "Could not find a Chrome executable. Install Google Chrome "
            "to run WebMind, or start Chrome manually with "
            f"--remote-debugging-port={CDP_PORT} --user-data-dir={PROFILE_DIR}"
        )

    print(f"Launching dedicated Chrome ({executable})...")

    flags = [
        executable,
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={PROFILE_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        CHATGPT_URL,
    ]

    creationflags = 0
    for flag_name in ("CREATE_NEW_PROCESS_GROUP", "DETACHED_PROCESS"):
        creationflags |= getattr(subprocess, flag_name, 0)

    proc = subprocess.Popen(
        flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    process_supervisor.register_process(proc)


def _wait_for_cdp(timeout=CHROME_START_TIMEOUT):
    """Poll until the CDP endpoint answers or the timeout expires."""
    start = time.time()

    while time.time() - start < timeout:

        if _cdp_available(timeout=1.0):
            return

        time.sleep(0.5)

    raise RuntimeError(
        f"Chrome did not expose its debugging endpoint on port {CDP_PORT} "
        f"within {timeout}s. Another Chrome may already be using "
        f"{PROFILE_DIR} without remote debugging enabled."
    )


# ============================================================
# TRANSIENT STATUS TEXT
# ============================================================
# ChatGPT shows short working-status placeholders (e.g. while it
# analyzes an attached image) that sit unchanged for seconds.
# They must never count as a finished response and must never
# be streamed as answer content. Add patterns here if the UI
# gains new status wording.

_TRANSIENT_STATUS_RE = re.compile(
    r"^\s*analyzing\s+image\s*[.\u2026]*\s*$",
    re.IGNORECASE
)

# Max time to tolerate a placeholder before giving up (seconds).
# The surrounding loops still enforce their own overall timeouts.
TRANSIENT_STATUS_TIMEOUT = 120


def _is_transient_status(text):
    """True if text is only a working-status placeholder."""
    return isinstance(text, str) and bool(
        _TRANSIENT_STATUS_RE.match(text)
    )


# Streaming cursor / zero-width chars ChatGPT web shows while
# generating (e.g. trailing "▍"). They appear/disappear between
# polls and must not break prefix matching.
_STREAM_CURSOR_CHARS = "▍▌▎▏│┃●◉○\u200b\u200c\u200d\ufeff"


def _clean_stream_text(text):
    """Strip streaming artifacts for stable delta computation."""
    if not isinstance(text, str):
        return ""
    return text.replace("\u200b", "").rstrip(_STREAM_CURSOR_CHARS).strip()


def _compute_stream_delta(sent, current):
    """Return only the newly-added suffix of current vs sent.

    Append-only SSE contract: never re-emit what was sent.
    Returns "" when there is nothing new (shrink/re-render
    glitch) instead of resending the full snapshot.
    """
    if not current:
        return ""
    if not sent:
        return current
    if current == sent:
        return ""
    if current.startswith(sent):
        return current[len(sent):]
    if sent.startswith(current):
        return ""
    common = 0
    limit = min(len(sent), len(current))
    while common < limit and sent[common] == current[common]:
        common += 1
    if common > 0:
        return current[common:]
    return ""


# ============================================================
# RESPONSE OBJECT
# ============================================================

class Response:

    def __init__(self, text, mode="INLINE"):
        self.text = text.strip()
        # "INLINE" = ChatGPT received the conversation as live text.
        # "ATTACHMENT" = it received an uploaded document instead.  Callers
        # must not trust the text of an ATTACHMENT response.
        self.mode = mode

    @property
    def markdown(self):
        return self.text

    @property
    def code(self):

        parts = self.text.split("```")
        blocks = []

        languages = {
            "python",
            "javascript",
            "typescript",
            "java",
            "c",
            "cpp",
            "c++",
            "c#",
            "html",
            "css",
            "json",
            "bash",
            "shell",
            "powershell",
            "sql",
            "rust",
            "go",
            "php",
            "ruby",
            "kotlin",
            "swift",
        }

        for i in range(1, len(parts), 2):

            block = parts[i].strip()
            lines = block.splitlines()

            if lines:
                language = lines[0].strip().lower()

                if language in languages:
                    block = "\n".join(lines[1:])

            blocks.append(block.strip())

        return "\n\n".join(blocks)


# ============================================================
# CHATGPT BROWSER
# ============================================================

class ChatGPTBrowser:

    def __init__(self):

        print("Starting dedicated ChatGPT browser...")

        self.playwright = sync_playwright().start()

        # Test-injectable clock / sleep and insertion-deadline tuning.
        self._clock = time.time
        self._poll_sleep_fn = time.sleep
        self._insert_deadline_base = INSERTION_DEADLINE_BASE
        self._insert_deadline_per_char = INSERTION_DEADLINE_PER_CHAR
        self._insert_deadline_cap = INSERTION_DEADLINE_CAP
        self.allow_attachment = ALLOW_ATTACHMENT_PROMPT

        # ----------------------------------------------------
        # Ensure the dedicated CDP Chrome is running, then
        # attach to it. We never launch an automated browser
        # context: Chrome (and its login) is a normal instance.
        # ----------------------------------------------------

        if not _cdp_available():
            print(
                f"No Chrome detected on {CDP_ENDPOINT}; "
                "starting the dedicated WebMind Chrome..."
            )
            _launch_chrome()
            _wait_for_cdp()

        print("Connecting to Chrome over CDP...")

        self.browser = self.playwright.chromium.connect_over_cdp(
            CDP_ENDPOINT
        )

        # Reuse the persistent context/pages Chrome already owns so the
        # authenticated session survives across WebMind restarts. We do
        # NOT create a fresh automated context.
        if not self.browser.contexts:
            raise RuntimeError(
                "Chrome exposed no browser context on the CDP endpoint."
            )

        self.context = self.browser.contexts[0]

        # Best-effort: mirror the old persistent-context permissions.
        try:
            self.context.grant_permissions(
                ["clipboard-read", "clipboard-write"]
            )
        except Exception:
            pass

        if self.context.pages:
            self.page = self.context.pages[0]
        else:
            self.page = self.context.new_page()

        # Identity of the client conversation the live ChatGPT chat is
        # currently serving.  None until the first identified request.
        self._conversation_id = None
        self._last_assistant_snapshot = None

        self.page.goto(
            CHATGPT_URL,
            wait_until="domcontentloaded"
        )

        self.page.wait_for_timeout(4000)

        self._check_login()


    # ========================================================
    # LOGIN
    # ========================================================

    def _check_login(self):

        _log_event("browser.login_check", page_id=id(self.page))

        body = self.page.locator("body").inner_text()

        if "Log in to get answers" in body:

            print()
            print("=" * 60)
            print("LOGIN REQUIRED")
            print("=" * 60)
            print()
            print("Log into ChatGPT in the opened Chrome window.")
            print("Your login will be stored in:")
            print(PROFILE_DIR)
            print()

            input("Press ENTER after you have logged in...")

            self.page.wait_for_timeout(3000)

        print("Login detected.")


    # ========================================================
    # FIND PROMPT BOX
    # ========================================================

    def _get_input(self):

        selectors = [
            "#prompt-textarea",
            'div[contenteditable="true"]',
            '[contenteditable="true"]',
            "textarea",
            '[role="textbox"]'
        ]

        for selector in selectors:

            try:

                elements = self.page.locator(selector)

                for i in range(elements.count()):

                    element = elements.nth(i)

                    if element.is_visible():

                        print(
                            f"Input found using: {selector}"
                        )

                        return element

            except Exception:
                continue

        raise ChatGPTInputError(
            "Could not find the ChatGPT prompt box."
        )

    def _locate_assistant_messages(self):
        """Locate assistant messages across modern and legacy ChatGPT Web DOM."""
        if not hasattr(self, "page") or not hasattr(self.page, "locator"):
            return None
        for sel in (
            '[data-markdown-text-style="assistant-message"]',
            '[data-chatgpt-search-unit-key*="assistant"]',
            '[data-content-search-unit-key*="assistant"]',
            'div[class*="MarkdownRoot"]',
            '[data-conversation-role="assistant"]',
            '[data-message-author-role="assistant"]',
            '.markdown',
        ):
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0:
                    if hasattr(loc, "last") and getattr(loc, "last", None) is None:
                        continue
                    return loc
            except Exception:
                pass
        return self.page.locator('[data-markdown-text-style="assistant-message"]')

    def _locate_user_messages(self):
        """Locate user messages across modern and legacy ChatGPT Web DOM."""
        if not hasattr(self, "page") or not hasattr(self.page, "locator"):
            return None
        for sel in (
            '[data-user-message-bubble="true"]',
            '[data-chatgpt-search-unit-key*="user"]',
            '[data-content-search-unit-key*="user"]',
            'div[class*="user-message"]',
            '[data-conversation-role="user"]',
            '[data-message-author-role="user"]',
        ):
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0:
                    if hasattr(loc, "last") and getattr(loc, "last", None) is None:
                        continue
                    return loc
            except Exception:
                pass
        return self.page.locator('[data-user-message-bubble="true"]')


    # ========================================================
    # ATTACH IMAGES (file upload, worker thread only)
    # ========================================================

    def _attach_images(self, image_paths):

        import os

        if not image_paths:
            return

        for path in image_paths:
            if not path or not os.path.exists(path):
                raise RuntimeError(
                    f"ChatGPT attachment failure: image file not found: {path}"
                )

        print(f"Attaching {len(image_paths)} image(s)...")

        # ----------------------------------------------------
        # Strategy 1: direct file input (no click needed).
        # ChatGPT keeps a hidden <input type="file"> in the
        # composer. Playwright can set files on it directly.
        # ----------------------------------------------------

        try:
            file_inputs = self.page.locator('input[type="file"]')
            count = file_inputs.count()

            for i in range(count):
                try:
                    file_inputs.nth(i).set_input_files(image_paths, timeout=10000)
                    print(f"Files set via input[type=file] #{i}.")

                    # Give ChatGPT time to render the preview chip.
                    self.page.wait_for_timeout(3000)
                    return

                except Exception as e:
                    print(f"File input #{i} failed: {e}")
                    continue

        except Exception as e:
            print(f"Direct file-input attach failed: {e}")

        # ----------------------------------------------------
        # Strategy 2: click attach/upload button, catch the
        # native file chooser Playwright opens.
        # ----------------------------------------------------

        attach_selectors = [
            'button[aria-label*="Attach" i]',
            'button[aria-label*="Upload" i]',
            'button[aria-label*="Add file" i]',
            'button[aria-label*="Add photo" i]',
            '[data-testid*="attach" i]',
            '[data-testid*="upload" i]',
            '[data-testid="composer-plus-btn"]',
        ]

        for selector in attach_selectors:
            try:
                buttons = self.page.locator(selector)

                for i in range(buttons.count()):
                    button = buttons.nth(i)

                    if not button.is_visible():
                        continue

                    try:
                        with self.page.expect_file_chooser(timeout=5000) as chooser:
                            button.click()

                        # A menu ("Upload from computer", ...) may
                        # appear after clicking "+". Look for it.
                        try:
                            menu_item = self.page.locator(
                                'text=/Upload from computer|Upload.*file|Attach.*file/i'
                            ).first

                            if menu_item.count() > 0 and menu_item.is_visible():
                                with self.page.expect_file_chooser(timeout=5000) as chooser2:
                                    menu_item.click()
                                chooser2.value.set_files(image_paths, timeout=10000)
                            else:
                                chooser2.value.set_files(image_paths, timeout=10000)

                        except Exception:
                            # If menu handling fails, files were
                            # already set via the first chooser.
                            pass

                        print(f"Files set via attach button: {selector}.")
                        self.page.wait_for_timeout(3000)
                        return

                    except Exception as e:
                        print(f"Attach button {selector} #{i} failed: {e}")
                        continue

            except Exception:
                continue

        raise RuntimeError(
            "ChatGPT attachment failure: could not locate file input or attach button. "
            "ChatGPT UI may have changed."
        )


    # ========================================================
    # SEND PROMPT
    # ========================================================

    def _send_prompt(self, prompt):
        """Insert and submit the prompt.  Returns "INLINE" or "ATTACHMENT".

        The mode is the single most important fact about a submission: an
        ATTACHMENT means ChatGPT received a FILE, not the conversation, and
        will answer as an outside reader of that file.  It is returned so
        the caller can refuse the result instead of forwarding a confident
        wrong answer to the client.
        """
        # Per-send correlation id for the insertion/submission lifecycle
        # logs.  (The FastAPI-side request_id lives across the worker
        # boundary and is not visible here).
        request_id = uuid.uuid4().hex

        _log_event(
            "browser.prompt_insertion_started",
            request_id=request_id,
            prompt_len=len(prompt or ""),
        )

        # Pre-flight refusal for runaway prompts.  Refusing here is
        # deterministic and costs no paste, no probe loop and no ChatGPT
        # generation.  It is NOT a conversion guard: prompts well under this
        # ceiling have been converted too, which is what the post-submit
        # check below is for.
        if MAX_INLINE_PROMPT_CHARS > 0 and len(prompt or "") > MAX_INLINE_PROMPT_CHARS:
            _log_event(
                "browser.prompt_too_large",
                request_id=request_id,
                prompt_len=len(prompt or ""),
                max_inline=MAX_INLINE_PROMPT_CHARS,
                reason="prompt exceeds the configured backstop; refusing "
                       "before paste so it cannot be submitted as an attachment",
            )
            raise PromptTooLargeError(
                f"PROMPT_TOO_LARGE: prompt is {len(prompt or '')} chars, over the "
                f"{MAX_INLINE_PROMPT_CHARS} char backstop. This is a runaway-prompt "
                f"guard, not the point where ChatGPT would start converting the "
                f"paste into a file: that conversion is content-dependent and has "
                f"been seen well below this size, so prompts that fit here can still "
                f"be caught by post-submit detection. Shorten the conversation or "
                f"the tool output, or raise WEBMIND_MAX_INLINE_PROMPT_CHARS if this "
                f"account allows more.",
                kind="over_ceiling",
            )

        # Early warning well below the hard ceiling.  The attachment
        # conversion is content-dependent and unpredictable, so this is a
        # heads-up for diagnosis, not a guarantee: if a submission later
        # comes back as an attachment, the log line below is the clue.
        if (
            MAX_INLINE_PROMPT_CHARS > 0
            and len(prompt or "") > PROMPT_BUDGET_WARN_RATIO * MAX_INLINE_PROMPT_CHARS
        ):
            _log_event(
                "browser.prompt_budget_exceeded",
                request_id=request_id,
                prompt_len=len(prompt or ""),
                max_inline=MAX_INLINE_PROMPT_CHARS,
                ratio=round(len(prompt or "") / MAX_INLINE_PROMPT_CHARS, 3),
                reason="prompt is close to the inline ceiling; the composer may "
                       "convert it to a pasted-text attachment",
            )

        box = self._get_input()

        expected = _normalize_prompt(prompt)
        if not expected:
            raise ChatGPTInputError(
                "ChatGPT input: refusing to send an empty prompt."
            )

        # The composer may already hold exactly the target text (e.g. a
        # previous attempt verified its insertion but its submit failed,
        # or an identical retry landed asynchronously).  Then there is
        # nothing to insert: go straight to the pre-submit gate.
        try:
            existing = _normalize_prompt(self._read_composer(box))
        except Exception:
            existing = ""

        if _is_prompt_match(existing, expected):
            self._submit_prompt(box, prompt, request_id, attempt=0)
            _log_event(
                "browser.prompt_insertion_completed",
                request_id=request_id,
                prompt_len=len(prompt or ""),
                mode="INLINE",
            )
            return "INLINE"

        # Clear a stale draft ONLY before dispatching our own insertion.
        # An EMPTY composer is left alone: it may still be an in-flight
        # ChatGPT pasted-text operation, and clearing it could break that
        # (invariants 3, 8).
        if existing:
            self._clear_prompt(box)

        hard_deadline = (
            self._now()
            + self._insertion_deadline(len(expected))
            + INSERTION_DEADLINE_GRACE
        )

        attempt = 1
        while True:
            attempt_deadline = min(
                self._now() + self._insertion_deadline(len(expected)),
                hard_deadline,
            )

            # Snapshot the attachment UI BEFORE this dispatch so a
            # newly-created pasted-text document is distinguishable from a
            # stale tile left over from a previous send.
            tiles_before = self._snapshot_attachment_tiles()

            state, observed = self._insert_and_verify(
                box, prompt, expected, request_id, attempt, attempt_deadline, tiles_before
            )

            if state == "VERIFIED_EXACT":
                break

            if state == "VERIFIED_ATTACHMENT_READY":
                # observed is the attachment info dict from the verifier.
                attachment_ctx = {
                    "tiles_before": tiles_before,
                    "label": (observed or {}).get("attachment_label") or "",
                }
                self._submit_prompt(
                    box, prompt, request_id, attempt=attempt, mode="ATTACHMENT",
                    attachment_ctx=attachment_ctx,
                )
                _log_event(
                    "browser.prompt_insertion_completed",
                    request_id=request_id,
                    prompt_len=len(prompt or ""),
                    mode="ATTACHMENT",
                )
                return "ATTACHMENT"

            if state == "UNKNOWN":
                # Final safety recovery: check if attachment is ready or inline text matches
                attach_check, ainfo_check = self._get_pasted_text_attachment_state(
                    expected, tiles_before
                )
                if attach_check == "READY" or (
                    ainfo_check.get("new_tile_detected")
                    and ainfo_check.get("send_aria_disabled") == "false"
                ):
                    attachment_ctx = {
                        "tiles_before": tiles_before,
                        "label": ainfo_check.get("attachment_label") or "",
                    }
                    self._submit_prompt(
                        box, prompt, request_id, attempt=attempt, mode="ATTACHMENT",
                        attachment_ctx=attachment_ctx,
                    )
                    _log_event(
                        "browser.prompt_insertion_completed",
                        request_id=request_id,
                        prompt_len=len(prompt or ""),
                        mode="ATTACHMENT",
                    )
                    return "ATTACHMENT"

                try:
                    cur_text = _normalize_prompt(self._read_composer(box))
                    if _is_prompt_match(cur_text, expected):
                        self._submit_prompt(box, prompt, request_id, attempt=attempt, mode="INLINE")
                        _log_event(
                            "browser.prompt_insertion_completed",
                            request_id=request_id,
                            prompt_len=len(prompt or ""),
                            mode="INLINE",
                        )
                        return "INLINE"
                except Exception:
                    pass

                _log_event(
                    "browser.prompt_insert_waiting",
                    request_id=request_id,
                    attempt=attempt,
                    state="UNKNOWN",
                    reason="insertion timed out while potentially processing; NOT retrying",
                    expected_len=len(expected),
                    expected_hash=_content_hash(expected),
                    observed_len=len(_normalize_prompt(observed)),
                    observed_hash=_content_hash(observed),
                )
                raise ChatGPTInputError(
                    "ChatGPT input: could not verify the prompt text was "
                    "inserted before sending (state=UNKNOWN: a previous "
                    "paste may still be processing)."
                )

            # state == FAILED: every strategy produced stable non-exact
            # content, so the previous insertions are no longer
            # processing and a fresh attempt is safe (invariant 8).
            if attempt >= PROMPT_INSERT_MAX_ATTEMPTS:
                _log_event(
                    "browser.prompt_insert_failed",
                    request_id=request_id,
                    attempt=attempt,
                    expected_len=len(expected),
                    expected_hash=_content_hash(expected),
                    observed_len=len(_normalize_prompt(observed)),
                    observed_hash=_content_hash(observed),
                    state="FAILED",
                    reason="max attempts reached",
                )
                raise ChatGPTInputError(
                    "ChatGPT input: could not verify the prompt text was "
                    "inserted before sending."
                )

            # Defensive: before retrying, confirm the failed attempt did not
            # leave a pasted-text attachment behind.  If it did, the failure
            # is not "demonstrably inactive" and we must not clear/retry.
            attach_check, _ = self._get_pasted_text_attachment_state(
                expected, tiles_before
            )
            if attach_check != "NONE":
                _log_event(
                    "browser.prompt_insert_failed",
                    request_id=request_id,
                    attempt=attempt,
                    expected_len=len(expected),
                    expected_hash=_content_hash(expected),
                    observed_len=len(_normalize_prompt(observed)),
                    observed_hash=_content_hash(observed),
                    state="FAILED",
                    reason="attachment from failed attempt still present; NOT retrying",
                )
                raise ChatGPTInputError(
                    "ChatGPT input: could not verify the prompt text was "
                    "inserted before sending (a pasted-text attachment from "
                    "the failed attempt is still present)."
                )

            attempt += 1
            _log_event(
                "browser.prompt_insert_retry",
                request_id=request_id,
                attempt=attempt,
                expected_len=len(expected),
                expected_hash=_content_hash(expected),
                observed_len=len(_normalize_prompt(observed)),
                observed_hash=_content_hash(observed),
                state="FAILED",
                reason="stable non-exact content across all strategies",
            )
            self._clear_prompt(box)

        self._submit_prompt(
            box, prompt, request_id, attempt=attempt, mode="INLINE"
        )
        _log_event(
            "browser.prompt_insertion_completed",
            request_id=request_id,
            prompt_len=len(prompt or ""),
            mode="INLINE",
        )
        return "INLINE"


    def _submit_prompt(
        self, box, prompt, request_id=None, attempt=None, mode="INLINE",
        attachment_ctx=None,
    ):
        """Submit the composed prompt and confirm it was sent.

        This is the single send choke point: nothing clicks Send or presses
        Enter except here.  The composer/attachment is re-verified
        immediately before the interaction and the interaction is REFUSED
        unless the state still holds:

          mode="INLINE"     -> composer normalized content == expected EXACTLY
          mode="ATTACHMENT" -> the pasted-text attachment is still READY
                               (exactly one new associated tile, send
                               aria-disabled="false")

        This makes

            SEND  =>  VERIFIED_EXACT | VERIFIED_ATTACHMENT_READY

        structurally unbreakable: reaching this method is necessary for
        every send, and the appropriate state gate must pass first.
        """
        expected = _normalize_prompt(prompt)
        expected_hash = _content_hash(expected)
        attachment_state = "N/A"
        attachment_count = 0

        if mode == "ATTACHMENT":
            tiles_before = set((attachment_ctx or {}).get("tiles_before") or [])
            attach_state, ainfo = self._get_pasted_text_attachment_state(
                expected, tiles_before
            )
            attachment_state = attach_state
            attachment_count = ainfo.get("tile_count_after", 0)
            _log_event(
                "browser.attachment_probe",
                request_id=request_id,
                attempt=attempt,
                state=attach_state,
                tile_count_before=ainfo["tile_count_before"],
                tile_count_after=ainfo["tile_count_after"],
                new_tile_detected=ainfo["new_tile_detected"],
                send_aria_disabled=ainfo["send_aria_disabled"],
                attachment_label=ainfo["attachment_label"],
                attachment_ready=ainfo["attachment_ready"],
                association_match=ainfo["association_match"],
            )
            if attach_state != "READY":
                _log_event(
                    "browser.prompt_submit_blocked",
                    request_id=request_id,
                    attempt=attempt,
                    expected_len=len(expected),
                    expected_hash=expected_hash,
                    observed_len=0,
                    observed_hash="",
                    state=attach_state,
                    reason=(
                        "pre-submit attachment re-probe: state="
                        + str(attach_state)
                        + " (expected READY)"
                    ),
                )
                raise ChatGPTInputError(
                    "ChatGPT input: pasted-text attachment no longer ready "
                    "immediately before submit; refusing to send."
                )
            # Insert companion grounding prompt into composer so ChatGPT does not
            # treat the attached file as a static document upload.
            self._insert_attachment_companion(
                box, ATTACHMENT_COMPANION_PROMPT, request_id=request_id
            )
        else:
            # Invariant 6: re-verify immediately before submit.  A few read
            # retries absorb transient DOM re-renders, never a content diff.
            observed = ""
            for _ in range(3):
                try:
                    observed = _normalize_prompt(self._read_composer(box))
                    break
                except Exception:
                    self._poll_sleep(VERIFY_POLL_INTERVAL)

            if not _is_prompt_match(observed, expected):
                _log_event(
                    "browser.prompt_submit_blocked",
                    request_id=request_id,
                    attempt=attempt,
                    expected_len=len(expected),
                    expected_hash=expected_hash,
                    observed_len=len(observed) if observed else 0,
                    observed_hash=_content_hash(observed) if observed else "",
                    state="FAILED",
                    reason="pre-submit exact-match verification failed",
                )
                raise ChatGPTInputError(
                    "ChatGPT input: composer content did not match the expected "
                    "prompt immediately before submit; refusing to send."
                )

        u_loc = self._locate_user_messages()
        old_user_count = u_loc.count() if u_loc else 0
        send_btn = None
        for sel in (
            'button[aria-label="Send"]',
            'button[aria-label="Send prompt"]',
            'button[data-testid="send-button"]',
            self.SEND_BUTTON_SELECTOR,
        ):
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0 and loc.first.is_visible():
                    send_btn = loc.first
                    break
            except Exception:
                pass
        _log_event(
            "browser.submit_started",
            request_id=request_id,
            attempt=attempt,
            old_user_count=old_user_count,
        )

        def _submitted():
            try:
                cur_u = self._locate_user_messages()
                if cur_u and cur_u.count() > old_user_count:
                    return True
                return len(self._get_input().inner_text().strip()) == 0
            except Exception:
                return False

        attempts = 0
        while attempts < 3:

            if send_btn and send_btn.is_visible():
                try:
                    send_btn.click()
                except Exception:
                    pass
            else:
                try:
                    box.press("Enter")
                except Exception:
                    pass

            # Give ChatGPT up to ~8s to register the message.
            deadline = time.time() + 8
            while time.time() < deadline:
                if _submitted():
                    _log_event(
                        "browser.submitted_content",
                        request_id=request_id,
                        attempt=attempt,
                        mode=mode,
                        expected_hash=expected_hash,
                        observed_hash=expected_hash,
                        exact_match=(mode == "INLINE"),
                        attachment_count=attachment_count,
                        attachment_ready=(mode == "ATTACHMENT" and attachment_state == "READY"),
                    )
                    _log_event(
                        "browser.submit_completed",
                        request_id=request_id,
                        attempt=attempt,
                        attempts=attempts + 1,
                    )
                    return
                time.sleep(0.25)

            attempts += 1

        _log_event(
            "browser.submit_failed",
            request_id=request_id,
            attempt=attempt,
            attempts=attempts,
        )
        raise ChatGPTInputError(
            "ChatGPT input: could not submit the prompt (composer "
            "did not clear and no new user message appeared)."
        )


    # ========================================================
    # ROBUST PROMPT INSERTION (ProseMirror / contenteditable)
    # ========================================================
    # ChatGPT's composer is a div[contenteditable=true] ProseMirror
    # editor. Playwright's locator.fill() times out on it because it is
    # not a native form control.  These helpers prefer fill() only where
    # it is supported (textarea/input) and otherwise insert via real
    # keyboard events, which ProseMirror handles.
    #
    # The old flow verified with substring matching, retried immediately
    # after a ~7.5s timeout (clear -> re-paste -> execCommand -> typing),
    # and accepted a duplicated/concatenated prompt ("ABCABC" for "ABC").
    # Large pastes are processed by ChatGPT asynchronously, so an empty
    # composer is NOT a failure: it may still be resolving.  The new flow
    # dispatches ONE strategy at a time and drives it to a definitive
    # outcome (VERIFIED_EXACT / FAILED / UNKNOWN), only escalating or
    # retrying after a stable FAILED (previous insertion no longer
    # processing).  A single dispatch is never stacked on an uncertain one.

    def _insert_and_verify(self, box, prompt, expected, request_id, attempt, deadline, tiles_before):
        """Dispatch one insertion strategy, then verify it to completion.

        Strategies are tried in order (paste -> execCommand -> typing),
        preserving the existing browser-interaction ladder, but the next
        strategy is only dispatched after the previous one produced a
        definitive FAILED result (stable non-exact content => that
        insertion is no longer processing).  If a strategy's outcome is
        still uncertain when its budget expires, the whole send aborts as
        UNKNOWN instead of stacking a second simultaneous insertion on
        top of the first (invariant 9).

        tiles_before is the attachment snapshot taken before dispatch, so
        the verifier can distinguish a newly-created pasted-text document
        (VERIFIED_ATTACHMENT_READY) from a stale tile.  Attachment
        processing is never escalated or cleared.

        Returns (state, observed):
          "VERIFIED_EXACT" (observed=normalized text)
          | "VERIFIED_ATTACHMENT_READY" (observed=attachment info dict)
          | "FAILED" | "UNKNOWN"
        """
        if self._is_native_input(box):
            strategies = ["fill"]
        else:
            strategies = ["paste", "insertText", "execCommand", "typing"]

        observed = ""

        for index, strategy in enumerate(strategies):
            _log_event(
                "browser.prompt_insert_started",
                request_id=request_id,
                attempt=attempt,
                strategy=strategy,
                expected_len=len(expected),
                expected_hash=_content_hash(expected),
            )

            self._dispatch_insert(box, prompt, strategy)

            state, observed = self._verify_insert(
                box, expected, request_id, attempt, strategy, deadline, tiles_before
            )

            if state in ("VERIFIED_EXACT", "VERIFIED_ATTACHMENT_READY"):
                return state, observed
            if state == "UNKNOWN":
                final_state, final_info = self._get_pasted_text_attachment_state(expected, tiles_before)
                if final_state == "READY":
                    return "VERIFIED_ATTACHMENT_READY", final_info
                try:
                    cur_obs = _normalize_prompt(self._read_composer(box))
                    if _is_prompt_match(cur_obs, expected):
                        return "VERIFIED_EXACT", cur_obs
                except Exception:
                    pass
                return state, observed

            # state == FAILED: this strategy's insertion finished with a
            # stable non-exact result and is no longer processing.  A
            # clear plus the next strategy is safe here.
            _log_event(
                "browser.prompt_insert_retry",
                request_id=request_id,
                attempt=attempt,
                strategy=strategy,
                expected_len=len(expected),
                expected_hash=_content_hash(expected),
                observed_len=len(_normalize_prompt(observed)),
                observed_hash=_content_hash(observed),
                state="FAILED",
                reason=f"strategy '{strategy}' produced stable non-exact content",
            )

            if index < len(strategies) - 1:
                self._clear_prompt(box)

        return "FAILED", observed

    def _dispatch_insert(self, box, prompt, strategy):

        if strategy == "fill":
            box.fill(prompt)
        elif strategy == "paste":
            self._paste_prompt(box, prompt)
        elif strategy == "insertText":
            try:
                box.click()
                if hasattr(box, "focus"):
                    box.focus()
                if hasattr(self, "page") and hasattr(self.page, "keyboard") and hasattr(self.page.keyboard, "insert_text"):
                    self.page.keyboard.insert_text(prompt)
            except Exception:
                pass
        elif strategy == "execCommand":
            try:
                box.click()
                self.page.evaluate(
                    "text => {"
                    "  const t = typeof text === 'string' ? text : String(text);"
                    "  document.execCommand('insertText', false, t);"
                    "}",
                    prompt,
                )
            except Exception:
                pass
        else:  # "typing" (last resort)
            self._type_via_keyboard(box, prompt)

    def _paste_prompt(self, box, prompt):
        """Insert the full prompt via clipboard paste (one DOM operation).

        Character-by-character typing (press_sequentially) is far too slow
        for long prompts.  Clipboard paste inserts the entire text in one
        shot, which ProseMirror handles natively.
        """
        try:
            self.page.evaluate(
                "async text => { await navigator.clipboard.writeText(text); }",
                prompt,
            )
            box.click()
            box.press("ControlOrMeta+V")
        except Exception:
            # Fallback: if clipboard API is unavailable (e.g. permission
            # denied), fall back to sequential typing.
            self._type_via_keyboard(box, prompt)

    def _type_via_keyboard(self, box, prompt):
        """Fallback: insert prompt via fast CDP text insertion or sequential typing."""
        try:
            if hasattr(box, "focus"):
                box.focus()
            if hasattr(self, "page") and hasattr(self.page, "keyboard") and hasattr(self.page.keyboard, "insert_text"):
                self.page.keyboard.insert_text(prompt)
                return
        except Exception:
            pass

        try:
            if hasattr(self, "page") and hasattr(self.page, "evaluate"):
                self.page.evaluate(
                    "text => { document.execCommand('insertText', false, text); }",
                    prompt,
                )
                return
        except Exception:
            pass

        lines = (prompt or "").split("\n")
        for i, line in enumerate(lines):
            if line:
                box.press_sequentially(line)
            if i < len(lines) - 1:
                box.press("Shift+Enter")

    def _verify_insert(self, box, expected, request_id, attempt, strategy, deadline, tiles_before):
        """Poll both the composer AND the attachment UI to a definitive state.

        State outcomes:
          VERIFIED_EXACT            -> inline exact match seen 2x in a row;
          VERIFIED_ATTACHMENT_READY -> a new pasted-text attachment READY
                                       (exists + sendable + associated);
          FAILED                    -> stable non-exact inline content with NO
                                       new attachment anywhere;
          UNKNOWN                   -> deadline reached while still processing.

        PROCESSING/INSERTING always keep waiting; this method NEVER clears
        the composer (invariants 3 and 8).  While a new attachment exists
        (PROCESSING, or an as-yet-unmatched/ambiguous tile) inline checks
        are suppressed so a still-loading pasted-text document can never be
        misread as a stable FAILED content and cleared.

        Returns (state, observed) where observed is the normalized composer
        text (VERIFIED_EXACT/FAILED/UNKNOWN) or the attachment info dict
        (VERIFIED_ATTACHMENT_READY).
        """
        prev = ""
        stable_reads = 0
        state = "PROCESSING"
        last_logged_waiting_key = None
        last_logged_waiting_time = 0.0
        last_logged_probe_key = None
        last_logged_probe_time = 0.0
        empty_start_time = None
        empty_composer_timeout = float(
            os.environ.get("WEBMIND_EMPTY_COMPOSER_TIMEOUT", "10.0")
        )

        def _log_waiting(w_state, w_reason, w_obs=""):
            nonlocal last_logged_waiting_key, last_logged_waiting_time
            now_curr = self._now()
            w_key = (w_state, w_reason)
            if w_key != last_logged_waiting_key or (now_curr - last_logged_waiting_time) >= 5.0:
                last_logged_waiting_key = w_key
                last_logged_waiting_time = now_curr
                _log_event(
                    "browser.prompt_insert_waiting",
                    request_id=request_id,
                    attempt=attempt,
                    strategy=strategy,
                    state=w_state,
                    reason=w_reason,
                    expected_len=len(expected),
                    expected_hash=_content_hash(expected),
                    observed_len=len(_normalize_prompt(w_obs)),
                    observed_hash=_content_hash(w_obs) if w_obs else "",
                )

        _log_event(
            "browser.prompt_insert_waiting",
            request_id=request_id,
            attempt=attempt,
            strategy=strategy,
            state=state,
            reason="awaiting first composer read",
            expected_len=len(expected),
            expected_hash=_content_hash(expected),
            observed_len=0,
            observed_hash="",
        )

        while True:

            if self._now() >= deadline:
                _log_event(
                    "browser.prompt_insert_waiting",
                    request_id=request_id,
                    attempt=attempt,
                    strategy=strategy,
                    state="UNKNOWN",
                    reason="insertion deadline reached while still potentially processing",
                    expected_len=len(expected),
                    expected_hash=_content_hash(expected),
                    observed_len=len(_normalize_prompt(prev)),
                    observed_hash=_content_hash(prev),
                )
                return "UNKNOWN", prev

            try:
                actual = _normalize_prompt(self._read_composer(box))
            except Exception:
                self._poll_sleep(VERIFY_POLL_INTERVAL)
                continue

            # Probe the pasted-text attachment UI every poll.
            attach_state, ainfo = self._get_pasted_text_attachment_state(
                expected, tiles_before
            )
            probe_key = (
                attach_state,
                ainfo["tile_count_after"],
                ainfo["new_tile_detected"],
                ainfo["attachment_ready"],
                ainfo["send_aria_disabled"],
            )
            now_t = self._now()
            if (
                attach_state != "NONE"
                or ainfo["new_tile_detected"]
                or probe_key != last_logged_probe_key
                or (now_t - last_logged_probe_time) >= 5.0
            ):
                last_logged_probe_key = probe_key
                last_logged_probe_time = now_t
                _log_event(
                    "browser.attachment_probe",
                    request_id=request_id,
                    attempt=attempt,
                    state=attach_state,
                    tile_count_before=ainfo["tile_count_before"],
                    tile_count_after=ainfo["tile_count_after"],
                    new_tile_detected=ainfo["new_tile_detected"],
                    send_aria_disabled=ainfo["send_aria_disabled"],
                    attachment_label=ainfo["attachment_label"],
                    attachment_ready=ainfo["attachment_ready"],
                    association_match=ainfo["association_match"],
                )

            if attach_state == "READY":
                _log_event(
                    "browser.prompt_insert_verified",
                    request_id=request_id,
                    attempt=attempt,
                    strategy=strategy,
                    mode="ATTACHMENT",
                    state="VERIFIED_ATTACHMENT_READY",
                    expected_len=len(expected),
                    expected_hash=_content_hash(expected),
                    observed_len=0,
                    observed_hash="",
                    attachment_label=ainfo["attachment_label"],
                )
                return "VERIFIED_ATTACHMENT_READY", ainfo

            if attach_state == "PROCESSING":
                # A new attachment is still loading: send is aria-disabled.
                # Keep waiting; inline checks must NOT run against an empty
                # composer while the pasted-text document is in flight.
                state = "PROCESSING"
                _log_waiting(
                    state,
                    "attachment processing (send aria-disabled=true); not clearing",
                    actual,
                )
                prev = actual
                self._poll_sleep(VERIFY_POLL_INTERVAL)
                continue

            if attach_state == "UNKNOWN" and ainfo["new_tile_detected"]:
                # A new tile exists but is unmatched or ambiguous.  It may
                # resolve shortly, and clearing could corrupt it: keep
                # waiting, fail safely when the deadline expires.
                state = "PROCESSING"
                _log_waiting(
                    state,
                    "new attachment detected but association unclear; keeping it",
                    actual,
                )
                prev = actual
                self._poll_sleep(VERIFY_POLL_INTERVAL)
                continue

            # No new attachment (NONE, or UNKNOWN without a tile): fall back
            # to pure inline verification.
            if _is_prompt_match(actual, expected):
                # Invariants 1-2: EXACT equality, stable across 2 reads.
                stable_reads += 1
                if stable_reads >= VERIFY_STABLE_READS:
                    _log_event(
                        "browser.prompt_insert_verified",
                        request_id=request_id,
                        attempt=attempt,
                        strategy=strategy,
                        mode="INLINE",
                        state="VERIFIED_EXACT",
                        expected_len=len(expected),
                        expected_hash=_content_hash(expected),
                        observed_len=len(actual),
                        observed_hash=_content_hash(actual),
                    )
                    return "VERIFIED_EXACT", actual
                state = "INSERTING"
                _log_event(
                    "browser.prompt_insert_observed",
                    request_id=request_id,
                    attempt=attempt,
                    strategy=strategy,
                    state=state,
                    expected_len=len(expected),
                    expected_hash=_content_hash(expected),
                    observed_len=len(actual),
                    observed_hash=_content_hash(actual),
                )
            else:
                stable_reads = 0
                if not actual:
                    # Empty composer with no (new) attachment: may still be
                    # an in-flight ChatGPT pasted-text document.  Keep
                    # waiting; never clear.
                    if empty_start_time is None:
                        empty_start_time = self._now()
                    elif (
                        attach_state == "NONE"
                        and not ainfo["new_tile_detected"]
                        and (self._now() - empty_start_time) >= empty_composer_timeout
                    ):
                        state = "FAILED"
                        _log_event(
                            "browser.prompt_insert_waiting",
                            request_id=request_id,
                            attempt=attempt,
                            strategy=strategy,
                            state=state,
                            reason="composer empty timeout reached without attachment detection",
                            expected_len=len(expected),
                            expected_hash=_content_hash(expected),
                            observed_len=0,
                            observed_hash="",
                        )
                        return "FAILED", ""

                    state = "PROCESSING"
                    _log_waiting(
                        state,
                        "composer empty and no new attachment (pasted-text doc may still be processing)",
                        actual,
                    )
                else:
                    empty_start_time = None
                    if actual == prev:
                        # Two consecutive identical non-exact reads with no
                        # attachment: the previous insertion is no longer
                        # processing and its result is wrong (duplicated,
                        # old+prompt, prompt+extra, or a partial prompt that
                        # stopped growing).
                        state = "FAILED"
                        _log_event(
                            "browser.prompt_insert_waiting",
                            request_id=request_id,
                            attempt=attempt,
                            strategy=strategy,
                            state=state,
                            reason="stable non-exact content observed twice (no attachment)",
                            expected_len=len(expected),
                            expected_hash=_content_hash(expected),
                            observed_len=len(actual),
                            observed_hash=_content_hash(actual),
                        )
                        return "FAILED", actual
                    elif expected.startswith(actual):
                        # Genuine prefix of the expected text still changing:
                        # the insertion is still being written.
                        state = "INSERTING"
                        _log_waiting(
                            state,
                            "partial content present (still inserting)",
                            actual,
                        )
                    else:
                        # Changing non-exact content: wait for it to settle
                        # before deciding.
                        state = "INSERTING"
                        _log_waiting(
                            state,
                            "content differs from expected and is still changing",
                            actual,
                        )

            prev = actual
            self._poll_sleep(VERIFY_POLL_INTERVAL)

    def _is_native_input(self, box):
        """True when the composer is a real <textarea>/<input> (fill() works)."""
        try:
            info = box.evaluate(
                "el => ({"
                "  tag: (el.tagName || '').toUpperCase(),"
                "  ce: el.getAttribute('contenteditable')"
                "})"
            )
        except Exception:
            return False
        tag = (info or {}).get("tag")
        contenteditable = (info or {}).get("ce")
        return tag in ("TEXTAREA", "INPUT") and contenteditable is None

    def _read_composer(self, box):
        """Read the composer, preferring the native value then inner text."""
        try:
            return box.input_value()
        except Exception:
            return box.inner_text()

    def _insert_attachment_companion(self, box, companion_text, request_id=None):
        """Insert active grounding text into composer alongside the attachment tile."""
        if not companion_text:
            return
        try:
            if hasattr(self, "page") and hasattr(self.page, "evaluate"):
                self.page.evaluate(
                    """([el, text]) => {
                        if (!el) return;
                        if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') {
                            el.value = text;
                            el.dispatchEvent(new Event('input', { bubbles: true }));
                            el.dispatchEvent(new Event('change', { bubbles: true }));
                        } else {
                            let p = el.querySelector('p');
                            if (!p) {
                                p = document.createElement('p');
                                el.appendChild(p);
                            }
                            p.textContent = text;
                            el.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'insertText', data: text }));
                        }
                    }""",
                    [box, companion_text],
                )
            _log_event(
                "browser.attachment_companion_inserted",
                request_id=request_id,
                companion_len=len(companion_text),
            )
        except Exception as e:
            _log_event(
                "browser.attachment_companion_error",
                request_id=request_id,
                error=str(e),
            )

    # ----- pasted-text document (attachment) probing -----
    #
    # ChatGPT sometimes turns a large paste into a "Pasted text" document
    # ATTACHMENT instead of inline composer text.  This used to be
    # documented as a >~43KB rule; that number was never measured.  Real
    # submissions show the conversion is CONTENT-DEPENDENT rather than
    # length-dependent -- 12042, 14124 and 14285 char real conversations
    # all converted, while a 6587 char retry stayed inline.  Treat it as
    # something that can happen at ANY size, which is why the submission
    # mode is detected and reported rather than predicted from a count.
    #
    # Real DOM (live-verified via CDP): the composer stays EMPTY and a
    # file tile appears with aria-label truncated from the pasted content's
    # first line (~18 chars + "..").  Readiness is signalled by the send
    # button: aria-disabled="true" while the document is being processed,
    # aria-disabled="false" (and a visible library-file-icon) once ready.
    #
    # Attachment verification is deliberately weaker than inline EXACT
    # matching: a READY attachment means the pasted content EXISTS and is
    # SENDABLE and its label ASSOCIATES with the expected first line.  It
    # is a separate state (VERIFIED_ATTACHMENT_READY), never conflated
    # with VERIFIED_EXACT.
    ATTACHMENT_TILE_SELECTOR = (
        '[class*="group/resource-card"], '
        '[class*="composer-attachment-surface"] [class*="group/file-tile"], '
        '[class*="composer-attachment-surface"] [role="group"], '
        '[class*="group/file-tile"][role="group"], '
        '[data-testid*="attachment-tile"], '
        '[data-testid*="file-tile"]'
    )
    SEND_BUTTON_SELECTOR = "#composer-submit-button"

    def _attachment_tile_labels(self):
        """Return the aria-labels of every attachment tile currently in the DOM."""
        try:
            loc = self.page.locator(self.ATTACHMENT_TILE_SELECTOR)
            count = loc.count()
            labels = []
            seen = set()
            for i in range(count):
                el = loc.nth(i)
                lbl = (el.get_attribute("aria-label") or "").strip()
                if not lbl and hasattr(el, "inner_text"):
                    try:
                        txt = el.inner_text().strip()
                        if txt:
                            lbl = txt.splitlines()[0].strip()
                    except Exception:
                        pass
                if lbl and lbl not in seen:
                    seen.add(lbl)
                    labels.append(lbl)
                elif not lbl and "unlabeled_tile" not in seen:
                    seen.add("unlabeled_tile")
                    labels.append("")
            return labels
        except Exception:
            return []

    def _snapshot_attachment_tiles(self):
        """Snapshot the current tile labels (taken BEFORE a dispatch)."""
        return set(self._attachment_tile_labels())

    def _send_button_aria_disabled(self):
        """Return the send button aria-disabled value ("true"/"false"/None)."""
        try:
            for sel in (
                'button[aria-label="Send"]',
                'button[aria-label="Send prompt"]',
                self.SEND_BUTTON_SELECTOR,
                'button[data-testid="send-button"]',
                'button[data-testid="composer-send-button"]',
                'button[data-testid*="send"]',
                'button[aria-label*="Send" i]',
                'button[aria-label*="发送"]',
            ):
                loc = self.page.locator(sel)
                if loc.count() > 0:
                    btn = loc.first
                    val = btn.get_attribute("aria-disabled")
                    if val is not None:
                        return val.strip().lower()
                    if hasattr(btn, "is_disabled"):
                        try:
                            return "true" if btn.is_disabled() else "false"
                        except Exception:
                            return "false"
                    return "false"
            return None
        except Exception:
            return None

    def _attachment_label_matches(self, label, expected):
        """Weak association: tile label starts with or relates to the expected text."""
        if not label:
            return True
        lbl_lower = label.lower().strip()
        generic_markers = (
            "pasted", "paste", "snippet", "document", "untitled",
            "text", ".txt", "attachment", "code", "file", "prompt", "upload"
        )
        if any(marker in lbl_lower for marker in generic_markers):
            return True

        if not expected:
            return True

        lines = [line.strip() for line in str(expected).splitlines() if line.strip()]
        if not lines:
            return True
        first_line = lines[0]
        if first_line.startswith("```"):
            first_line = lines[1] if len(lines) > 1 else first_line.lstrip("`")
        first_line = first_line.strip()

        prefix = first_line[:24].strip(",.;:!? \t\r\n`'\"")
        if prefix:
            if prefix.lower() in lbl_lower or lbl_lower.startswith(prefix.lower()):
                return True
            if lbl_lower.rstrip(". ") in prefix.lower():
                return True

        tokens = [t.lower() for t in re.findall(r"\w{4,}", first_line)]
        for token in tokens[:5]:
            if token in lbl_lower:
                return True

        return True

    def _get_pasted_text_attachment_state(self, expected, tiles_before):
        """Probe the attachment UI and classify the pasted-document state.

        Returns (state, info):
          NONE       - no new attachment tile was created by this insertion;
          PROCESSING - a new tile exists but send is aria-disabled="true";
          READY      - exactly one new associated tile + send aria-disabled="false";
          UNKNOWN    - read failure, >1 new tile, or a new tile that does not
                       associate with the expected first line (never submit).
        info carries every observed field for browser.attachment_probe.
        """
        info = dict(
            tile_count_before=len(tiles_before or []),
            tile_count_after=0,
            new_tile_detected=False,
            send_aria_disabled=None,
            attachment_label="",
            attachment_ready=False,
            association_match=False,
        )
        labels = self._attachment_tile_labels()
        info["tile_count_after"] = len(labels)

        new_labels = [lbl for lbl in labels if lbl not in (tiles_before or set())]
        if not new_labels and info["tile_count_after"] > info["tile_count_before"]:
            new_labels = [labels[-1]]
        info["new_tile_detected"] = bool(new_labels)

        if not new_labels:
            return "NONE", info

        unique_new_labels = []
        for l in new_labels:
            if l not in unique_new_labels:
                unique_new_labels.append(l)

        if len(unique_new_labels) > 1:
            return "UNKNOWN", info

        label = unique_new_labels[0]
        info["attachment_label"] = label
        info["association_match"] = self._attachment_label_matches(label, expected)

        send_aria = self._send_button_aria_disabled()
        info["send_aria_disabled"] = send_aria
        if send_aria == "true":
            # aria-disabled="true": the pasted document is still processing.
            return "PROCESSING", info
        if not info["association_match"]:
            return "UNKNOWN", info

        info["attachment_ready"] = True
        return "READY", info

    def _insertion_deadline(self, length):
        """Per-dispatch insertion deadline: base + per-character allowance.

        State-checking means the loop exits as soon as content settles
        (exact or stable wrong), so this only bites while the pasted-text
        document is genuinely still processing.  It scales with prompt
        size but stays capped so a hung paste fails closed quickly.
        """
        return min(
            self._insert_deadline_base + length * self._insert_deadline_per_char,
            self._insert_deadline_cap,
        )

    def _now(self):
        return self._clock()

    def _poll_sleep(self, seconds):
        self._poll_sleep_fn(seconds)


    def _clear_prompt(self, box):

        # Remove any lingering attachment tiles from composer
        try:
            if hasattr(self, "page") and hasattr(self.page, "locator"):
                remove_btns = self.page.locator('button[aria-label*="Remove"]')
                for _ in range(5):
                    if remove_btns.count() == 0:
                        break
                    remove_btns.first.click()
                    if hasattr(self, "_poll_sleep"):
                        self._poll_sleep(0.1)
        except Exception:
            pass

        box.click()
        box.press("ControlOrMeta+A")
        box.press("Delete")

        # Confirm the composer is actually empty; retry a couple of
        # times in case ProseMirror re-rendered mid-clear.
        for _ in range(2):
            try:
                if not box.inner_text().strip():
                    return
            except Exception:
                pass
            box.click()
            box.press("ControlOrMeta+A")
            box.press("Delete")
            time.sleep(0.2)


    # ========================================================
    # GET ASSISTANT MESSAGE
    # ========================================================

    def _extract_code_blocks(self, container):
        if not hasattr(container, "locator"):
            return None
        try:
            code = container.locator("pre code")
            c_count = code.count() if hasattr(code, "count") else (len(code._nodes) if hasattr(code, "_nodes") else 0)
            if c_count > 1:
                blocks = []
                for i in range(c_count):
                    c_node = code.nth(i) if hasattr(code, "nth") else (code._nodes[i] if hasattr(code, "_nodes") else code.first)
                    c_text = c_node.text_content()
                    if c_text and c_text.strip():
                        blocks.append(f"```json\n{c_text.strip()}\n```")
                if blocks:
                    return "\n\n".join(blocks)
            elif c_count == 1:
                c_node = code.first if hasattr(code, "first") else (code._nodes[0] if hasattr(code, "_nodes") else None)
                if c_node:
                    c_text = c_node.text_content()
                    if c_text is not None and c_text.strip():
                        return c_text.strip()
        except Exception:
            pass
        return None

    def _extract_markdown_element(self, m):
        if m is None:
            return ""
        if hasattr(m, "locator"):
            code_text = self._extract_code_blocks(m)
            if code_text:
                return code_text
        try:
            text = m.text_content()
            if text is not None:
                return text.strip()
        except Exception:
            pass
        return ""

    def _extract_node_text(self, node):
        if node is None:
            return ""
        try:
            if hasattr(node, "locator"):
                md = node.locator('.markdown, [class*="MarkdownRoot"], [data-markdown-text-style="assistant-message"]')
                md_count = md.count() if hasattr(md, "count") else (len(md._nodes) if hasattr(md, "_nodes") else 0)
                if md_count > 0:
                    blocks = []
                    for i in range(md_count):
                        m = md.nth(i) if hasattr(md, "nth") else (md._nodes[i] if hasattr(md, "_nodes") else md.first)
                        m_text = self._extract_markdown_element(m)
                        if m_text and m_text.strip():
                            blocks.append(m_text.strip())
                    if blocks:
                        return "\n\n".join(blocks)
                code_text = self._extract_code_blocks(node)
                if code_text:
                    return code_text
        except Exception:
            pass
        try:
            raw_text = (node.text_content() or "").strip()
            if raw_text.startswith("ChatGPT said:"):
                raw_text = raw_text[len("ChatGPT said:"):].strip()
            return raw_text
        except Exception:
            return ""

    def _get_assistant_message(self, since_count=None):

        messages = self._locate_assistant_messages()

        if not messages:
            return ""

        total = messages.count() if hasattr(messages, "count") else (len(messages._messages) if hasattr(messages, "_messages") else 0)
        if total == 0:
            return ""

        if since_count is not None and total > since_count:
            msg_parts = []
            for idx in range(since_count, total):
                try:
                    node = messages.nth(idx) if hasattr(messages, "nth") else (messages._messages[idx] if hasattr(messages, "_messages") else messages.last)
                    text = self._extract_node_text(node)
                    if text and text.strip():
                        msg_parts.append(text.strip())
                except Exception:
                    pass
            if msg_parts:
                return "\n\n".join(msg_parts)

        if not hasattr(messages, "last") or getattr(messages, "last", None) is None:
            return ""

        return self._extract_node_text(messages.last)

    def _get_assistant_turn_text(self, since_count=None):
        """Retrieve assistant turn text, falling back to 0-arg call if mocked."""
        try:
            return self._get_assistant_message(since_count=since_count)
        except TypeError:
            return self._get_assistant_message()

    def _is_send_button_visible(self):
        """Check if any send or idle-composer submit button is currently visible in the DOM."""
        if not hasattr(self, "page") or not hasattr(self.page, "locator"):
            return False
        for sel in (
            '[data-testid="send-button"]:visible',
            'button[aria-label="Send"]:visible',
            'button[aria-label="Send prompt"]:visible',
            'button[aria-label*="Send"]:visible',
            'button[aria-label*="发送"]:visible',
            '#composer-submit-button:visible',
            'button[aria-label="Dictate"]:visible',
            'button[aria-label="Start Voice"]:visible',
        ):
            try:
                if self.page.locator(sel).count() > 0:
                    return True
            except Exception:
                pass
        return False

    def _is_generating(self):
        """Check if ChatGPT is still actively generating a response."""
        if not hasattr(self, "page") or not hasattr(self.page, "locator"):
            return False

        # Active stop button present
        for sel in (
            '[data-testid="stop-button"]:visible',
            'button[aria-label*="Stop"]:visible',
            'button[aria-label*="stop"]:visible',
            'button[aria-label*="停止"]:visible',
        ):
            try:
                if self.page.locator(sel).count() > 0:
                    return True
            except Exception:
                pass

        # If the send/idle button has reappeared and no stop button exists, ChatGPT is idle
        if self._is_send_button_visible():
            return False

        # Active streaming or active thinking indicator (excluding past-tense static buttons)
        for sel in (
            '.result-thinking:visible',
            'button:has-text("Thinking..."):visible',
            'button:has-text("Searching..."):visible',
            '[data-testid="cursor"]:visible',
            '.result-streaming:visible',
            '[class*="result-streaming"]:visible',
            '[data-markdown-text-style="assistant-message"] [class~="cursor"]:visible',
            '.markdown [class~="cursor"]:visible',
            'div[class*="MarkdownRoot"] [class~="cursor"]:visible',
        ):
            try:
                if self.page.locator(sel).count() > 0:
                    return True
            except Exception:
                pass
        return False

    def _wait_for_first_assistant_message(
        self, old_assistant_count, prompt, sleep_s=0.3, is_stream=False, wall_cap=None, old_last_text=None
    ):
        """Wait for a NEW assistant message node to appear.

        Phase 1 of the two-phase wait; the caller owns the generation /
        stability phase that follows.  Three classes of failure are handled
        instead of a blind timeout:

          * Submit never registered: ChatGPT went idle with no new message
            while the composer STILL holds exactly our prompt.  Re-submit
            once (nothing reached ChatGPT, so re-sending cannot duplicate)
            and keep waiting.
          * Input eaten: submit consumed the composer but ChatGPT never
            rendered our turn as the last user message (large pasted-text
            attachments can be dropped by the submit).  Re-insert the
            prompt from scratch once and re-submit.
          * Turn received but silent (user message rendered, no reply): if
            it stays idle for RESPONSE_WAIT_TIMEOUT, log a state snapshot
            and raise.  The error message and the logged event carry
            whether ChatGPT ever showed our turn, so the next diagnosis
            distinguishes "long thinking misread as dead" from "paste
            lost".

        Active generation (stop button / thinking indicator) always keeps
        waiting: the timeout only accrues while ChatGPT is provably idle.
        """
        start = time.time()
        request_id = uuid.uuid4().hex
        resubmitted = False
        was_generating = False

        if old_last_text is None:
            old_last_text = getattr(self, "_last_assistant_snapshot", None)

        while True:
            # The rate-limit dialog can arrive after submission, while this
            # request waits for its assistant message.
            self._dismiss_rate_limit_modal()

            # Absolute per-request cap: even active generation must stop
            # once the request deadline passes (the HTTP layer is the
            # backstop against a wedged browser holding the client open).
            if wall_cap is not None and time.time() >= wall_cap:
                _log_event(
                    "browser.response_wait_deadline",
                    elapsed_ms=int((time.time() - start) * 1000),
                    is_stream=is_stream,
                )
                raise TimeoutError(
                    "ChatGPT did not produce a response (request "
                    "wall-clock deadline exceeded)."
                )

            a_loc = self._locate_assistant_messages()
            count = a_loc.count() if a_loc else 0

            # Signal 1: Message count increased in the DOM
            if count > old_assistant_count:
                print("Response detected.")
                _log_event(
                    "browser.generation_detected",
                    elapsed_ms=int((time.time() - start) * 1000),
                    assistant_count=count,
                    is_stream=is_stream,
                    trigger="count_increase",
                )
                return

            current_text = _clean_stream_text(self._get_assistant_turn_text(since_count=old_assistant_count))

            # Signal 2: Content changed from pre-submission snapshot
            # (handles virtualization where old message unmounted keeping count the same)
            if current_text:
                if old_last_text is not None and current_text != old_last_text:
                    print("Response detected (content changed).")
                    _log_event(
                        "browser.generation_detected",
                        elapsed_ms=int((time.time() - start) * 1000),
                        assistant_count=count,
                        is_stream=is_stream,
                        trigger="content_changed",
                    )
                    return
                if old_assistant_count == 0:
                    print("Response detected (first turn text).")
                    _log_event(
                        "browser.generation_detected",
                        elapsed_ms=int((time.time() - start) * 1000),
                        assistant_count=count,
                        is_stream=is_stream,
                        trigger="first_turn_text",
                    )
                    return

            is_gen = self._is_generating()
            if is_gen:
                was_generating = True
                time.sleep(sleep_s)
                continue

            # Signal 3: Generation was active and has now stopped, with new content present
            if was_generating and current_text and (old_last_text is None or current_text != old_last_text):
                print("Response detected (generation finished).")
                _log_event(
                    "browser.generation_detected",
                    elapsed_ms=int((time.time() - start) * 1000),
                    assistant_count=count,
                    is_stream=is_stream,
                    trigger="generation_finished",
                )
                return

            # Signal 4: Send button is visible (ChatGPT returned to idle state) with new content
            if (
                self._is_send_button_visible()
                and current_text
                and (old_last_text is not None and current_text != old_last_text)
            ):
                print("Response detected (idle with new response).")
                _log_event(
                    "browser.generation_detected",
                    elapsed_ms=int((time.time() - start) * 1000),
                    assistant_count=count,
                    is_stream=is_stream,
                    trigger="send_button_idle",
                )
                return

            if time.time() - start < RESPONSE_WAIT_TIMEOUT:
                time.sleep(sleep_s)
                continue

            diag = self._idle_timeout_diagnostics(prompt)

            if not resubmitted:
                resubmitted = True

                if diag.get("composer_still_holds_prompt"):
                    # Submit never registered: the composer still carries our
                    # exact prompt, so nothing reached ChatGPT.  Re-submit the
                    # already-verified text once and keep waiting.
                    heal = self._heal_resubmit(
                        prompt, request_id, is_stream, elapsed_ms=int((time.time() - start) * 1000), diag=diag
                    )
                    if heal:
                        start = time.time()
                        was_generating = False
                        old_last_text = _clean_stream_text(self._get_assistant_message())
                        continue

                if (
                    not diag.get("composer_still_holds_prompt")
                    and diag.get("last_user_hash")
                    and diag.get("last_user_hash") != diag.get("prompt_hash")
                    and not diag.get("composer_error")
                ):
                    # Submit consumed the composer but ChatGPT never rendered
                    # our turn as a user message (last-user mismatch).  The
                    # input was eaten, not answered: re-insert from scratch
                    # (composer is empty, so no duplication) and re-submit.
                    heal = self._heal_resend(
                        prompt, is_stream, elapsed_ms=int((time.time() - start) * 1000), diag=diag
                    )
                    if heal:
                        start = time.time()
                        was_generating = False
                        old_last_text = _clean_stream_text(self._get_assistant_message())
                        continue

            _log_event(
                "browser.response_wait_timeout",
                elapsed_ms=int((time.time() - start) * 1000),
                reason="no new assistant message",
                is_stream=is_stream,
                **diag,
            )
            raise TimeoutError(
                "ChatGPT did not produce a response."
                + self._timeout_suffix(diag)
            )

    def _heal_resubmit(self, prompt, request_id, is_stream, elapsed_ms, diag):
        """Re-send the prompt still sitting verified in the composer."""
        _log_event(
            "browser.response_wait_resubmit",
            elapsed_ms=elapsed_ms,
            is_stream=is_stream,
            **diag,
        )
        try:
            self._submit_prompt(self._get_input(), prompt, request_id, attempt=1)
            return True
        except Exception as e:
            _log_event(
                "browser.response_wait_resubmit_failed",
                error=str(e),
                error_type=type(e).__name__,
                is_stream=is_stream,
            )
            return False

    def _heal_resend(self, prompt, is_stream, elapsed_ms, diag):
        """When the submit ate the input without ChatGPT rendering the
        turn, re-run the full insertion pipeline (composer is empty, so no
        duplication is possible) and re-submit."""
        _log_event(
            "browser.response_wait_resend",
            elapsed_ms=elapsed_ms,
            is_stream=is_stream,
            **diag,
        )
        try:
            self._send_prompt(prompt)
            return True
        except Exception as e:
            _log_event(
                "browser.response_wait_resend_failed",
                error=str(e),
                error_type=type(e).__name__,
                is_stream=is_stream,
            )
            return False

    def _idle_timeout_diagnostics(self, prompt):
        """Snapshot why a wait timed out, so a rerun of the same request can
        tell a submit that never registered from a turn ChatGPT received but
        left silently."""
        diag = {}
        try:
            diag["stop_button_visible"] = (
                self.page.locator('[data-testid="stop-button"]:visible').count() > 0
            )
        except Exception as e:
            diag["stop_button_error"] = str(e)
        try:
            diag["thinking_visible"] = (
                self.page.locator(".result-thinking:visible").count() > 0
            )
        except Exception as e:
            diag["thinking_error"] = str(e)
        try:
            a_loc = self._locate_assistant_messages()
            diag["assistant_messages"] = a_loc.count() if a_loc else 0
            u_loc = self._locate_user_messages()
            diag["user_messages"] = u_loc.count() if u_loc else 0
        except Exception as e:
            diag["count_error"] = str(e)
        try:
            composer = _normalize_prompt(self._read_composer(self._get_input()))
            diag["composer_len"] = len(composer)
            diag["composer_still_holds_prompt"] = (
                composer == _normalize_prompt(prompt)
            )
        except Exception as e:
            diag["composer_error"] = str(e)
        try:
            u_loc = self._locate_user_messages()
            last_user = u_loc.last if u_loc and u_loc.count() > 0 else None
            if last_user:
                diag["last_user_hash"] = _content_hash(
                    last_user.text_content() or ""
                )[:16]
        except Exception as e:
            diag["last_user_error"] = str(e)
        diag["prompt_hash"] = _content_hash(prompt)[:16]
        return diag

    @staticmethod
    def _timeout_suffix(diag):
        if diag.get("composer_still_holds_prompt"):
            composer = "prompt-still-in-composer (submit never registered)"
        elif diag.get("composer_len", -1) == 0:
            composer = "composer-empty (turn submitted)"
        else:
            composer = "composer-other"
        last = diag.get("last_user_hash")
        if last:
            received = "yes" if last == diag.get("prompt_hash") else "no"
        else:
            received = "unknown"
        return (
            f" [composer={composer}, "
            f"ChatGPT-showed-our-turn-as-last-user-message={received}]"
        )


    # ========================================================
    # WAIT FOR RESPONSE
    # ========================================================

    def _wait_for_response(self, old_assistant_count, prompt, wall_cap=None, old_last_text=None):
        if old_last_text is None:
            old_last_text = getattr(self, "_last_assistant_snapshot", None)

        print("Waiting for ChatGPT...")
        _log_event("browser.response_wait_started", old_assistant_count=old_assistant_count)

        # ----------------------------------------------------
        # Wait for a new assistant message
        # ----------------------------------------------------

        self._wait_for_first_assistant_message(
            old_assistant_count,
            prompt,
            sleep_s=0.3,
            is_stream=False,
            wall_cap=wall_cap,
            old_last_text=old_last_text,
        )


        # ----------------------------------------------------
        # Wait for streaming to finish
        # ----------------------------------------------------

        previous = ""
        stable_count = 0
        stable_since = None
        placeholder_since = None
        hard_deadline = time.time() + 600   # absolute max 10 min
        if wall_cap is not None:
            hard_deadline = min(hard_deadline, wall_cap)
        deadline = time.time() + 300

        start = time.time()

        last_assistant_count = old_assistant_count

        while time.time() < hard_deadline:
            # ChatGPT can render this acknowledgement dialog mid-request.
            self._dismiss_rate_limit_modal()

            current = _clean_stream_text(self._get_assistant_turn_text(since_count=old_assistant_count))

            # Detect new assistant bubble appearing mid-turn
            try:
                a_loc = self._locate_assistant_messages()
                raw_count = a_loc.count() if a_loc and hasattr(a_loc, "count") else None
                count_now = raw_count if isinstance(raw_count, int) else last_assistant_count
            except Exception:
                count_now = last_assistant_count

            if count_now > last_assistant_count:
                _log_event(
                    "browser.new_assistant_message_detected",
                    previous_count=last_assistant_count,
                    new_count=count_now,
                    elapsed_ms=int((time.time() - start) * 1000),
                )
                last_assistant_count = count_now
                stable_count = 0
                stable_since = None

            # Check if the model is still actively generating.
            # ChatGPT shows a stop-button during generation; also
            # the "result-thinking" class marks reasoning models.
            # If generation is still active, extend the stability
            # deadline but never beyond the hard deadline.
            generating = self._is_generating()
            if generating:
                new_deadline = time.time() + 300
                deadline = min(new_deadline, hard_deadline)
                stable_count = 0
                stable_since = None

            # Working-status placeholder (e.g. "Analyzing image"):
            # never counts as finished, never returned as answer.
            if _is_transient_status(current):

                if placeholder_since is None:
                    placeholder_since = time.time()

                elif time.time() - placeholder_since > TRANSIENT_STATUS_TIMEOUT:
                    raise TimeoutError(
                        "ChatGPT image analysis timed out."
                    )

                stable_count = 0
                stable_since = None
                previous = current

                time.sleep(0.35)
                continue

            placeholder_since = None

            if current and current == previous:

                if stable_since is None:
                    stable_since = time.time()

                stable_count += 1

            else:

                stable_count = 0
                stable_since = None

            previous = current

            # Response hasn't changed AND the model is idle:
            # the answer has settled, return it. If the send button has
            # reappeared, ChatGPT has completely finished rendering and is
            # ready for new input: settle in 4 checks (~1.4s) instead of
            # forcing a 2.1s or 15s wait, safely bridging multi-bubble pauses.
            # Otherwise fall back to 6 checks.
            send_visible = self._is_send_button_visible()
            if (
                current
                and not generating
                and (
                    (stable_count >= 4 and send_visible)
                    or stable_count >= 6
                )
            ):
                _log_event(
                    "browser.generation_finished",
                    elapsed_ms=int((time.time() - start) * 1000),
                    chars=len(current or ""),
                    send_button_observed=send_visible,
                    assistant_count=last_assistant_count,
                )
                return current

            # Versatile idle safety net: if text has stayed frozen for >= 5.5s (stable_count >= 16)
            # and no active stop button is visible on page, generation is definitely finished.
            has_stop_btn = False
            for s_sel in (
                '[data-testid="stop-button"]:visible',
                'button[aria-label*="Stop"]:visible',
                'button[aria-label*="stop"]:visible',
                'button[aria-label*="停止"]:visible',
            ):
                try:
                    if self.page.locator(s_sel).count() > 0:
                        has_stop_btn = True
                        break
                except Exception:
                    pass

            if current and not has_stop_btn and stable_count >= 16:
                _log_event(
                    "browser.generation_finished_via_idle_backstop",
                    elapsed_ms=int((time.time() - start) * 1000),
                    chars=len(current or ""),
                    send_button_observed=send_visible,
                )
                return current

            if stable_since is not None and time.time() - stable_since >= 15.0:
                _log_event(
                    "browser.generation_stalled_finishing",
                    elapsed_ms=int((time.time() - start) * 1000),
                    chars=len(current or ""),
                    stalled_us_while_generating=bool(generating),
                )
                return current

            time.sleep(0.35)


        _log_event("browser.response_wait_timeout", elapsed_ms=int((time.time() - start) * 1000), reason="multiple-response-stability-timeout")
        raise TimeoutError(
            "ChatGPT response took too long."
        )


    # ========================================================
    # ASK
    # ========================================================

    def ask(self, prompt, chat_url=None, image_paths=None, conversation_id=None, deadline=None):

        _page_health = None
        try:
            _page_health = {
                "closed": self.page.is_closed(),
                "url": self.page.url,
            }
        except Exception as e:
            _page_health = {"health_check_error": str(e)}
        _log_event(
            "browser.ask_started",
            prompt_len=len(prompt or ""),
            chat_url=bool(chat_url),
            images=len(image_paths or []),
            page_health=_page_health,
            page_id=id(self.page) if self.page else None,
        )

        try:
            return self._ask_impl(
                prompt,
                chat_url,
                image_paths,
                conversation_id,
                _deadline_wall(deadline),
            )
        except Exception as e:
            _log_event("browser.ask_failed", error=str(e), error_type=type(e).__name__)
            raise

    def _rotate_chat_if_oversized(self):
        """Bounded persistent-chat rotation (Phase 3, survival #7).

        ChatGPT keeps ONE live conversation in the browser until the user
        starts a fresh one, so an unbounded number of WebMind turns makes
        the accumulated scrollback (and therefore every continuation
        prompt built by build_prompt) grow forever.  Once the live chat
        exceeds MAX_TURNS_BEFORE_NEW_CHAT assistant turns, THIS method
        navigates the browser to a BRAND-NEW ChatGPT chat before the next
        turn is dispatched, bounding how large the next continuation
        prompt can ever get.

        Continuity: after rotation build_prompt (protocol.py) re-serialises
        the ENTIRE conversation history into the next prompt, so no turn
        is lost by starting a fresh chat.  The rotation only changes WHERE
        the next turn is typed, never what is said.

        MUST be failure-safe:
          * It ONLY navigates to a new chat URL and NEVER pastes anything,
            so a rotation that throws leaves the prompt flow untouched and
            the caller proceeds normally in the current chat.
          * It NEVER clears or attaches and NEVER throws to the caller.
          * It is invoked ONLY before a new (non-pinned) turn is
            dispatched, never mid-answer and never while a stream is in
            flight.
          * It does NOT run when the caller pinned a chat_url, because a
            pinned chat_url is an explicit "continue THIS conversation"
            signal that rotation must respect.
        """
        if MAX_TURNS_BEFORE_NEW_CHAT <= 0:
            return

        try:
            a_loc = self._locate_assistant_messages()
            old_count = a_loc.count() if a_loc else 0
        except Exception as e:
            _log_event(
                "browser.chat_rotation_count_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return

        if old_count <= MAX_TURNS_BEFORE_NEW_CHAT:
            return

        _log_event(
            "browser.chat_rotation_started",
            old_count=old_count,
            max_turns=MAX_TURNS_BEFORE_NEW_CHAT,
        )

        try:
            self._rotate_to_fresh_chat()
            self._conversation_id = None
            _log_event("browser.chat_rotation_completed")
        except Exception as e:
            print(
                f"[rotation] failed to open a fresh chat ({e}); "
                "continuing in the existing chat.",
                flush=True,
            )
            _log_event(
                "browser.chat_rotation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    def _rotate_to_fresh_chat(self):
        """Navigate to a clean, empty ChatGPT chat with 0 messages."""
        try:
            # 1. If "New chat" button is visible, click it (fastest & cleanest SPA reset)
            if hasattr(self.page, "locator"):
                btn = self.page.locator(
                    '[data-testid="create-new-chat-button"], a[aria-label="New chat"]'
                ).first
                if hasattr(btn, "is_visible") and btn.is_visible(timeout=800):
                    btn.click()
                    if hasattr(self.page, "wait_for_timeout"):
                        self.page.wait_for_timeout(1000)
                else:
                    self.page.goto(CHATGPT_URL, wait_until="domcontentloaded")
                    if hasattr(self.page, "wait_for_timeout"):
                        self.page.wait_for_timeout(2500)
            else:
                self.page.goto(CHATGPT_URL, wait_until="domcontentloaded")
                if hasattr(self.page, "wait_for_timeout"):
                    self.page.wait_for_timeout(2500)
        except Exception:
            self.page.goto(CHATGPT_URL, wait_until="domcontentloaded")
            if hasattr(self.page, "wait_for_timeout"):
                self.page.wait_for_timeout(2500)

        # 2. Confirm assistant count is 0; if still > 0, full goto CHATGPT_URL
        try:
            a_loc = self._locate_assistant_messages()
            if a_loc and a_loc.count() > 0:
                self.page.goto(CHATGPT_URL, wait_until="domcontentloaded")
                if hasattr(self.page, "wait_for_timeout"):
                    self.page.wait_for_timeout(2500)
        except Exception:
            pass

        # 3. Clear any leftover draft in composer if present
        try:
            if hasattr(self, "_get_input") and hasattr(self, "_clear_prompt"):
                box = self._get_input()
                if box:
                    self._clear_prompt(box)
        except Exception:
            pass

    def _ensure_conversation_chat(self, conversation_id):
        """Isolate each client conversation in its own ChatGPT chat.

        ChatGPT holds ONE live conversation in the browser, so appending a
        request from a DIFFERENT conversation (a new harness session, the
        session-title call, any unrelated client) to the current chat leaks
        the earlier turns into the new prompt and the model can answer with
        stale context.  When the incoming conversation id differs from the
        one the live chat serves, navigate to a fresh chat first.
        build_prompt re-serialises the full history into each turn, so
        nothing is lost by starting the new chat empty.

        A falsy conversation_id preserves the caller's explicit chat_url or
        the legacy single-chat behavior.  Failure to navigate is logged and
        ignored: the turn proceeds in the current chat instead.
        """
        if not conversation_id:
            return

        if conversation_id == self._conversation_id:
            return

        _log_event(
            "browser.conversation_rotation_started",
            previous=self._conversation_id,
            incoming=conversation_id,
        )

        try:
            self._rotate_to_fresh_chat()
        except Exception as e:
            print(
                f"[conversation] failed to open a fresh chat ({e}); "
                "continuing in the existing chat.",
                flush=True,
            )
            _log_event(
                "browser.conversation_rotation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            return

        self._conversation_id = conversation_id
        _log_event(
            "browser.conversation_rotation_completed",
            conversation_id=conversation_id,
        )

    def _dismiss_rate_limit_modal(self):
        """Dismiss ChatGPT's rate-limit dialog when it is visible."""
        now_t = time.time()
        last_t = getattr(self, "_last_rate_limit_check_time", 0.0)
        if (now_t - last_t) < 1.0:
            return False
        self._last_rate_limit_check_time = now_t
        try:
            dialogs = self.page.locator('[role="dialog"]')
            for index in range(dialogs.count()):
                dialog = dialogs.nth(index)
                text = (dialog.inner_text() or "").lower()
                if "too many requests" not in text:
                    continue

                _log_event("browser.rate_limit_modal_detected")
                try:
                    dialog.get_by_role(
                        "button", name=re.compile(r"^got it$", re.IGNORECASE)
                    ).click()
                    _log_event("browser.rate_limit_modal_dismissed")
                    return True
                except Exception as error:
                    _log_event(
                        "browser.rate_limit_modal_dismiss_failed",
                        error=str(error),
                        error_type=type(error).__name__,
                    )
                return False
        except RuntimeError:
            raise
        except Exception:
            return False
        return False

    def _ask_impl(self, prompt, chat_url=None, image_paths=None, conversation_id=None, wall_cap=None):

        if not isinstance(prompt, str):
            raise TypeError(
                "Prompt must be a string."
            )

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )


        # ----------------------------------------------------
        # Open specified chat
        # ----------------------------------------------------

        if chat_url:

            print("Opening chat...")

            self.page.goto(
                chat_url,
                wait_until="domcontentloaded"
            )

            self.page.wait_for_timeout(3000)


        # ----------------------------------------------------
        # Count existing assistant messages
        # ----------------------------------------------------

        if not chat_url:
            self._ensure_conversation_chat(conversation_id)
            self._rotate_chat_if_oversized()

        self._dismiss_rate_limit_modal()

        a_loc = self._locate_assistant_messages()
        old_count = a_loc.count() if a_loc else 0
        old_last_text = _clean_stream_text(self._get_assistant_message())
        self._last_assistant_snapshot = old_last_text


        # ----------------------------------------------------
        # Attach images first (worker thread owns Playwright)
        # ----------------------------------------------------

        if image_paths:
            try:
                self._attach_images(image_paths)
            except Exception as e:
                msg = str(e).lower()
                if "multiple" in msg or len(image_paths) > 1 and "attach" in msg:
                    raise RuntimeError(
                        f"MULTI_ATTACH_UNSUPPORTED: {e}"
                    )
                raise


        # ----------------------------------------------------
        # Send prompt
        # ----------------------------------------------------

        print("Prompt:", prompt)

        mode = self._send_prompt(prompt)

        # ----------------------------------------------------
        # Detection of pasted-text attachment
        # ----------------------------------------------------
        if mode != "INLINE":
            if not getattr(self, "allow_attachment", ALLOW_ATTACHMENT_PROMPT):
                _log_event(
                    "browser.attachment_answer_refused",
                    prompt_len=len(prompt or ""),
                    mode=mode,
                    reason="prompt was submitted as a pasted-text attachment; refusing "
                           "the answer so the caller can retry with a shorter prompt "
                           "rather than accept a document-reader response",
                )
                raise PromptTooLargeError(
                    f"PROMPT_TOO_LARGE: the {len(prompt or '')} char prompt was "
                    f"submitted as a pasted-text attachment, so ChatGPT answered as "
                    f"an outside reader of that file rather than as the agent. "
                    f"Refusing to return that answer. Shorten the conversation or "
                    f"the tool output.",
                    kind="attachment",
                )
            _log_event(
                "browser.attachment_answer_accepted",
                prompt_len=len(prompt or ""),
                mode=mode,
                reason="prompt was submitted as a pasted-text attachment; "
                       "allowing answer from capable model",
            )


        # ----------------------------------------------------
        # Wait for answer
        # ----------------------------------------------------

        answer = self._wait_for_response(
            old_count,
            prompt,
            wall_cap=wall_cap,
            old_last_text=old_last_text,
        )


        # ----------------------------------------------------
        # Return clean Response object
        # ----------------------------------------------------

        return Response(answer, mode=mode)


    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        print("Disconnecting from Chrome (session kept alive)...")

        try:

            # CDP connection: do NOT close the browser, or Chrome would
            # lose the authenticated session. Just detach Playwright.
            pass

        finally:

            self.playwright.stop()


    # ========================================================
    # STREAM RESPONSE (yields text deltas)
    # ========================================================

    def _stream_response(self, old_assistant_count, prompt, wall_cap=None, old_last_text=None):
        """
        Generator yielding newly-added text as the assistant
        message grows in the DOM.

        These are UI-observed deltas, NOT model tokens.
        Chunk sizes are arbitrary.
        """
        if old_last_text is None:
            old_last_text = getattr(self, "_last_assistant_snapshot", None)

        _log_event("browser.stream_started", old_assistant_count=old_assistant_count)

        # ----------------------------------------------------
        # Wait for a new assistant message
        # ----------------------------------------------------

        self._wait_for_first_assistant_message(
            old_assistant_count,
            prompt,
            sleep_s=0.15,
            is_stream=True,
            wall_cap=wall_cap,
            old_last_text=old_last_text,
        )

        # Re-anchor so the streaming phase below gets its own full budget.
        start = time.time()

        # ----------------------------------------------------
        # Yield deltas as the message grows (append-only).
        # Never re-emit snapshots: deltas are suffixes only.
        # ----------------------------------------------------

        sent = ""
        last_seen = ""
        last_change = time.time()
        placeholder_since = None
        a_loc = self._locate_assistant_messages()
        live_count = a_loc.count() if a_loc else 0

        streaming_budget_end = start + 300
        if wall_cap is not None:
            streaming_budget_end = min(streaming_budget_end, wall_cap)

        while time.time() < streaming_budget_end:
            # Check independently of assistant-message updates.
            self._dismiss_rate_limit_modal()

            raw = self._get_assistant_turn_text(since_count=old_assistant_count)

            # Working-status placeholder (e.g. "Analyzing image"):
            # suppress from the stream and never treat as finished.
            if _is_transient_status(raw):

                if placeholder_since is None:
                    placeholder_since = time.time()

                elif time.time() - placeholder_since > TRANSIENT_STATUS_TIMEOUT:
                    raise TimeoutError(
                        "ChatGPT image analysis timed out."
                    )

                last_seen = raw
                last_change = time.time()

                time.sleep(0.15)
                continue

            placeholder_since = None

            # New assistant node appeared mid-stream (split answer):
            # track new bubble count and update activity timestamp.
            try:
                a_loc = self._locate_assistant_messages()
                raw_count = a_loc.count() if a_loc and hasattr(a_loc, "count") else None
                count_now = raw_count if isinstance(raw_count, int) else live_count
            except Exception:
                count_now = live_count
            if count_now > live_count:
                live_count = count_now
                last_change = time.time()

            current = _clean_stream_text(raw)

            if current != last_seen:
                last_seen = current
                last_change = time.time()

                delta = _compute_stream_delta(sent, current)
                if delta:
                    sent = current
                    _log_event(
                        "browser.stream_delta",
                        delta_len=len(delta),
                        total_len=len(current),
                        elapsed_ms=int((time.time() - start) * 1000),
                    )
                    yield delta

            # Generation finished detection:
            # If the model is idle and the send button has reappeared, ChatGPT
            # is ready for the next prompt: finish in ~1.4s of stability instead
            # of stalling, safely bridging inter-bubble pauses.
            # Otherwise fall back to 2.0s stability.
            if current and not self._is_generating():
                elapsed_stable = time.time() - last_change
                send_visible = self._is_send_button_visible()
                if (elapsed_stable >= 1.4 and send_visible) or elapsed_stable >= 2.0:
                    _log_event(
                        "browser.generation_finished",
                        elapsed_ms=int((time.time() - start) * 1000),
                        chars=len(current or ""),
                        is_stream=True,
                        send_button_observed=send_visible,
                    )
                    return
                if time.time() - last_change >= 15.0:
                    _log_event(
                        "browser.generation_stalled_finishing",
                        elapsed_ms=int((time.time() - start) * 1000),
                        chars=len(current or ""),
                        is_stream=True,
                        stalled_us_while_generating=bool(self._is_generating()),
                    )
                    return

            # Versatile idle safety net for streaming:
            # If text has stopped changing for >= 6.0s and no active stop button is visible:
            if current and (time.time() - last_change >= 6.0):
                has_stop = False
                for s_sel in (
                    '[data-testid="stop-button"]:visible',
                    'button[aria-label*="Stop"]:visible',
                    'button[aria-label*="stop"]:visible',
                    'button[aria-label*="停止"]:visible',
                ):
                    try:
                        if self.page.locator(s_sel).count() > 0:
                            has_stop = True
                            break
                    except Exception:
                        pass
                if not has_stop:
                    _log_event(
                        "browser.stream_finished_via_idle_backstop",
                        elapsed_ms=int((time.time() - start) * 1000),
                        chars=len(current or ""),
                        is_stream=True,
                    )
                    return

            time.sleep(0.15)

        _log_event("browser.stream_timed_out", elapsed_ms=int((time.time() - start) * 1000))
        raise TimeoutError(
            "ChatGPT response took too long."
        )

    # ========================================================
    # ASK (STREAMING GENERATOR)
    # ========================================================

    def ask_stream(self, prompt, chat_url=None, image_paths=None, conversation_id=None, deadline=None):
        """
        Send a prompt and yield text deltas as ChatGPT streams.

        Usage:
            for delta in browser.ask_stream("hello"):
                print(delta, end="", flush=True)
        """

        _page_health = None
        try:
            _page_health = {
                "closed": self.page.is_closed(),
                "url": self.page.url,
            }
        except Exception as e:
            _page_health = {"health_check_error": str(e)}
        _log_event(
            "browser.ask_stream_started",
            prompt_len=len(prompt or ""),
            chat_url=bool(chat_url),
            images=len(image_paths or []),
            page_health=_page_health,
            page_id=id(self.page) if self.page else None,
        )

        try:
            yield from self._ask_stream_impl(
                prompt,
                chat_url,
                image_paths,
                conversation_id,
                _deadline_wall(deadline),
            )
            _log_event("browser.ask_stream_completed")
        except Exception as e:
            _log_event(
                "browser.ask_stream_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _ask_stream_impl(self, prompt, chat_url=None, image_paths=None, conversation_id=None, wall_cap=None):

        if not isinstance(prompt, str):
            raise TypeError(
                "Prompt must be a string."
            )

        if not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        # ----------------------------------------------------
        # Open specified chat
        # ----------------------------------------------------

        if chat_url:

            print("Opening chat...")

            self.page.goto(
                chat_url,
                wait_until="domcontentloaded"
            )

            self.page.wait_for_timeout(3000)

        # ----------------------------------------------------
        # Count existing assistant messages
        # ----------------------------------------------------

        if not chat_url:
            self._ensure_conversation_chat(conversation_id)
            self._rotate_chat_if_oversized()

        self._dismiss_rate_limit_modal()

        a_loc = self._locate_assistant_messages()
        old_count = a_loc.count() if a_loc else 0
        old_last_text = _clean_stream_text(self._get_assistant_message())
        self._last_assistant_snapshot = old_last_text

        # ----------------------------------------------------
        # Attach images first (worker thread owns Playwright)
        # ----------------------------------------------------

        if image_paths:
            try:
                self._attach_images(image_paths)
            except Exception as e:
                msg = str(e).lower()
                if "multiple" in msg or len(image_paths) > 1 and "attach" in msg:
                    raise RuntimeError(
                        f"MULTI_ATTACH_UNSUPPORTED: {e}"
                    )
                raise

        # ----------------------------------------------------
        # Send prompt
        # ----------------------------------------------------

        print("Prompt:", prompt)

        mode = self._send_prompt(prompt)

        if mode != "INLINE":
            if not getattr(self, "allow_attachment", ALLOW_ATTACHMENT_PROMPT):
                _log_event(
                    "browser.attachment_answer_refused",
                    prompt_len=len(prompt or ""),
                    mode=mode,
                    is_stream=True,
                    reason="prompt was submitted as a pasted-text attachment; refusing "
                           "before yielding any delta so caller can retry",
                )
                raise PromptTooLargeError(
                    f"PROMPT_TOO_LARGE: the {len(prompt or '')} char prompt was "
                    f"submitted as a pasted-text attachment, so ChatGPT answered as "
                    f"an outside reader of that file rather than as the agent. "
                    f"Refusing to stream that answer. Shorten the conversation or "
                    f"the tool output.",
                    kind="attachment",
                )
            _log_event(
                "browser.attachment_answer_accepted",
                prompt_len=len(prompt or ""),
                mode=mode,
                is_stream=True,
                reason="prompt was submitted as a pasted-text attachment; "
                       "streaming model response from attachment",
            )

        # ----------------------------------------------------
        # Stream answer deltas
        # ----------------------------------------------------

        yield from self._stream_response(old_count, prompt, wall_cap=wall_cap, old_last_text=old_last_text)

    def ensure_healthy(self) -> bool:
        """Verifies session health; reconnects if the browser closed."""
        if not browser_guard.is_browser_healthy(getattr(self, "browser", None), getattr(self, "page", None)):
            return self.reconnect()
        return True

    def reconnect(self) -> bool:
        """Relaunches and reconnects to ChatGPT Chrome session."""
        return browser_guard.relaunch_and_reconnect(
            self,
            "ChatGPT",
            CDP_PORT,
            PROFILE_DIR,
            CHATGPT_URL,
        )
# ============================================================
# GLOBAL INSTANCE
# ============================================================

_instance = None


def start():

    global _instance

    if _instance is None:

        _instance = ChatGPTBrowser()

    return _instance


_REQUEST_DEADLINE_SECONDS = float(
    os.environ.get("WEBMIND_ASK_DEADLINE", "180.0")
)


def _deadline_value(deadline):
    if deadline is None:
        return None
    if deadline <= 0:
        return time.monotonic()
    return time.monotonic() + deadline


def _deadline_wall(deadline):
    """Convert an absolute monotonic deadline to a wall-clock instant, or None.

    The browser wait loops compare against ``time.time()``, so the per-request
    monotonic deadline (from ``_deadline_value``) is translated once at the
    entry of ask/ask_stream into the same clock the loops already use.
    """
    if deadline is None:
        return None
    return time.time() + max(0.0, deadline - time.monotonic())


def ask(prompt, chat_url=None, image_paths=None, conversation_id=None, deadline=None):

    global _instance

    if _instance is None:
        start()

    return browser_guard.with_recovery(
        _instance,
        "ChatGPT",
        CDP_PORT,
        PROFILE_DIR,
        CHATGPT_URL,
        _instance.ask,
        prompt,
        chat_url,
        image_paths,
        conversation_id,
        deadline=_deadline_value(deadline),
    )


def ask_stream(prompt, chat_url=None, image_paths=None, conversation_id=None, deadline=None):

    global _instance

    if _instance is None:
        start()

    if hasattr(_instance, "ensure_healthy"):
        _instance.ensure_healthy()

    yield from _instance.ask_stream(
        prompt,
        chat_url,
        image_paths,
        conversation_id,
        deadline=_deadline_value(deadline),
    )


def stop():

    global _instance

    if _instance is not None:

        _instance.close()

        _instance = None


# ============================================================
# DIRECT PROGRAM MODE
# ============================================================

if __name__ == "__main__":

    try:

        start()

        while True:

            prompt = input(
                "\nPrompt (or 'exit'): "
            )

            if prompt.strip().lower() == "exit":
                break

            try:

                response = ask(prompt)

                print()
                print("=" * 70)
                print("RESPONSE")
                print("=" * 70)
                print(response.text)
                print("=" * 70)

            except Exception as e:

                print()
                print("ERROR:")
                print(e)

    finally:

        stop()
