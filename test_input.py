import time
import unittest

import main as main_mod


class FakeElement:
    """A resolved locator slot (mirrors Playwright's single Locator)."""

    def __init__(self, selector, tag="DIV", contenteditable="true", visible=True):
        self.selector = selector
        self.tag = tag
        self.contenteditable = contenteditable
        self.visible = visible
        self.text = ""
        self.pressed = []
        self.sent = None
        self.fill_calls = 0
        self.native_fill_ok = True
        self.sequentially_calls = 0
        self.paste_handler = None
        self.submit_handler = None
        self.click_calls = 0
        self.is_send_button = False
        self.aria_label = None
        self.aria_disabled = None

    def is_visible(self):
        return self.visible

    def get_attribute(self, name):
        if name == "aria-label":
            return self.aria_label
        if name == "aria-disabled":
            return self.aria_disabled
        return None

    def evaluate(self, expr):
        return {"tag": self.tag, "ce": self.contenteditable}

    def click(self):
        self.click_calls += 1
        if self.is_send_button and self.submit_handler:
            self.submit_handler()

    def fill(self, text):
        self.fill_calls += 1
        if self.native_fill_ok:
            self.text = text
            return
        raise TimeoutError(
            "locator.fill: Timeout waiting for element to be filled. "
            "Element is not an <input>, <textarea> or [contenteditable] element "
            "and cannot be filled."
        )

    def press_sequentially(self, text, **kwargs):
        self.sequentially_calls += 1
        self.text += text

    def press(self, key):
        self.pressed.append(key)
        if key == "Shift+Enter":
            self.text += "\n"
        elif key == "ControlOrMeta+V":
            if self.paste_handler:
                self.paste_handler()
        elif key == "Enter":
            self.sent = self.text
            if self.submit_handler:
                self.submit_handler()
        elif key in ("ControlOrMeta+A", "Delete"):
            pass

    def inner_text(self):
        return self.text

    def input_value(self):
        if self.tag in ("TEXTAREA", "INPUT") and self.contenteditable is None:
            return self.text
        raise TimeoutError("Input is not an <input> or <textarea> element.")


class FakePage:
    """Mirrors Playwright page surface used by main.py."""

    def __init__(self, selectors):
        self.selectors = selectors
        self.clipboard = ""
        self.user_count = 0
        self.last_clicked = None
        self.exec_calls = 0
        # Pasted-text attachment simulation.
        self.attachments = []          # live DOM tile list (FakeTile)
        self._scheduled_tiles = []     # [FakeTile] not yet in the DOM
        self.send_aria_disabled = "false"
        self.submit_probe = FakeSendProbe(self)

    def locator(self, selector):
        if selector == '[data-message-author-role="user"]':
            return FakeLocator(selector, [], fixed_count=self.user_count)
        if selector == main_mod.ChatGPTBrowser.SEND_BUTTON_SELECTOR:
            return FakeLocator(selector, [self.submit_probe])
        if selector == main_mod.ChatGPTBrowser.ATTACHMENT_TILE_SELECTOR:
            return FakeLocator(selector, self.attachments)
        return FakeLocator(selector, self.selectors.get(selector, []))

    def evaluate(self, expression, *args):
        # Simulate navigator.clipboard.writeText(text)
        if "clipboard.writeText" in expression and args:
            self.clipboard = args[0]
            return None
        # Simulate document.execCommand('insertText', false, text)
        if "execCommand" in expression and args:
            self.exec_calls += 1
            box = self.last_clicked
            if box is not None:
                mode = getattr(box, "mode", None)
                if mode == "duplicate":
                    box.text = args[0] + args[0]
                elif mode == "partial":
                    box.text = args[0][: max(1, len(args[0]) // 3)]
                elif mode == "old_text":
                    box.text = "old " + args[0]
                else:
                    box.text = args[0]
            return None
        return None

    def paste_into(self, box):
        # Simulate a Ctrl+V paste: the editor receives clipboard content.
        if getattr(box, "mode", None) == "attachment":
            # Large paste -> ChatGPT turns it into a pasted-text document.
            self._start_attachment(box)
            return
        if getattr(box, "paste_delay_reads", 0) > 0:
            box.pending_paste = self.clipboard
            box.pending_after_read = box.reads + box.paste_delay_reads
            return
        mode = getattr(box, "mode", None)
        if mode == "duplicate":
            box.text = self.clipboard + self.clipboard
        elif mode == "partial":
            box.text = self.clipboard[: max(1, len(self.clipboard) // 3)]
        elif mode == "old_text":
            box.text = "old " + self.clipboard
        else:
            box.text += self.clipboard

    def _attachment_label(self, box):
        first = (self.clipboard or "").splitlines()[0]
        if not first:
            return "Pasted text.."
        return first[:18] + ".."

    def _start_attachment(self, box):
        count = getattr(box, "attachment_tile_count", 1)
        for i in range(count):
            tile = FakeTile(self._attachment_label(box) if i == 0 else self._attachment_label(box) + str(i))
            tile.appear_read = box.reads + getattr(box, "attachment_delay_reads", 0)
            tile.ready_read = tile.appear_read + getattr(box, "attachment_processing_reads", 0)
            vanish = getattr(box, "attachment_vanish_reads", None)
            if vanish is not None:
                tile.vanish_read = tile.appear_read + vanish
            tile.page = self
            tile.self_removing = bool(getattr(box, "attachment_self_removing", False))
            self._scheduled_tiles.append(tile)

    def _advance_attachments(self, reads):
        for tile in list(self._scheduled_tiles):
            if reads >= tile.appear_read and tile not in self.attachments:
                self.attachments.append(tile)
        for tile in list(self.attachments):
            if tile.vanish_read is not None and reads >= tile.vanish_read:
                self.attachments.remove(tile)
        pending = [t for t in self.attachments if reads < t.ready_read]
        self.send_aria_disabled = "true" if pending else "false"

    def submit(self, box):
        # Simulate ChatGPT accepting the message: a new user row appears
        # and the composer (and its attachments) are cleared.
        box.sent = box.text
        self.user_count += 1
        box.text = ""
        self.attachments = []
        self._scheduled_tiles = []
        self.send_aria_disabled = "false"


class FakeClock:
    """Controllable clock: advance(dt) walks _now() forward."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class PromptEditor(FakeElement):
    """Contenteditable fake with read-count / delayed-paste / race controls."""

    def __init__(self, selector, page=None, **kw):
        super().__init__(selector, **kw)
        self.page = page
        self.reads = 0
        self.read_overrides = {}     # read index -> forced inner_text value
        self.paste_delay_reads = 0   # >0: pasted text appears after N reads
        self.pending_paste = None
        self.pending_after_read = 0
        self.mode = None             # None|"duplicate"|"partial"|"old_text"

    def click(self):
        if self.page is not None:
            self.page.last_clicked = self
        self.click_calls += 1
        if self.is_send_button and self.submit_handler:
            self.submit_handler()

    def press_sequentially(self, text, **kwargs):
        self.sequentially_calls += 1
        if self.mode == "duplicate":
            self.text = text + text
        elif self.mode == "partial":
            self.text = text[: max(1, len(text) // 3)]
        elif self.mode == "old_text":
            self.text = "old " + text
        else:
            self.text += text

    def inner_text(self):
        self.reads += 1
        if self.page is not None:
            self.page._advance_attachments(self.reads)
        forced = self.read_overrides.get(self.reads)
        if forced is not None:
            self.text = forced
        if self.pending_paste is not None and self.reads >= self.pending_after_read:
            self.text = self.pending_paste
            self.pending_paste = None
        return self.text


class FakeTile(FakeElement):
    """Attachment file tile: `[class*="group/file-tile"][role="group"]`."""

    def __init__(self, label):
        super().__init__(main_mod.ChatGPTBrowser.ATTACHMENT_TILE_SELECTOR)
        self.aria_label = label
        self.appear_read = 0
        self.ready_read = 0
        self.vanish_read = None
        self.page = None
        self.self_removing = False   # remove from DOM on the first label read

    def get_attribute(self, name):
        value = super().get_attribute(name)
        if (
            name == "aria-label"
            and self.self_removing
            and self.page is not None
            and self in self.page.attachments
        ):
            self.page.attachments.remove(self)
        return value


class FakeSendProbe(FakeElement):
    """#composer-submit-button: exposes aria-disabled load state."""

    def __init__(self, page):
        super().__init__(main_mod.ChatGPTBrowser.SEND_BUTTON_SELECTOR, "BUTTON", None, visible=True)
        self.page = page

    def get_attribute(self, name):
        if name == "aria-disabled":
            return self.page.send_aria_disabled
        return super().get_attribute(name)


class FakeLocator:
    """Mirrors the locator surface used by main.py input handling."""

    def __init__(self, selector, elements, fixed_count=None):
        self.selector = selector
        self.elements = elements
        self._fixed_count = fixed_count

    @property
    def first(self):
        return self.elements[0]

    def count(self):
        if self._fixed_count is not None:
            return self._fixed_count
        return len(self.elements)

    def is_visible(self):
        return any(
            getattr(el, "visible", False) for el in self.elements
        ) if self.elements else False

    def click(self):
        if self._fixed_count is not None and self._fixed_count <= 0:
            raise Exception("clicked with no elements")
        for el in self.elements:
            el.click()

    def nth(self, index):
        return self.elements[index]


def make_browser(page):
    browser = main_mod.ChatGPTBrowser.__new__(main_mod.ChatGPTBrowser)
    browser.page = page
    browser._clock = time.time
    browser._poll_sleep_fn = time.sleep
    browser._insert_deadline_base = main_mod.INSERTION_DEADLINE_BASE
    browser._insert_deadline_per_char = main_mod.INSERTION_DEADLINE_PER_CHAR
    browser._insert_deadline_cap = main_mod.INSERTION_DEADLINE_CAP
    return browser


def fast_browser(page, deadline=1.0):
    """Browser wired to a fake clock so insertion polls run instantly."""
    browser = make_browser(page)
    clock = FakeClock()
    browser._clock = clock
    browser._poll_sleep_fn = clock.advance
    browser._insert_deadline_base = deadline
    browser._insert_deadline_per_char = 0.0
    browser._insert_deadline_cap = deadline
    return browser, clock


class TestPromptInputHandling(unittest.TestCase):
    """Regression: ChatGPT's ProseMirror div[contenteditable=true] editor."""

    def setUp(self):
        # #prompt-textarea now resolves to a contenteditable ProseMirror div;
        # fill() times out on it, so clipboard paste must be used.
        self.editor = FakeElement("#prompt-textarea", "DIV", "true")
        self.editor.native_fill_ok = False
        self.editor.submit_handler = lambda: self.page.submit(self.editor)
        self.send_button = FakeElement(
            'button[data-testid="send-button"]', "BUTTON", None, visible=True
        )
        self.send_button.is_send_button = True
        self.send_button.submit_handler = lambda: self.page.submit(self.editor)
        self.page = FakePage({
            "#prompt-textarea": [self.editor],
            'div[contenteditable="true"]': [self.editor],
            '[contenteditable="true"]': [self.editor],
            "textarea": [],
            '[role="textbox"]': [self.editor],
            'button[data-testid="send-button"]': [self.send_button],
        })
        self.editor.paste_handler = lambda: self.page.paste_into(self.editor)

    def test_prosemirror_uses_clipboard_paste_not_fill(self):
        browser = make_browser(self.page)

        browser._send_prompt("What is my CPU usage?")

        # The ProseMirror contenteditable editor must NEVER be fill()ed.
        self.assertEqual(self.editor.fill_calls, 0)
        # Must NOT use press_sequentially (character-by-character typing).
        self.assertEqual(self.editor.sequentially_calls, 0)
        # Must use clipboard paste: evaluate wrote to clipboard, Ctrl+V pressed.
        self.assertEqual(self.page.clipboard, "What is my CPU usage?")
        self.assertIn("ControlOrMeta+V", self.editor.pressed)
        # Text was pasted into the editor, then submitted.
        self.assertEqual(self.editor.sent, "What is my CPU usage?")
        self.assertEqual(self.page.user_count, 1)
        self.assertGreater(self.send_button.click_calls, 0)

    def test_prosemirror_multiline_uses_clipboard_paste(self):
        browser = make_browser(self.page)

        browser._send_prompt("line one\nline two")

        self.assertEqual(self.editor.fill_calls, 0)
        self.assertEqual(self.editor.sequentially_calls, 0)
        # Multiline prompt pasted as a single clipboard operation.
        self.assertEqual(self.page.clipboard, "line one\nline two")
        self.assertIn("ControlOrMeta+V", self.editor.pressed)
        self.assertEqual(self.editor.sent, "line one\nline two")
        self.assertEqual(self.page.user_count, 1)

    def test_unknown_insertion_raises_without_retry(self):
        # A dispatch that never lands (e.g. an async pasted-text document
        # that does not resolve in time) must NOT trigger a blind
        # clear + repaste and must NOT stack a second simultaneous
        # insertion on the first (invariants 3, 8, 9).
        browser, clock = fast_browser(self.page)
        dispatches = []

        def broken_dispatch(box, prompt, strategy):
            dispatches.append(strategy)
            # Simulate insertion never landing: composer stays empty.

        browser._dispatch_insert = broken_dispatch
        with self.assertRaises(RuntimeError):
            browser._send_prompt("hello")
        self.assertIsNone(self.editor.sent)
        self.assertEqual(self.page.user_count, 0)
        # Exactly one strategy was dispatched - no escalation, no retry.
        self.assertEqual(dispatches, ["paste"])
        # The composer was never cleared after the failed dispatch.
        self.assertNotIn("ControlOrMeta+A", self.editor.pressed)


class TestPromptInputTextarea(unittest.TestCase):
    """Native textarea still uses fill() (legacy variant)."""

    def setUp(self):
        self.box = FakeElement("textarea", "TEXTAREA", None)
        self.box.submit_handler = lambda: self.page.submit(self.box)
        self.send_button = FakeElement(
            'button[data-testid="send-button"]', "BUTTON", None, visible=True
        )
        self.send_button.is_send_button = True
        self.send_button.submit_handler = lambda: self.page.submit(self.box)
        self.page = FakePage({
            "#prompt-textarea": [],
            'div[contenteditable="true"]': [],
            '[contenteditable="true"]': [],
            "textarea": [self.box],
            '[role="textbox"]': [],
            'button[data-testid="send-button"]': [self.send_button],
        })

    def test_textarea_uses_fill(self):
        browser = make_browser(self.page)

        browser._send_prompt("plain question")

        self.assertEqual(self.box.fill_calls, 1)
        self.assertEqual(self.box.sent, "plain question")
        self.assertEqual(self.page.user_count, 1)


class TestPromptInputSelection(unittest.TestCase):
    """_get_input prefers the actual editable/contenteditable element."""

    def test_contenteditable_editor_selected_before_other_slots(self):
        editor = FakeElement("#prompt-textarea", "DIV", "true")
        manager = FakeElement('[role="textbox"]', "DIV", "true")
        page = FakePage({
            "#prompt-textarea": [editor],
            'div[contenteditable="true"]': [editor],
            '[contenteditable="true"]': [editor],
            "textarea": [manager],
            '[role="textbox"]': [manager],
        })
        browser = make_browser(page)

        box = browser._get_input()

        self.assertIs(box, editor)


class TestPromptInsertionRace(unittest.TestCase):
    """Regression: the large-paste race / verified-insertion invariants.

    Invariants under test:
      - delayed paste that eventually succeeds
      - timeout while the original paste is still processing
      - duplicated / partial / old-text content is REJECTED (exact match)
      - exact stable prompt is verified (>=2 consecutive reads)
      - verify-again-immediately-before-submit blocks a late mutation
    """

    def setUp(self):
        self.events = []
        original = main_mod._log_event

        def spy(event, **kw):
            self.events.append((event, kw))

        main_mod._log_event = spy
        self._original_log_event = original
        self.addCleanup(
            lambda: setattr(main_mod, "_log_event", self._original_log_event)
        )

    def _make_page(self, editor):
        editor.native_fill_ok = False
        editor.submit_handler = lambda: self.page.submit(editor)
        send_button = FakeElement(
            'button[data-testid="send-button"]', "BUTTON", None, visible=True
        )
        send_button.is_send_button = True
        send_button.submit_handler = lambda: self.page.submit(editor)
        page = FakePage({
            "#prompt-textarea": [editor],
            'div[contenteditable="true"]': [editor],
            '[contenteditable="true"]': [editor],
            "textarea": [],
            '[role="textbox"]': [editor],
            'button[data-testid="send-button"]': [send_button],
        })
        self.page = page
        editor.page = page
        editor.paste_handler = lambda: page.paste_into(editor)
        return editor, page, send_button

    def _events_of(self, name):
        return [kw for (evt, kw) in self.events if evt == name]

    def test_delayed_paste_eventually_succeeds(self):
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        editor.paste_delay_reads = 3   # text lands on the 3rd read after Ctrl+V
        browser, clock = fast_browser(page, deadline=5.0)

        browser._send_prompt("make the race go away")

        self.assertEqual(editor.sent, "make the race go away")
        self.assertEqual(page.user_count, 1)
        self.assertGreater(send.click_calls, 0)
        # Paste won: no strategy escalation, no typing fallback.
        self.assertEqual(editor.sequentially_calls, 0)
        self.assertEqual(page.exec_calls, 0)
        # It waited through EMPTY reads before verifying.
        waiting = self._events_of("browser.prompt_insert_waiting")
        self.assertTrue(
            any(
                kw.get("state") == "PROCESSING"
                and "empty" in kw.get("reason", "")
                for kw in waiting
            )
        )
        self.assertEqual(len(self._events_of("browser.prompt_insert_verified")), 1)
        submitted = self._events_of("browser.submitted_content")
        self.assertEqual(len(submitted), 1)
        self.assertTrue(submitted[0]["exact_match"])
        self.assertEqual(submitted[0]["expected_hash"], submitted[0]["observed_hash"])

    def test_timeout_while_paste_still_processing(self):
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        editor.paste_delay_reads = 10 ** 9   # never lands within the deadline
        browser, clock = fast_browser(page, deadline=1.0)

        with self.assertRaises(main_mod.ChatGPTInputError) as ctx:
            browser._send_prompt("hello")
        self.assertIn("UNKNOWN", str(ctx.exception))
        self.assertIsNone(editor.sent)
        self.assertEqual(page.user_count, 0)
        # Exactly one paste dispatched: no stacked insertion, no typing.
        self.assertEqual(
            [k for k in editor.pressed if k == "ControlOrMeta+V"],
            ["ControlOrMeta+V"],
        )
        self.assertEqual(editor.sequentially_calls, 0)
        self.assertEqual(page.exec_calls, 0)
        # The composer was NEVER cleared after the timeout.
        self.assertNotIn("ControlOrMeta+A", editor.pressed)
        # The timeout logged UNKNOWN, not a blind retry.
        self.assertFalse(self._events_of("browser.prompt_insert_retry"))

    def test_duplicated_prompt_rejected(self):
        # expected "ABC", actual "ABCABC" must FAIL verification.
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        editor.mode = "duplicate"
        browser, clock = fast_browser(page, deadline=1.0)

        with self.assertRaises(main_mod.ChatGPTInputError):
            browser._send_prompt("ABC")
        self.assertIsNone(editor.sent)
        self.assertEqual(page.user_count, 0)
        self.assertEqual(send.click_calls, 0)
        self.assertEqual(self._events_of("browser.prompt_insert_verified"), [])
        self.assertTrue(self._events_of("browser.prompt_insert_retry"))

    def test_partial_prompt_rejected(self):
        # expected "ABCDEFGHIJ", actual "ABC" (stable, stopped growing) FAILS.
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        editor.mode = "partial"
        browser, clock = fast_browser(page, deadline=1.0)

        with self.assertRaises(main_mod.ChatGPTInputError):
            browser._send_prompt("ABCDEFGHIJ")
        self.assertIsNone(editor.sent)
        self.assertEqual(page.user_count, 0)
        self.assertEqual(send.click_calls, 0)
        self.assertEqual(self._events_of("browser.prompt_insert_verified"), [])
        failing = self._events_of("browser.prompt_insert_waiting")
        self.assertTrue(any(kw.get("state") == "FAILED" for kw in failing))

    def test_unrelated_old_text_rejected(self):
        # expected "My request", actual "old My request" must FAIL.
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        editor.mode = "old_text"
        browser, clock = fast_browser(page, deadline=1.0)

        with self.assertRaises(main_mod.ChatGPTInputError):
            browser._send_prompt("My request")
        self.assertIsNone(editor.sent)
        self.assertEqual(page.user_count, 0)
        self.assertEqual(send.click_calls, 0)
        self.assertEqual(self._events_of("browser.prompt_insert_verified"), [])
        self.assertTrue(self._events_of("browser.prompt_insert_retry"))

    def test_exact_stable_prompt_verified(self):
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        browser, clock = fast_browser(page, deadline=5.0)

        browser._send_prompt("exactly this text")

        self.assertEqual(editor.sent, "exactly this text")
        self.assertEqual(page.user_count, 1)
        self.assertGreater(send.click_calls, 0)
        # Exact match must be observed (>=2 consecutive reads) before verify.
        self.assertEqual(len(self._events_of("browser.prompt_insert_observed")), 1)
        self.assertEqual(len(self._events_of("browser.prompt_insert_verified")), 1)
        self.assertGreaterEqual(editor.reads, 3)
        self.assertEqual(self._events_of("browser.prompt_submit_blocked"), [])

    def test_verify_before_submit_blocks_on_late_change(self):
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        # Reads: 1 = _send_prompt probe, 2-3 = exact verify (VERIFIED_EXACT),
        # 4 = the immediate pre-submit re-read, which now shows ChatGPT's
        # "Pasted text(...)" helper chip appended after verification.
        editor.read_overrides = {4: "make a cake\n\nPasted text(0123).txt"}
        browser, clock = fast_browser(page, deadline=5.0)

        with self.assertRaises(main_mod.ChatGPTInputError) as ctx:
            browser._send_prompt("make a cake")
        self.assertIn("refusing to send", str(ctx.exception))
        self.assertIsNone(editor.sent)
        self.assertEqual(page.user_count, 0)
        self.assertEqual(send.click_calls, 0)
        blocked = self._events_of("browser.prompt_submit_blocked")
        self.assertEqual(len(blocked), 1)
        self.assertEqual(
            blocked[0]["reason"],
            "pre-submit exact-match verification failed",
        )


class TestPromptDualModeInsertion(unittest.TestCase):
    """Pasted-text attachment mode: large pastes become attachments.

    ChatGPT turns huge pastes into a "Pasted text" document instead of
    inline composer text.  Verification is a separate, weaker state
    (VERIFIED_ATTACHMENT_READY: tile exists + sendable + label association),
    submission is gated on the state still holding immediately before the
    click, and a processing attachment is NEVER cleared, repasted, or used
    to justify a second insertion.

    This class covers the BACKSTOP layer: the paste genuinely converts and
    the verifier/submission path must stay correct.  The pre-flight size
    refusal in _send_prompt is an earlier, separate layer that normally
    stops such a prompt before it is ever pasted, so it is disabled here to
    let these tests reach the verifier.  The pre-flight itself is covered by
    TestPromptTooLargePreflight.
    """

    def setUp(self):
        self.events = []
        original = main_mod._log_event

        def spy(event, **kw):
            self.events.append((event, kw))

        main_mod._log_event = spy
        self._original_log_event = original
        self.addCleanup(
            lambda: setattr(main_mod, "_log_event", self._original_log_event)
        )

        # Let the oversized prompts below actually reach the paste/verifier.
        original_limit = main_mod.MAX_INLINE_PROMPT_CHARS
        main_mod.MAX_INLINE_PROMPT_CHARS = 0
        self.addCleanup(
            lambda: setattr(
                main_mod, "MAX_INLINE_PROMPT_CHARS", original_limit
            )
        )

    def _make_page(self, editor, preexisting_tiles=None):
        editor.native_fill_ok = False
        editor.submit_handler = lambda: self.page.submit(editor)
        send_button = FakeElement(
            'button[data-testid="send-button"]', "BUTTON", None, visible=True
        )
        send_button.is_send_button = True
        send_button.submit_handler = lambda: self.page.submit(editor)
        page = FakePage({
            "#prompt-textarea": [editor],
            'div[contenteditable="true"]': [editor],
            '[contenteditable="true"]': [editor],
            "textarea": [],
            '[role="textbox"]': [editor],
            'button[data-testid="send-button"]': [send_button],
        })
        for tile in (preexisting_tiles or []):
            tile.page = page
            page.attachments.append(tile)
        self.page = page
        editor.page = page
        editor.paste_handler = lambda: page.paste_into(editor)
        return editor, page, send_button

    def _events_of(self, name):
        return [kw for (evt, kw) in self.events if evt == name]

    def test_large_paste_attachment_readies_and_sends(self):
        # (1) Large paste -> attachment tile created -> VERIFIED_ATTACHMENT_READY
        #     -> exactly ONE send of the attachment (not a composed message).
        editor = PromptEditor("#prompt-textarea")
        editor.mode = "attachment"
        editor, page, send = self._make_page(editor)
        prompt = "WM_PROBE_9f81 :: " + ("x" * 70000)
        browser, clock = fast_browser(page, deadline=5.0)

        browser._send_prompt(prompt)

        self.assertEqual(page.user_count, 1)
        self.assertGreater(send.click_calls, 0)
        submitted = self._events_of("browser.submitted_content")
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["mode"], "ATTACHMENT")
        self.assertTrue(submitted[0]["attachment_ready"])
        self.assertFalse(submitted[0]["exact_match"])
        verified = self._events_of("browser.prompt_insert_verified")
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["mode"], "ATTACHMENT")
        self.assertEqual(verified[0]["state"], "VERIFIED_ATTACHMENT_READY")
        # Exactly one paste dispatch - no escalation while attaching.
        self.assertEqual(
            [k for k in editor.pressed if k == "ControlOrMeta+V"],
            ["ControlOrMeta+V"],
        )
        self.assertEqual(editor.sequentially_calls, 0)
        # Grounding companion prompt inserted alongside attachment tile.
        companion_events = self._events_of("browser.attachment_companion_inserted")
        self.assertEqual(len(companion_events), 1)
        self.assertGreater(companion_events[0]["companion_len"], 0)
        self.assertEqual(page.exec_calls, 0)

    def test_long_processing_never_repastes(self):
        # (2) The attachment keeps processing well past 7.5s: still NO
        #     second paste, NO clear, NO typing fallback, NO failure retry.
        editor = PromptEditor("#prompt-textarea")
        editor.mode = "attachment"
        editor.attachment_processing_reads = 1000   # never becomes ready
        editor, page, send = self._make_page(editor)
        browser, clock = fast_browser(page, deadline=1.0)

        with self.assertRaises(main_mod.ChatGPTInputError) as ctx:
            browser._send_prompt("WM_LIVE_7f3a :: " + ("y" * 60000))
        self.assertIn("UNKNOWN", str(ctx.exception))
        self.assertEqual(page.user_count, 0)
        self.assertIsNone(editor.sent)
        self.assertEqual(
            [k for k in editor.pressed if k == "ControlOrMeta+V"],
            ["ControlOrMeta+V"],
        )
        self.assertEqual(editor.sequentially_calls, 0)
        self.assertEqual(page.exec_calls, 0)
        self.assertNotIn("ControlOrMeta+A", editor.pressed)
        self.assertFalse(self._events_of("browser.prompt_insert_retry"))
        probes = self._events_of("browser.attachment_probe")
        self.assertTrue(any(kw.get("state") == "PROCESSING" for kw in probes))
        self.assertTrue(all(kw.get("state") != "READY" for kw in probes))

    def test_processing_then_ready_transition(self):
        # (3) PROCESSING -> READY: the verifier waits through LOADING
        #     (aria-disabled=true) and only sends once the button enables.
        editor = PromptEditor("#prompt-textarea")
        editor.mode = "attachment"
        editor.attachment_processing_reads = 3
        editor, page, send = self._make_page(editor)
        browser, clock = fast_browser(page, deadline=5.0)

        browser._send_prompt("WM_READY_aa11 :: " + ("z" * 90000))

        self.assertEqual(page.user_count, 1)
        probes = self._events_of("browser.attachment_probe")
        states = [kw["state"] for kw in probes]
        self.assertIn("PROCESSING", states)
        self.assertIn("READY", states)
        self.assertLess(states.index("PROCESSING"), states.index("READY"))
        submitted = self._events_of("browser.submitted_content")
        self.assertTrue(submitted[0]["attachment_ready"])
        self.assertEqual(
            [k for k in editor.pressed if k == "ControlOrMeta+V"],
            ["ControlOrMeta+V"],
        )

    def test_existing_unrelated_tile_plus_new_attachment(self):
        # (4) A stale tile from a previous action must NOT be mistaken for
        #     this send's attachment; only the newly-created tile counts.
        editor = PromptEditor("#prompt-textarea")
        editor.mode = "attachment"
        stale = FakeTile("old-research.doc")
        editor, page, send = self._make_page(editor, preexisting_tiles=[stale])
        browser, clock = fast_browser(page, deadline=5.0)

        browser._send_prompt("WM_NEW_42a :: " + ("q" * 50000))

        self.assertEqual(page.user_count, 1)
        submitted = self._events_of("browser.submitted_content")
        self.assertEqual(submitted[0]["mode"], "ATTACHMENT")
        ready_probes = [
            kw for kw in self._events_of("browser.attachment_probe")
            if kw["state"] == "READY"
        ]
        self.assertTrue(ready_probes)
        probe = ready_probes[0]
        self.assertTrue(probe["new_tile_detected"])
        self.assertTrue(probe["association_match"])
        self.assertEqual(probe["tile_count_before"], 1)

    def test_attachment_disappears_before_submit_blocks(self):
        # (5) READY during verification, but the tile vanishes before the
        #     pre-submit re-probe: the send MUST be refused.
        editor = PromptEditor("#prompt-textarea")
        editor.mode = "attachment"
        editor.attachment_self_removing = True
        editor, page, send = self._make_page(editor)
        browser, clock = fast_browser(page, deadline=5.0)

        with self.assertRaises(main_mod.ChatGPTInputError) as ctx:
            browser._send_prompt("WM_GHOST_5c :: " + ("w" * 45000))
        self.assertIn("refusing to send", str(ctx.exception))
        self.assertEqual(page.user_count, 0)
        self.assertIsNone(editor.sent)
        self.assertEqual(send.click_calls, 0)
        blocked = self._events_of("browser.prompt_submit_blocked")
        self.assertEqual(len(blocked), 1)
        self.assertIn("attachment", blocked[0]["reason"].lower())

    def test_two_new_attachment_tiles_blocked(self):
        # (6) ChatGPT creating TWO tiles for one paste is ambiguous: never
        #     submit, fail safely (UNKNOWN), never treat as READY.
        editor = PromptEditor("#prompt-textarea")
        editor.mode = "attachment"
        editor.attachment_tile_count = 2
        editor, page, send = self._make_page(editor)
        browser, clock = fast_browser(page, deadline=1.0)

        with self.assertRaises(main_mod.ChatGPTInputError) as ctx:
            browser._send_prompt("WM_DUP_77 :: " + ("d" * 60000))
        self.assertIn("UNKNOWN", str(ctx.exception))
        self.assertEqual(page.user_count, 0)
        self.assertIsNone(editor.sent)
        self.assertEqual(send.click_calls, 0)
        probes = self._events_of("browser.attachment_probe")
        self.assertFalse(any(kw["state"] == "READY" for kw in probes))
        self.assertTrue(any(kw.get("new_tile_detected") for kw in probes))

    def test_inline_small_prompt_verified_exact_and_sends(self):
        # (7) Small prompts still take the INLINE path: VERIFIED_EXACT send.
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        browser, clock = fast_browser(page, deadline=5.0)

        browser._send_prompt("short inline request")

        self.assertEqual(editor.sent, "short inline request")
        self.assertEqual(page.user_count, 1)
        verified = self._events_of("browser.prompt_insert_verified")
        self.assertEqual(verified[0]["mode"], "INLINE")
        self.assertEqual(verified[0]["state"], "VERIFIED_EXACT")
        submitted = self._events_of("browser.submitted_content")
        self.assertEqual(submitted[0]["mode"], "INLINE")
        self.assertTrue(submitted[0]["exact_match"])

    def test_duplicated_inline_content_rejected(self):
        # (8) Duplicated inline content ("ABCABC") destroys the exact
        #     match: no send, and the failure is retryable by the server
        #     handoff (ChatGPTInputError with "could not verify").
        editor = PromptEditor("#prompt-textarea")
        editor.mode = "duplicate"
        editor, page, send = self._make_page(editor)
        browser, clock = fast_browser(page, deadline=1.0)

        with self.assertRaises(main_mod.ChatGPTInputError) as ctx:
            browser._send_prompt("ABC")
        self.assertIsNone(editor.sent)
        self.assertEqual(page.user_count, 0)
        self.assertEqual(send.click_calls, 0)
        self.assertIn("could not verify", str(ctx.exception))
        self.assertEqual(self._events_of("browser.prompt_insert_verified"), [])
        self.assertTrue(self._events_of("browser.prompt_insert_retry"))

    def test_pre_submit_inline_change_blocked(self):
        # (9) EXACT verified, but the composer mutates between verification
        #     and the immediate pre-submit re-read: send must be refused.
        editor = PromptEditor("#prompt-textarea")
        editor, page, send = self._make_page(editor)
        editor.read_overrides = {4: "make a cake\n\nPasted text(0123).txt"}
        browser, clock = fast_browser(page, deadline=5.0)

        with self.assertRaises(main_mod.ChatGPTInputError) as ctx:
            browser._send_prompt("make a cake")
        self.assertIn("refusing to send", str(ctx.exception))
        self.assertEqual(page.user_count, 0)
        self.assertEqual(send.click_calls, 0)
        blocked = self._events_of("browser.prompt_submit_blocked")
        self.assertEqual(len(blocked), 1)


class TestPromptTooLargePreflight(unittest.TestCase):
    """The inline-limit backstop and the submission-mode contract.

    A pasted-text ATTACHMENT means ChatGPT received a FILE rather than the
    conversation, and answers as an outside reader of that file.  The bug
    this guards was silent: the submission reported success and a confident
    wrong answer ("the material available in this turn contains the
    transcript, not the repository") reached the client as if it were the
    agent's own.

    Measurement showed the conversion is CONTENT-DEPENDENT, not
    length-dependent, so these tests pin the two dependable behaviours:
      * a prompt over the ceiling is refused BEFORE any paste, and
      * an ATTACHMENT submission is refused rather than answered.
    """

    def setUp(self):
        self.events = []
        original = main_mod._log_event
        main_mod._log_event = lambda event, **kw: self.events.append((event, kw))
        self.addCleanup(
            lambda: setattr(main_mod, "_log_event", original)
        )

        original_limit = main_mod.MAX_INLINE_PROMPT_CHARS
        self.addCleanup(
            lambda: setattr(main_mod, "MAX_INLINE_PROMPT_CHARS", original_limit)
        )
        main_mod.MAX_INLINE_PROMPT_CHARS = 1000

    def _events_of(self, name):
        return [kw for (evt, kw) in self.events if evt == name]

    def _inline_page(self):
        editor = FakeElement("#prompt-textarea", "DIV", "true")
        editor.native_fill_ok = False
        editor.submit_handler = lambda: self.page.submit(editor)
        send = FakeElement(
            'button[data-testid="send-button"]', "BUTTON", None, visible=True
        )
        send.is_send_button = True
        send.submit_handler = lambda: self.page.submit(editor)
        page = FakePage({
            "#prompt-textarea": [editor],
            'div[contenteditable="true"]': [editor],
            '[contenteditable="true"]': [editor],
            "textarea": [],
            '[role="textbox"]': [editor],
            'button[data-testid="send-button"]': [send],
        })
        editor.paste_handler = lambda: page.paste_into(editor)
        self.page = page
        return editor, page, send

    def test_oversized_prompt_refused_before_any_paste(self):
        # The whole point of the pre-flight: no paste, no file tile, no
        # ChatGPT generation, and a deterministic error.
        editor, page, send = self._inline_page()
        browser = make_browser(page)

        with self.assertRaises(main_mod.PromptTooLargeError) as ctx:
            browser._send_prompt("x" * 1001)

        self.assertIn("PROMPT_TOO_LARGE", str(ctx.exception))
        self.assertFalse(page.clipboard, "must not paste an oversized prompt")
        self.assertEqual(editor.pressed, [], "must not touch the composer")
        self.assertEqual(page.user_count, 0, "must not submit")
        self.assertEqual(send.click_calls, 0)
        self.assertEqual(page.attachments, [], "must not create a file tile")
        self.assertEqual(len(self._events_of("browser.prompt_too_large")), 1)

    def test_prompt_at_the_ceiling_is_still_allowed(self):
        # Boundary is inclusive: exactly at the limit must go through, or a
        # prompt could never legitimately reach the ceiling.
        editor, page, send = self._inline_page()
        browser = make_browser(page)

        mode = browser._send_prompt("x" * 1000)

        self.assertEqual(mode, "INLINE")
        self.assertEqual(page.user_count, 1)
        self.assertEqual(self._events_of("browser.prompt_too_large"), [])

    def test_pre_flight_is_skipped_when_limit_disabled(self):
        # MAX_INLINE_PROMPT_CHARS=0 disables the backstop entirely; used by
        # the measurement harness and by the attachment backstop tests.
        main_mod.MAX_INLINE_PROMPT_CHARS = 0
        editor, page, send = self._inline_page()
        browser = make_browser(page)

        mode = browser._send_prompt("x" * 5000)

        self.assertEqual(mode, "INLINE")
        self.assertEqual(page.user_count, 1)
        self.assertEqual(self._events_of("browser.prompt_too_large"), [])

    def test_budget_warning_near_ceiling_does_not_block(self):
        # The warning is a heads-up for diagnosis, NOT a refusal: the
        # conversion is unpredictable, so blocking here would just throw
        # away working turns.
        editor, page, send = self._inline_page()
        browser = make_browser(page)
        warn_at = int(main_mod.PROMPT_BUDGET_WARN_RATIO * 1000)

        mode = browser._send_prompt("x" * (warn_at + 10))

        self.assertEqual(mode, "INLINE")
        self.assertEqual(page.user_count, 1, "must still submit")
        warned = self._events_of("browser.prompt_budget_exceeded")
        self.assertEqual(len(warned), 1)
        self.assertEqual(warned[0]["max_inline"], 1000)
        self.assertEqual(self._events_of("browser.prompt_too_large"), [])

    def test_no_budget_warning_well_below_ceiling(self):
        editor, page, send = self._inline_page()
        browser = make_browser(page)

        browser._send_prompt("x" * 100)

        self.assertEqual(self._events_of("browser.prompt_budget_exceeded"), [])

    def test_response_carries_submission_mode(self):
        # The mode has to survive on the Response so a caller can never
        # mistake an attachment answer for a real one.
        self.assertEqual(main_mod.Response("hi").mode, "INLINE")
        self.assertEqual(main_mod.Response("hi", mode="ATTACHMENT").mode, "ATTACHMENT")

    def _stub_browser_for_ask(self, mode):
        editor, page, send = self._inline_page()
        browser = make_browser(page)
        for name in (
            "_ensure_conversation_chat",
            "_rotate_chat_if_oversized",
            "_dismiss_rate_limit_modal",
        ):
            setattr(browser, name, lambda *a, **k: None)
        browser._send_prompt = lambda prompt: mode
        return browser

    def test_ask_refuses_an_attachment_answer(self):
        # The regression itself: this used to return the document summary
        # to the client as if it were the agent's answer.
        browser = self._stub_browser_for_ask("ATTACHMENT")
        browser.allow_attachment = False
        reached = []
        browser._wait_for_response = lambda *a, **k: reached.append("waited")

        with self.assertRaises(main_mod.PromptTooLargeError) as ctx:
            browser._ask_impl("please inspect the repo")

        self.assertIn("PROMPT_TOO_LARGE", str(ctx.exception))
        self.assertEqual(reached, [], "must not wait on an untrusted answer")
        self.assertEqual(len(self._events_of("browser.attachment_answer_refused")), 1)

    def test_ask_stream_refuses_before_yielding_any_delta(self):
        browser = self._stub_browser_for_ask("ATTACHMENT")
        browser.allow_attachment = False
        reached = []
        browser._stream_response = lambda *a, **k: iter(reached.append("streamed"))

        with self.assertRaises(main_mod.PromptTooLargeError):
            list(browser._ask_stream_impl("please inspect the repo"))

        self.assertEqual(reached, [], "must not stream a document summary")
        refused = self._events_of("browser.attachment_answer_refused")
        self.assertEqual(len(refused), 1)
        self.assertTrue(refused[0]["is_stream"])

    def test_ask_accepts_attachment_answer_when_allowed(self):
        browser = self._stub_browser_for_ask("ATTACHMENT")
        browser.allow_attachment = True
        browser._wait_for_response = lambda *a, **k: "agent response"
        resp = browser._ask_impl("please inspect the repo")
        self.assertEqual(resp.text, "agent response")
        self.assertEqual(resp.mode, "ATTACHMENT")
        self.assertEqual(len(self._events_of("browser.attachment_answer_accepted")), 1)

    def test_ask_stream_accepts_attachment_when_allowed(self):
        browser = self._stub_browser_for_ask("ATTACHMENT")
        browser.allow_attachment = True
        browser._stream_response = lambda *a, **k: iter(["delta1", "delta2"])
        deltas = list(browser._ask_stream_impl("please inspect the repo"))
        self.assertEqual(deltas, ["delta1", "delta2"])
        accepted = self._events_of("browser.attachment_answer_accepted")
        self.assertEqual(len(accepted), 1)
        self.assertTrue(accepted[0]["is_stream"])


class TestPromptNormalizationAndMatching(unittest.TestCase):
    def test_normalize_prompt_cleans_zero_width_characters(self):
        text = "Hello\u200bWorld\ufeff\u200cTest\u2060"
        normalized = main_mod._normalize_prompt(text)
        self.assertEqual(normalized, "Hello World Test")

    def test_normalize_prompt_cleans_non_breaking_spaces(self):
        text = "Hello\xa0World \t \n Test"
        normalized = main_mod._normalize_prompt(text)
        self.assertEqual(normalized, "Hello World Test")

    def test_is_prompt_match_exact(self):
        self.assertTrue(main_mod._is_prompt_match("hello world", "hello world"))
        self.assertTrue(main_mod._is_prompt_match("  hello   world  ", "hello world"))

    def test_is_prompt_match_with_prosemirror_zero_width_difference(self):
        base = "A" * 100
        with_artifact = base + "\u200b"
        self.assertTrue(main_mod._is_prompt_match(with_artifact, base))

    def test_is_prompt_match_tolerance_for_trailing_breaks(self):
        base = "You are continuing a multi-turn tool loop on the live workspace."
        with_extra_space = base + " "
        self.assertTrue(main_mod._is_prompt_match(with_extra_space, base))

    def test_is_prompt_match_rejects_actual_differences(self):
        self.assertFalse(main_mod._is_prompt_match("hello", "goodbye"))
        self.assertFalse(main_mod._is_prompt_match("ABC" * 30, "XYZ" * 30))
        self.assertFalse(main_mod._is_prompt_match("", "something"))


if __name__ == "__main__":
    unittest.main()
