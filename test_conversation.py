"""Unit tests for per-conversation ChatGPT chat isolation.

Two layers are covered without any browser or network:

  1. server._conversation_id -- the stable id derived from a request's
     first human message, used to tell one client conversation from
     another across the harness's full-history replays.
  2. main.ChatGPTBrowser._ensure_conversation_chat -- the rotation that
     opens a fresh ChatGPT chat when the incoming id changes, and only
     then.
"""

import inspect
import os
import tempfile
import time
import unittest
from unittest import mock

from main import ChatGPTBrowser, CHATGPT_URL
from main import _deadline_value, _deadline_wall
from protocol import build_prompt
import server
from server import ChatRequest, _conversation_id, _record_turn_completed, webmind


def make_request(messages, user=None):
    return ChatRequest(
        model="webmind",
        messages=messages,
        stream=True,
        user=user,
    )


MAIN_TURN_1 = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "Create smoke_hello.txt"},
]
MAIN_TURN_2 = MAIN_TURN_1 + [
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
]
TITLE_REQUEST = [
    {"role": "system", "content": "You generate session titles."},
    {"role": "user", "content": "Generate the session title from this JSON: [...]"},
]


class ConversationIdTests(unittest.TestCase):
    def setUp(self):
        self._orig_persistent = os.environ.get("WEBMIND_PERSISTENT_CHAT")
        os.environ["WEBMIND_PERSISTENT_CHAT"] = "0"
        with server._session_lock:
            server._active_session_id = None
            server._active_opener_hash = None
            server._session_generation = 0
            server._session_turns_completed.clear()

    def tearDown(self):
        if self._orig_persistent is None:
            os.environ.pop("WEBMIND_PERSISTENT_CHAT", None)
        else:
            os.environ["WEBMIND_PERSISTENT_CHAT"] = self._orig_persistent
        with server._session_lock:
            server._active_session_id = None
            server._active_opener_hash = None
            server._session_generation = 0
            server._session_turns_completed.clear()

    def test_stable_across_history_replay(self):
        first = _conversation_id(make_request(MAIN_TURN_1))
        second = _conversation_id(make_request(MAIN_TURN_2))
        self.assertIsNotNone(first)
        self.assertEqual(first, second)

    def test_different_opener_is_different_conversation(self):
        main = _conversation_id(make_request(MAIN_TURN_1))
        reply = _conversation_id(make_request(MAIN_TURN_2))
        other = _conversation_id(make_request([
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Read README.md"},
        ]))
        self.assertNotEqual(main, other)
        self.assertEqual(main, reply)

    def test_title_request_is_its_own_conversation(self):
        main = _conversation_id(make_request(MAIN_TURN_1))
        title = _conversation_id(make_request(TITLE_REQUEST))
        self.assertNotEqual(main, title)

    def test_repeated_opener_after_turn_completed_advances_generation(self):
        req1 = make_request([{"role": "user", "content": "Build an FPS game"}])
        cid1 = _conversation_id(req1)
        _record_turn_completed(cid1)

        # New session starts with identical opener prompt
        req2 = make_request([{"role": "user", "content": "Build an FPS game"}])
        cid2 = _conversation_id(req2)
        self.assertNotEqual(cid1, cid2, "Repeated opener after completed turn must produce a new session id")

    def test_explicit_session_header(self):
        req = make_request(MAIN_TURN_1)
        fake_raw = mock.Mock()
        fake_raw.headers = {"x-session-id": "session-xyz-123"}
        cid = _conversation_id(req, raw_request=fake_raw)
        self.assertIn("conv:sess_", cid)

    def test_explicit_user_parameter(self):
        req = make_request(MAIN_TURN_1, user="task-session-abc")
        cid = _conversation_id(req)
        self.assertIn("conv:user_", cid)

    def test_no_human_message_returns_none(self):
        req = make_request([
            {"role": "system", "content": "You are a coding agent."},
            {"role": "assistant", "content": "hello"},
        ])
        self.assertIsNone(_conversation_id(req))

    def test_empty_messages_returns_none(self):
        self.assertIsNone(_conversation_id(make_request([])))

    def test_rate_limit_modal_is_dismissed(self):
        page = _FakeRateLimitPage()
        browser = make_browser(page)
        self.assertTrue(browser._dismiss_rate_limit_modal())
        self.assertTrue(page.dialog.button.clicked)



class PersistentChatTests(unittest.TestCase):
    def setUp(self):
        self._orig_persistent = os.environ.get("WEBMIND_PERSISTENT_CHAT")
        os.environ["WEBMIND_PERSISTENT_CHAT"] = "1"
        with server._session_lock:
            server._active_session_id = None
            server._active_opener_hash = None
            server._session_generation = 0
            server._session_turns_completed.clear()

    def tearDown(self):
        if self._orig_persistent is None:
            os.environ.pop("WEBMIND_PERSISTENT_CHAT", None)
        else:
            os.environ["WEBMIND_PERSISTENT_CHAT"] = self._orig_persistent
        with server._session_lock:
            server._active_session_id = None
            server._active_opener_hash = None
            server._session_generation = 0
            server._session_turns_completed.clear()

    def test_persistent_chat_maintains_same_session_across_turns(self):
        req1 = make_request([{"role": "user", "content": "Write a python script"}])
        cid1 = _conversation_id(req1)
        self.assertIsNotNone(cid1)
        _record_turn_completed(cid1)

        req2 = make_request([{"role": "user", "content": "Now add tests for it"}])
        cid2 = _conversation_id(req2)
        self.assertEqual(cid1, cid2, "Persistent chat must keep the same session id across turns")

    def test_persistent_chat_shields_auxiliary_title_request(self):
        req1 = make_request(MAIN_TURN_1)
        cid1 = _conversation_id(req1)
        _record_turn_completed(cid1)

        title_req = make_request(TITLE_REQUEST)
        cid_title = _conversation_id(title_req)
        self.assertEqual(cid1, cid_title, "Auxiliary title request must reuse the active chat session")

    def test_explicit_reset_header_forces_new_chat(self):
        req1 = make_request(MAIN_TURN_1)
        cid1 = _conversation_id(req1)
        _record_turn_completed(cid1)

        fake_raw = mock.Mock()
        fake_raw.headers = {"x-reset-chat": "true"}
        req2 = make_request([{"role": "user", "content": "Start something completely new"}])
        cid2 = _conversation_id(req2, raw_request=fake_raw)
        self.assertNotEqual(cid1, cid2, "x-reset-chat must force a new conversation generation")

class _FakePage:
    def __init__(self, fail=False):
        self.goto_calls = []
        self.waits = []
        self._fail = fail

    def goto(self, url, wait_until=None):
        if self._fail:
            raise RuntimeError("navigation failed")
        self.goto_calls.append((url, wait_until))

    def wait_for_timeout(self, ms):
        self.waits.append(ms)


class _FakeRateLimitButton:
    def __init__(self):
        self.clicked = False

    def click(self):
        self.clicked = True


class _FakeRateLimitDialog:
    def __init__(self):
        self.button = _FakeRateLimitButton()

    def inner_text(self):
        return "Too many requests\nYou are making requests too quickly."

    def get_by_role(self, role, name):
        return self.button


class _FakeRateLimitDialogs:
    def __init__(self, dialog):
        self.dialog = dialog

    def count(self):
        return 1

    def nth(self, index):
        return self.dialog


class _FakeRateLimitPage:
    def __init__(self):
        self.dialog = _FakeRateLimitDialog()

    def locator(self, selector):
        return _FakeRateLimitDialogs(self.dialog)


def make_browser(page, conversation_id=None):
    browser = ChatGPTBrowser.__new__(ChatGPTBrowser)
    browser.page = page
    browser._conversation_id = conversation_id
    return browser


class _FakeTextNode:
    def __init__(self, text):
        self._text = text

    def text_content(self):
        return self._text


class _FakeMdLocator:
    def __init__(self, nodes):
        self._nodes = nodes

    def count(self):
        return len(self._nodes)

    @property
    def first(self):
        return self._nodes[0] if self._nodes else None

    @property
    def last(self):
        return self._nodes[-1] if self._nodes else None

    def nth(self, i):
        return self._nodes[i]

    def __getitem__(self, i):
        return self._nodes[i]


class _FakeMsgNode:
    def __init__(self, markdown_nodes, body_text):
        self._md = _FakeMdLocator(markdown_nodes)
        self._body = body_text

    def locator(self, selector):
        return self._md

    def text_content(self):
        return self._body


class _FakeMsgLocator:
    def __init__(self, message_nodes):
        if isinstance(message_nodes, list):
            self._messages = message_nodes
        elif message_nodes is None:
            self._messages = []
        else:
            self._messages = [message_nodes]

    def count(self):
        return len(self._messages)

    @property
    def first(self):
        return self._messages[0] if self._messages else None

    @property
    def last(self):
        return self._messages[-1] if self._messages else None

    def nth(self, i):
        return self._messages[i]

    def __getitem__(self, i):
        return self._messages[i]


class _FakeCapturePage:
    """ChatGPT message DOM with a markdown body and action chrome."""

    def __init__(self, message_node):
        self._message = message_node

    def locator(self, selector):
        if selector in (
            '[data-markdown-text-style="assistant-message"]',
            '[data-conversation-role="assistant"]',
            '[data-message-author-role="assistant"]',
        ):
            return _FakeMsgLocator(self._message)
        return _FakeMsgLocator(None)


class AssistantMessageCaptureTests(unittest.TestCase):
    def test_markdown_body_preferred_no_layout_newlines(self):
        chrome = "Copy code\nRegenerate"
        md_text = "path=index.html;\nhtml=html.Replace('a','b');"
        page = _FakeCapturePage(
            _FakeMsgNode([_FakeTextNode(md_text)], md_text + "\n" + chrome)
        )
        browser = make_browser(page)
        self.assertEqual(
            browser._get_assistant_message(),
            "path=index.html;\nhtml=html.Replace('a','b');",
        )

    def test_no_markdown_wrapper_falls_back_to_message_text(self):
        body = "plain answer with no code block"
        page = _FakeCapturePage(_FakeMsgNode([], body))
        browser = make_browser(page)
        self.assertEqual(browser._get_assistant_message(), body)

    def test_fallback_reads_text_content_not_layout(self):
        # Fallback reads textContent of the whole message, which NEVER
        # injects a block-boundary newline per character the way innerText
        # does for ChatGPT's fine-grained stream/edit nodes.
        page = _FakeCapturePage(
            _FakeMsgNode([], "path=index.html;Set-Content -Path index.html")
        )
        browser = make_browser(page)
        self.assertEqual(
            browser._get_assistant_message(),
            "path=index.html;Set-Content -Path index.html",
        )

    def test_multiple_markdown_blocks_in_single_message_joined(self):
        md1 = _FakeTextNode("First markdown block.")
        md2 = _FakeTextNode("Second markdown block.")
        page = _FakeCapturePage(
            _FakeMsgNode([md1, md2], "First markdown block.\nSecond markdown block.")
        )
        browser = make_browser(page)
        self.assertEqual(
            browser._get_assistant_message(),
            "First markdown block.\n\nSecond markdown block.",
        )

    def test_multiple_assistant_messages_since_count_joined(self):
        msg1 = _FakeMsgNode([_FakeTextNode("Bubble 1 content")], "Bubble 1 content")
        msg2 = _FakeMsgNode([_FakeTextNode("Bubble 2 content")], "Bubble 2 content")
        msg3 = _FakeMsgNode([_FakeTextNode("Bubble 3 content")], "Bubble 3 content")
        page = _FakeCapturePage([msg1, msg2, msg3])
        browser = make_browser(page)
        self.assertEqual(
            browser._get_assistant_message(since_count=0),
            "Bubble 1 content\n\nBubble 2 content\n\nBubble 3 content",
        )

    def test_multiple_assistant_messages_since_count_partial(self):
        msg_old = _FakeMsgNode([_FakeTextNode("Prior turn answer")], "Prior turn answer")
        msg1 = _FakeMsgNode([_FakeTextNode("Turn 2 bubble 1")], "Turn 2 bubble 1")
        msg2 = _FakeMsgNode([_FakeTextNode("Turn 2 bubble 2")], "Turn 2 bubble 2")
        page = _FakeCapturePage([msg_old, msg1, msg2])
        browser = make_browser(page)
        self.assertEqual(
            browser._get_assistant_message(since_count=1),
            "Turn 2 bubble 1\n\nTurn 2 bubble 2",
        )

    def test_assistant_messages_without_since_count_falls_back_to_last(self):
        msg1 = _FakeMsgNode([_FakeTextNode("Bubble 1")], "Bubble 1")
        msg2 = _FakeMsgNode([_FakeTextNode("Bubble 2")], "Bubble 2")
        page = _FakeCapturePage([msg1, msg2])
        browser = make_browser(page)
        self.assertEqual(
            browser._get_assistant_message(),
            "Bubble 2",
        )


class EnsureConversationChatTests(unittest.TestCase):
    def test_none_never_navigates(self):
        page = _FakePage()
        browser = make_browser(page)
        browser._ensure_conversation_chat(None)
        self.assertEqual(page.goto_calls, [])
        self.assertIsNone(browser._conversation_id)

    def test_first_conversation_opens_fresh_chat(self):
        page = _FakePage()
        browser = make_browser(page)
        browser._ensure_conversation_chat("conv:a")
        self.assertEqual(page.goto_calls, [(CHATGPT_URL, "domcontentloaded")])
        self.assertEqual(browser._conversation_id, "conv:a")

    def test_same_conversation_does_not_rotate(self):
        page = _FakePage()
        browser = make_browser(page, conversation_id="conv:a")
        browser._ensure_conversation_chat("conv:a")
        self.assertEqual(page.goto_calls, [])

    def test_different_conversation_rotates(self):
        page = _FakePage()
        browser = make_browser(page, conversation_id="conv:a")
        browser._ensure_conversation_chat("conv:b")
        self.assertEqual(page.goto_calls, [(CHATGPT_URL, "domcontentloaded")])
        self.assertEqual(browser._conversation_id, "conv:b")

    def test_rotation_failure_keeps_previous_id(self):
        page = _FakePage(fail=True)
        browser = make_browser(page, conversation_id="conv:a")
        browser._ensure_conversation_chat("conv:b")
        self.assertEqual(browser._conversation_id, "conv:a")


# ============================================================
# FIRST-ASSISTANT-MESSAGE WAIT (self-heal + diagnostics)
# ============================================================
# _wait_for_first_assistant_message waits for a NEW assistant message node.
# Its two behaviours worth pinning down browser-free:
#
#   1. When the timeout fires while the composer STILL holds our exact
#      prompt (the earlier submit never registered), it re-submits ONCE and
#      keeps waiting instead of failing the turn.
#   2. When the composer is consumed/empty and ChatGPT stays silent, it
#      raises TimeoutError whose message states whether ChatGPT ever showed
#      our turn as the last user message.

FAKE_WAIT_PROMPT = (
    "call the read tool on C:/repo/index.html, then continue with edit"
)


class _WaitCountLocator:
    def __init__(self, submitted):
        self._submitted = submitted

    def count(self):
        return 1 if self._submitted["value"] else 0


class _WaitFakePage:
    """Minimal page: assistant-node count goes 0 -> 1 once submitted."""

    def __init__(self):
        self.submitted = {"value": False}

    def locator(self, selector):
        return _WaitCountLocator(self.submitted)


def _patched_browser(page, diag, submit_recorder, send_prompt_recorder=None):
    browser = make_browser(page)
    browser._is_generating = lambda: False
    browser._idle_timeout_diagnostics = lambda prompt: diag
    browser._get_input = lambda: object()
    browser._submit_prompt = submit_recorder
    browser._send_prompt = send_prompt_recorder or (lambda prompt: None)
    return browser


class FirstAssistantMessageWaitTests(unittest.TestCase):

    def _patches(self):
        tmp_log = os.path.join(tempfile.gettempdir(), "webmind_test_runtime.log")
        return (
            mock.patch("main.RESPONSE_WAIT_TIMEOUT", 0.001),
            mock.patch("main.time.sleep", lambda _: None),
            mock.patch("main.RUNTIME_LOG_PATH", tmp_log),
        )

    def test_submit_never_registered_resubmits_once_then_proceeds(self):
        page = _WaitFakePage()
        submits = []

        def submit(*args, **kwargs):
            submits.append((args, kwargs))
            page.submitted["value"] = True

        diag = {"composer_still_holds_prompt": True}
        browser = _patched_browser(page, diag, submit)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            browser._wait_for_first_assistant_message(0, FAKE_WAIT_PROMPT)
        self.assertEqual(len(submits), 1)
        self.assertTrue(page.submitted["value"])

    def test_silent_turn_raises_with_diagnostics(self):
        page = _WaitFakePage()
        diag = {
            "composer_still_holds_prompt": False,
            "composer_len": 0,
            "last_user_hash": "aabbccdd",
            "prompt_hash": "aabbccdd",
        }
        browser = _patched_browser(page, diag, lambda *a, **k: None)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            with self.assertRaises(TimeoutError) as ctx:
                browser._wait_for_first_assistant_message(0, FAKE_WAIT_PROMPT)
        msg = str(ctx.exception)
        self.assertIn("ChatGPT did not produce a response", msg)
        self.assertIn("ChatGPT-showed-our-turn-as-last-user-message=yes", msg)
        self.assertIn("composer-empty (turn submitted)", msg)

    def test_resubmit_once_even_when_still_silent_then_raises(self):
        page = _WaitFakePage()
        submits = []
        diag = {"composer_still_holds_prompt": True}

        def submit(*args, **kwargs):
            submits.append(True)
            diag["composer_still_holds_prompt"] = False

        browser = _patched_browser(page, diag, submit)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            with self.assertRaises(TimeoutError):
                browser._wait_for_first_assistant_message(0, FAKE_WAIT_PROMPT)
        self.assertEqual(len(submits), 1)

    def test_eaten_submit_is_reinserted_from_scratch_once(self):
        # Submit consumed the composer but ChatGPT never rendered the turn
        # (last-user mismatch): the prompt is re-inserted from scratch and
        # submitted once instead of failing the turn.
        page = _WaitFakePage()
        sends = []

        def send_prompt(prompt):
            sends.append(prompt)
            page.submitted["value"] = True

        diag = {
            "composer_still_holds_prompt": False,
            "last_user_hash": "abc",
            "prompt_hash": "def",
        }
        browser = _patched_browser(
            page, diag, lambda *a, **k: None, send_prompt_recorder=send_prompt
        )
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            browser._wait_for_first_assistant_message(0, FAKE_WAIT_PROMPT)
        self.assertEqual(sends, [FAKE_WAIT_PROMPT])

    def test_eaten_submit_raises_when_resend_fails(self):
        page = _WaitFakePage()

        def send_prompt(prompt):
            raise RuntimeError("insertion failed")

        diag = {
            "composer_still_holds_prompt": False,
            "last_user_hash": "abc",
            "prompt_hash": "def",
        }
        browser = _patched_browser(
            page, diag, lambda *a, **k: None, send_prompt_recorder=send_prompt
        )
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            with self.assertRaises(TimeoutError):
                browser._wait_for_first_assistant_message(0, FAKE_WAIT_PROMPT)

    def test_rendered_user_message_without_reply_raises_no_resend(self):
        # Turn received (=yes) but ChatGPT silent: never re-insert, so a
        # slow-but-real reply can't be duplicated.
        page = _WaitFakePage()
        sends = []
        diag = {
            "composer_still_holds_prompt": False,
            "last_user_hash": "abc",
            "prompt_hash": "abc",
        }
        browser = _patched_browser(page, diag, lambda *a, **k: None,
                                   send_prompt_recorder=sends.append)
        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            with self.assertRaises(TimeoutError) as ctx:
                browser._wait_for_first_assistant_message(0, FAKE_WAIT_PROMPT)
        self.assertEqual(sends, [])
        self.assertIn("=yes", str(ctx.exception))

    def test_timeout_suffix_states(self):
        held = ChatGPTBrowser._timeout_suffix(
            {"composer_still_holds_prompt": True}
        )
        self.assertIn("submit never registered", held)
        received_yes = ChatGPTBrowser._timeout_suffix(
            {"composer_len": 0, "last_user_hash": "abc", "prompt_hash": "abc"}
        )
        self.assertIn("=yes", received_yes)
        received_no = ChatGPTBrowser._timeout_suffix(
            {"composer_len": 0, "last_user_hash": "abc", "prompt_hash": "def"}
        )
        self.assertIn("=no", received_no)
        unknown = ChatGPTBrowser._timeout_suffix({"composer_len": 5})
        self.assertIn("=unknown", unknown)

    def test_virtualized_dom_same_count_detected_by_content_change(self):
        # When ChatGPT scrolls and earlier messages unmount from the DOM,
        # message node count remains equal to old_assistant_count (e.g. 5 == 5).
        # Multi-signal detection must identify the new response from content change.
        page = _WaitFakePage()
        diag = {"composer_still_holds_prompt": False, "composer_len": 0}
        browser = _patched_browser(page, diag, lambda *a, **k: None)
        browser._locate_assistant_messages = lambda: mock.Mock(count=lambda: 5)
        browser._get_assistant_message = lambda: "New response from ChatGPT"

        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            # Must return cleanly (not raise TimeoutError)
            browser._wait_for_first_assistant_message(
                5, FAKE_WAIT_PROMPT, old_last_text="Old response from previous turn"
            )

    def test_was_generating_and_finished_detected_with_new_text(self):
        # Model was generating (stop button or thinking indicator), then
        # finished. Even if message count didn't increase due to virtualization,
        # the transition from generating -> idle with content returns immediately.
        page = _WaitFakePage()
        diag = {"composer_still_holds_prompt": False, "composer_len": 0}
        browser = _patched_browser(page, diag, lambda *a, **k: None)
        browser._locate_assistant_messages = lambda: mock.Mock(count=lambda: 4)
        browser._get_assistant_message = lambda: "Final answer"

        gen_states = [True, False]
        browser._is_generating = lambda: gen_states.pop(0) if gen_states else False

        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            browser._wait_for_first_assistant_message(
                4, FAKE_WAIT_PROMPT, old_last_text="Prior answer"
            )

    def test_send_button_visible_with_new_text_detected(self):
        # If ChatGPT is already back to idle with Send button visible and new text,
        # detect immediately without waiting for any timeout.
        page = _WaitFakePage()
        diag = {"composer_still_holds_prompt": False, "composer_len": 0}
        browser = _patched_browser(page, diag, lambda *a, **k: None)
        browser._locate_assistant_messages = lambda: mock.Mock(count=lambda: 3)
        browser._is_send_button_visible = lambda: True
        browser._get_assistant_message = lambda: "Completed answer"

        p1, p2, p3 = self._patches()
        with p1, p2, p3:
            browser._wait_for_first_assistant_message(
                3, FAKE_WAIT_PROMPT, old_last_text="Old answer"
            )

    def test_is_generating_requires_visible_stop_button(self):
        # A hidden stop button (display: none) retained in the DOM must NOT
        # cause _is_generating to return True.
        browser = make_browser(mock.Mock())

        def locator_mock(selector):
            loc = mock.Mock()
            if ":visible" in selector:
                loc.count.return_value = 0
            else:
                loc.count.return_value = 1
            return loc

        browser.page.locator = locator_mock
        self.assertFalse(browser._is_generating())

        # When visible stop button is present, returns True
        def visible_locator_mock(selector):
            loc = mock.Mock()
            if '[data-testid="stop-button"]:visible' in selector:
                loc.count.return_value = 1
            else:
                loc.count.return_value = 0
            return loc

        browser.page.locator = visible_locator_mock
        self.assertTrue(browser._is_generating())

    def test_is_generating_ignores_sidebar_cursor_interaction(self):
        # Sidebar items with class 'cursor-interaction' or global cursor spans
        # must NOT trigger _is_generating = True. Only stop buttons or scoped cursors do.
        browser = make_browser(mock.Mock())

        def locator_mock(selector):
            loc = mock.Mock()
            loc.count.return_value = 0
            return loc

        browser.page.locator = locator_mock
        self.assertFalse(browser._is_generating())

    def test_is_send_button_visible_recognizes_dictate_and_voice(self):
        # When composer text is empty, ChatGPT displays Dictate or Start Voice buttons
        browser = make_browser(mock.Mock())

        def locator_mock(selector):
            loc = mock.Mock()
            if 'button[aria-label="Dictate"]:visible' in selector:
                loc.count.return_value = 1
            else:
                loc.count.return_value = 0
            return loc

        browser.page.locator = locator_mock
        self.assertTrue(browser._is_send_button_visible())

    def test_extract_node_text_strips_chatgpt_said_header(self):
        # Modern search unit wrappers contain 'ChatGPT said:\n\nHello', should be cleanly stripped
        browser = make_browser(mock.Mock())
        node = mock.Mock()
        node.locator.return_value.count.return_value = 0
        node.text_content.return_value = "ChatGPT said:\n\nHello from WebMind"

        extracted = browser._extract_node_text(node)
        self.assertEqual(extracted, "Hello from WebMind")

    def test_wait_for_response_fast_settle_with_visible_send_button(self):
        # In _wait_for_response, when send button is visible, answer settles in
        # 2 stable cycles (~0.7s) instead of 6 cycles (2.1s) or 15s timeout.
        browser = make_browser(mock.Mock())
        browser._wait_for_first_assistant_message = lambda *a, **k: None
        browser._dismiss_rate_limit_modal = lambda: None
        browser._get_assistant_message = lambda: "Settled answer"
        browser._is_generating = lambda: False
        browser._is_send_button_visible = lambda: True

        with mock.patch("main.time.sleep", lambda _: None):
            res = browser._wait_for_response(0, "prompt")
            self.assertEqual(res, "Settled answer")

    def test_wait_for_response_multi_bubble_settlement(self):
        # When a turn spawns multiple bubbles, _wait_for_response resets stability
        # upon detecting new bubbles and collects all bubbles for the turn.
        browser = make_browser(mock.Mock())
        browser._wait_for_first_assistant_message = lambda *a, **k: None
        browser._dismiss_rate_limit_modal = lambda: None
        browser._is_generating = lambda: False
        browser._is_send_button_visible = lambda: True

        counts = [1, 1, 2, 2, 2, 2, 2]
        browser._locate_assistant_messages = lambda: mock.Mock(
            count=lambda: counts.pop(0) if counts else 2
        )

        responses = [
            "Bubble 1",
            "Bubble 1",
            "Bubble 1\n\nBubble 2",
            "Bubble 1\n\nBubble 2",
            "Bubble 1\n\nBubble 2",
            "Bubble 1\n\nBubble 2",
        ]
        browser._get_assistant_turn_text = lambda since_count=None: responses.pop(0) if responses else "Bubble 1\n\nBubble 2"

        with mock.patch("main.time.sleep", lambda _: None):
            res = browser._wait_for_response(0, "prompt")
            self.assertEqual(res, "Bubble 1\n\nBubble 2")


# ============================================================
# REQUEST WALL-CLOCK DEADLINE CAP (per-request backstop)
# ============================================================
# WebMind's request deadline (server.py) is threaded into main.py as an
# absolute MONOTONIC instant, converted by _deadline_wall to the wall clock
# the wait loops use.  A passed/expired cap must abort the wait immediately --
# it is the backstop against a wedged browser holding the SSE stream open.
# These tests cover (a) the conversion helpers, (b) that the public browser
# entry points accept deadline/wall_cap, and (c) that an expired cap raises
# before any page access (proving the check runs first).


class RequestWallClockCapTests(unittest.TestCase):

    def _rants(self):
        """Patches that avoid real sleeps + log writes."""
        tmp_log = os.path.join(tempfile.gettempdir(), "webmind_test_runtime.log")
        return (
            mock.patch("main.time.sleep", lambda _: None),
            mock.patch("main.RUNTIME_LOG_PATH", tmp_log),
        )

    def test_deadline_value_seconds_to_monotonic(self):
        self.assertIsNone(_deadline_value(None))
        now = time.monotonic()
        self.assertGreaterEqual(_deadline_value(5), now)
        self.assertLess(_deadline_value(5) - now, 6)
        # non-positive means "already expired"
        self.assertLessEqual(_deadline_value(0), time.monotonic())

    def test_deadline_wall_converts_monotonic_to_wall(self):
        self.assertIsNone(_deadline_wall(None))
        wall = _deadline_wall(time.monotonic() + 5)
        self.assertGreaterEqual(wall, time.time() + 4.9)
        self.assertLessEqual(wall, time.time() + 6)
        expired = _deadline_wall(time.monotonic() - 10)
        self.assertLessEqual(expired, time.time())

    def test_ask_and_ask_stream_entry_points_accept_deadline(self):
        # Regressions: smoke failed with `ChatGPTBrowser.ask_stream() got an
        # unexpected keyword argument 'deadline'`.  The public entry points
        # and every wait layer must accept the threaded value.
        self.assertIn("deadline", inspect.signature(ChatGPTBrowser.ask).parameters)
        self.assertIn(
            "deadline", inspect.signature(ChatGPTBrowser.ask_stream).parameters
        )
        self.assertIn("wall_cap", inspect.signature(ChatGPTBrowser._ask_impl).parameters)
        self.assertIn(
            "wall_cap", inspect.signature(ChatGPTBrowser._ask_stream_impl).parameters
        )
        self.assertIn(
            "wall_cap", inspect.signature(ChatGPTBrowser._wait_for_response).parameters
        )
        self.assertIn(
            "wall_cap", inspect.signature(ChatGPTBrowser._stream_response).parameters
        )
        self.assertIn(
            "wall_cap",
            inspect.signature(
                ChatGPTBrowser._wait_for_first_assistant_message
            ).parameters,
        )

    def test_expired_wall_cap_aborts_before_page_access(self):
        # page=None would explode on locator(); an expired cap must raise
        # TimeoutError on the FIRST loop iteration, before touching the page.
        browser = make_browser(None)
        with self.assertRaises(TimeoutError) as ctx:
            browser._wait_for_first_assistant_message(
                0, FAKE_WAIT_PROMPT, wall_cap=time.time() - 1
            )
        self.assertIn("wall-clock deadline", str(ctx.exception))

    def test_wait_for_response_propagates_wall_cap(self):
        browser = make_browser(None)
        with self.assertRaises(TimeoutError) as ctx:
            browser._wait_for_response(
                0, FAKE_WAIT_PROMPT, wall_cap=time.time() - 1
            )
        self.assertIn("wall-clock deadline", str(ctx.exception))

    def test_stream_response_propagates_wall_cap(self):
        browser = make_browser(None)
        with self.assertRaises(TimeoutError) as ctx:
            list(
                browser._stream_response(
                    0, FAKE_WAIT_PROMPT, wall_cap=time.time() - 1
                )
            )
        self.assertIn("wall-clock deadline", str(ctx.exception))


class OrphanedToolCallTests(unittest.TestCase):
    def test_orphaned_tool_call_notice_injected(self):
        messages = [
            {"role": "user", "content": "read files"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_dropped_1", "function": {"name": "bash", "arguments": '{"command":"Get-Content a.txt"}'}}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_ok_2", "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]},
            {"role": "tool", "tool_call_id": "call_ok_2", "content": "C:\\repo"},
        ]
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
        prompt, _ = build_prompt(messages, tools)
        self.assertIn("call_dropped_1", prompt)
        self.assertIn("MISSING from the conversation history", prompt)
        self.assertIn("Do NOT assume this tool executed", prompt)


if __name__ == "__main__":
    unittest.main()
