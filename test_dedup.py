"""Unit tests for the request idempotency guard (server.py).

These tests exercise the guard in isolation -- no browser, no network.
They validate the four behaviors requested:

  1. Identical request twice -> second is skipped (replayed).
  2. Two different requests  -> both are processed.
  3. Same text, different metadata -> NOT deduplicated.
  4. Tool-call requests      -> still work (key uses tools + messages).
"""

import unittest
import threading
import time
from types import SimpleNamespace

import server as server_mod
from server import (
    ChatRequest,
    IdempotencyGuard,
    _has_tools,
    _REPLAY_TOOLCALL_MESSAGE,
    _replay_response,
)


def make_request(
    text="hello",
    stream=False,
    tools=None,
    tool_choice=None,
    chat_url=None,
    request_id=None,
):
    messages = [{"role": "user", "content": text}]
    req = ChatRequest(
        model="webmind",
        messages=messages,
        stream=stream,
        tools=tools,
        tool_choice=tool_choice,
        chat_url=chat_url,
        request_id=request_id,
    )
    return req


def headers(**kw):
    return SimpleNamespace(get=lambda name, default=None: kw.get(name, default))


class KeyConstructionTest(unittest.TestCase):

    def test_same_request_same_key(self):
        guard = IdempotencyGuard()
        a = make_request("hello world")
        b = make_request("hello world")
        self.assertEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))

    def test_different_request_different_key(self):
        guard = IdempotencyGuard()
        a = make_request("hello")
        b = make_request("goodbye")
        self.assertNotEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))

    def test_same_text_different_stream_flag_distinct(self):
        guard = IdempotencyGuard()
        a = make_request("same text", stream=False)
        b = make_request("same text", stream=True)
        self.assertNotEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))

    def test_same_text_different_tools_distinct(self):
        guard = IdempotencyGuard()
        tool_a = [{"type": "function", "function": {"name": "fa", "parameters": {}}}]
        tool_b = [{"type": "function", "function": {"name": "fb", "parameters": {}}}]
        a = make_request("same text", tools=tool_a)
        b = make_request("same text", tools=tool_b)
        self.assertNotEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))

    def test_same_text_different_tool_choice_distinct(self):
        guard = IdempotencyGuard()
        a = make_request("same text", tools=[], tool_choice="auto")
        b = make_request("same text", tools=[], tool_choice="none")
        self.assertNotEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))

    def test_same_text_different_chat_url_distinct(self):
        guard = IdempotencyGuard()
        a = make_request("same text", chat_url="https://chatgpt.com/c/abc")
        b = make_request("same text", chat_url="https://chatgpt.com/c/xyz")
        self.assertNotEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))

    def test_request_id_wins_over_hash(self):
        guard = IdempotencyGuard()
        a = make_request("text-a", request_id="req-123")
        b = make_request("text-b", request_id="req-123")  # different text, same id
        self.assertEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))
        self.assertTrue(guard._key_for(a, headers()).startswith("request:"))

    def test_idempotency_header_wins_over_hash(self):
        guard = IdempotencyGuard()
        a = make_request("text-a")
        b = make_request("text-b")  # different text, same header key
        h = headers(**{"Idempotency-Key": "same-client-key"})
        self.assertEqual(guard._key_for(a, h), guard._key_for(b, h))
        self.assertTrue(guard._key_for(a, h).startswith("idempotency:"))

    def test_messages_are_part_of_hash(self):
        guard = IdempotencyGuard()
        a = make_request("single user msg")
        b = ChatRequest(
            model="webmind",
            messages=[
                {"role": "user", "content": "single user msg"},
                {"role": "assistant", "content": "then an extra turn"},
            ],
        )
        self.assertNotEqual(guard._key_for(a, headers()), guard._key_for(b, headers()))


class GuardBehaviorTest(unittest.TestCase):

    def test_identical_second_request_is_skipped(self):
        guard = IdempotencyGuard()
        req = make_request("What is 2+2?")
        key, rec, is_new1 = guard.start(req, headers())
        self.assertTrue(is_new1)

        # Simulate the worker completing the request.
        guard.finish(key, {"kind": "success", "content": "4"})
        self.assertTrue(rec.event.is_set())
        self.assertEqual(rec.response["content"], "4")

        # Second identical request -> retry, not new.
        req2 = make_request("What is 2+2?")
        key2, rec2, is_new2 = guard.start(req2, headers())
        self.assertFalse(is_new2)
        self.assertEqual(key2, key)
        self.assertTrue(rec2.event.is_set())
        self.assertEqual(rec2.response["content"], "4")

    def test_two_different_requests_both_new(self):
        guard = IdempotencyGuard()
        _, _, new1 = guard.start(make_request("first"), headers())
        _, _, new2 = guard.start(make_request("second"), headers())
        self.assertTrue(new1)
        self.assertTrue(new2)

    def test_in_flight_duplicate_waits(self):
        guard = IdempotencyGuard()
        req = make_request("long request")
        key, rec, new = guard.start(req, headers())
        self.assertTrue(new)
        self.assertFalse(rec.event.is_set())

        def _complete_later(k, r):
            time.sleep(0.2)
            guard.finish(k, {"kind": "success", "content": "done"})
            self.assertTrue(r.event.is_set())

        t = threading.Thread(target=_complete_later, args=(key, rec))
        t.daemon = True
        t.start()

        # Retry arrives while original is in flight.
        req2 = make_request("long request")
        key2, rec2, new2 = guard.start(req2, headers())
        self.assertFalse(new2)
        # It should complete once the original finishes.
        self.assertTrue(rec2.event.wait(2.0))
        self.assertEqual(rec2.response["content"], "done")
        t.join(3.0)

    def test_failed_original_is_replayed_as_error(self):
        guard = IdempotencyGuard()
        key, rec, new = guard.start(make_request("boom"), headers())
        self.assertTrue(new)
        guard.fail(key, "ChatGPT input: could not submit the prompt...")
        self.assertTrue(rec.event.is_set())
        self.assertEqual(rec.error, "ChatGPT input: could not submit the prompt...")

        key2, rec2, new2 = guard.start(make_request("boom"), headers())
        self.assertFalse(new2)
        self.assertTrue(rec2.event.is_set())
        self.assertIsNotNone(rec2.error)

    def test_ttl_expiry_allows_reprocessing(self):
        guard = IdempotencyGuard(ttl=0.05)
        req = make_request("expiring")
        key, _rec, new = guard.start(req, headers())
        self.assertTrue(new)
        guard.finish(key, {"kind": "success", "content": "x"})

        # Wait for TTL to pass, then a fresh identical request is new again.
        time.sleep(0.1)
        req2 = make_request("expiring")
        key2, _rec2, new2 = guard.start(req2, headers())
        self.assertTrue(new2)

    def test_max_entries_bounded(self):
        guard = IdempotencyGuard(max_entries=3)
        for i in range(5):
            guard.start(make_request(f"req-{i}"), headers())
        self.assertLessEqual(len(guard._records), 3)

    def test_evicts_oldest_when_full(self):
        guard = IdempotencyGuard(max_entries=2)
        key_oldest, _, _ = guard.start(make_request("oldest"), headers())
        key_middle, _, _ = guard.start(make_request("middle"), headers())
        key_newest, _, _ = guard.start(make_request("newest"), headers())
        keys = set(guard._records.keys())
        # FIFO eviction: the oldest entry is dropped.
        self.assertNotIn(key_oldest, keys)
        self.assertIn(key_middle, keys)
        self.assertIn(key_newest, keys)

    def test_retryable_failure_can_be_removed_and_rerun(self):
        """A transient pre-send failure must not permanently poison a key.

        fail(..., retryable=True) records the error, but remove() lets a
        retry re-register as a fresh request instead of replaying the
        cached 500.
        """
        guard = IdempotencyGuard()
        key, rec, new = guard.start(make_request("flake"), headers())
        self.assertTrue(new)

        guard.fail(
            key,
            "ChatGPT input: could not verify the prompt text was inserted...",
            retryable=True,
        )
        self.assertTrue(rec.event.is_set())
        self.assertTrue(rec.retryable)
        self.assertIsNone(rec.response)

        # A retry that removes the record becomes new again.
        self.assertTrue(guard.remove(key))
        key2, rec2, new2 = guard.start(make_request("flake"), headers())
        self.assertTrue(new2)
        self.assertFalse(rec2.retryable)
        self.assertIsNone(rec2.error)

    def test_remove_missing_key_returns_false(self):
        guard = IdempotencyGuard()
        self.assertFalse(guard.remove("hash:does-not-exist"))

    def test_fatal_failure_marks_non_retryable_by_default(self):
        """fail() without retryable stays fatal: retries replay the error."""
        guard = IdempotencyGuard()
        key, rec, new = guard.start(make_request("fatal"), headers())
        self.assertTrue(new)
        guard.fail(key, "browser died mid-request")
        self.assertFalse(rec.retryable)
        self.assertEqual(rec.error, "browser died mid-request")

    def test_retryable_exception_classified(self):
        from server import _is_retryable_error
        import main as main_mod

        self.assertTrue(
            _is_retryable_error(
                main_mod.ChatGPTInputError("could not verify the prompt text")
            )
        )
        self.assertFalse(_is_retryable_error(RuntimeError("browser closed")))
        self.assertFalse(
            _is_retryable_error(
                RuntimeError("Locator.count: Target page has been closed")
            )
        )

    def test_prompt_too_large_is_not_retryable(self):
        # An identical retry re-pastes an identical prompt and gets the same
        # verdict, so marking it retryable would make the guard loop on a
        # condition only the caller can fix by sending less.
        from server import _is_retryable_error
        import main as main_mod

        self.assertFalse(
            _is_retryable_error(
                main_mod.PromptTooLargeError("PROMPT_TOO_LARGE: 20000 chars")
            )
        )

    def test_prompt_too_large_maps_to_413(self):
        # 413 says "well-formed but too big", which a client can act on.
        # A 500 is opaque and reads like a WebMind bug.
        #
        # Only the pre-flight refusal qualifies: it never pasted, so the
        # prompt really is over the backstop and shrinking IS the fix.
        from server import _error_status
        import main as main_mod

        self.assertEqual(
            _error_status(
                main_mod.PromptTooLargeError("PROMPT_TOO_LARGE", kind="over_ceiling")
            ),
            413,
        )
        # The pre-existing mappings must not have shifted.
        self.assertEqual(_error_status(RuntimeError("x")), 500)
        self.assertEqual(_error_status(TimeoutError("x")), 504)

    def test_attachment_refusal_does_not_emit_the_413_compaction_cue(self):
        # An attachment refusal reaches _error_status only AFTER the one-shot
        # smaller retry has been spent.  Answering 413 there hands the client
        # the "context no longer fits" signal, so it compacts and re-sends a
        # prompt that is still large enough to convert -- 413 -> compact ->
        # 413, forever.  Withhold 413 so the loop cannot be driven from here.
        from server import _error_status
        import main as main_mod

        exc = main_mod.PromptTooLargeError(
            "PROMPT_TOO_LARGE: submitted as a pasted-text attachment",
            kind="attachment",
        )
        self.assertTrue(exc.is_attachment)
        self.assertEqual(_error_status(exc), 500)


class ReplayResponseTest(unittest.TestCase):

    def test_nonstream_plain_replay(self):
        resp = _replay_response(
            {"kind": "success", "content": "4"},
            make_request("What is 2+2?"),
        )
        self.assertIsInstance(resp, dict)
        self.assertEqual(resp["choices"][0]["message"]["content"], "4")
        self.assertEqual(resp["choices"][0]["finish_reason"], "stop")

    def test_stream_plain_replay_is_sse(self):
        from fastapi.responses import StreamingResponse

        resp = _replay_response(
            {"kind": "success", "content": "streamed text"},
            make_request("stream it", stream=True),
        )
        self.assertIsInstance(resp, StreamingResponse)

    def test_nonstream_tool_mode_replay_never_reissues(self):
        # Phase 3 contract (survival "no executor double-fire"): a
        # recorded tool_call is NEVER re-issued on a retry.  The original
        # tool_call was already delivered to the caller, so re-sending it
        # here would make the external client execute the same tool twice.
        # Instead the retry receives a safe message (finish_reason "stop").
        tool = {
            "type": "function",
            "function": {"name": "calc", "parameters": {"type": "object"}},
        }
        import json

        content = json.dumps(
            {
                "webmind_response": "tool_call",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "calc",
                        "arguments": {"x": 1, "y": 2},
                    }
                ],
            }
        )
        req = make_request("compute", tools=[tool])
        resp = _replay_response({"kind": "success", "content": content}, req)
        self.assertEqual(resp["choices"][0]["finish_reason"], "stop")
        message = resp["choices"][0]["message"]
        self.assertEqual(message["content"], _REPLAY_TOOLCALL_MESSAGE)
        self.assertNotIn("tool_calls", message)

    def test_invalid_format_raises(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException):
            _replay_response({"kind": "bogus"}, make_request())
class AttachmentRetryTest(unittest.TestCase):
    """A pasted-text attachment must still yield a real answer.

    Detection alone returns an error, which is honest but useless: the
    caller asked a question and gets told nothing.  Because the conversion
    is content-dependent, no character budget can prevent it, so the HTTP
    layer re-serializes the SAME conversation more compactly and asks once
    more.
    """

    def _call(self, run, rebuild, prompt):
        import asyncio

        async def _go():
            return await server_mod._dispatch_with_attachment_retry(
                run, rebuild, prompt, request_id="r1", path="non_stream"
            )

        return asyncio.run(_go())

    def test_attachment_is_retried_once_with_a_smaller_prompt(self):
        import main as main_mod

        seen = []
        reply = SimpleNamespace(text="the real answer")

        async def run(prompt):
            seen.append(prompt)
            if len(seen) == 1:
                raise main_mod.PromptTooLargeError("too big", kind="attachment")
            return reply

        result = self._call(run, lambda target: "small prompt", "x" * 14000)

        self.assertIs(result, reply)
        self.assertEqual(seen, ["x" * 14000, "small prompt"])

    def test_a_second_attachment_refusal_surfaces(self):
        # Exactly one retry: this must not loop.
        import main as main_mod

        seen = []

        async def run(prompt):
            seen.append(prompt)
            raise main_mod.PromptTooLargeError("too big", kind="attachment")

        with self.assertRaises(main_mod.PromptTooLargeError):
            self._call(run, lambda target: "small prompt", "x" * 14000)
        self.assertEqual(len(seen), 2)

    def test_over_ceiling_is_not_retried(self):
        # The pre-flight already knows the size; shrinking it is the
        # caller's call, so this must surface immediately.
        import main as main_mod

        seen = []

        async def run(prompt):
            seen.append(prompt)
            raise main_mod.PromptTooLargeError("over the limit", kind="over_ceiling")

        with self.assertRaises(main_mod.PromptTooLargeError):
            self._call(run, lambda target: "small prompt", "x" * 99000)
        self.assertEqual(seen, ["x" * 99000], "must not re-dispatch")

    def test_no_retry_when_reserialization_does_not_shrink(self):
        # An identical resend hits the identical verdict, so spending a
        # generation on it is pure waste.
        import main as main_mod

        seen = []

        async def run(prompt):
            seen.append(prompt)
            raise main_mod.PromptTooLargeError("too big", kind="attachment")

        with self.assertRaises(main_mod.PromptTooLargeError):
            self._call(run, lambda target: "x" * 14000, "x" * 14000)
        self.assertEqual(len(seen), 1)

    def test_other_errors_are_untouched(self):
        seen = []

        async def run(prompt):
            seen.append(prompt)
            raise RuntimeError("browser closed")

        with self.assertRaises(RuntimeError):
            self._call(run, lambda target: "small", "x" * 14000)
        self.assertEqual(len(seen), 1)


class LegacyAttachmentRetryTest(unittest.TestCase):
    """opencode's auto-compaction sends the whole history as ONE user message
    with no tools.  Capping it was not enough: a 12042-char capped prompt was
    still converted into a pasted-text attachment, so the no-tools path needs
    the same one-shot shrink the tool path already had."""

    def _call(self, run, prompt, target):
        import asyncio

        from protocol import cap_legacy_prompt

        async def _go():
            return await server_mod._dispatch_with_attachment_retry(
                run,
                lambda chars: cap_legacy_prompt(prompt, chars),
                prompt,
                request_id="r1",
                path="non_stream_legacy",
                target=target,
            )

        return asyncio.run(_go())

    def test_legacy_attachment_is_retried_shorter_and_can_succeed(self):
        import main as main_mod
        from protocol import LEGACY_RETRY_MAX_CHARS

        seen = []
        reply = SimpleNamespace(text="the compacted summary")

        async def run(prompt):
            seen.append(prompt)
            if len(seen) == 1:
                raise main_mod.PromptTooLargeError("file", kind="attachment")
            return reply

        capped = "HEAD\n" + ("transcript line\n" * 4000)
        result = self._call(run, capped, LEGACY_RETRY_MAX_CHARS)

        self.assertIs(result, reply)
        self.assertEqual(len(seen), 2)
        self.assertLess(len(seen[1]), len(seen[0]))
        self.assertTrue(seen[1].startswith("HEAD"))

    def test_legacy_retry_happens_exactly_once(self):
        import main as main_mod
        from protocol import LEGACY_RETRY_MAX_CHARS

        seen = []

        async def run(prompt):
            seen.append(prompt)
            raise main_mod.PromptTooLargeError("file", kind="attachment")

        with self.assertRaises(main_mod.PromptTooLargeError):
            self._call(run, "x" * 20000, LEGACY_RETRY_MAX_CHARS)
        self.assertEqual(len(seen), 2)

    def test_legacy_over_ceiling_is_not_retried(self):
        import main as main_mod
        from protocol import LEGACY_RETRY_MAX_CHARS

        seen = []

        async def run(prompt):
            seen.append(prompt)
            raise main_mod.PromptTooLargeError("ceiling", kind="over_ceiling")

        with self.assertRaises(main_mod.PromptTooLargeError):
            self._call(run, "x" * 20000, LEGACY_RETRY_MAX_CHARS)
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
