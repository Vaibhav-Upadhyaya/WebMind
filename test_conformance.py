"""Protocol conformance: WebMind as a generic OpenAI-compatible provider.

This suite models an ARBITRARY OpenAI-compatible harness talking to WebMind.
It never assumes DeepSeek-harness or OpenCode internals. The simulated pieces
are, in order:

    harness (wire messages) -> WebMind build_prompt -> model text (ChatGPT Web)
    -> WebMind parse/validate -> OpenAI chat.completion (JSON or SSE) -> harness

The harness owns its own agent loop, tool execution, and tool results. WebMind
only ever translates: it never runs a tool and never fabricates a tool call.

Imports protocol.py only (no browser, no server worker), so the suite runs
with zero network and zero Playwright.
"""

import json
import unittest
from collections import OrderedDict
from types import SimpleNamespace

from protocol import (
    build_prompt,
    build_correction_directive,
    build_nonstream_response,
    parse_chatgpt_response,
    recover_tool_call,
    sse_message_chunks,
    sse_tool_call_chunks,
)


# ============================================================
# GENERIC TOOL DEFINITIONS (OpenAI wire format)
# ============================================================

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the project workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string"},
            },
            "required": ["command"],
        },
    },
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
}

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
}

TOOLS = [BASH_TOOL, READ_TOOL, WRITE_TOOL]


# ============================================================
# SIMULATED COMPONENTS
# ============================================================

def _as_message(wire: dict):
    """Convert an OpenAI wire-format message to the object form build_prompt
    reads. Mirrors what WebMind's request layer does to an incoming request."""
    return SimpleNamespace(
        role=str(wire.get("role", "user")) or "user",
        content=wire.get("content"),
        tool_calls=wire.get("tool_calls"),
        tool_call_id=wire.get("tool_call_id"),
        name=wire.get("name"),
        reasoning_content=wire.get("reasoning_content"),
    )


class ModelBackend:
    """The ChatGPT Web side WebMind reads from: raw assistant text per prompt.

    Scripting it with the exact envelopes (and malformations) ChatGPT web
    actually produces is what makes these tests deterministic.
    """

    def __init__(self, script=None):
        self._script = list(script) if script else []
        self._index = 0
        self.prompts = []

    def respond(self, prompt):
        self.prompts.append(prompt)
        if self._index >= len(self._script):
            raise AssertionError("ModelBackend exhausted: script too short")
        raw = self._script[self._index]
        self._index += 1
        return raw


def _envelope_tool_call(call_id, name, arguments):
    return {
        "webmind_response": "tool_call",
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
    }


def _envelope_message(content):
    return {"webmind_response": "message", "content": content}


class Harness:
    """Generic OpenAI-compatible agent loop.

    Deliberately product-agnostic: history is OpenAI wire-format dicts, and
    tool execution happens HERE (the harness) after WebMind returns tool_calls.
    WebMind never runs a tool; WebMind never sees the executor.
    """

    def __init__(self, tools, backend):
        self.tools = tools
        self.backend = backend
        self.history = []

    def user(self, text):
        self.history.append({"role": "user", "content": text})

    def _current_prompt(self):
        return build_prompt([_as_message(m) for m in self.history], self.tools)

    def chat_once(self, stream=False, recover=False):
        """One WebMind round trip: build the prompt, let the model speak,
        parse/validate, and return the OpenAI-compatible response.

        With recover=True, a validation-failure envelope triggers ONE model
        correction round trip (the backend doubles as the ask function),
        then the final parsed result shapes the response.
        """
        prompt, _ = self._current_prompt()
        raw = self.backend.respond(prompt)
        parsed = parse_chatgpt_response(
            raw,
            self.tools,
            history=[_as_message(m) for m in self.history],
        )
        if recover:
            parsed = recover_tool_call(
                parsed,
                self.backend.respond,
                self.tools,
                history=[_as_message(m) for m in self.history],
            )
        chat_id = "chatcmpl-test"
        created = 123
        if not stream:
            return build_nonstream_response(parsed, chat_id, created)
        if parsed["type"] == "tool_call":
            return sse_tool_call_chunks(chat_id, created, parsed["tool_calls"])
        return sse_message_chunks(chat_id, created, parsed["content"])

    def run_tool_calls(self, response, executor):
        """Harness-side execution. Returns the list of executed (name, args)."""
        message = response["choices"][0]["message"]
        executed = []
        for tc in (message.get("tool_calls") or []):
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            result = executor(name, args)
            executed.append((name, args))
            self.history.append(
                {"role": "assistant", "content": None, "tool_calls": [tc]}
            )
            self.history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
        return executed


def sse_to_chat_message(chunks):
    """Reconstruct an OpenAI chat message from SSE chunks exactly the way a
    streaming harness consumer does (index-grouped tool-call fragments)."""
    deltas = []
    finish_reason = None
    for chunk in chunks:
        if not chunk.startswith("data: "):
            continue
        body = chunk[len("data: "):].strip()
        if body == "[DONE]":
            continue
        obj = json.loads(body)
        choice = obj["choices"][0]
        finish_reason = choice.get("finish_reason")
        deltas.append(choice["delta"])

    role = None
    content_parts = []
    calls = OrderedDict()
    for d in deltas:
        if "role" in d:
            role = d["role"]
        if d.get("content") is not None:
            content_parts.append(d["content"])
        for piece in d.get("tool_calls", []):
            index = piece.get("index", 0)
            entry = calls.setdefault(
                index,
                {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
            )
            if piece.get("id") is not None:
                entry["id"] = piece["id"]
            if piece.get("type") is not None:
                entry["type"] = piece["type"]
            fn = piece.get("function") or {}
            if fn.get("name") is not None:
                entry["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                entry["function"]["arguments"] += fn["arguments"]

    content = "".join(content_parts)
    message = {
        "role": role or "assistant",
        "content": content if content else None,
        "tool_calls": [calls[i] for i in sorted(calls)] or None,
    }
    return message, finish_reason


def _noop_executor(name, args):
    """Stand-in for a real harness's tool runner: records and returns a result."""
    return {"exit_code": 0, "name": name}


# ============================================================
# CONFORMANCE TESTS
# ============================================================

class TestRequestTranslation(unittest.TestCase):
    """Incoming tools schemas must reach the model prompt verbatim."""

    def test_tool_schemas_are_translated_into_the_prompt(self):
        harness = Harness(TOOLS, ModelBackend([]))
        harness.user("what is in the workspace?")
        prompt, _ = harness._current_prompt()
        self.assertIn("TOOL DEFINITIONS:", prompt)
        self.assertIn('"bash"', prompt)
        self.assertIn('"command"', prompt)
        self.assertIn('"read"', prompt)
        self.assertIn('"write"', prompt)

    def test_tool_choice_is_translated_into_the_prompt(self):
        choice = {"type": "function", "function": {"name": "read"}}
        prompt, _ = build_prompt(
            [_as_message({"role": "user", "content": "pick one"})],
            TOOLS,
            tool_choice=choice,
        )
        self.assertIn(json.dumps(choice), prompt)

    def test_tool_result_is_represented_on_the_next_request(self):
        harness = Harness(TOOLS, ModelBackend([]))
        harness.user("list the files")
        tc = {
            "id": "call_01",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "dir"}'},
        }
        harness.history.append(
            {"role": "assistant", "content": None, "tool_calls": [tc]}
        )
        harness.history.append(
            {
                "role": "tool",
                "tool_call_id": "call_01",
                "name": "bash",
                "content": '{"exit_code": 0, "stdout": "a.txt b.txt"}',
            }
        )
        prompt, _ = harness._current_prompt()
        self.assertIn("[role: tool]", prompt)
        self.assertIn("tool_name: bash", prompt)
        self.assertIn("tool_call_id: call_01", prompt)
        self.assertIn("[role: assistant]", prompt)
        self.assertIn("call_01", prompt)


class TestResponseConformance(unittest.TestCase):
    """The OpenAI-compatible response shapes a harness actually consumes."""

    def test_normal_text_response(self):
        backend = ModelBackend([json.dumps(_envelope_message("all good"))])
        harness = Harness(TOOLS, backend)
        harness.user("status?")
        response = harness.chat_once()
        choice = response["choices"][0]
        self.assertEqual(choice["message"]["content"], "all good")
        self.assertIsNone(choice["message"].get("tool_calls"))
        self.assertEqual(choice["finish_reason"], "stop")

    def test_single_tool_call(self):
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_1", "bash", {"command": "pwd"})),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("where am I?")
        response = harness.chat_once()
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        message = choice["message"]
        self.assertIsNone(message["content"])
        (tc,) = message["tool_calls"]
        self.assertEqual(tc["id"], "call_1")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "bash")
        self.assertEqual(
            json.loads(tc["function"]["arguments"]), {"command": "pwd"}
        )

    def test_multiple_tool_calls_in_one_response(self):
        backend = ModelBackend([
            json.dumps({
                "webmind_response": "tool_call",
                "tool_calls": [
                    {"id": "call_a", "name": "bash", "arguments": {"command": "dir"}},
                    {"id": "call_b", "name": "read", "arguments": {"path": "a.txt"}},
                ],
            }),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("inspect both")
        response = harness.chat_once()
        calls = response["choices"][0]["message"]["tool_calls"]
        self.assertEqual(
            [c["function"]["name"] for c in calls], ["bash", "read"]
        )
        self.assertEqual([c["id"] for c in calls], ["call_a", "call_b"])
        self.assertEqual(
            json.loads(calls[1]["function"]["arguments"]), {"path": "a.txt"}
        )

    def test_complex_json_arguments_round_trip(self):
        complex_args = {
            "command": "git commit",
            "env": {"cwd": "C:\\repo", "flags": ["-m", "add feature"]},
        }
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_c", "bash", complex_args)),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("commit the change")
        response = harness.chat_once()
        tc = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(json.loads(tc["function"]["arguments"]), complex_args)

    def test_windows_paths_backslashes_survive(self):
        # ChatGPT web emits Windows paths with SINGLE backslashes, which are
        # invalid JSON escapes ("\U", "\b"...). Validation must keep them
        # literal (never turn "\b" into a backspace) and still deliver an
        # executable tool call. The envelope below is written the way ChatGPT
        # actually emits it: raw single backslashes, not escaped pairs.
        raw = (
            '{"webmind_response":"tool_call","tool_calls":['
            '{"id":"call_d","name":"read","arguments":{"path":"C:\\Users\\Dev\\Documents\\notes.txt"}},'
            '{"id":"call_e","name":"read","arguments":{"path":".\\brain.py"}}]}'
        )
        backend = ModelBackend([raw])
        harness = Harness(TOOLS, backend)
        harness.user("read both files")
        response = harness.chat_once()
        calls = response["choices"][0]["message"]["tool_calls"]
        paths = [json.loads(c["function"]["arguments"])["path"] for c in calls]
        self.assertEqual(
            paths,
            ["C:\\Users\\Dev\\Documents\\notes.txt", ".\\brain.py"],
        )
        self.assertNotIn("\x08", json.dumps(paths))  # no control chars leak

    def test_malformed_json_repaired_safely(self):
        # Dropped root closer + raw HTML attribute quotes (the live regression
        # shape) must recover into a real tool call, not leak as text.
        raw = ('{"webmind_response":"tool_call","tool_calls":['
               '{"id":"call_f","name":"write","arguments":'
               '{"file_path":"index.html","content":"<html lang="en" '
               'charset="UTF-8"><title>Todo</title></html>"}]}')
        backend = ModelBackend([raw])
        harness = Harness(TOOLS, backend)
        harness.user("create the page")
        response = harness.chat_once()
        choices = response["choices"][0]
        self.assertEqual(choices["finish_reason"], "tool_calls")
        tc = choices["message"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "write")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(
            args["content"], '<html lang="en" charset="UTF-8"><title>Todo</title></html>'
        )

    def test_truncated_envelope_stays_plain_text(self):
        # A structurally-open fragment like '{...[' must NOT be repaired into a
        # fabricated empty tool call; it stays ordinary content.
        raw = '{"webmind_response": "tool_call", "tool_calls": ['
        backend = ModelBackend([raw])
        harness = Harness(TOOLS, backend)
        harness.user("go")
        response = harness.chat_once()
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], raw)
        self.assertIsNone(choice["message"].get("tool_calls"))


class TestRejectionConformance(unittest.TestCase):
    """Invalid output must degrade to a message envelope, never a tool call."""

    def test_invalid_unknown_tool_name_rejected(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_x", "name": "rm_rf_system", "arguments": {"cmd": "boom"}},
            ],
        })
        backend = ModelBackend([raw])
        harness = Harness(TOOLS, backend)
        harness.user("clean up")
        response = harness.chat_once()
        choice = response["choices"][0]
        # A syntactically valid but unvalidatable envelope is a clean error
        # message now -- never a plain-text leak of the raw JSON envelope.
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIsNone(choice["message"].get("tool_calls"))
        self.assertIn("WebMind", choice["message"]["content"])
        self.assertIn("unknown tool", choice["message"]["content"])
        self.assertNotEqual(choice["message"]["content"], raw)
        self.assertNotIn("webmind_response", choice["message"]["content"])

    def test_tool_name_case_sensitive(self):
        raw = json.dumps(_envelope_tool_call("call_y", "Bash", {"command": "pwd"}))
        backend = ModelBackend([raw])
        harness = Harness(TOOLS, backend)
        harness.user("pwd")
        response = harness.chat_once()
        self.assertIsNone(response["choices"][0]["message"].get("tool_calls"))

    def test_invalid_arguments_rejected(self):
        raw = json.dumps(_envelope_tool_call("call_z", "bash", {"timeout": 5}))
        backend = ModelBackend([raw])
        harness = Harness(TOOLS, backend)
        harness.user("run it")
        response = harness.chat_once()
        self.assertIsNone(response["choices"][0]["message"].get("tool_calls"))

    def test_no_fabricated_tool_call_from_ordinary_text(self):
        # Prose that merely mentions the envelope shape is never a tool call.
        backend = ModelBackend([
            'here is a json example: {"webmind_response": "tool_call", '
            '"tool_calls": [{"name": "bash"}]} but that was not a real answer',
        ])
        harness = Harness(TOOLS, backend)
        harness.user("example?")
        response = harness.chat_once()
        message = response["choices"][0]["message"]
        self.assertIsNone(message.get("tool_calls"))
        self.assertEqual(message["content"], backend._script[0])

    def test_completed_tool_call_may_be_reissued_faithfully(self):
        # A re-issued identical call is a legitimate agent decision (the
        # underlying state may have changed since the result was recorded).
        # WebMind must forward it exactly, never inject its own refusal.
        harness = Harness(TOOLS, ModelBackend([]))
        harness.user("what does the workspace contain?")
        tc = {
            "id": "call_completed",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "dir"}'},
        }
        harness.history.append(
            {"role": "assistant", "content": None, "tool_calls": [tc]}
        )
        harness.history.append(
            {
                "role": "tool",
                "tool_call_id": "call_completed",
                "name": "bash",
                "content": '{"exit_code": 0, "stdout": "Makefile src"}',
            }
        )

        # The model re-proposes the IDENTICAL completed call.
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_dup", "bash", {"command": "dir"})),
        ])
        harness.backend = backend
        response = harness.chat_once()
        message = response["choices"][0]["message"]
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        (tc_out,) = message["tool_calls"]
        self.assertEqual(tc_out["id"], "call_dup")
        self.assertEqual(tc_out["function"]["name"], "bash")
        executed = harness.run_tool_calls(response, _noop_executor)
        self.assertEqual(executed, [("bash", {"command": "dir"})])


class TestAgenticSequenceConformance(unittest.TestCase):
    """tool call -> tool result -> next tool call, across many turns."""

    def test_tool_call_result_then_second_tool_call(self):
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_1", "bash", {"command": "dir"})),
            json.dumps(_envelope_tool_call("call_2", "read", {"path": "a.txt"})),
        ])
        harness = Harness(TOOLS, backend)
        executed = []

        harness.user("what is in a.txt?")
        first = harness.chat_once()
        self.assertEqual(
            first["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "bash",
        )
        executed += harness.run_tool_calls(first, _noop_executor)

        second = harness.chat_once()
        self.assertEqual(
            second["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read",
        )
        executed += harness.run_tool_calls(second, _noop_executor)

        self.assertEqual(
            [name for name, _ in executed], ["bash", "read"]
        )
        # The second request carried the first tool result back to the model.
        self.assertIn("tool_call_id: call_1", harness.backend.prompts[1])
        self.assertIn("[role: tool]", harness.backend.prompts[1])

    def test_continuation_prompt_carries_purpose_and_answer_rule(self):
        """The live regression, asserted on the prompt the harness BUILDS.

        On a continuation turn the model used to receive bare
        ``name(arg: type)`` signatures -- no indication of what a tool was
        FOR -- while being told never to invent a tool name. The result was a
        bail-out message ("I cannot safely invent another tool call") instead
        of a tool call or an answer. Assert both fixes are actually present in
        the real continuation prompt.
        """
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_1", "bash", {"command": "dir"})),
            json.dumps(_envelope_message("here is the answer")),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("how do I fix this?")

        first = harness.chat_once()
        harness.run_tool_calls(first, _noop_executor)
        harness.chat_once()

        prompt = backend.prompts[1]

        # (a) The message envelope is redefined as the answer, in both the
        #     always-present instruction block and the continuation directive.
        self.assertIn("must BE the answer to the user's request", prompt)
        self.assertIn("message envelope IS the user's answer", prompt)

        # (b) Full schemas are still replaced by compact signatures, and every
        #     described tool now carries a purpose clause.
        self.assertNotIn("TOOL DEFINITIONS:\n", prompt)
        start = prompt.find("Compact parameter signatures")
        self.assertNotEqual(start, -1)
        signatures = prompt[start:prompt.find("CONVERSATION (chronological")]
        self.assertIn("\u2014", signatures)
        for name in ("bash", "read", "write"):
            self.assertIn(f"{name}(", signatures)

    def test_at_least_five_consecutive_agentic_turns(self):
        script = [
            json.dumps(_envelope_tool_call("call_1", "bash", {"command": "new-item todo.html"})),
            json.dumps(_envelope_tool_call("call_2", "write", {
                "file_path": "todo.html",
                "content": "<html><body><ul id=todo></ul></body></html>",
            })),
            json.dumps(_envelope_tool_call("call_3", "read", {"path": "todo.html"})),
            json.dumps(_envelope_tool_call("call_4", "bash", {"command": "dir"})),
            json.dumps(_envelope_tool_call("call_5", "write", {
                "file_path": "todo.html",
                "content": "<html><body><ul id=todo></ul><script>init()</script></body></html>",
            })),
            json.dumps(_envelope_message("done")),
        ]
        backend = ModelBackend(script)
        harness = Harness(TOOLS, backend)
        executed = []

        harness.user("build me a todo page")

        tools_seen = []
        for _ in range(5):
            response = harness.chat_once()
            self.assertEqual(
                response["choices"][0]["finish_reason"], "tool_calls"
            )
            tools_seen.append(
                response["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
            )
            executed += harness.run_tool_calls(response, _noop_executor)

        final = harness.chat_once()
        self.assertEqual(final["choices"][0]["finish_reason"], "stop")
        self.assertEqual(final["choices"][0]["message"]["content"], "done")
        self.assertEqual(len(executed), 5)
        self.assertEqual(len(backend.prompts), 6)

        # Every continuation prompt is a real prompt (never empty), and the
        # previous tool result is always represented on the following turn.
        for i in range(1, len(backend.prompts)):
            self.assertIn("TOOL DEFINITIONS were already provided", backend.prompts[i])
            self.assertIn(
                "bash(command: string [required], cwd?: string)",
                backend.prompts[i],
            )
            self.assertIn("[role: tool]", backend.prompts[i])

    def test_webmind_never_executes_a_tool(self):
        # The harness's executor is the ONLY place a tool can run. If WebMind
        # executed tools itself, the executed list would be populated even
        # when the harness never calls run_tool_calls.
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_only", "bash", {"command": "pwd"})),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("pwd")
        response = harness.chat_once()
        self.assertIsNotNone(response["choices"][0]["message"]["tool_calls"])
        # The harness simply did not run them yet; nothing on WebMind's side did.
        self.assertEqual(len(harness.history), 1)


class TestStreamingConformance(unittest.TestCase):
    """SSE output must speak delta.tool_calls, never delta.content."""

    def test_streaming_tool_call_uses_delta_tool_calls(self):
        backend = ModelBackend([
            json.dumps({
                "webmind_response": "tool_call",
                "tool_calls": [
                    {"id": "call_s1", "name": "bash", "arguments": {"command": "dir"}},
                    {"id": "call_s2", "name": "read", "arguments": {"path": "a.txt"}},
                ],
            }),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("list and read")
        chunks = harness.chat_once(stream=True)

        self.assertEqual(chunks[-1].strip(), "data: [DONE]")

        # No assistant content may flow through a tool-call stream. The
        # terminal data chunk (empty delta, finish_reason/usage) is skipped.
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            obj = json.loads(chunk[len("data: "):])
            choice = obj["choices"][0]
            if choice.get("finish_reason") is not None:
                continue
            delta = choice["delta"]
            self.assertIsNone(delta.get("content"))
            tool_calls = delta.get("tool_calls") or []
            self.assertTrue(tool_calls or "role" in delta)

        message, finish = sse_to_chat_message(chunks)
        self.assertEqual(finish, "tool_calls")
        self.assertIsNone(message["content"])
        calls = message["tool_calls"]
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["id"], "call_s1")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"command": "dir"}
        )
        self.assertEqual(calls[1]["id"], "call_s2")
        self.assertEqual(
            json.loads(calls[1]["function"]["arguments"]), {"path": "a.txt"}
        )

    def test_streaming_message_uses_delta_content(self):
        backend = ModelBackend([
            json.dumps(_envelope_message("no tools needed")),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("say hello")
        chunks = harness.chat_once(stream=True)

        message, finish = sse_to_chat_message(chunks)
        self.assertEqual(finish, "stop")
        self.assertEqual(message["content"], "no tools needed")
        self.assertIsNone(message["tool_calls"])

        # No delta may carry a tool_calls key at all.
        for chunk in chunks:
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                obj = json.loads(chunk[len("data: "):])
                self.assertNotIn("tool_calls", obj["choices"][0]["delta"])

    def test_stream_reconstructed_message_feeds_next_request(self):
        # A real harness converts the SSE stream back to an assistant message
        # with tool_calls and then posts the result; that object must survive
        # the next build_prompt round trip.
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_r", "bash", {"command": "dir"})),
            json.dumps(_envelope_message("listed")),
        ])
        harness = Harness(TOOLS, backend)
        harness.user("list")
        chunks = harness.chat_once(stream=True)
        message, _ = sse_to_chat_message(chunks)

        harness.history.append(message)
        harness.history.append(
            {
                "role": "tool",
                "tool_call_id": "call_r",
                "name": "bash",
                "content": '{"exit_code": 0}',
            }
        )
        final = harness.chat_once()
        self.assertEqual(final["choices"][0]["message"]["content"], "listed")

        self.assertIn("call_r", harness.backend.prompts[1])
        self.assertIn("tool_call_id: call_r", harness.backend.prompts[1])


class TestToolCallRecoveryConformance(unittest.TestCase):
    """Validation-failure recovery: one correction attempt, clean errors."""

    FS_READ_TOOL = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    }

    RECOVERY_TOOLS = [BASH_TOOL, FS_READ_TOOL]

    def test_invalid_read_path_request_classified(self):
        raw = json.dumps(_envelope_tool_call("call_r", "read", {"path": "index.html"}))
        parsed = parse_chatgpt_response(raw, self.RECOVERY_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertIn("read: missing required property 'file_path'", parsed["errors"])
        self.assertIn("WebMind", parsed["content"])
        self.assertNotIn("webmind_response", parsed["content"])

    def test_correction_to_file_path_emits_tool_call(self):
        invalid = json.dumps(_envelope_tool_call("call_r1", "read", {"path": "index.html"}))
        corrected = json.dumps(_envelope_tool_call("call_r2", "read", {"file_path": "index.html"}))
        backend = ModelBackend([invalid, corrected])
        harness = Harness(self.RECOVERY_TOOLS, backend)
        harness.user("go through index.html")
        response = harness.chat_once(recover=True)
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        tc = choice["message"]["tool_calls"][0]
        self.assertEqual(tc["id"], "call_r2")
        self.assertEqual(
            json.loads(tc["function"]["arguments"]), {"file_path": "index.html"}
        )
        # The correction directive must have been visible to the model.
        correction_prompt = backend.prompts[1]
        self.assertIn(
            "read: missing required property 'file_path'", correction_prompt
        )
        self.assertIn("single allowed correction", correction_prompt)
        self.assertEqual(len(backend.prompts), 2)

    def test_unknown_tool_clean_failure_no_loop(self):
        invalid1 = json.dumps(_envelope_tool_call("call_x", "rm_rf", {"cmd": "boom"}))
        invalid2 = json.dumps(_envelope_tool_call("call_x2", "rm_rf", {"cmd": "boom"}))
        backend = ModelBackend([invalid1, invalid2])
        harness = Harness(self.RECOVERY_TOOLS, backend)
        harness.user("clean up")
        response = harness.chat_once(recover=True)
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIsNone(choice["message"].get("tool_calls"))
        content = choice["message"]["content"]
        self.assertIn("unknown tool 'rm_rf'", content)
        self.assertNotIn("webmind_response", content)
        self.assertEqual(len(backend.prompts), 2)

    def test_invalid_argument_type_recovered(self):
        bad = json.dumps(_envelope_tool_call("c1", "bash", {"command": 42}))
        fixed = json.dumps(_envelope_tool_call("c2", "bash", {"command": "pwd"}))
        backend = ModelBackend([bad, fixed])
        harness = Harness(self.RECOVERY_TOOLS, backend)
        harness.user("run pwd")
        response = harness.chat_once(recover=True)
        tc = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(tc["id"], "c2")
        self.assertEqual(json.loads(tc["function"]["arguments"]), {"command": "pwd"})

    def test_repeated_invalid_correction_yields_clean_error(self):
        bad1 = json.dumps(_envelope_tool_call("r1", "read", {"path": "a"}))
        bad2 = json.dumps(_envelope_tool_call("r2", "read", {"path": "b"}))
        backend = ModelBackend([bad1, bad2])
        harness = Harness(self.RECOVERY_TOOLS, backend)
        harness.user("read a")
        response = harness.chat_once(recover=True)
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIsNone(choice["message"].get("tool_calls"))
        self.assertIn("one permitted correction", choice["message"]["content"])
        self.assertNotIn("webmind_response", choice["message"]["content"])
        self.assertEqual(len(backend.prompts), 2)

    def test_valid_tool_call_skips_recovery(self):
        valid = json.dumps(_envelope_tool_call("ok1", "bash", {"command": "pwd"}))
        backend = ModelBackend([valid])
        harness = Harness(self.RECOVERY_TOOLS, backend)
        harness.user("pwd")
        response = harness.chat_once(recover=True)
        tc = response["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(tc["id"], "ok1")
        self.assertEqual(len(backend.prompts), 1)

    def test_message_response_skips_recovery(self):
        msg = json.dumps(_envelope_message("done"))
        backend = ModelBackend([msg])
        harness = Harness(self.RECOVERY_TOOLS, backend)
        harness.user("finished?")
        response = harness.chat_once(recover=True)
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["choices"][0]["message"]["content"], "done")
        self.assertEqual(len(backend.prompts), 1)

    def test_reissued_call_passes_recovery_unmodified(self):
        """A re-issued (previously completed) valid call is forwarded as a
        tool_call and never routes into the validation-failure recovery
        path."""
        harness = Harness(self.RECOVERY_TOOLS, ModelBackend([]))
        harness.user("list")
        tc = {
            "id": "call_c",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "dir"}'},
        }
        harness.history.append(
            {"role": "assistant", "content": None, "tool_calls": [tc]}
        )
        harness.history.append(
            {
                "role": "tool",
                "tool_call_id": "call_c",
                "name": "bash",
                "content": '{"exit_code": 0}',
            }
        )
        backend = ModelBackend([
            json.dumps(_envelope_tool_call("call_dup", "bash", {"command": "dir"})),
        ])
        harness.backend = backend
        response = harness.chat_once(recover=True)
        message = response["choices"][0]["message"]
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(message["tool_calls"][0]["id"], "call_dup")
        self.assertEqual(len(backend.prompts), 1)

    def test_multi_turn_tool_execution_with_recovery(self):
        invalid = json.dumps(_envelope_tool_call("call_m1", "read", {"path": "data.txt"}))
        corrected = json.dumps(_envelope_tool_call("call_m2", "read", {"file_path": "data.txt"}))
        final_msg = json.dumps(_envelope_message("Got it."))
        backend = ModelBackend([invalid, corrected, final_msg])
        harness = Harness(self.RECOVERY_TOOLS, backend)

        harness.user("read data.txt")
        first = harness.chat_once(recover=True)
        self.assertEqual(first["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(
            first["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read",
        )
        executed = harness.run_tool_calls(first, _noop_executor)
        self.assertEqual(
            executed,
            [("read", {"file_path": "data.txt"})],
        )

        second = harness.chat_once()
        self.assertEqual(second["choices"][0]["finish_reason"], "stop")
        self.assertEqual(second["choices"][0]["message"]["content"], "Got it.")
        self.assertEqual(len(backend.prompts), 3)
        # The second prompt carries the tool result from the corrected call.
        self.assertIn("tool_call_id: call_m2", backend.prompts[2])
        self.assertIn("[role: tool]", backend.prompts[2])

    def test_corrected_tool_call_streams_as_delta_tool_calls(self):
        invalid = json.dumps(_envelope_tool_call("s1", "read", {"path": "a.txt"}))
        corrected = json.dumps(_envelope_tool_call("s2", "read", {"file_path": "a.txt"}))
        backend = ModelBackend([invalid, corrected])
        harness = Harness(self.RECOVERY_TOOLS, backend)
        harness.user("read the file")
        chunks = harness.chat_once(stream=True, recover=True)

        message, finish = sse_to_chat_message(chunks)
        self.assertEqual(finish, "tool_calls")
        self.assertIsNone(message["content"])
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(message["tool_calls"][0]["id"], "s2")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "read")
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]),
            {"file_path": "a.txt"},
        )

    def test_correction_directive_carries_specific_errors(self):
        directive = build_correction_directive(
            ["read: missing required property 'file_path'"]
        )
        self.assertIn("read: missing required property 'file_path'", directive)
        self.assertIn("single allowed correction", directive)
        self.assertIn("TOOL DEFINITIONS", directive)


if __name__ == "__main__":
    unittest.main()
