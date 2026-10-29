import json
import re
import unittest
from types import SimpleNamespace

from protocol import (
    _select_continuation_blocks,
    build_correction_directive,
    build_prompt,
    build_nonstream_response,
    parse_chatgpt_response,
    recover_tool_call,
    sse_message_chunks,
    sse_tool_call_chunks,
    validate_arguments,
)
import server as server_mod
from server import _extract_prompt_and_images, _has_tools


def msg(role, content=None, tool_calls=None, tool_call_id=None, name=None):
    return SimpleNamespace(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        name=name,
    )


BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a Windows PowerShell command.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
            },
            "required": ["command"],
        },
    },
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    },
}

TOOLS = [BASH_TOOL, READ_TOOL]


class TestPromptTranslation(unittest.TestCase):
    """Scenarios 1, 2, 3, 9, 10 (request translation side)."""

    def test_message_envelope_rule_present_in_every_prompt(self):
        # The rule must ride on EVERY prompt, not just continuations:
        # tool_mode_instruction is prepended unconditionally, so this is the
        # stable anchor that redirects a bail-out message into an answer.
        first_turn, _ = build_prompt([msg("user", "hi")], TOOLS)
        self.assertIn("must BE the answer to the user's request", first_turn)

        # The continuation directive is appended only when the transcript
        # ends in a tool result (role "tool"), so that is the shape that has
        # to carry the mirrored clause.
        continuation, _ = build_prompt(
            [
                msg("user", "fix it"),
                msg("assistant", tool_calls=[{"id": "c1", "function": {
                    "name": "bash", "arguments": '{"command":"pwd"}'}}]),
                msg("tool", "exit_code 0", tool_call_id="c1"),
            ],
            TOOLS,
        )
        self.assertIn("must BE the answer to the user's request", continuation)
        self.assertIn("message envelope IS the user's answer", continuation)

    def test_truncated_tool_result_instructs_to_keep_searching(self):
        # Live failure: glob **/* returned ~11K chars, attachment retry capped
        # the tool block at 2000, and the model treated the truncation marker
        # as a hard stop ("listing incomplete, need the full glob").
        listing = "\n".join(
            f"C:/Users/Developer/Documents/Projects/file_{i}.py"
            for i in range(200)
        )
        self.assertGreater(len(listing), 2000)
        prompt, _ = build_prompt(
            [
                msg("user", "map the entire folder structure"),
                msg("assistant", tool_calls=[{"id": "c1", "function": {
                    "name": "bash", "arguments": '{"command":"glob"}'}}]),
                msg("tool", listing, tool_call_id="c1", name="glob"),
            ],
            TOOLS,
            tool_role_max_chars=2000,
        )
        self.assertIn("[truncated: kept", prompt)
        self.assertIn("[truncated result]", prompt)
        self.assertIn("PREFIX", prompt)
        self.assertIn("NARROWER", prompt)
        self.assertNotIn(
            "if the result is missing, empty, or clearly erroneous, state that "
            "in a message envelope instead of calling any tool",
            prompt,
        )
        # The original request must still be in the prompt so the model
        # continues THAT job rather than inventing a new one.
        self.assertIn("map the entire folder structure", prompt)

    def test_untruncated_tool_result_does_not_get_truncated_directive(self):
        prompt, _ = build_prompt(
            [
                msg("user", "list files"),
                msg("assistant", tool_calls=[{"id": "c1", "function": {
                    "name": "bash", "arguments": '{"command":"ls"}'}}]),
                msg("tool", "a.txt\nb.txt", tool_call_id="c1"),
            ],
            TOOLS,
        )
        self.assertNotIn("[truncated result]", prompt)
        self.assertNotIn("...[truncated: kept", prompt)

    def test_message_envelope_rule_still_allows_a_question(self):
        # The rule redirects toward the task; it must not read as "never say
        # you cannot", or it would suppress legitimate blocker reports.
        prompt, _ = build_prompt([msg("user", "hi")], TOOLS)
        self.assertIn("a question you need answered", prompt)

    def test_1_prompt_includes_full_history_not_just_last_user(self):
        messages = [
            msg("system", "You are a coding agent."),
            msg("user", "first question"),
            msg("assistant", "earlier answer"),
            msg("user", "what is my cpu usage"),
        ]
        prompt, images = build_prompt(messages, TOOLS)
        self.assertIn("first question", prompt)
        self.assertIn("earlier answer", prompt)
        self.assertIn("what is my cpu usage", prompt)
        self.assertIn("TOOL DEFINITIONS:", prompt)
        self.assertIn('"bash"', prompt)
        self.assertEqual(images, [])

    def test_2_single_tool_is_serialized(self):
        messages = [msg("user", "run something")]
        prompt, _ = build_prompt(messages, [BASH_TOOL])
        self.assertIn("bash", prompt)
        self.assertIn('"command"', prompt)
        self.assertIn('"required"', prompt)

    def test_3_multiple_tools_are_serialized(self):
        messages = [msg("user", "do both")]
        prompt, _ = build_prompt(messages, TOOLS)
        self.assertIn("bash", prompt)
        self.assertIn("read", prompt)

    def test_9_assistant_tool_calls_preserved_in_history(self):
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }
        ]
        messages = [
            msg("user", "pwd"),
            msg("assistant", "", tool_calls=tool_calls),
            msg("tool", '{"exit_code":0}', name="bash", tool_call_id="call_123"),
            msg("user", "now what"),
        ]
        prompt, _ = build_prompt(messages, TOOLS)
        self.assertIn("[role: assistant]", prompt)
        self.assertIn("tool_calls:", prompt)
        self.assertIn("call_123", prompt)

    def test_10_tool_role_preserved_in_history(self):
        messages = [
            msg("user", "pwd"),
            msg("assistant", "ok", tool_calls=[]),
            msg("tool", '{"exit_code": 0, \"stdout\": \"C:\\\\\\\\x\"}', name="bash", tool_call_id="call_abc"),
        ]
        prompt, _ = build_prompt(messages, TOOLS)
        self.assertIn("[role: tool]", prompt)
        self.assertIn("tool_name: bash", prompt)
        self.assertIn("tool_call_id: call_abc", prompt)

    def test_image_urls_collected_from_last_user_message_only(self):
        messages = [
            msg("user", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]),
            msg(
                "user",
                [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            ),
        ]
        prompt, images = build_prompt(messages, TOOLS)
        self.assertEqual(images, ["http://x/y.png"])
        self.assertIn("look", prompt)

    def test_empty_messages_rejected(self):
        with self.assertRaises(ValueError):
            build_prompt([], TOOLS)


class TestEnvelopeParsing(unittest.TestCase):
    """Scenarios 4, 5, 6, 7, 8 (parsing / validation side)."""

    def test_4_valid_tool_call_envelope(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_111", "name": "bash", "arguments": {"command": "Get-Date"}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        tc = parsed["tool_calls"][0]
        self.assertEqual(tc["name"], "bash")
        self.assertEqual(tc["id"], "call_111")
        self.assertEqual(tc["arguments"], {"command": "Get-Date"})

    def test_4_b_arguments_as_json_string_accepted(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_2", "name": "bash", "arguments": '{"command":"ls"}'},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"command": "ls"})

    def test_4_c_code_fence_wrapped_envelope_accepted(self):
        raw = '```json\n{"webmind_response":"message","content":"hi there"}\n```'
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed, {"type": "message", "content": "hi there"})

    def test_4_d_single_object_tool_call_accepted(self):
        raw = ('```json\n{"webmind_response":"tool_call",'
               '"tool_calls":{"id":"call_single","name":"bash",'
               '"arguments":{"command":"Get-Date"}}}\n```')
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["id"], "call_single")
        self.assertEqual(parsed["tool_calls"][0]["name"], "bash")

    def test_4_e_fenced_tool_call_with_preamble_accepted(self):
        # Scenario from user: ChatGPT generates normal text commentary first,
        # then emits ```json { "webmind_response": "tool_call", ... } ```
        raw = (
            "We've located the three core files in web fps: index.html, game.js, and style.css. "
            "I'll inspect the entry point and game initialization together.\n"
            "```json\n"
            "{\n"
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {\n'
            '      "id": "call_fps_read_201",\n'
            '      "name": "read",\n'
            '      "arguments": {"path": "index.html"}\n'
            '    },\n'
            '    {\n'
            '      "id": "call_fps_read_202",\n'
            '      "name": "read",\n'
            '      "arguments": {"path": "game.js"}\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "```"
        )
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 2)
        self.assertEqual(parsed["tool_calls"][0]["id"], "call_fps_read_201")
        self.assertEqual(parsed["tool_calls"][0]["name"], "read")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"path": "index.html"})
        self.assertEqual(parsed["tool_calls"][1]["id"], "call_fps_read_202")
        self.assertEqual(parsed["tool_calls"][1]["name"], "read")
        self.assertEqual(parsed["tool_calls"][1]["arguments"], {"path": "game.js"})

    def test_4_f_fenced_tool_call_with_preamble_no_newline(self):
        # Attached directly without newline: "...check.```json\n{...}\n```"
        raw = (
            "I will patch the confirmed failure and run a focused check.```json\n"
            "{\n"
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {"id": "c1", "name": "bash", "arguments": {"command": "dir"}}\n'
            '  ]\n'
            "}\n"
            "```"
        )
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "bash")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"command": "dir"})

    def test_4_g_fenced_tool_call_with_postamble_accepted(self):
        raw = (
            "```json\n"
            "{\n"
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {"id": "c2", "name": "read", "arguments": {"path": "app.py"}}\n'
            '  ]\n'
            "}\n"
            "```\n"
            "Let me know if you want me to read more files."
        )
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "read")
        self.assertEqual(parsed["tool_calls"][0]["arguments"], {"path": "app.py"})

    def test_5_invalid_json_returns_raw_text_as_message(self):
        raw = '{"webmind_response": "tool_call", "tool_calls": ['
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(parsed["content"], raw)

    def test_5_c_invalid_json_escapes_are_repaired(self):
        # ChatGPT emitted \P and \% (invalid JSON escapes); the repair must
        # preserve the literal backslash and produce a valid tool_call.
        raw = ('{"webmind_response":"tool_call","tool_calls":['
               '{"id":"call_7f3a9c21","name":"bash",'
               '"arguments":{"command":"Get-Counter \''
               '\\\\Processor(_Total)\\\\% Processor Time\'"}}]}')
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        args = parsed["tool_calls"][0]["arguments"]
        self.assertEqual(args["command"], "Get-Counter '\\\\Processor(_Total)\\% Processor Time'")

    def test_5_c_2_raw_interior_quotes_in_message_repaired(self):
        # ChatGPT wrote prose with unescaped quotes inside the envelope's
        # content string, making the object invalid JSON.  The quote repair
        # must recover the intended message instead of passing the raw
        # envelope through as plain text.
        raw = ('{"webmind_response":"message","content":'
               '"The result says only "alpha" rather than what was asked."}')
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(
            parsed["content"],
            'The result says only "alpha" rather than what was asked.',
        )

    def test_5_c_3_raw_interior_quotes_in_tool_arguments_repaired(self):
        # A quoted argument value ("echo "hello world"") is invalid JSON;
        # the repair must still yield the executable tool call.
        raw = ('{"webmind_response":"tool_call","tool_calls":['
               '{"id":"call_q","name":"bash","arguments":'
               '{"command":"echo "hello world""}}]}')
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(
            parsed["tool_calls"][0]["arguments"]["command"],
            'echo "hello world"',
        )

    def test_5_d_windows_path_backslash_kept_literal(self):
        # Regression: ChatGPT emits a Windows path with SINGLE backslashes
        # (invalid JSON). "\b" in "\brain.py" is a VALID JSON escape (a
        # backspace char) -- it must be kept as a literal path separator,
        # not decoded into ...\x08rain.py which breaks fs.read.  The "\b"
        # below is a single backslash + "b" (a real path separator), NOT a
        # Python backspace escape.
        raw = ('{"webmind_response":"tool_call","tool_calls":['
               '{"id":"call_w","name":"read",'
               '"arguments":{"path":".\\\\brain.py"}}]}')
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        args = parsed["tool_calls"][0]["arguments"]
        self.assertEqual(
            args["path"],
            ".\\brain.py",
        )
        # No control characters may survive the decode.
        self.assertNotIn("\x08", args["path"])

    def test_5_e_windows_path_in_arguments_json_string(self):
        # The same, but ChatGPT handed us arguments as a JSON-encoded string
        # (doubled backslashes in the envelope, singles after decode).
        raw = ('{"webmind_response":"tool_call","tool_calls":['
               '{"id":"call_r","name":"read",'
               '"arguments":"{\\\\"path\\\\":\\\\"C:\\\\\\\\Users\\\\\\\\Developer\\\\\\\\Documents\\\\\\\\WebMind\\\\\\\\brain.py\\\\"}"}]}')
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        args = parsed["tool_calls"][0]["arguments"]
        self.assertEqual(
            args["path"],
            "C:\\Users\\Developer\\Documents\\WebMind\\brain.py",
        )
        self.assertNotIn("\x08", args["path"])

    def test_5_b_plain_prose_returns_as_message(self):
        raw = "I can't run that for you, but here is how you could do it yourself."
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(parsed["content"], raw)

    def test_6_unsupplied_tool_is_classified_as_validation_failure(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_x", "name": "exec", "arguments": {"cmd": "evil"}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertIn("unknown tool", parsed["errors"][0])
        self.assertNotIn("webmind_response", parsed["content"])
        self.assertNotIn("tool_calls", parsed["content"])
        self.assertIn("WebMind", parsed["content"])

    def test_7_malformed_arguments_classified_as_validation_failure(self):
        # Missing required 'command'.
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_y", "name": "bash", "arguments": {"timeout": 5}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertIn("missing required property 'command'", parsed["errors"][0])

    def test_7_b_wrong_typed_argument_classified_as_validation_failure(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_z", "name": "bash", "arguments": {"command": 42}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertIn("property 'command' must be", parsed["errors"][0])

    def test_7_c_arguments_not_object_classified_as_validation_failure(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_w", "name": "bash", "arguments": "nope"},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertIn("could not be parsed", parsed["errors"][0])

    def test_validation_failure_content_is_clean_not_raw(self):
        # The raw envelope must never leak into the assistant message.
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_l", "name": "bash", "arguments": {"timeout": 5}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertEqual(parsed["content"].strip(), parsed["content"])
        self.assertNotIn("webmind_response", parsed["content"])
        self.assertNotIn("tool_calls", parsed["content"])

    def test_8_multiple_tool_calls(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_a", "name": "bash", "arguments": {"command": "pwd"}},
                {"id": "call_b", "name": "read", "arguments": {"path": "main.py"}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual([t["id"] for t in parsed["tool_calls"]], ["call_a", "call_b"])

    def test_generated_id_when_missing(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"name": "bash", "arguments": {"command": "pwd"}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertTrue(parsed["tool_calls"][0]["id"].startswith("call_"))

    def test_message_envelope_unwrapped(self):
        raw = json.dumps({"webmind_response": "message", "content": "hello world"})
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed, {"type": "message", "content": "hello world"})

    def test_unknown_envelope_falls_back_to_text(self):
        raw = json.dumps({"foo": "bar"})
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(parsed["content"], raw)

    def test_tool_name_case_sensitive_exact_match(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"name": "Bash", "arguments": {"command": "pwd"}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertIn("unknown tool 'Bash'", parsed["errors"][0])

    def test_message_content_decodes_over_escaped_newlines(self):
        # The model wrote "\\n" in the envelope source, so json.loads
        # yields a literal backslash-n. Left alone, the whole answer
        # renders on one line with visible escape markers.
        raw = json.dumps({
            "webmind_response": "message",
            "content": "Step one.\\\\nStep two.\\\\n\\\\tindented",
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(
            parsed["content"], "Step one.\nStep two.\n\tindented"
        )

    def test_message_content_preserves_windows_path_segments(self):
        # The repair is positional: a drive-letter path stays byte-exact
        # across EVERY segment while ordinary escapes in the same string
        # still decode. A one-segment-only guard would pass C:\new but
        # corrupt the second separator in C:\new\notes.txt.
        original = (
            "line one\\\\nline two\\\\\n"
            "open C:\\\\new\\\\notes.txt and C:\\\\node_modules\\\\pkg then\\\\nlast"
        )
        expected = (
            "line one\nline two\n"
            "open C:\\\\new\\\\notes.txt and C:\\\\node_modules\\\\pkg then\nlast"
        )
        raw = json.dumps({"webmind_response": "message", "content": original})
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["content"], expected)

    def test_message_content_does_not_treat_prose_colon_as_drive(self):
        # "todo:" and "a: 30" are prose, not drive letters, so their
        # escapes must still decode.
        original = "todo:\\\\nthen at a: 30\\\\no'clock"
        expected = "todo:\nthen at a: 30\no'clock"
        raw = json.dumps({"webmind_response": "message", "content": original})
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["content"], expected)

    def test_tool_call_arguments_are_left_byte_exact(self):
        # Tool-call arguments are never routed through the cosmetic
        # repair: the model writes those escapes correctly, so decoding
        # them here would alter tool payloads that work today.
        arguments = {"command": "line one\\\\nline two"}
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_nl", "name": "bash", "arguments": arguments},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(
            parsed["tool_calls"][0]["arguments"], {"command": "line one\\\\nline two"}
        )


class TestToolCallRecovery(unittest.TestCase):
    """One-attempt correction of a validation-failed tool call envelope."""

    READ_FILE_TOOL = {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        },
    }
    RECOVERY_TOOLS = [BASH_TOOL, READ_FILE_TOOL]

    def _invalid_read(self):
        return json.dumps(_envelope_tool_call("call_r", "read", {"path": "index.html"}))

    def test_validation_failure_on_path_instead_of_file_path(self):
        # The live failure: the supplied schema requires file_path, the model
        # proposed path.  This must be classified, not degraded to text.
        parsed = parse_chatgpt_response(self._invalid_read(), self.RECOVERY_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertEqual(parsed["errors"], ["read: missing required property 'file_path'"])

    def test_corrected_call_emits_tool_calls(self):
        def ask(prompt):
            return json.dumps(_envelope_tool_call("call_r2", "read", {"file_path": "index.html"}))

        parsed = parse_chatgpt_response(self._invalid_read(), self.RECOVERY_TOOLS)
        recovered = recover_tool_call(parsed, ask, self.RECOVERY_TOOLS)
        self.assertEqual(recovered["type"], "tool_call")
        self.assertEqual(
            recovered["tool_calls"][0],
            {
                "id": "call_r2",
                "name": "read",
                "arguments": {"file_path": "index.html"},
            },
        )

    def test_no_rename_or_fabrication_of_arguments(self):
        # The correction must never silently rewrite path -> file_path.  A
        # corrected envelope that still does not satisfy the schema (the file
        # path hidden under another key) must NOT become a tool call.
        def ask(prompt):
            return json.dumps(_envelope_tool_call("call_r3", "read", {"pair": {"path": "index.html"}}))

        parsed = parse_chatgpt_response(self._invalid_read(), self.RECOVERY_TOOLS)
        recovered = recover_tool_call(parsed, ask, self.RECOVERY_TOOLS)
        self.assertEqual(recovered["type"], "message")
        self.assertIn("one permitted correction", recovered["content"])
        self.assertIn("file_path", recovered["content"])

    def test_repeated_invalid_correction_yields_clean_error(self):
        def ask(prompt):
            return self._invalid_read()

        parsed = parse_chatgpt_response(self._invalid_read(), self.RECOVERY_TOOLS)
        recovered = recover_tool_call(parsed, ask, self.RECOVERY_TOOLS)
        self.assertEqual(recovered["type"], "message")
        self.assertIn("one permitted correction", recovered["content"])
        self.assertNotIn("webmind_response", recovered["content"])
        self.assertNotIn("tool_calls", recovered["content"])

    def test_ask_exception_falls_back_to_clean_error(self):
        def ask(prompt):
            raise RuntimeError("browser hiccup")

        parsed = parse_chatgpt_response(self._invalid_read(), self.RECOVERY_TOOLS)
        recovered = recover_tool_call(parsed, ask, self.RECOVERY_TOOLS)
        self.assertEqual(recovered["type"], "message")
        self.assertIn("WebMind", recovered["content"])
        self.assertIn("file_path", recovered["content"])

    def test_correction_directive_carries_specific_error(self):
        directive = build_correction_directive(
            ["read: missing required property 'file_path'"]
        )
        self.assertIn(
            "read: missing required property 'file_path'", directive
        )
        self.assertIn("single allowed correction", directive)

    def test_non_validation_result_passes_through(self):
        def ask(prompt):
            raise AssertionError("must not be called")

        parsed = parse_chatgpt_response(
            json.dumps(_envelope_tool_call("call_ok", "bash", {"command": "pwd"})),
            self.RECOVERY_TOOLS,
        )
        recovered = recover_tool_call(parsed, ask, self.RECOVERY_TOOLS)
        self.assertEqual(recovered["type"], "tool_call")
        self.assertEqual(recovered["tool_calls"][0]["name"], "bash")


def _envelope_tool_call(call_id, name, arguments):
    return {
        "webmind_response": "tool_call",
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
    }


class TestArgumentValidation(unittest.TestCase):
    def test_required_and_types(self):
        schema = {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
                "paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["command"],
        }
        self.assertEqual(validate_arguments({"command": "ls"}, schema), [])
        self.assertNotEqual(validate_arguments({"timeout": 5}, schema), [])
        self.assertNotEqual(validate_arguments({"command": 5}, schema), [])
        self.assertNotEqual(validate_arguments({"command": "ls", "timeout": "5"}, schema), [])
        self.assertEqual(
            validate_arguments({"command": "ls", "paths": ["a", "b"]}, schema), []
        )
        self.assertEqual(validate_arguments({}, schema)[0].startswith("missing"), True)


class TestResponseBuilders(unittest.TestCase):
    """Scenarios 11, 12 (server-facing output shapes)."""

    def test_11_nonstream_tool_call_response(self):
        parsed = {
            "type": "tool_call",
            "tool_calls": [
                {"id": "call_900", "name": "bash", "arguments": {"command": "pwd"}},
            ],
        }
        payload = build_nonstream_response(parsed, "chatcmpl-test", 123)
        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        message = choice["message"]
        self.assertIsNone(message["content"])
        tc = message["tool_calls"][0]
        self.assertEqual(tc["id"], "call_900")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "bash")
        self.assertEqual(tc["function"]["arguments"], '{"command": "pwd"}')
        self.assertIsInstance(tc["function"]["arguments"], str)

    def test_11_b_nonstream_message_response(self):
        parsed = {"type": "message", "content": "plain answer"}
        payload = build_nonstream_response(parsed, "chatcmpl-test", 123)
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["choices"][0]["message"]["content"], "plain answer")

    def test_12_sse_tool_call_sequence(self):
        parsed = {
            "type": "tool_call",
            "tool_calls": [
                {"id": "call_a", "name": "bash", "arguments": {"command": "pwd"}},
            ],
        }
        chunks = sse_tool_call_chunks("chatcmpl-x", 123, parsed["tool_calls"])
        self.assertEqual(chunks[-1].strip(), "data: [DONE]")

        first = json.loads(chunks[0][len("data: "):])
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")

        role_delta = json.loads(chunks[1][len("data: "):])["choices"][0]["delta"]
        tc = role_delta["tool_calls"][0]
        self.assertEqual(tc["index"], 0)
        self.assertEqual(tc["id"], "call_a")
        self.assertEqual(tc["type"], "function")
        self.assertEqual(tc["function"]["name"], "bash")

        args_delta = json.loads(chunks[2][len("data: "):])["choices"][0]["delta"]
        self.assertEqual(
            json.loads(args_delta["tool_calls"][0]["function"]["arguments"]),
            {"command": "pwd"},
        )

        finish = json.loads(chunks[-2][len("data: "):])["choices"][0]
        self.assertEqual(finish["finish_reason"], "tool_calls")

    def test_13_sse_message_sequence_unchanged_shape(self):
        chunks = sse_message_chunks("chatcmpl-x", 123, "hello world")
        self.assertEqual(chunks[-1].strip(), "data: [DONE]")
        texts = []
        for chunk in chunks:
            if chunk.startswith("data: "):
                body = chunk[len("data: "):]
                if body.strip() == "[DONE]":
                    continue
                obj = json.loads(body)
                texts.append(obj["choices"][0]["delta"].get("content", ""))
        self.assertEqual("".join(texts), "hello world")
        finish = json.loads(chunks[-2][len("data: "):])["choices"][0]
        self.assertEqual(finish["finish_reason"], "stop")


class TestServerLegacyPaths(unittest.TestCase):
    """Scenarios 1 & 14: existing text/image behavior untouched."""

    def test_tool_mode_detection(self):
        req = SimpleNamespace(tools=[BASH_TOOL])
        self.assertTrue(_has_tools(req))
        req2 = SimpleNamespace(tools=None)
        self.assertFalse(_has_tools(req2))
        req3 = SimpleNamespace(tools=[])
        self.assertFalse(_has_tools(req3))

    def test_legacy_last_user_text_prompt(self):
        request = server_mod.ChatRequest(
            model="webmind",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ],
        )
        prompt, images = _extract_prompt_and_images(request)
        self.assertEqual(prompt, "three")
        self.assertEqual(images, [])

    def test_14_legacy_image_only_prompt(self):
        request = server_mod.ChatRequest(
            model="webmind",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                    ],
                }
            ],
        )
        prompt, images = _extract_prompt_and_images(request)
        self.assertEqual(prompt, server_mod.DEFAULT_IMAGE_PROMPT)
        self.assertEqual(images, ["http://x/a.png"])

    def test_14_b_legacy_text_plus_image_prompt(self):
        request = server_mod.ChatRequest(
            model="webmind",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is in this image"},
                        {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                    ],
                }
            ],
        )
        prompt, images = _extract_prompt_and_images(request)
        self.assertEqual(prompt, "what is in this image")
        self.assertEqual(images, ["http://x/a.png"])


    def test_oversized_single_message_is_capped_not_refused(self):
        # Regression, hit live: opencode's auto-compaction posts the whole
        # history as ONE user message (61029 chars observed) with no tools.
        # That path had no budget, so the prompt reached the browser and was
        # rejected by its hard backstop -- compaction could then never
        # succeed, and opencode reported "Conversation history too large to
        # compact" on every attempt.
        import main as main_mod
        from protocol import LEGACY_PROMPT_MAX_CHARS

        body = "SUMMARIZE THE FOLLOWING CONVERSATION:\n" + ("transcript line\n" * 5000)
        request = server_mod.ChatRequest(
            model="webmind",
            messages=[{"role": "user", "content": body}],
        )
        prompt, _ = _extract_prompt_and_images(request)

        self.assertLess(len(prompt), main_mod.MAX_INLINE_PROMPT_CHARS)
        self.assertLessEqual(len(prompt), LEGACY_PROMPT_MAX_CHARS + 80)
        # The instruction leads these requests, so the head must survive...
        self.assertTrue(prompt.startswith("SUMMARIZE THE FOLLOWING CONVERSATION:"))
        # ...and the cut must be declared, never silent.
        self.assertIn("[truncated: kept", prompt)

    def test_normal_single_message_is_untouched_by_the_cap(self):
        request = server_mod.ChatRequest(
            model="webmind",
            messages=[{"role": "user", "content": "just a normal question"}],
        )
        prompt, _ = _extract_prompt_and_images(request)
        self.assertEqual(prompt, "just a normal question")

    def test_cap_legacy_prompt_honors_an_explicit_limit(self):
        # The retry needs a materially smaller prompt, so the helper takes a
        # limit instead of hardcoding the pre-flight cap.
        from protocol import LEGACY_PROMPT_MAX_CHARS, cap_legacy_prompt

        body = "HEAD MARKER\n" + ("filler line\n" * 5000)
        smaller = cap_legacy_prompt(body, 4000)

        self.assertLess(len(smaller), len(cap_legacy_prompt(body)))
        self.assertLessEqual(len(smaller), 4000 + 80)
        self.assertTrue(smaller.startswith("HEAD MARKER"))
        self.assertIn("[truncated: kept", smaller)

    def test_recapping_an_already_capped_prompt_does_not_stack_markers(self):
        # The retry re-caps the already-capped prompt, so the first marker
        # must fall outside the new cut instead of being duplicated.
        from protocol import cap_legacy_prompt

        body = "HEAD MARKER\n" + ("filler line\n" * 5000)
        once = cap_legacy_prompt(body)
        twice = cap_legacy_prompt(once, 4000)

        self.assertEqual(twice.count("[truncated: kept"), 1)
        self.assertLess(len(twice), len(once))

    def test_legacy_retry_target_is_materially_smaller_than_the_cap(self):
        # A retry at the same size would earn the identical attachment
        # verdict, so the second attempt must be well under the first.
        from protocol import LEGACY_PROMPT_MAX_CHARS, LEGACY_RETRY_MAX_CHARS

        self.assertLess(LEGACY_RETRY_MAX_CHARS, LEGACY_PROMPT_MAX_CHARS)
        self.assertLessEqual(LEGACY_RETRY_MAX_CHARS, 5000)


class TestLenientRepairRegression(unittest.TestCase):
    """Regression for the live failure: ChatGPT's envelope was structurally
    and lexically malformed (raw HTML attribute quotes, literal newlines, and
    a missing closing brace), which rejected parsing so the envelope leaked
    through SSE as ordinary assistant text.  Each fix below must recover the
    envelope into a real tool_call / message instead."""

    def test_missing_closing_brace_repaired_to_tool_call(self):
        # The real submitted envelope: raw interior HTML attribute quotes AND
        # a dropped root closer -- text still starts '{' and ends '}', but
        # the JSON is one brace short.
        raw = ('{"webmind_response":"tool_call","tool_calls":['
               '{"id":"call_todo","name":"bash","arguments":'
               '{"command":"<html lang=\"en\" charset=\"UTF-8\">'
               '<title>Todo</title></html>"}]}')
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        args = parsed["tool_calls"][0]["arguments"]
        self.assertEqual(
            args["command"],
            '<html lang="en" charset="UTF-8"><title>Todo</title></html>',
        )

    def test_literal_newlines_inside_string_repaired(self):
        raw = (
            '{"webmind_response":"tool_call","tool_calls":['
            '{"id":"call_n","name":"bash","arguments":'
            '{"command":"echo hi'
            '\n'
            'there"}}]}'
        )
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(
            parsed["tool_calls"][0]["arguments"]["command"],
            "echo hi\nthere",
        )

    def test_missing_brace_raw_quotes_and_newline_together(self):
        raw = (
            '{"webmind_response":"tool_call","tool_calls":['
            '{"id":"call_c","name":"bash","arguments":'
            '{"command":"create todo:\\\\n'
            '<html lang=\"en\" charset=\"UTF-8\"></html>"}]}'
        )
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(
            parsed["tool_calls"][0]["arguments"]["command"],
            'create todo:\n<html lang="en" charset="UTF-8"></html>',
        )

    def test_balanced_payload_unchanged_by_repairs(self):
        # A well-formed envelope must pass through byte-identical: the new
        # repair stages must never alter already-valid input.
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_ok", "name": "bash", "arguments": {"command": "pwd"}},
            ],
        })
        self.assertEqual(parse_chatgpt_response(raw, TOOLS)["type"], "tool_call")
        self.assertEqual(
            parse_chatgpt_response(raw, TOOLS)["tool_calls"],
            [{"id": "call_ok", "name": "bash", "arguments": {"command": "pwd"}}],
        )

    def test_json_copy_download_bare_envelope_repaired(self):
        raw = 'jsonCopyDownload{\n  "webmind_response": "tool_call",\n  "tool_calls": [\n    {"id": "call_1", "name": "read", "arguments": {"path": "game.js"}}\n  ]\n}'
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "read")

    def test_json_copy_download_fenced_envelope_repaired(self):
        raw = 'jsonCopyDownload```json\n{\n  "webmind_response": "tool_call",\n  "tool_calls": [\n    {"id": "call_2", "name": "read", "arguments": {"path": "game.js"}}\n  ]\n}\n```'
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "read")

    def test_truncated_envelope_still_plain_text(self):
        # test_5 invariant: a reply that does not END like a JSON object
        # ('{...[') stays plain text -- balancing must not turn it into a
        # (fake) empty tool_call.
        raw = '{"webmind_response": "tool_call", "tool_calls": ['
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(parsed["content"], raw)


class TestSseTranslationPipeline(unittest.TestCase):
    """End-to-end: a raw ChatGPT envelope -> OpenAI-compatible SSE stream,
    proving tool_call envelopes become real choices[].delta.tool_calls.  This
    is exactly the leak that used to happen: an unparseable envelope was
    piped through sse_message_chunks and the raw JSON spoke as assistant
    text."""

    def _parse_chunks(self, chunks):
        return [json.loads(c[len("data: "):]) for c in chunks if c.startswith("data: ") and c.strip() != "data: [DONE]"]

    def stream_text(self, chunks):
        return "".join(
            (o["choices"][0]["delta"].get("content") or "")
            for o in self._parse_chunks(chunks)
        )

    def test_message_envelope_streams_as_content_not_tool_calls(self):
        raw = json.dumps({"webmind_response": "message", "content": "hello world"})
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "message")
        chunks = sse_message_chunks("chatcmpl-1", 123, parsed["content"])
        self.assertEqual(chunks[-1].strip(), "data: [DONE]")
        for obj in self._parse_chunks(chunks):
            self.assertNotIn("tool_calls", obj["choices"][0]["delta"])
        self.assertEqual(self.stream_text(chunks), "hello world")
        finish = json.loads(chunks[-2][len("data: "):])["choices"][0]
        self.assertEqual(finish["finish_reason"], "stop")

    def test_tool_call_envelope_streams_as_tool_calls_not_content(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_a", "name": "bash", "arguments": {"command": "pwd"}},
                {"id": "call_b", "name": "read", "arguments": {"path": "main.py"}},
            ],
        })
        parsed = parse_chatgpt_response(raw, TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        chunks = sse_tool_call_chunks("chatcmpl-2", 123, parsed["tool_calls"])
        self.assertEqual(chunks[-1].strip(), "data: [DONE]")

        objs = self._parse_chunks(chunks)
        # Every delta is a tool-call delta; none emits assistant content.
        for obj in objs[:-1]:
            delta = obj["choices"][0]["delta"]
            if "tool_calls" in delta:
                for tc in delta["tool_calls"]:
                    self.assertIsNone(delta.get("content"))
                    if "id" in tc:
                        self.assertEqual(tc["type"], "function")
                    else:
                        # The arguments-only fragment carries valid JSON.
                        json.loads(tc["function"]["arguments"])

        role_deltas = [o["choices"][0]["delta"] for o in objs[:-1]]
        first = role_deltas[1]["tool_calls"][0]
        self.assertEqual(first["index"], 0)
        self.assertEqual(first["id"], "call_a")
        self.assertEqual(first["function"]["name"], "bash")

        args_chunk = role_deltas[2]["tool_calls"][0]
        self.assertEqual(
            json.loads(args_chunk["function"]["arguments"]),
            {"command": "pwd"},
        )

        second = role_deltas[3]["tool_calls"][0]
        self.assertEqual(second["index"], 1)
        self.assertEqual(second["id"], "call_b")
        self.assertEqual(second["function"]["name"], "read")

        args_chunk = role_deltas[4]["tool_calls"][0]
        self.assertEqual(
            json.loads(args_chunk["function"]["arguments"]),
            {"path": "main.py"},
        )

        finish = json.loads(chunks[-2][len("data: "):])["choices"][0]
        self.assertEqual(finish["finish_reason"], "tool_calls")
        self.assertEqual(finish["delta"], {})

    def test_malformed_envelope_still_streams_as_real_tool_calls(self):
        # The original failure, end-to-end: the broken envelope (raw HTML
        # attribute quotes, literal newline in content, missing root brace)
        # must now surface as delta.tool_calls and NOT as assistant content.
        raw = (
            '{"webmind_response":"tool_call","tool_calls":['
            '{"id":"call_ff","name":"write","arguments":'
            '{"file_path":"index.html","content":"<html lang=\"en\">'
            '\n'
            '<head></head></html>"}}]}'
        )
        tools = TOOLS + [{
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        }]
        parsed = parse_chatgpt_response(raw, tools)
        self.assertEqual(parsed["type"], "tool_call")
        tc = parsed["tool_calls"][0]
        self.assertEqual(tc["name"], "write")
        self.assertEqual(
            tc["arguments"]["content"],
            '<html lang="en">\n<head></head></html>',
        )

        chunks = sse_tool_call_chunks("chatcmpl-3", 123, parsed["tool_calls"])
        self.assertEqual(chunks[-1].strip(), "data: [DONE]")
        objs = self._parse_chunks(chunks)
        self.assertEqual(
            objs[1]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_ff",
        )

        joined = self.stream_text(chunks)
        # The raw envelope bytes never reach the assistant-content stream.
        self.assertNotIn("webmind_response", joined)
        self.assertNotIn("tool_calls", joined)


class TestSpuriousSandboxAndPhantomFix(unittest.TestCase):
    EDIT_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file contents.",
                "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Edit file contents.",
                "parameters": {"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}},
            },
        },
    ]

    def test_sandbox_link_triggers_validation_failure(self):
        raw = "I fixed the deployment blockers, including the invalid player spawn position, and rebuilt the game files. Download the fixed project: sandbox:/mnt/data/web-fps-fixed.zip"
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("sandbox" in e or "Code Interpreter" in e for e in parsed["errors"]))

    def test_phantom_fix_without_edits_triggers_validation_failure(self):
        raw = '{"webmind_response": "message", "content": "I fixed the deployment blockers and rebuilt the game files."}'
        history = [
            {"role": "user", "content": "fix the game"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "read"}}]},
            {"role": "tool", "content": "file contents"},
        ]
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS, history=history)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("without calling edit or write" in e for e in parsed["errors"]))

    def test_fix_message_allowed_after_actual_edit(self):
        raw = '{"webmind_response": "message", "content": "I fixed the deployment blockers and rebuilt the game files."}'
        history = [
            {"role": "user", "content": "fix the game"},
            {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "edit"}}]},
            {"role": "tool", "content": "ok"},
        ]
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS, history=history)
        self.assertEqual(parsed["type"], "message")
        self.assertIn("I fixed the deployment blockers", parsed["content"])

    def test_correction_directive_mentions_live_workspace(self):
        directive = build_correction_directive(["model hallucinated a sandbox:/mnt/data or Code Interpreter download link"])
        self.assertIn("NO python sandbox", directive)
        self.assertIn("LIVE local workspace", directive)
        self.assertIn("edit", directive)

    def test_tool_deprivation_phrase_triggers_validation_failure(self):
        raw = (
            "I’m blocked from applying the fix because this session does not expose the live workspace edit/read tools. "
            "The supplied session confirms the project files are index.html, game.js, and style.css, but I cannot safely modify "
            "or verify them without filesystem access. :contentReferenceoaicite:0{index=0} :contentReferenceoaicite:1{index=1}"
        )
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("filesystem access or workspace tools" in e for e in parsed["errors"]))

    def test_deprivation_correction_directive_lists_real_tools(self):
        directive = build_correction_directive(["model falsely claimed it lacks filesystem access or workspace tools"])
        self.assertIn("You DO have full workspace tools exposed", directive)
        self.assertIn("read", directive)
        self.assertIn("edit", directive)
        self.assertIn("write", directive)

    def test_build_prompt_neutralizes_prior_deprivation_refusal(self):
        messages = [
            {"role": "user", "content": "fix the game"},
            {
                "role": "tool",
                "content": "I’m blocked from applying the fix because this session does not expose the live workspace edit/read tools.",
                "tool_call_id": "c1",
            },
        ]
        prompt, _ = build_prompt(messages, self.EDIT_TOOLS)
        self.assertIn("[important notice]", prompt)
        self.assertIn("were false hallucinations", prompt)
        self.assertIn("All workspace tools", prompt)

    def test_post_edit_pasted_transcript_refusal_triggers_validation_failure(self):
        raw = "I can’t continue the live workspace edit from this turn because the workspace tool session is no longer available; the pasted transcript only contains the previous WebMind session, not an editable workspace connection."
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")

    def test_post_edit_correction_directive_requests_summary(self):
        history = [
            {"role": "user", "content": "fix game"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "edit"}}]},
            {"role": "tool", "content": "ok"},
        ]
        directive = build_correction_directive(
            ["model falsely claimed tool session is no longer available"],
            history=history,
        )
        self.assertIn("ALREADY successfully edited", directive)
        self.assertIn("summarizing the fixes", directive)

    def test_recover_tool_call_post_edit_clean_summary_fallback(self):
        history = [
            {"role": "user", "content": "fix game"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "edit"}}]},
            {"role": "tool", "content": "ok"},
        ]
        parsed = {
            "type": "validation_failure",
            "errors": ["model falsely claimed tool session is no longer available"],
        }
        # Model stubbornly returns the refusal again
        stubborn_refusal = "I can’t continue because the pasted transcript is not an editable workspace."
        recovered = recover_tool_call(parsed, lambda _: stubborn_refusal, self.EDIT_TOOLS, history=history)
        self.assertEqual(recovered["type"], "message")
        self.assertIn("completed the requested changes", recovered["content"])

    def test_bash_file_read_redirected_to_read_tool(self):
        raw = '```json\n{"webmind_response": "tool_call", "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "Get-Content -Raw .\\\\game.js"}}]}\n```'
        parsed = parse_chatgpt_response(raw, [BASH_TOOL, READ_TOOL])
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("read" in e and "preferred" in e for e in parsed["errors"]))

    def test_chained_bash_rejected(self):
        raw = '```json\n{"webmind_response": "tool_call", "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "cd dir; dir; node --version"}}]}\n```'
        parsed = parse_chatgpt_response(raw, [BASH_TOOL])
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("chains" in e for e in parsed["errors"]))

    def test_meta_report_detected_as_spurious(self):
        raw = "I inspected the available execution results. However, the attempted file-content inspection did not return the actual source code, so I cannot yet reliably identify the runtime bugs. No files have been modified."
        parsed = parse_chatgpt_response(raw, [BASH_TOOL, READ_TOOL])
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("narrated tool failure" in e for e in parsed["errors"]))

    def test_escape_wrapped_tool_call_parsed(self):
        raw = '<escape>{"webmind_response": "tool_call", "tool_calls": [{"id": "c1", "name": "edit", "arguments": {"filePath": "game.js", "oldString": "a", "newString": "b"}}]}</escape>'
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "edit")
        self.assertEqual(parsed["tool_calls"][0]["arguments"]["filePath"], "game.js")

    def test_escape_wrapped_message_unwrapped(self):
        raw = '<escape>\n{\n  "webmind_response": "message",\n  "content": "Here is the explanation."\n}\n</escape>'
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(parsed["content"], "Here is the explanation.")
        self.assertNotIn("<escape>", parsed["content"])

    def test_escape_tags_stripped_from_plain_text(self):
        raw = "<escape>Here is a direct answer without json.</escape>"
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(parsed["content"], "Here is a direct answer without json.")
        self.assertNotIn("<escape>", parsed["content"])

    def test_prose_containing_envelope_remains_message(self):
        raw = 'Here is the result: {"webmind_response": "message", "content": "Embedded message."}'
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "message")
        self.assertEqual(parsed["content"], raw)

    def test_workspace_editing_tools_unavailable_detected_as_spurious(self):
        raw = (
            "I've identified a definite CSS mismatch in your game. "
            "I can't safely apply a fix to your live Windows project from this session because "
            "the workspace editing tools aren't available here. I won't claim I've modified your files."
        )
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("filesystem access or workspace tools" in e for e in parsed["errors"]))

    def test_directory_not_accessible_detected_as_spurious(self):
        raw = (
            '<escape>{\n'
            '  "webmind_response": "message",\n'
            '  "content": "I can identify the CSS bug, but I cannot honestly claim to have fixed your live project yet: '
            'the Windows project directory is not accessible in this execution environment. '
            'I will not claim that files were modified without being able to edit and test the actual project."\n'
            '}</escape>'
        )
        parsed = parse_chatgpt_response(raw, self.EDIT_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("filesystem access or workspace tools" in e for e in parsed["errors"]))

    def test_directory_accessibility_correction_directive(self):
        errors = ["model falsely claimed it lacks filesystem access or workspace tools"]
        directive = build_correction_directive(errors)
        self.assertIn("directly accessible", directive)
        self.assertIn("edit", directive)
        self.assertIn("write", directive)

    def test_openai_style_tool_calls_with_function_and_aliases_parsed(self):
        # As seen in the screenshot: ChatGPT emits OpenAI-native tool_calls format
        # with function.name = "read_file" and stringified JSON arguments.
        tools_with_offset = [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filePath": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["filePath"],
                    },
                },
            }
        ]
        raw = json.dumps({
            "tool_calls": [
                {
                    "id": "call_read_gamejs_tail01",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "filePath": "C:\\game.js",
                            "startLine": 80,
                            "endLine": 240,
                        }),
                    },
                }
            ]
        })
        parsed = parse_chatgpt_response(raw, tools_with_offset)
        self.assertEqual(parsed["type"], "tool_call")
        tc = parsed["tool_calls"][0]
        self.assertEqual(tc["name"], "read")
        self.assertEqual(tc["arguments"]["filePath"], "C:\\game.js")
        self.assertEqual(tc["arguments"]["offset"], 80)
        self.assertEqual(tc["arguments"]["limit"], 161)

    def test_correction_directive_includes_full_context_and_signatures(self):
        history = [{"role": "user", "content": "the game is not loading, fix it"}]
        directive = build_correction_directive(
            ["model narrated tool failure / diagnostic state in plain text"],
            history=history,
            tools=self.EDIT_TOOLS,
        )
        self.assertIn("REQUIRED RESPONSE FORMAT", directive)
        self.assertIn("USER GOAL", directive)
        self.assertIn("the game is not loading", directive)
        self.assertIn("TOOL DEFINITIONS", directive)

    def test_raw_tool_calls_in_correction_never_leak_as_content(self):
        parsed = {
            "type": "validation_failure",
            "errors": ["some error"],
        }
        # Model returns broken tool calls that cannot validate
        broken = '{"tool_calls": [{\"name\": \"non_existent_tool_xyz\"}]}'
        recovered = recover_tool_call(parsed, lambda _: broken, self.EDIT_TOOLS)
        self.assertEqual(recovered["type"], "message")
        self.assertNotIn("tool_calls", recovered["content"])
        self.assertIn("WebMind:", recovered["content"])

    def test_multi_file_read_retention_under_luna_profile(self):
        """Under luna profile, sequential reads of multiple files (e.g. game.js, index.html, style.css)
        must all be retained in prompt context rather than pruned out."""
        game_js_content = "function updatePlayer() { player.x += player.vx; player.y += player.vy; }\n" * 40
        index_html_content = "<!DOCTYPE html><html><body><canvas id='c'></canvas></body></html>\n" * 30
        style_css_content = "body { margin: 0; padding: 0; background: #111; }\n" * 30

        messages = [
            {"role": "user", "content": "fix the game"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_read_1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "game.js"})}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_read_1", "content": game_js_content},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_read_2", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "index.html"})}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_read_2", "content": index_html_content},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_read_3", "type": "function", "function": {"name": "read", "arguments": json.dumps({"filePath": "style.css"})}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_read_3", "content": style_css_content},
        ]

        prompt, _ = build_prompt(messages, self.EDIT_TOOLS, model="gpt-5.6-luna")
        self.assertIn("updatePlayer", prompt)
        self.assertIn("<!DOCTYPE html>", prompt)
        self.assertIn("background: #111", prompt)

    def test_file_inspection_prioritized_over_large_bash_output(self):
        """When budget is constrained, read tool outputs must be prioritized over bash dumps."""
        game_code = "const GAME_LOOP_RUNNING = true; function tick() {}\n" * 20
        large_bash_dump = "Directory listing:\n" + ("mode: rw-r--r-- size: 1024 name: dummy\n" * 150)

        blocks = [
            "[role: assistant]\ntool_calls: [{\"id\": \"c1\", \"name\": \"bash\"}]",
            f"[role: tool]\n{large_bash_dump}\ntool_name: bash\ntool_call_id: c1",
            "[role: assistant]\ntool_calls: [{\"id\": \"c2\", \"name\": \"read\"}]",
            f"[role: tool]\n{game_code}\ntool_name: read\ntool_call_id: c2",
        ]

        # Budget large enough for read + tool_calls, but not large enough for the bulky bash dump
        tail_budget = len(game_code) + 800
        selected = _select_continuation_blocks(blocks, tail_budget=tail_budget)
        joined = "\n".join(selected)

        self.assertIn("GAME_LOOP_RUNNING", joined)
        self.assertNotIn("mode: rw-r--r--", joined)

    def test_spurious_meta_report_lack_of_source_code_rejected(self):
        """Model refusing to work due to claimed lack of source code must trigger validation_failure."""
        samples = [
            "The available conversation contains only a partial CSS excerpt and prior inspection attempts, not enough source code to identify the exact failure without guessing.",
            "The available evidence confirms that index.html, game.js, and style.css exist, but it does not contain the complete JavaScript source. I cannot responsibly identify or patch the root cause without that evidence. No code changes have been made yet.",
            "The next step is to read game.js to see the implementation.",
            "The next diagnostic steps are to inspect the HTML and JS files.",
        ]
        for s in samples:
            parsed = parse_chatgpt_response(s, self.EDIT_TOOLS)
            self.assertEqual(parsed["type"], "validation_failure", f"Failed to reject sample: {s}")
            self.assertTrue(any("narrated tool failure" in err for err in parsed["errors"]))

    def test_spurious_supplied_transcript_deprivation_refusal_rejected(self):
        """Refusals claiming direct edit access is unavailable or that editing transcript does not modify game must be rejected."""
        samples = [
            "the available tools in this session do not provide direct read/edit access to your live Windows project directory. The supplied transcript contains the project code, but editing that transcript would not modify your actual game.",
            "the editing interface required to modify your live Windows project is not available through the tools in this session, so I cannot truthfully claim to have edited or tested the project.",
            "I cannot modify your actual game files from this session because editing the transcript would not modify your live project.",
        ]
        for s in samples:
            parsed = parse_chatgpt_response(s, self.EDIT_TOOLS)
            self.assertEqual(parsed["type"], "validation_failure", f"Failed to reject: {s}")
            self.assertTrue(
                any("lacks filesystem access" in err or "only has a transcript" in err for err in parsed["errors"]),
                f"Unexpected errors: {parsed.get('errors')}",
            )

    def test_build_prompt_redacts_assistant_deprivation_refusal(self):
        """Prior assistant deprivation excuses in history must be redacted to avoid self-reinforcing hallucinations."""
        messages = [
            {"role": "user", "content": "inspect the game"},
            {
                "role": "assistant",
                "content": "the available tools in this session do not provide direct read/edit access to your live Windows project directory. The supplied transcript contains the project code, but editing that transcript would not modify your actual game.",
            },
            {"role": "user", "content": "fix the game"},
        ]
        prompt, _ = build_prompt(messages, self.EDIT_TOOLS)
        self.assertIn("[prior assistant refusal message omitted", prompt)
        self.assertNotIn("do not provide direct read/edit access", prompt)
        self.assertIn("[important notice]", prompt)
        self.assertIn("were false hallucinations", prompt)
        self.assertIn("All workspace tools", prompt)


class TestToolCallLeakAndMultiEnvelopeHandling(unittest.TestCase):
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Edit file contents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run shell command.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "workdir": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                },
            },
        },
    ]

    def test_user_snippet_bare_json_tool_calls_with_sextuple_backtick_message(self):
        """User reproduction: bare JSON tool_calls followed by sextuple backtick message envelope."""
        snippet = (
            '{\n'
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {\n'
            '      "id": "call_fix_muzzle_spawn_scope_1226",\n'
            '      "name": "edit",\n'
            '      "arguments": {\n'
            '        "path": "C:\\\\Users\\\\Developer\\\\Documents\\\\Projects\\\\fps_game\\\\game.js",\n'
            '        "oldString": "player.ammo--;shootCooldown=.105;tracers.push({x:W*.64+weaponSway+Math.cos(-.075)*184*Math.min(1,W/1050,H/700),y:H*.80+Math.sin(-.075)*184*Math.min(1,W/1050,H/700)-2*Math.min(1,W/1050,H/700),life:.11,maxLife:.11});",\n'
            '        "newString": "player.ammo--;shootCooldown=.105;const scale=Math.min(1,W/1050,H/700),sway=playing&&!paused?Math.sin(performance.now()*.006)*(keys.KeyW||keys.KeyA||keys.KeyS||keys.KeyD?3:.7):0;tracers.push({x:W*.64+sway+Math.cos(-.075)*184*scale,y:H*.80+Math.sin(-.075)*184*scale-2*scale,life:.11,maxLife:.11});"\n'
            '      }\n'
            '    },\n'
            '    {\n'
            '      "id": "call_remove_static_tracer_line_1226",\n'
            '      "name": "edit",\n'
            '      "arguments": {\n'
            '        "path": "C:\\\\Users\\\\Developer\\\\Documents\\\\Projects\\\\fps_game\\\\game.js",\n'
            '        "oldString": "// Tracer originates at the transformed muzzle and converges on the crosshair.",\n'
            '        "newString": "// Moving tracer particles are rendered after the world, from the muzzle toward the crosshair."\n'
            '      }\n'
            '    },\n'
            '    {\n'
            '      "id": "call_validate_tracer_fix_1226",\n'
            '      "name": "shell",\n'
            '      "arguments": {\n'
            '        "command": "node --check game.js",\n'
            '        "workdir": "C:\\\\Users\\\\Developer\\\\Documents\\\\Projects\\\\fps_game",\n'
            '        "timeout": 120000\n'
            '      }\n'
            '    }\n'
            '  ]\n'
            '}\n'
            '``````json id="webmind_tracer_fix_handoff_1227"\n'
            '{\n'
            '  "webmind_response": "message",\n'
            '  "content": "The next validation checks the firing-scope bug: `shoot()` was referencing `weaponSway`."\n'
            '}\n'
            '``````'
        )

        parsed = parse_chatgpt_response(snippet, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 3)
        self.assertEqual(parsed["tool_calls"][0]["name"], "edit")
        self.assertEqual(parsed["tool_calls"][1]["name"], "edit")
        self.assertEqual(parsed["tool_calls"][2]["name"], "bash")
        self.assertIn("The next validation checks the firing-scope bug", parsed.get("content", ""))

        # Non-stream response verifies content is preserved with finish_reason: tool_calls
        nonstream = build_nonstream_response(parsed, "chat-123", 1000)
        choice = nonstream["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIn("The next validation checks the firing-scope bug", choice["message"]["content"])
        self.assertEqual(len(choice["message"]["tool_calls"]), 3)

        # Streaming SSE chunks verify chunk 0 carries content and terminal chunk has finish_reason: tool_calls
        chunks = sse_tool_call_chunks("chat-123", 1000, parsed["tool_calls"], content=parsed.get("content"))
        first_payload = json.loads(chunks[0][len("data: "):])
        self.assertIn("The next validation checks the firing-scope bug", first_payload["choices"][0]["delta"]["content"])
        self.assertEqual(first_payload["choices"][0]["delta"]["role"], "assistant")

        terminal_payload = json.loads(chunks[-2][len("data: "):])
        self.assertEqual(terminal_payload["choices"][0]["finish_reason"], "tool_calls")

    def test_multiple_tool_call_envelopes_merged(self):
        """ChatGPT emitting multiple distinct tool_call code blocks merges them into one tool_call."""
        raw = (
            '```json\n'
            '{"webmind_response": "tool_call", "tool_calls": [{\"id\": \"c1\", \"name\": \"read\", \"arguments\": {\"path\": \"a.js\"}}]}\n'
            '```\n\n'
            '```json\n'
            '{"webmind_response": "tool_call", "tool_calls": [{\"id\": \"c2\", \"name\": \"read\", \"arguments\": {\"path\": \"b.js\"}}]}\n'
            '```'
        )
        parsed = parse_chatgpt_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 2)
        self.assertEqual(parsed["tool_calls"][0]["id"], "c1")
        self.assertEqual(parsed["tool_calls"][1]["id"], "c2")

    def test_tool_call_with_arbitrary_fences_and_attributes(self):
        """Code fences with 4-6 backticks and attributes like id=\"...\" are parsed cleanly."""
        for ticks in ("````", "`````", "``````"):
            raw = (
                f'{ticks}json id="step_42"\n'
                '{\n'
                '  "webmind_response": "tool_call",\n'
                '  "tool_calls": [{\"id\": \"c1\", \"name\": \"read\", \"arguments\": {\"path\": \"main.py\"}}]\n'
                '}\n'
                f'{ticks}'
            )
            parsed = parse_chatgpt_response(raw, self.TOOLS)
            self.assertEqual(parsed["type"], "tool_call", f"Failed for ticks: {ticks}")
            self.assertEqual(len(parsed["tool_calls"]), 1)
            self.assertEqual(parsed["tool_calls"][0]["name"], "read")


class TestEditFailureRecoveryDirective(unittest.TestCase):
    def test_chatgpt_prompt_appends_edit_failure_recovery(self):
        msgs = [
            {"role": "user", "content": "Fix the bug in strings.c"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "edit", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Could not find oldString in src/strings.c. It must match exactly, including whitespace and indentation."},
        ]
        prompt, _ = build_prompt(msgs, [READ_TOOL, BASH_TOOL], None)
        self.assertIn("[edit failure recovery]", prompt)
        self.assertIn("The previous `edit` tool call failed because `oldString` was not found", prompt)
        self.assertIn("EXACT STRING EDITING", prompt)

    def test_deepseek_prompt_appends_edit_failure_recovery(self):
        import deepseek_protocol
        msgs = [
            {"role": "user", "content": "Fix dp.c"},
            {"role": "tool", "tool_call_id": "c1", "content": "Could not find oldString in src/dp.c. It must match exactly."},
        ]
        prompt = deepseek_protocol.build_deepseek_prompt(msgs, [{"type": "function", "function": {"name": "edit", "parameters": {}}}])
        self.assertIn("[edit failure recovery]", prompt)
        self.assertIn("EXACT STRING EDITING", prompt)

    def test_kimi_prompt_appends_edit_failure_recovery(self):
        import kimi_protocol
        msgs = [
            {"role": "user", "content": "Fix engine.c"},
            {"role": "tool", "tool_call_id": "c1", "content": "Could not find oldString in src/engine.c. It must match exactly."},
        ]
        prompt = kimi_protocol.build_kimi_prompt(msgs, [{"type": "function", "function": {"name": "edit", "parameters": {}}}])
        self.assertIn("[edit failure recovery]", prompt)
        self.assertIn("EXACT STRING EDITING", prompt)

    def test_glm_prompt_appends_edit_failure_recovery(self):
        import glm_protocol
        msgs = [
            {"role": "user", "content": "Fix strings.c"},
            {"role": "tool", "tool_call_id": "c1", "content": "Could not find oldString in src/strings.c."},
        ]
        prompt = glm_protocol.build_glm_prompt(msgs, [{"type": "function", "function": {"name": "edit", "parameters": {}}}])
        self.assertIn("[edit failure recovery]", prompt)
        self.assertIn("EXACT STRING EDITING", prompt)


if __name__ == "__main__":
    unittest.main()
