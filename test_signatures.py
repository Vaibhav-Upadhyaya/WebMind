"""Regression suite for continuation-prompt compact tool signatures.

Proves the fix for the live turn-27 failure: on continuation turns the
prompt now carries each tool's EXACT argument names (verbatim from the
supplied OpenAI wire schemas) plus required/optional markers, without
aliases, without touching validation, and without regressing first-turn
or recovery behavior.
"""

import json
import unittest
from types import SimpleNamespace

from protocol import (
    PROMPT_TARGET_CHARS,
    TOOL_PURPOSE_MAX_CHARS,
    _TRUNCATED_TOOL_DIRECTIVE,
    _tool_signatures,
    build_prompt,
    parse_chatgpt_response,
    recover_tool_call,
    validate_arguments,
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


WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write a UTF-8 text file.",
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


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a UTF-8 text file and return line-numbered content.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["file_path"],
        },
    },
}


def cont_msgs():
    return [
        SimpleNamespace(role="system", content="You are a helpful assistant."),
        SimpleNamespace(role="user", content="build a todo page"),
        SimpleNamespace(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
            }],
        ),
        SimpleNamespace(role="user", content='{"exit_code":0,"stdout":"ok"}', tool_call_id="call_1"),
        SimpleNamespace(role="user", content="now read index.html"),
    ]


def realistic_26_tools():
    names = [
        "ask_user_question", "create_goal", "edit", "exit_plan_mode",
        "get_goal", "glob", "grep", "interrupt_agent", "job_kill",
        "job_list", "job_output", "list_agents", "present", "pwsh",
        "read", "read_image", "send_message", "skill", "subagent",
        "subagent_fork", "todo_write", "update_goal", "web_fetch",
        "web_search", "workflow", "write",
    ]
    tools = []
    for name in names:
        if name == "read":
            tools.append(READ_FILE_TOOL)
        else:
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": "placeholder",
                    "parameters": {
                        "type": "object",
                        "properties": {"input": {"type": "string"}},
                        "required": ["input"],
                    },
                },
            })
    return tools


def verbose_26_tools():
    """26 tools with realistically WORDY descriptions.

    realistic_26_tools() uses "placeholder", so its budget test cannot catch a
    purpose clause that overshoots. Real OpenCode tool descriptions are long,
    so the cap has to be measured against descriptions like these.
    """
    verbose = (
        "Use this tool when you need to replace an exact string in a file. "
        "The oldString must match exactly and appear once in the file, and "
        "you must read the file immediately before editing it so the content "
        "you are matching against is current."
    )
    tools = []
    for name in ["edit_%02d" % i for i in range(26)]:
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": verbose,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "oldString": {"type": "string"},
                        "newString": {"type": "string"},
                    },
                    "required": ["path", "oldString", "newString"],
                },
            },
        })
    return tools


class TestContinuationSignatures(unittest.TestCase):

    def test_continuation_prompt_contains_exact_parameter_names(self):
        tools = [READ_FILE_TOOL, BASH_TOOL, WRITE_TOOL]
        prompt, _ = build_prompt(cont_msgs(), tools, None)
        # The exact read signature, as derived from the schema.
        self.assertIn(
            "read(file_path: string [required], offset?: integer, limit?: integer)",
            prompt,
        )

    def test_continuation_marks_required_and_optional(self):
        prompt, _ = build_prompt(cont_msgs(), [READ_FILE_TOOL], None)
        self.assertIn("file_path: string [required]", prompt)
        self.assertIn("offset?: integer", prompt)
        self.assertIn("limit?: integer", prompt)
        self.assertNotIn("offset: integer [required]", prompt)

    def test_all_supplied_tools_are_represented(self):
        tools = [BASH_TOOL, WRITE_TOOL, READ_FILE_TOOL]
        prompt, _ = build_prompt(cont_msgs(), tools, None)
        for t in tools:
            self.assertIn(f"{t['function']['name']}(", prompt)
        self.assertIn("bash(command: string [required], timeout?: integer)", prompt)
        self.assertIn(
            "write(file_path: string [required], content: string [required])",
            prompt,
        )

    def test_continuation_stays_within_budget(self):
        tools = realistic_26_tools()
        full_defs = "TOOL DEFINITIONS:\n" + json.dumps(tools, indent=2)
        prompt, _ = build_prompt(cont_msgs(), tools, None)
        start = prompt.find("TOOL DEFINITIONS were already provided")
        end = prompt.find("CONVERSATION (chronological, oldest first):")
        self.assertNotEqual(start, -1)
        self.assertNotEqual(end, -1)
        note = prompt[start:end]
        # Comfortably below the non-continuation full schema block.
        self.assertLess(len(note), 4000)
        self.assertLess(len(note), len(full_defs) // 4)
        self.assertIn(
            "read(file_path: string [required], offset?: integer, limit?: integer)",
            note,
        )

    def test_purpose_clause_stays_within_budget_with_wordy_descriptions(self):
        # The other budget test uses "placeholder" descriptions, so it cannot
        # detect a purpose clause that overshoots. Measure against realistic
        # multi-sentence descriptions for a 26-tool set.
        tools = verbose_26_tools()
        prompt, _ = build_prompt(cont_msgs(), tools, None)
        start = prompt.find("TOOL DEFINITIONS were already provided")
        end = prompt.find("CONVERSATION (chronological, oldest first):")
        note = prompt[start:end]
        self.assertLess(len(note), 4000)

    def test_signature_carries_tool_purpose(self):
        prompt, _ = build_prompt(cont_msgs(), [READ_FILE_TOOL], None)
        self.assertIn("Read a UTF-8 text file", prompt)

    def test_purpose_is_first_sentence_only(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Replace an exact string in a file. The oldString must match exactly.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        line = _tool_signatures(tools)[0]
        self.assertIn("Replace an exact string in a file", line)
        self.assertNotIn("oldString must match exactly", line)

    def test_purpose_collapses_whitespace_and_is_capped(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "edit",
                "description": "   Multi\n  line   collapsed.  ",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        line = _tool_signatures(tools)[0]
        self.assertIn("Multi line collapsed", line)
        self.assertNotIn("\n", line)

        long_tools = [{
            "type": "function",
            "function": {
                "name": "edit",
                "description": "A very long description that rambles well past the cap.",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        long_line = _tool_signatures(long_tools)[0]
        purpose = long_line.split("\u2014", 1)[1].strip()
        self.assertLessEqual(len(purpose), TOOL_PURPOSE_MAX_CHARS)

    def test_undescribed_tool_still_renders_bare_name(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "mystery",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        self.assertEqual(_tool_signatures(tools), ["mystery()"])

    def test_first_turn_full_schema_unchanged(self):
        tools = [READ_FILE_TOOL, BASH_TOOL]
        msgs = [
            SimpleNamespace(role="system", content="You are a helpful assistant."),
            SimpleNamespace(role="user", content="read index.html"),
        ]
        prompt, _ = build_prompt(msgs, tools, None)
        self.assertIn("TOOL DEFINITIONS:\n" + json.dumps(tools, indent=2), prompt)
        self.assertNotIn("were already provided in full earlier", prompt)
        self.assertNotIn("[required]", prompt)

    def test_read_file_path_validates_normally(self):
        schema = READ_FILE_TOOL["function"]["parameters"]
        self.assertEqual(validate_arguments({"file_path": "index.html"}, schema), [])

    def test_read_path_still_rejected_not_aliased(self):
        schema = READ_FILE_TOOL["function"]["parameters"]
        errors = validate_arguments({"path": "index.html"}, schema)
        self.assertIn("missing required property 'file_path'", errors)
        parsed = parse_chatgpt_response(
            json.dumps({
                "webmind_response": "tool_call",
                "tool_calls": [{
                    "id": "call_1",
                    "name": "read",
                    "arguments": {"path": "index.html"},
                }],
            }),
            [READ_FILE_TOOL],
        )
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertEqual(parsed["errors"], ["read: missing required property 'file_path'"])
        self.assertIsNone(parsed.get("tool_calls"))

    def test_recovery_still_corrects_to_file_path(self):
        invalid = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [{
                "id": "call_1",
                "name": "read",
                "arguments": {"path": "index.html"},
            }],
        })
        corrected = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [{
                "id": "call_2",
                "name": "read",
                "arguments": {"file_path": "index.html"},
            }],
        })
        parsed = parse_chatgpt_response(invalid, [READ_FILE_TOOL])
        recovered = recover_tool_call(
            parsed, lambda prompt: corrected, [READ_FILE_TOOL]
        )
        self.assertEqual(recovered["type"], "tool_call")
        self.assertEqual(
            recovered["tool_calls"][0]["arguments"], {"file_path": "index.html"}
        )


class TestPromptSizeGracefulDegradation(unittest.TestCase):
    """A too-big prompt must still get an ANSWER.

    The browser cannot submit a prompt past a certain size as live text --
    ChatGPT converts the paste into an uploaded file and answers as an
    outside reader of it.  Returning a 413 is honest but useless to the
    caller, so build_prompt spends the OLDEST context first to fit under
    protocol.PROMPT_TARGET_CHARS, and main.PromptTooLargeError stays as the
    last-resort backstop for prompts that cannot be shrunk (a single
    enormous pending tool result).
    """

    def test_target_sits_below_the_browser_backstop(self):
        # The ordering is the whole mechanism: if the soft target ever
        # reaches or exceeds the hard ceiling, large requests fail instead
        # of degrading.  Pin it so the two numbers cannot drift apart.
        #
        # Slack, not a flat 2000: the budget covers the conversation, and
        # the assembled prompt adds a little on top of it -- the "[earlier
        # conversation omitted...]" marker and the "\n\n" between blocks
        # are both inserted after the selection pass.  PROMPT_TARGET_CHARS
        # is therefore a near-bound rather than an exact one, and the margin
        # below is what absorbs that.  A generous constant here would hide a
        # genuine regression, so keep it small and explain it.
        import main as main_mod

        self.assertLess(PROMPT_TARGET_CHARS, main_mod.MAX_INLINE_PROMPT_CHARS)
        # ~75 chars for the omission marker, plus a few separator pairs.
        self.assertLessEqual(
            PROMPT_TARGET_CHARS + 128, main_mod.MAX_INLINE_PROMPT_CHARS
        )

    def _bulky_history(self, body_chars, turns=6):
        body = "z" * body_chars
        msgs = [SimpleNamespace(role="system", content="You are a helpful assistant.")]
        for i in range(turns):
            msgs.append(SimpleNamespace(role="user", content=f"step {i}: {body}"))
            msgs.append(SimpleNamespace(role="assistant", content=f"ack {i}"))
        # A pending call + result, which must survive verbatim.
        msgs.append(SimpleNamespace(
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read", "arguments": '{\"file_path\": \"index.html\"}'},
            }],
        ))
        msgs.append(SimpleNamespace(
            role="tool", tool_call_id="call_1",
            content='{\"exit_code\":0,\"stdout\":\"ok\"}',
        ))
        return msgs

    def test_oversized_history_is_trimmed_to_target(self):
        msgs = self._bulky_history(4000)
        prompt, _ = build_prompt(msgs, realistic_26_tools(), None)
        # Without the soft target this lands far over the ceiling and gets
        # refused with no answer at all.
        self.assertLessEqual(len(prompt), PROMPT_TARGET_CHARS + 128)
        self.assertLess(len(prompt), main_prompt_ceiling())

    def test_trim_is_explicit_not_silent(self):
        # The model must never be quietly deprived of context it believes
        # it has, or it will answer from a fiction.
        prompt, _ = build_prompt(self._bulky_history(4000), realistic_26_tools(), None)
        self.assertIn("earlier conversation omitted for size", prompt)

    def test_pending_call_and_result_survive_trimming(self):
        # Dropping these breaks the continuation loop entirely, so they are
        # never a trimming candidate.  Asserted on substrings rather than
        # exact JSON: tool_calls render with arguments as an escaped string,
        # which is incidental to what is being pinned here.
        prompt, _ = build_prompt(self._bulky_history(4000), realistic_26_tools(), None)
        self.assertIn("index.html", prompt)
        self.assertIn("exit_code", prompt)
        # The directive only fires when the last message is a tool result.
        self.assertIn("You are CONTINUING an existing multi-turn tool loop", prompt)

    def test_first_turn_with_many_tools_is_not_refused(self):
        # Regression, hit live: a 26-tool OPENING request serialized 19298
        # chars and was rejected with PROMPT_TOO_LARGE, so the very first
        # message of a session could not be answered at all.  The soft
        # budget used to apply only to continuations.
        msgs = [
            SimpleNamespace(
                role="system",
                content="You are WebMind, the browser-side brain of an "
                        "OpenAI-compatible coding agent.",
            ),
            SimpleNamespace(
                role="user",
                content="ok then inspect the brain and figure out what you "
                        "just mentioned",
            ),
        ]
        prompt, _ = build_prompt(msgs, realistic_26_tools(), None)
        import main as main_mod

        self.assertLess(len(prompt), main_mod.MAX_INLINE_PROMPT_CHARS)
        # Degrade to the compact signature note rather than the full JSON.
        self.assertNotIn('"type": "function"', prompt)
        self.assertIn("read(file_path: string [required]", prompt)
        # The actual question is still there, and still the newest turn.
        self.assertIn("inspect the brain", prompt)

    def test_first_turn_keeps_its_history_when_it_fits(self):
        # The compact-schema fallback must not become a reason to throw away
        # context that would have fitted: a regression here silently emptied
        # the conversation section.
        msgs = [SimpleNamespace(
            role="system", content="You are WebMind, the browser-side brain."
        )]
        for i in range(4):
            msgs.append(SimpleNamespace(role="user", content=f"turn {i}: " + "context " * 60))
            msgs.append(SimpleNamespace(role="assistant", content=f"ack {i}: " + "reply " * 60))
        msgs.append(SimpleNamespace(role="user", content="SENTINEL: what port does it bind?"))
        prompt, _ = build_prompt(msgs, realistic_26_tools(), None)
        for i in range(4):
            self.assertIn(f"turn {i}:", prompt)
        self.assertIn("SENTINEL", prompt)
        self.assertNotIn("earlier conversation omitted for size", prompt)

    def test_the_instruction_that_caused_the_pending_call_survives(self):
        # Regression, found live: trimming the head could drop the user's
        # actual request while keeping its tool result, and the model then
        # replied that it could not infer the task from the file alone.
        # Same shape as the original bug -- a confident answer to a question
        # nobody asked.
        msgs = self._bulky_history(4000)
        msgs[-3] = SimpleNamespace(
            role="user", content="SENTINEL_TASK: report the port that server.py binds."
        )
        prompt, _ = build_prompt(msgs, realistic_26_tools(), None)
        self.assertIn("SENTINEL_TASK", prompt)
        # Older turns were still sacrificed to make room for it.
        self.assertIn("earlier conversation omitted for size", prompt)

    def test_small_history_is_untouched_by_the_soft_target(self):
        # The trim must not add an omission marker to prompts that already
        # fit comfortably.
        prompt, _ = build_prompt(cont_msgs(), realistic_26_tools(), None)
        self.assertNotIn("earlier conversation omitted for size", prompt)

    def test_retry_budget_must_actually_shrink_the_prompt(self):
        # A retry that barely shrinks is worse than useless: it spends a
        # generation to arrive at almost the same prompt, which is refused
        # again.  This is a real regression -- a single large pending tool
        # result dominates the prompt and lives in the TAIL, which the head
        # budget can never touch, so lowering the target alone moved a 13764
        # char prompt only to 13538.  The tool-result cap is the lever.
        msgs = self._bulky_history(4000)
        msgs[-1] = SimpleNamespace(
            role="tool", tool_call_id="call_1",
            content='{\"exit_code\":0,\"stdout\":\"' + "x" * 9000 + '\"}',
        )
        first, _ = build_prompt(msgs, realistic_26_tools(), None)

        # Target alone: barely moves, which is the failure mode.
        target_only, _ = build_prompt(
            msgs, realistic_26_tools(), None, target_chars=5000
        )
        # Target plus tool-result cap: genuinely smaller.
        both, _ = build_prompt(
            msgs, realistic_26_tools(), None,
            target_chars=5000, tool_role_max_chars=2000,
        )

        self.assertLess(len(both), len(first) * 0.75)
        self.assertLess(len(both), len(target_only))
        # The pending call and a truncated-but-honest result both survive.
        self.assertIn("index.html", both)
        self.assertIn("[truncated: kept", both)

    def test_target_chars_does_not_leak_across_calls(self):
        # A per-call override must not become the new default for a later
        # request, or one huge request would silently shrink every
        # subsequent conversation.
        msgs = self._bulky_history(4000)
        small, _ = build_prompt(
            msgs, realistic_26_tools(), None,
            target_chars=5000, tool_role_max_chars=2000,
        )
        normal, _ = build_prompt(msgs, realistic_26_tools(), None)
        self.assertGreater(len(normal), len(small))

    def test_full_tree_glob_retry_keeps_a_prefix_and_says_keep_going(self):
        # The reported failure: OpenCode maps a whole repo, Glob returns
        # ~11.6K chars, the paste converts to a pasted-text file, and the
        # attachment retry re-serializes with ATTACHMENT_RETRY_TOOL_RESULT_CHARS.
        #
        # The model must be handed a PREFIX plus an explicit instruction to
        # continue with a narrower glob, NOT a result that reads as missing
        # and invites "the scan was insufficient, stopping here".
        from server import (
            ATTACHMENT_RETRY_TARGET_CHARS,
            ATTACHMENT_RETRY_TOOL_RESULT_CHARS,
        )

        listing = "\n".join(
            f"src/module_{i}/file_{i}.py" for i in range(400)
        )
        listing = ((listing + "\n") * 12)[:11673]
        total = len(listing)
        self.assertGreater(total, 11000)

        msgs = [
            SimpleNamespace(
                role="system", content="You are a repository mapping agent."
            ),
            SimpleNamespace(
                role="user",
                content="Map the entire folder structure and what files are in it.",
            ),
            SimpleNamespace(
                role="assistant",
                content=None,
                tool_calls=[{
                    "id": "call_1", "type": "function",
                    "function": {
                        "name": "glob", "arguments": '{\"pattern\":\"**/*\"}',
                    },
                }],
            ),
            SimpleNamespace(
                role="tool", content=listing, tool_call_id="call_1", name="glob",
            ),
        ]

        prompt, _ = build_prompt(
            msgs, realistic_26_tools(), "auto",
            target_chars=ATTACHMENT_RETRY_TARGET_CHARS,
            tool_role_max_chars=ATTACHMENT_RETRY_TOOL_RESULT_CHARS,
        )

        # Truncated, and honestly marked as such.  The cap and the reported
        # total are both measured on the whole rendered block (the
        # "[role: tool]" header and the tool_call_id entry are part of it),
        # so the total is slightly larger than the bare tool content.
        marker = f"[truncated: kept {ATTACHMENT_RETRY_TOOL_RESULT_CHARS} of "
        idx = prompt.find(marker)
        self.assertNotEqual(idx, -1, "retry must mark the cut explicitly")
        reported_total = int(
            prompt[idx + len(marker):].split(" ", 1)[0]
        )
        self.assertGreater(reported_total, len(listing))
        # The retry must be genuinely smaller than what failed, or it would
        # convert to an attachment a second time.
        normal, _ = build_prompt(msgs, realistic_26_tools(), "auto")
        self.assertLess(len(prompt), len(normal) * 0.75)
        # The directive that drives the recovery must be present and must
        # tell the model to continue rather than report incompleteness.
        self.assertIn(_TRUNCATED_TOOL_DIRECTIVE, prompt)
        self.assertIn("NARROWER", prompt)
        self.assertIn("Do NOT reply that the scan is insufficient", prompt)
        # The pending call survives, so a narrower follow-up is expressible.
        self.assertIn("call_1", prompt)

    def test_tiered_continuation_shrinks_by_turn(self):
        msgs = [
            SimpleNamespace(role="user", content="check code"),
            SimpleNamespace(role="assistant", content="", tool_calls=[{"id": "c1", "function": {"name": "bash", "arguments": '{\"command\":\"pwd\"}'}}]),
            SimpleNamespace(role="tool", content="ok", tool_call_id="c1"),
        ]
        tools = [BASH_TOOL]
        # Turn 1 -> full directive
        prompt_t1, _ = build_prompt(msgs, tools, turn_count=1)
        # Turn 5 -> medium directive (shorter)
        prompt_t5, _ = build_prompt(msgs, tools, turn_count=5)
        # Turn 10 -> minimal directive (shortest)
        prompt_t10, _ = build_prompt(msgs, tools, turn_count=10)
        # Turn 8 -> full directive refresh
        prompt_t8, _ = build_prompt(msgs, tools, turn_count=8)

        self.assertGreater(len(prompt_t1), len(prompt_t5))
        self.assertGreater(len(prompt_t5), len(prompt_t10))
        self.assertEqual(len(prompt_t1), len(prompt_t8))
        self.assertIn("message envelope IS the user's answer", prompt_t10)


def main_prompt_ceiling():
    import main as main_mod

    return main_mod.MAX_INLINE_PROMPT_CHARS


if __name__ == "__main__":
    unittest.main()
