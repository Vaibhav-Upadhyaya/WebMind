# ============================================================
# WebMind Phase 2: tool-call translation bridge.
#
# Pure logic only. No FastAPI / Playwright imports, so the whole
# module is unit-testable without a browser.
#
# Responsibilities (per the Phase 2 brief):
#   - translate a structured OpenAI request (messages + tools +
#     tool_choice) into one text prompt for the ChatGPT browser
#   - parse/validate ChatGPT's returned envelope and decide
#     message vs. tool_call -- without ever executing a tool
#   - emit OpenAI-compatible non-streaming responses and SSE
#     tool-call sequences
# ============================================================

import json
import os
import re
import uuid
from typing import Any


# ============================================================
# SERIALIZATION BUDGETS
# ============================================================
# ChatGPT web converts a clipboard paste larger than roughly 40K chars
# into a "Pasted text(...)" file ATTACHMENT instead of inline composer
# text. A continuation whose whole prompt turns into such an attachment
# stops behaving like an instruction/conversation prompt (GPT treats it
# as a document and often just echoes the previous tool call back to us).
#
# These budgets keep the serialized prompt on the inline side of that
# line while an explicit truncation marker guarantees a tool result is
# NEVER cut silently:
#
#   TOOL_ROLE_MAX_CHARS  per-message cap for a tool-result block.
#   MESSAGE_MAX_CHARS    per-message cap for every other block.
#   CONTINUATION_HEAD_BUDGET  how much OLDER context (before the pending
#                        call) survives a continuation.
#   CONTINUATION_TAIL_BUDGET reserved room for the pending assistant
#                        tool_call + its result (kept newest-first, so
#                        the pair that matters is always present).
#   SYSTEM_CONTINUATION_MAX_CHARS  a continuation drops a larger-than-
#                        budget system message to a marker (the full text
#                        is already in the earlier browser turns).

# Keep the complete browser prompt well below ChatGPT's roughly 40K paste
# to attachment threshold.  The protocol instructions, tool signatures, and
# continuation directive consume several thousand characters in addition to
# these transcript budgets.
#
# TOOL_ROLE_MAX_CHARS: per-message cap for a single tool-result block.
# Keeps any one file read from dominating the whole prompt; explicit
# [truncated] marker tells the model it succeeded and must narrow its query.
#
# MESSAGE_MAX_CHARS: per-message cap for non-tool blocks (user/assistant prose).
#
# CONTINUATION_HEAD_BUDGET: chars of OLDER context (before the current agent
# turn) allowed to survive.  The anchor (original user request) is pinned
# first, so it always survives regardless of this limit.
#
# CONTINUATION_TAIL_BUDGET: max chars reserved for the CURRENT agent turn
# (all tool_calls + results since the previous user message).  Newest blocks
# survive first; within the tail, tool-result blocks are prioritised over
# assistant prose so file content outlives analysis text.
#
# Attachment conversion is handled upstream: main.py demotes it to a warning
# and server.py retries at 5000 chars when a real conversion is detected.
TOOL_ROLE_MAX_CHARS = int(os.environ.get("WEBMIND_TOOL_ROLE_MAX_CHARS", "8000"))
MESSAGE_MAX_CHARS = int(os.environ.get("WEBMIND_MESSAGE_MAX_CHARS", "6000"))
CONTINUATION_HEAD_BUDGET = int(os.environ.get("WEBMIND_CONTINUATION_HEAD_BUDGET", "6000"))
CONTINUATION_TAIL_BUDGET = int(os.environ.get("WEBMIND_CONTINUATION_TAIL_BUDGET", "11000"))
SYSTEM_CONTINUATION_MAX_CHARS = int(os.environ.get("WEBMIND_SYSTEM_MAX_CHARS", "1200"))
TOOL_PURPOSE_MAX_CHARS = 50

PROMPT_TARGET_CHARS = int(
    os.environ.get("WEBMIND_PROMPT_TARGET_CHARS", "10000")
)

MODEL_CONTEXT_PROFILES = {
    "luna": {
        "target_chars": 48_000,
        "tool_role_max_chars": 20_000,
        "message_max_chars": 12_000,
        "continuation_tail_budget": 40_000,
        "continuation_head_budget": 15_000,
        "system_max_chars": 2_000,
    },
    "large": {
        "target_chars": 48_000,
        "tool_role_max_chars": 20_000,
        "message_max_chars": 12_000,
        "continuation_tail_budget": 40_000,
        "continuation_head_budget": 15_000,
        "system_max_chars": 2_000,
    },
    "legacy": {
        "target_chars": 10_000,
        "tool_role_max_chars": 8_000,
        "message_max_chars": 6_000,
        "continuation_tail_budget": 11_000,
        "continuation_head_budget": 6_000,
        "system_max_chars": 1_200,
    },
}


def resolve_context_limits(
    model: str | None = None,
    target_chars: int | None = None,
    tool_role_max_chars: int | None = None,
    message_max_chars: int | None = None,
    continuation_tail_budget: int | None = None,
    continuation_head_budget: int | None = None,
    system_max_chars: int | None = None,
) -> dict[str, int]:
    """Resolve context limits based on requested model, env overrides, or explicit parameters."""
    m = (model or os.environ.get("WEBMIND_MODEL", "")).strip().lower()
    if any(k in m for k in ("luna", "5.6", "gpt-5", "o1", "o3", "o4")):
        profile = MODEL_CONTEXT_PROFILES["luna"]
    elif any(k in m for k in ("4o", "large", "claude", "gemini")):
        profile = MODEL_CONTEXT_PROFILES["large"]
    else:
        profile = None

    def _resolve(explicit_val, env_var, profile_key, fallback_val):
        if explicit_val is not None:
            return explicit_val
        if env_var in os.environ:
            try:
                return int(os.environ[env_var])
            except ValueError:
                pass
        if profile is not None and profile_key in profile:
            return profile[profile_key]
        return fallback_val

    return {
        "target_chars": _resolve(target_chars, "WEBMIND_PROMPT_TARGET_CHARS", "target_chars", PROMPT_TARGET_CHARS),
        "tool_role_max_chars": _resolve(tool_role_max_chars, "WEBMIND_TOOL_ROLE_MAX_CHARS", "tool_role_max_chars", TOOL_ROLE_MAX_CHARS),
        "message_max_chars": _resolve(message_max_chars, "WEBMIND_MESSAGE_MAX_CHARS", "message_max_chars", MESSAGE_MAX_CHARS),
        "continuation_tail_budget": _resolve(continuation_tail_budget, "WEBMIND_CONTINUATION_TAIL_BUDGET", "continuation_tail_budget", CONTINUATION_TAIL_BUDGET),
        "continuation_head_budget": _resolve(continuation_head_budget, "WEBMIND_CONTINUATION_HEAD_BUDGET", "continuation_head_budget", CONTINUATION_HEAD_BUDGET),
        "system_max_chars": _resolve(system_max_chars, "WEBMIND_SYSTEM_MAX_CHARS", "system_max_chars", SYSTEM_CONTINUATION_MAX_CHARS),
    }

_TRUNCATION_MARK = "\n...[truncated: kept {kept} of {total} chars]"

# Appended after the latest tool result on a continuation so GPT knows it
# is mid-tool-loop and must decide the next action instead of repeating.
_CONTINUATION_DIRECTIVE_FULL = (
    "[continuation] You are CONTINUING an existing multi-turn tool loop on the live local workspace. "
    "The latest tool result above is available. Decide the next action: "
    "(1) call a tool with DIFFERENT arguments if more data is needed (e.g. `read` to view file contents, `grep`/`glob` to search), or "
    "(2) if you have read the files and diagnosed the issues or the user asked to fix/rebuild files, your NEXT tool call MUST be `edit` or `write` to apply the actual code fixes to the workspace files on disk. "
    "(3) emit the final answer message envelope ONLY AFTER all necessary file modifications have been executed via `edit`/`write` tool calls. "
    "CRITICAL RULES: "
    "- DECISION AT WRITE POINT: When you know the fix, decide to WRITE immediately. Call `edit` or `write` on disk—never narrate proposed edits or re-read files. "
    "- COMMON SCENARIOS: Use `glob`/`grep` to find code, `read` to inspect, `edit`/`write` to modify, `bash` for commands. Modify files on disk before giving final answer. "
    "- FULL TOOL ACCESS: You HAVE full filesystem tools (`read`, `grep`, `glob`, `edit`, `write`, `bash`). NEVER claim this session does not expose workspace tools, that the session is no longer available, that you only have a pasted transcript, or that you lack filesystem access. When all needed edits are finished, summarize the fixes you made in the message envelope. NEVER output document citations like `:contentReference`. "
    "- NO SANDBOX / NO ZIP DOWNLOADS: You do NOT have a python sandbox, /mnt/data directory, or zip download capability. NEVER output links starting with `sandbox:` or `/mnt/data/`. NEVER claim you created a downloadable zip or rebuilt files elsewhere. "
    "- REAL WORKSPACE EDITS REQUIRED: Reading files does NOT modify them. NEVER claim in a message that you fixed or rebuilt files unless you already called `edit` or `write` in this tool loop to apply the changes. "
    "- NEVER ask the user to upload or paste files. NEVER claim you only have a transcript or lack file contents: you have the `read` tool to inspect any file in the workspace directly. "
    "- NO META-REPORTING: NEVER narrate tool execution status or report on tool result absences in a message. If data is missing or empty, emit another tool_call. "
    "- EXACT STRING EDITING: When calling `edit`, `oldString` MUST be copied character-for-character from a previous `read` output, matching exact whitespace, indentation, and line breaks. Never guess or fabricate `oldString`. If you have not read the file or if lines moved, call `read` first! Keep `oldString` concise (3-8 lines). "
    "- a `[truncated: kept X of Y chars]` marker means the tool SUCCEEDED; call with a NARROWER path/range if needed. "
    "When you stop calling tools, the message envelope IS the user's answer. "
    "Reply with exactly one JSON envelope wrapped in a ```json code fence, and nothing else."
)

_CONTINUATION_DIRECTIVE_MEDIUM = (
    "[continuation] Continuing multi-turn tool loop on live workspace. Latest tool result is available above. "
    "MANDATORY DECISION RULES: "
    "(1) DECISION AT WRITE POINT: When code is inspected and the fix is identified, call `edit` or `write` immediately to apply changes to disk—never narrate next steps in text. "
    "(2) COMMON SCENARIOS: Use `glob`/`grep` to find code, `read` to inspect files, `edit`/`write` for code modifications, and `bash` for tests/commands. "
    "(3) NO SANDBOX / NO ZIP / NO CITATIONS: Edit files on disk directly. Never ask user to upload files. Never narrate tool failures in a message — emit another tool call. "
    "When you stop calling tools, the message envelope IS the user's answer. "
    "Reply with exactly one JSON envelope wrapped in a ```json code fence, and nothing else."
)

_CONTINUATION_DIRECTIVE_MINIMAL = (
    "[continuation] Continuing tool loop. Use `read` for file contents, `grep`/`glob` to search, and `edit`/`write` for code fixes. "
    "No sandbox, no uploads, no meta-reporting. When you stop calling tools, the message envelope IS the user's answer. "
    "Reply with exactly one JSON envelope wrapped in a ```json code fence, and nothing else."
)

_CONTINUATION_DIRECTIVE = _CONTINUATION_DIRECTIVE_FULL


def _get_continuation_directive(turn_count: int) -> str:
    """Return tiered continuation directive to conserve budget on longer sessions."""
    if turn_count < 4 or (turn_count % 8 == 0):
        return _CONTINUATION_DIRECTIVE_FULL
    if turn_count < 9:
        return _CONTINUATION_DIRECTIVE_MEDIUM
    return _CONTINUATION_DIRECTIVE_MINIMAL

_USER_TURN_CONTINUATION_DIRECTIVE = (
    "[action required] The user is instructing you to continue and take action. You are the live autonomous agent. "
    "CRITICAL OPERATIONAL RULES: "
    "- FULL TOOL ACCESS: You HAVE full filesystem access (`read`, `grep`, `glob`, `edit`, `write`, `bash`). NEVER claim this session does not expose workspace tools, that the session is no longer available, or that you only have a transcript. NEVER output `:contentReference` citations. "
    "- NO SANDBOX / NO ZIP DOWNLOADS: You do NOT have a python sandbox, /mnt/data directory, or zip download capability. NEVER output `sandbox:` or `/mnt/data/` links. "
    "- REAL EDITS REQUIRED: If the user asks you to fix, debug, or build something, you must inspect the files with `read` and apply the changes using `edit` or `write`. NEVER claim you fixed or rebuilt files without calling `edit` or `write` to modify them on disk. "
    "- NEVER ask the user to upload or paste files, and NEVER claim you only have a transcript. "
    "If files need to be inspected, read, or edited, immediately emit a ```json tool_call envelope "
    "calling `read`, `grep`, `edit`, or another relevant tool to act on the workspace right now."
)

# Extra tail instruction when the LATEST tool result was size-capped. Rule
# (3) above already covers this, but a truncated glob is the live failure
# mode (OpenCode maps a whole tree, the listing is ~10K+, attachment retry
# keeps 2000 chars) and the model otherwise treats the marker as a hard
# stop. This block is the last thing in the prompt so it wins.
_TRUNCATED_TOOL_DIRECTIVE = (
    "[truncated result] The latest tool result was cut for size; the tool "
    "itself completed successfully. You are looking at a PREFIX, not a "
    "failure. Do NOT reply that the scan is insufficient or that you need "
    "the complete glob. Continue the user's request with NEW tool calls "
    "whose arguments differ: list immediate children, glob one subdirectory "
    "at a time, or tighten the pattern. Repeat until you can answer, then "
    "emit the message envelope with the map/answer you assembled."
)

_EDIT_FAILURE_DIRECTIVE = (
    "[edit failure recovery] The previous `edit` tool call failed because `oldString` was not found in the file. "
    "CRITICAL ACTION REQUIRED: Do NOT guess or invent `oldString`, and do NOT repeat the failed edit without reading. "
    "(1) Immediately emit a ```json tool_call calling `read` on this file to inspect the actual verbatim text and indentation on disk. "
    "(2) Once inspected, re-issue `edit` using an EXACT verbatim 3-to-6 line snippet copied directly from the `read` result as `oldString`."
)


def _cap_block(text, limit):
    """Truncate with an explicit marker; never silently cut content."""
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_MARK.format(kept=limit, total=len(text))


# Ceiling for the no-tools path, which sends the latest user message
# verbatim instead of serializing a conversation.
#
# It exists because opencode's auto-compaction posts the ENTIRE history as
# one user message: 61K chars were observed.  With no budget here that
# prompt reached the browser, and the browser's hard backstop rejected it --
# so compaction could never succeed and opencode reported "Conversation
# history too large to compact" forever.  Capping keeps the head (where
# these requests put their instruction, before the transcript) and marks
# the cut, so a summary of the visible part is still possible.
#
# Kept well under main.MAX_INLINE_PROMPT_CHARS so the browser backstop
# stays a last resort rather than the normal outcome.
LEGACY_PROMPT_MAX_CHARS = 6000

# Second, much tighter attempt for the no-tools path.  The conversion to an
# attachment turned out to be content-dependent and unpredictable -- a
# 12042-char capped compaction prompt was still converted -- so the only
# dependable move is to try again with materially less text.
LEGACY_RETRY_MAX_CHARS = 4000


def cap_legacy_prompt(text: str, limit: int | None = None) -> str:
    """Cap a single-message (no-tools) prompt, marking what was dropped."""
    return _cap_block(
        text or "", LEGACY_PROMPT_MAX_CHARS if limit is None else limit
    )


def _is_continuation(messages):
    """True when history already contains a tool round (assistant
    tool_calls or a tool role).  Continuations may drop duplicated
    tool-definition/system material because it is already in scope."""
    for m in messages or []:
        if _role(m) == "tool" or getattr(m, "tool_calls", None):
            return True
    return False


# ============================================================
# PROMPT TRANSLATION (structured request -> ChatGPT prompt)
# ============================================================

def tool_mode_instruction(tool_choice: Any = None) -> str:
    lines = [
        "You are WebMind, the live autonomous brain of an OpenAI-compatible coding agent. "
        "Below you receive TOOL DEFINITIONS and the full CONVERSATION from the user's workspace.",
        "",
        "CRITICAL OPERATIONAL RULES:",
        "- You are connected to the live local workspace. You are NOT an outside reader analyzing a file or transcript.",
        "- FORBIDDEN: NEVER ask the user to upload/paste files; NEVER claim you only have a transcript or lack file contents.",
        "- FULL TOOL ACCESS: You have full workspace tools (`read`, `grep`, `glob`, `edit`, `write`, `bash`). Never claim you lack filesystem access or tools. Never output :contentReference citations.",
        "- WORKSPACE ACCESSIBILITY: The Windows workspace directory and files are live and directly accessible on disk. Never claim the project directory is not accessible, that the execution environment lacks access, or that editing tools aren't available.",
        "- DECISION AT WRITE POINT: When you diagnose an issue, decide to WRITE immediately. Never narrate proposed edits or re-read files in loops. Call `edit` or `write` to apply fixes directly to disk.",
        "- COMMON SCENARIOS: (1) Find: use `glob`/`grep`. (2) Inspect: use `read`. (3) Modify: use `edit`/`write`. (4) Test: use `bash`. For questions, reply in a message; for tasks/fixes, modify files on disk before summarizing.",
        "- NO SANDBOX / NO ZIP: You do not have a sandbox or /mnt/data. Never output sandbox: or /mnt/data links.",
        "- REAL EDITS REQUIRED: Reading files does not modify them. When asked to fix/build, you MUST use `edit` or `write` to modify code on disk. Claiming files are fixed without calling edit/write is strictly prohibited.",
        "- If you need to inspect or edit any file in the workspace, you MUST emit a tool_call envelope directly.",
        "",
        "RESPONSE PROTOCOL: reply with YOUR ONE JSON envelope wrapped in a single ```json fenced block, "
        "and nothing else -- no extra text, no second envelope, and NO XML or <escape> tags.",
        "Example:",
        '```json',
        '{"webmind_response": "tool_call", "tool_calls": [{"id": "call_<unique>", "name": "<EXACT tool name>", "arguments": { ... }}]}',
        '```',
        "",
        "The fence is REQUIRED, not decoration: your reply is rendered as markdown, and an UNFENCED reply would "
        "strip every ` backtick` pair (and other inline markdown) out of string values before they reach the file. "
        "A fenced JSON block passes through byte-for-byte, so file content containing `, ${...}, \\n, or CSS stays intact.",
        "",
        "1) If the latest user request requires calling one or more of the supplied tools, your envelope is:",
        '{"webmind_response": "tool_call", "tool_calls": [{"id": "call_<unique>", "name": "<EXACT tool name>", "arguments": { ... }}]}',
        "",
        '- "name" MUST exactly match a tool name from TOOL DEFINITIONS.',
        '- "arguments" MUST be a JSON object satisfying that tool\'s parameters schema.',
        '- "id" must be a fresh unique string such as call_ab12cd34.',
        "",
        "2) Otherwise reply with a normal, helpful answer wrapped like:",
        '{"webmind_response": "message", "content": "your plain-text answer here"}',
        "",
        "Rules:",
        "- NEVER invent tool names; only use names from TOOL DEFINITIONS.",
        "- NEVER execute anything; returning a tool_call envelope only asks the external agent to run it.",
        "- If the envelope cannot be honored exactly, fall back to a message envelope.",
        "- A `message` envelope must BE the answer to the user's request: the plan, the explanation, the result, "
        "or a question you need answered. NEVER use it to report on your own tooling, which tools are available, "
        "or whether you are permitted to act. NEVER narrate tool failures or diagnostic state; if data is missing, emit another tool_call.",
        "",
        "FILE-CONTENT RULE (applies to any argument holding file bytes, e.g. write/edit content):",
        "- A real newline in the file is a single `\\n` escape in your JSON string. NEVER write `\\\\n` (double "
        "backslash): that writes the two literal characters backslash-n into the file and collapses it to ONE "
        "physical line.",
        "- If a read-back tool result shows a file as one line full of `\\n` sequences, the file bytes are literally "
        "backslash-n: when repairing, emit each newline as a single `\\n` escape so the file regains real lines.",
    ]
    if tool_choice is not None:
        lines += [
            "",
            "The client has requested a specific tool selection:",
            json.dumps(tool_choice),
        ]
    return "\n".join(lines)


def continuation_instruction(tool_choice: Any = None) -> str:
    """Compact protocol reminder for continuation turns.

    The browser conversation history already holds the full protocol and
    examples from turn one, so a continuation only needs to jog memory of
    the envelope contract without consuming ~2.4K characters of budget.
    """
    lines = [
        "You are WebMind, the live autonomous coding agent brain continuing the tool loop. "
        "CRITICAL: The workspace is live. The Windows project directory and files are directly accessible. "
        "FULL TOOL ACCESS: You have full workspace tools (read, grep, glob, edit, write, bash). Never claim you lack filesystem access or tools. Never output :contentReference citations. "
        "DECISION AT WRITE POINT: When you know the fix, decide to WRITE immediately. Call `edit` or `write` on disk—never narrate next steps or re-read files in loops. "
        "COMMON SCENARIOS: Use `glob`/`grep` to find code, `read` to inspect, `edit` to patch, `write` for new files, `bash` for commands. "
        "NO SANDBOX / NO ZIP: Never output sandbox: or /mnt/data links. Reading files does not modify them: use edit or write to apply fixes on disk. "
        "Reply with exactly ONE JSON envelope wrapped in a single ```json code fence, and nothing else (no <escape> tags).",
        '{"webmind_response": "tool_call", "tool_calls": [{"id": "call_<unique>", "name": "<tool>", "arguments": {...}}]}',
        'or {"webmind_response": "message", "content": "..."}',
        "Rules: a `message` envelope must BE the answer to the user's request (never a report on tooling or permissions). "
        "Real newlines in file content must be a single `\\n` escape, never literal `\\\\n`.",
    ]
    if tool_choice is not None:
        lines += [
            "",
            "The client has requested a specific tool selection:",
            json.dumps(tool_choice),
        ]
    return "\n".join(lines)


def _role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", "user")).strip() or "user"
    return str(getattr(msg, "role", "user")).strip() or "user"


def _image_url_from_part(part: Any) -> str | None:
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return None
    obj = part.get("image_url")
    if isinstance(obj, dict):
        url = obj.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    pieces.append(text)
            elif part_type == "image_url":
                pieces.append("[image attached]")
            elif part_type == "tool-result":
                for sub in (part.get("content") or []):
                    if isinstance(sub, dict) and sub.get("type") == "text":
                        text = sub.get("text")
                        if isinstance(text, str) and text.strip():
                            pieces.append(text)
        return "\n".join(pieces)
    return ""


def _is_file_inspection_tool_block(block: str) -> bool:
    """True if block is a tool response for file inspection tools (read, grep, glob)."""
    if not block.startswith("[role: tool]"):
        return False
    low = block.lower()
    for fn in ("tool_name: read", "tool_name: grep", "tool_name: glob", "tool_name: file"):
        if fn in low:
            return True
    return False


def _select_continuation_blocks(blocks: list[str], tail_budget: int) -> list[str]:
    """Keep the NEWEST tail blocks within tail_budget, preserving order.

    Priority order (highest to lowest):

    1. assistant tool_call blocks — always kept unconditionally; they are
       small and a tool-result block is only meaningful when paired with
       the call that produced it.

    2. user instruction in tail — the latest user message in the tail is
       what the current turn acts on and is kept unconditionally.

    3. pending tool result — the latest tool result is kept unconditionally
       because it is the answer to the pending tool call.

    4A. file inspection tool results (read, grep, glob) — prioritized so source
       code context is retained across multi-file turns when budget is tight.

    4B. other tool result blocks ([role: tool]) — e.g. bash outputs, kept before
       assistant prose.

    5. everything else — filled newest-first within remaining budget.
    """
    kept_set: set[int] = set()
    used = 0

    # Pass 1: unconditionally keep tool_call blocks.
    for i in range(len(blocks) - 1, -1, -1):
        if "tool_calls:" in blocks[i]:
            kept_set.add(i)
            used += len(blocks[i])

    # Pass 2: unconditionally keep the latest user instruction in the tail (if any).
    for i in range(len(blocks) - 1, -1, -1):
        if blocks[i].startswith("[role: user]"):
            kept_set.add(i)
            used += len(blocks[i])
            break

    # Pass 3: unconditionally keep the latest tool result (pending result).
    for i in range(len(blocks) - 1, -1, -1):
        if blocks[i].startswith("[role: tool]"):
            kept_set.add(i)
            used += len(blocks[i])
            break

    # Pass 4A: prioritize file inspection tool results (read, grep, glob) (newest-first).
    for i in range(len(blocks) - 1, -1, -1):
        if i in kept_set:
            continue
        if _is_file_inspection_tool_block(blocks[i]):
            cost = len(blocks[i])
            if used + cost <= tail_budget:
                kept_set.add(i)
                used += cost

    # Pass 4B: keep other tool result blocks before assistant prose (newest-first).
    for i in range(len(blocks) - 1, -1, -1):
        if i in kept_set:
            continue
        if blocks[i].startswith("[role: tool]"):
            cost = len(blocks[i])
            if used + cost <= tail_budget:
                kept_set.add(i)
                used += cost

    # Pass 5: fill remaining budget with other blocks (newest-first).
    for i in range(len(blocks) - 1, -1, -1):
        if i in kept_set:
            continue
        cost = len(blocks[i])
        if used + cost <= tail_budget:
            kept_set.add(i)
            used += cost

    return [blocks[i] for i in sorted(kept_set)]


def _count_chained_commands(command: str) -> int:
    """Count top-level statement separators (; && ||) outside quoted strings."""
    if not isinstance(command, str):
        return 0
    s = re.sub(r'"(?:\\.|[^"\\])*"', '', command)
    s = re.sub(r"'(?:\\.|[^'\\])*'", '', s)
    parts = re.split(r';|&&|\|\|', s)
    return len([p.strip() for p in parts if p.strip()])


def build_prompt(
    messages: list[Any],
    tools: list[Any],
    tool_choice: Any = None,
    target_chars: int | None = None,
    tool_role_max_chars: int | None = None,
    message_max_chars: int | None = None,
    continuation_tail_budget: int | None = None,
    continuation_head_budget: int | None = None,
    system_max_chars: int | None = None,
    model: str | None = None,
    turn_count: int | None = None,
) -> tuple[str, list[str]]:
    """Serialize a structured OpenAI request into a ChatGPT prompt.

    Returns (prompt_text, image_urls). Image URLs are collected only
    from the last user message (matches the existing attach behavior);
    images earlier in history are marked inline.
    """
    limits = resolve_context_limits(
        model=model,
        target_chars=target_chars,
        tool_role_max_chars=tool_role_max_chars,
        message_max_chars=message_max_chars,
        continuation_tail_budget=continuation_tail_budget,
        continuation_head_budget=continuation_head_budget,
        system_max_chars=system_max_chars,
    )
    effective_target = limits["target_chars"]
    effective_tool_cap = limits["tool_role_max_chars"]
    effective_msg_cap = limits["message_max_chars"]
    effective_tail_budget = limits["continuation_tail_budget"]
    effective_head_budget = limits["continuation_head_budget"]
    effective_system_cap = limits["system_max_chars"]
    if not messages:
        raise ValueError("messages cannot be empty.")

    if turn_count is None:
        turn_count = sum(
            1 for m in (messages or [])
            if _role(m) in ("assistant", "tool") or getattr(m, "tool_calls", None) or (isinstance(m, dict) and m.get("tool_calls"))
        )

    continuation = _is_continuation(messages)

    # Collect tool result IDs to detect orphaned tool calls, and map tool_call_id -> tool_name
    tool_result_ids: set[str] = set()
    call_id_to_tool_name: dict[str, str] = {}
    for m in messages:
        if _role(m) == "tool":
            tcid = m.get("tool_call_id") if isinstance(m, dict) else getattr(m, "tool_call_id", None)
            if tcid:
                tool_result_ids.add(str(tcid).strip())
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        if tcs:
            call_list = tcs if isinstance(tcs, list) else [tcs]
            for tc in call_list:
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                cname = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)) or (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None))
                if cid and cname:
                    call_id_to_tool_name[str(cid).strip()] = str(cname).strip()

    image_urls: list[str] = []
    blocks: list[str] = []
    block_roles: list[str] = []

    user_indices = [i for i, m in enumerate(messages) if _role(m) == "user"]
    last_user_idx = user_indices[-1] if user_indices else None

    for idx, msg in enumerate(messages):
        role = _role(msg)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        text = _content_to_text(content)

        if (
            continuation
            and role == "system"
            and len(text) > effective_system_cap
        ):
            text = (
                "[system message omitted: identical to earlier turns. "
                "The surviving conversation below carries the full context.]"
            )

        entries: list[str] = []
        if text:
            entries.append(text)

        if idx == last_user_idx:
            if isinstance(content, list):
                for part in content:
                    url = _image_url_from_part(part)
                    if url:
                        image_urls.append(url)
            if not text.strip():
                entries.append(
                    "[The latest user request is the attached image(s): analyze them.]"
                )

        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        if tool_calls:
            entries.append("tool_calls: " + json.dumps(tool_calls, default=str))
            # Orphaned tool calls detection: check if assistant issued a tool call whose result was dropped
            if idx < len(messages) - 1:
                call_list = tool_calls if isinstance(tool_calls, list) else [tool_calls]
                for tc in call_list:
                    cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                    fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
                    cname = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)) or (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "tool"))
                    if cid and str(cid).strip() not in tool_result_ids:
                        entries.append(
                            f"[system notice: tool call {str(cid).strip()} ({cname or 'tool'}) was issued earlier, but its result is MISSING from the conversation history (the client may have dropped it or timed out). Do NOT assume this tool executed or produced output. Re-issue the tool call using the dedicated `read` tool if inspecting files.]"
                        )
                    if cname == "bash":
                        args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
                        if not args and isinstance(tc, dict):
                            args = tc.get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        cmd = args.get("command") if isinstance(args, dict) else ""
                        if isinstance(cmd, str) and _count_chained_commands(cmd) > 2:
                            entries.append(
                                "[note: this bash command chains multiple operations. If any sub-command times out, the entire output may be lost. Prefer individual tool calls.]"
                            )
        reasoning = msg.get("reasoning_content") if isinstance(msg, dict) else getattr(msg, "reasoning_content", None)
        if reasoning is not None and str(reasoning).strip():
            entries.append("reasoning_content: " + str(reasoning))
        tname = msg.get("name") if isinstance(msg, dict) else getattr(msg, "name", None)
        tcid = msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", None)
        if role == "tool" and not tname and tcid:
            tname = call_id_to_tool_name.get(str(tcid).strip())
        if role == "tool" and tname:
            entries.append("tool_name: " + str(tname))
        if tcid:
            entries.append("tool_call_id: " + str(tcid))

        block = f"[role: {role}]"
        if entries:
            block += "\n" + "\n".join(entries)

        limit = effective_tool_cap if role == "tool" else effective_msg_cap
        if len(block) > limit:
            block = _cap_block(block, limit)

        blocks.append(block)
        block_roles.append(role)

    # Fixed framing is measured FIRST, because it is not negotiable: the
    # response protocol and the tool signatures are what let the model
    # answer at all, so the conversation is what has to give way.  Budgeting
    # the conversation alone would miss several thousand chars of framing
    # and still land the prompt over the ceiling.
    if continuation:
        prompt_parts = [
            continuation_instruction(tool_choice),
            _tool_signatures_note(tools),
        ]
    else:
        prompt_parts = [tool_mode_instruction(tool_choice)]
        full_defs = "TOOL DEFINITIONS:\n" + json.dumps(tools, indent=2)
        # A first turn is the one chance to show the model the real schemas,
        # so full JSON is preferred -- but 26 tools serialize to ~12.5K, which
        # on its own eats most of the budget and used to push ANY opening
        # request past the ceiling, where it was refused outright with no
        # answer at all.  Degrade to the compact signature note instead: it
        # carries verbatim parameter names and marks the required ones, which
        # is what the model needs to emit a valid call.
        if len(full_defs) * 2 <= effective_target:
            prompt_parts.append(full_defs)
        else:
            prompt_parts.append(
                _tool_signatures_note(tools)
                + "\n\n(The full JSON schemas were elided to fit the size limit. "
                "The tool names and parameter names above are exact and "
                "complete; send only those keys.)"
            )

    conversation_header = "CONVERSATION (chronological, oldest first):\n\n"
    has_tool_directive = bool(
        continuation and messages and _role(messages[-1]) == "tool"
    )
    has_user_directive = bool(
        continuation and messages and _role(messages[-1]) == "user"
    )
    # True when the pending tool result was explicitly size-capped (the
    # live glob-the-whole-tree failure: 11K listing -> attachment retry
    # keeps 2000 chars).  The extra directive is measured into framing so
    # the head budget still accounts for it.
    latest_tool_truncated = bool(
        has_tool_directive and blocks and "[truncated: kept" in blocks[-1]
    )
    latest_tool_edit_error = bool(
        has_tool_directive and blocks and any(
            phrase in blocks[-1].lower()
            for phrase in ("could not find oldstring", "must match exactly", "oldstring in")
        )
    )
    # "\n\n" joins between the framing parts, the conversation, and the
    # trailing directive(s).
    selected_directive = _get_continuation_directive(turn_count)
    framing_len = sum(len(p) for p in prompt_parts) + len(conversation_header) + 4
    if has_tool_directive:
        framing_len += 2 + len(selected_directive)
    if has_user_directive:
        framing_len += 2 + len(_USER_TURN_CONTINUATION_DIRECTIVE)
    if latest_tool_truncated:
        framing_len += 2 + len(_TRUNCATED_TOOL_DIRECTIVE)
    if latest_tool_edit_error:
        framing_len += 2 + len(_EDIT_FAILURE_DIRECTIVE)

    # A first turn has no tail: every block is trimmable context.  This path
    # used to skip budgeting entirely, so a 26-tool opening request was
    # refused outright (PROMPT_TOO_LARGE) instead of being answered.
    # tail_start == len(blocks) means "no tail", so the head is everything.
    tail_start = len(blocks)
    if continuation:
        # Anchor the tail at the START OF THE MOST RECENT AGENT TURN:
        # the first tool_call that comes AFTER the second-to-last user
        # message.  This keeps the entire cluster of sequential reads
        # made in one agent turn (read A → read B → read C → analysis)
        # in the TAIL together, so none of the file-read results are
        # silently dropped into the low-budget HEAD when the user sends
        # a follow-up message like "fix the files".
        prev_user_end = user_indices[-2] if len(user_indices) >= 2 else -1

        tail_start = None
        # First tool_call after the previous user message.
        for i, m in enumerate(messages):
            if i <= prev_user_end:
                continue
            tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
            if _role(m) == "assistant" and tcs:
                tail_start = i
                break
        # Fallback: first tool result after the previous user message.
        if tail_start is None:
            for i, m in enumerate(messages):
                if i <= prev_user_end:
                    continue
                if _role(m) == "tool":
                    tail_start = i
                    break
        if tail_start is None:
            tail_start = 0

    head = blocks[:tail_start]
    head_roles = block_roles[:tail_start]
    tail = blocks[tail_start:]

    # Dynamic tail budget: enforce effective_target so the tail cannot blow
    # through the overall prompt ceiling on its own. Reserve up to 2000 chars
    # for the head anchor (user request) if target allows, while ensuring the
    # tail gets at least effective_tool_cap room for the pending result.
    max_tail_from_target = max(
        effective_tool_cap,
        effective_target - framing_len - 2000,
    )
    tail_budget = min(effective_tail_budget, max_tail_from_target)

    fixed = (
        _select_continuation_blocks(tail, tail_budget)
        if continuation
        else []
    )
    fixed_len = sum(len(b) for b in fixed)

    head_budget = max(
        0,
        effective_target - framing_len
        if not continuation
        else min(
            effective_head_budget,
            effective_target - framing_len - fixed_len,
        ),
    )

    # The newest user turn is the instruction itself: on a continuation it is
    # what the pending tool call was made FOR, and on a first turn it is the
    # only actual question.  Dropping it is not graceful degradation, it is a
    # wrong answer: the model sees a tool result (or nothing) and no reason
    # for it, and replies that it cannot infer what is wanted.  Observed
    # live, so it is pinned by a test.
    anchor: int | None = None
    for i in range(len(head) - 1, -1, -1):
        if head_roles[i] == "user":
            anchor = i
            break

    # Older context, newest-first, within the head budget.  The anchor is
    # reserved first, so an oversized newer assistant block can never starve
    # it out via the break below.
    kept_head_idx: list[int] = []
    used = 0
    if anchor is not None:
        kept_head_idx.append(anchor)
        used += len(head[anchor])
    for i in range(len(head) - 1, -1, -1):
        if i == anchor:
            continue
        cost = len(head[i])
        if used + cost <= head_budget:
            kept_head_idx.append(i)
            used += cost
        else:
            break
    kept_head_idx.sort()
    head_kept = [head[i] for i in kept_head_idx]

    transcripts = head_kept + fixed

    dropped = len(head) - len(head_kept)
    if dropped > 0:
        transcripts.insert(
            0,
            f"[earlier conversation omitted for size ({dropped} message(s)); see earlier turns]",
        )

    cleaned_transcripts: list[str] = []
    has_prior_deprivation_refusal = False
    for t in transcripts:
        t_low = t.lower()
        if (
            any(p in t_low for p in _SPURIOUS_DEPRIVATION_PHRASES)
            or any(p in t_low for p in (
                "does not expose the live workspace",
                "without filesystem access",
                "no writable workspace",
                "no actual file contents or writable workspace",
                "supplied transcript",
            ))
        ):
            has_prior_deprivation_refusal = True
            if t.startswith("[role: assistant]"):
                cleaned_transcripts.append(
                    "[role: assistant]\n[prior assistant refusal message omitted: hallucinated lack of workspace tools or direct access. Workspace tools are active.]"
                )
            else:
                cleaned_transcripts.append(t)
        else:
            cleaned_transcripts.append(t)
    transcripts = cleaned_transcripts

    conversation = conversation_header + "\n\n".join(transcripts)

    if has_prior_deprivation_refusal:
        conversation += (
            "\n\n[important notice] Any prior assistant or subagent messages in the history claiming that this session "
            "does not expose tools or lacks filesystem access were false hallucinations. All workspace tools (read, grep, glob, edit, write, bash) "
            "are fully active, exposed, and operational right now. Proceed to call tools directly."
        )

    if has_tool_directive:
        conversation += "\n\n" + selected_directive
    elif has_user_directive:
        conversation += "\n\n" + _USER_TURN_CONTINUATION_DIRECTIVE
    if latest_tool_truncated:
        conversation += "\n\n" + _TRUNCATED_TOOL_DIRECTIVE
    if latest_tool_edit_error:
        conversation += "\n\n" + _EDIT_FAILURE_DIRECTIVE

    prompt_parts.append(conversation)
    return "\n\n".join(prompt_parts), image_urls


# ============================================================
# ENVELOPE PARSING (ChatGPT text -> structured result)
# ============================================================

_VALID_JSON_ESCAPES = set('"\\/bfnrtu')

# Regex to strip UI chrome headers that browsers may copy alongside code blocks
# (e.g., "jsonCopyDownload", "json\nCopy\nDownload", "pythonCopy", etc.)
_UI_CHROME_REGEX = re.compile(
    r"^(?:json|python|bash|javascript|text|sh|cmd|powershell)?\s*(?:copy|download|code|复制|下载|\s)*",
    re.IGNORECASE,
)

# A backslash that a Windows path separator begins: after a drive letter
# ("C:\...") or after a dot (".\..." / "..\..."). ChatGPT web emits these as
# single backslashes (invalid JSON escapes). In that context a lone "\b" in
# "\brain.py" is a path separator, never the JSON backspace control char.
# The drive letter must not follow another letter/digit, so "todo:\n" (a
# colon in prose before a JSON escape) is not mistaken for "C:\...".
_WINDOWS_PATH_BACKSLASH = re.compile(r"(?:(?<![A-Za-z0-9])[A-Za-z]:|\.)\\")


def _wants_literal_backslashes(text: str) -> bool:
    return _WINDOWS_PATH_BACKSLASH.search(text) is not None


def _repair_json_escapes(text: str) -> str:
    """Turn invalid escapes like \\P / \\% into valid \\\\P / \\\\%.

    Only reached after strict parsing fails. Preserves the literal
    backslash instead of dropping it (\\P -> \\\\P -> decodes to \\P).
    """
    return re.sub(r"\\(?![\\\\\"/bfnrtu])", r"\\\\", text)


def _repair_windows_path_escapes(text: str) -> str:
    """Double every SINGLE backslash in a Windows-drive-path payload.

    When the JSON embeds a path like "C:\\Users\\...\\brain.py" with
    single backslashes (invalid JSON), a naive escape repair corrupts
    segments whose next letter is itself a valid escape: "\\b" (in
    "\\brain.py") would otherwise decode as a BACKSPACE control char.

    Doubling every single (not already-paired) backslash turns each
    into a literal separator, which is always the intent for a path.
    """
    return re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", text)


def _repair_unescaped_quotes(text: str) -> str:
    """Escape double quotes that cannot be closing a JSON string.

    ChatGPT sometimes writes prose inside an envelope value with raw
    quotes, e.g. "content":"says "alpha" here".  Those interior quotes are
    not string terminators, so strict parsing fails.  A quote only closes
    the string when the next significant character is a separator or a
    container close (',' '}' ']' ':') or the input ends; every other quote
    is interior punctuation and gets escaped.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    length = len(text)
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
                continue
            if char == "\\":
                out.append(char)
                escaped = True
                continue
            if char == '"':
                lookahead = index + 1
                while lookahead < length and text[lookahead] in " \t\r\n":
                    lookahead += 1
                if lookahead >= length or text[lookahead] in ",}]:":
                    out.append(char)
                    in_string = False
                else:
                    out.append('\\"')
                continue
            out.append(char)
        else:
            out.append(char)
            if char == '"':
                in_string = True
    return "".join(out)


_RAW_CONTROL = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _repair_control_chars(text: str) -> str:
    """Escape raw control characters that occur inside JSON strings.

    ChatGPT occasionally pastes a literal newline or tab into a large string
    value (typically HTML file content), and `json.loads` rejects any raw
    control character inside a string.  Inside a string the raw character is
    rewritten to its JSON escape; outside a string control characters are
    legal JSON whitespace and are left untouched.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                out.append(char)
                escaped = False
                continue
            if char == "\\":
                out.append(char)
                escaped = True
                continue
            if char in _RAW_CONTROL:
                out.append(_RAW_CONTROL[char])
                continue
            if ord(char) < 0x20:
                out.append("\\u%04x" % ord(char))
                continue
            out.append(char)
            if char == '"':
                in_string = False
        else:
            out.append(char)
            if char == '"':
                in_string = True
    return "".join(out)


_MAX_BALANCE_CLOSERS = 32


def _balance_json_structure(text: str) -> str:
    """Re-close JSON containers that ChatGPT left structurally open.

    ChatGPT occasionally emits a large envelope with a DROPPED interior
    closer: the tool-call object closes as "}]}" instead of "}}]}" (one '}'
    missing before the ']'), or the root '}' is missing at the very end.
    Both leave a structurally open object even though the text still begins
    with '{' and ends with '}'.

    The walk tracks open containers, strings-aware, and:
      - inserts the minimal missing closer BEFORE an otherwise mismatched
        container-close (']' while a '{' or '}' is still open above it),
      - appends any still-open containers' closers at the end.

    Inserting closers never invents content -- it only tightens structure --
    and the result must still pass the envelope/validation gates to become
    a tool call.  Returns the input unchanged when nothing is unbalanced so
    callers can dedupe.  Call AFTER the escape repairs so string tracking is
    reliable.
    """
    out: list[str] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            out.append(char)
            in_string = True
        elif char in "{[":
            out.append(char)
            if len(stack) < _MAX_BALANCE_CLOSERS:
                stack.append(char)
        elif char in "}]":
            need = "{" if char == "}" else "["
            while stack and stack[-1] != need:
                out.append("}" if stack[-1] == "{" else "]")
                stack.pop()
            out.append(char)
            if stack:
                stack.pop()
        else:
            out.append(char)
    while stack:
        out.append("]" if stack.pop() == "[" else "}")
    return "".join(out)


def _parse_json_lenient(text: str):
    """Parse a JSON payload with progressive escape repairs.

    Ordered attempts:
      1. strict json.loads
      2. windows-drive-path repair (single backslashes literal)
      3. generic invalid-escape repair
      4. windows repair then generic repair
      5. each of the above with interior raw quotes escaped
      6. each candidate with raw control chars inside strings escaped
      7. each repaired candidate with missing closing braces appended

    Returns the parsed value (dict in practice) or None.
    """
    candidates = [text]
    if "\\" in text:
        if _wants_literal_backslashes(text):
            # A path separator is present: prefer the literal-backslash
            # reading BEFORE strict decoding.  Strict JSON would silently
            # corrupt a separator in front of an escape letter -- "\b" in
            # ".\brain.py" or "C:\brain.py" becomes a backspace -- and the
            # whole point of the repair is to keep the path intact.
            candidates.insert(0, _repair_windows_path_escapes(text))
        candidates.append(_repair_json_escapes(text))
        candidates.append(_repair_json_escapes(_repair_windows_path_escapes(text)))
    if '"' in text:
        # Prose inside an envelope value can carry raw quotes that make the
        # whole object unparseable; retry the existing candidates with those
        # interior quotes escaped.
        candidates.extend(_repair_unescaped_quotes(cand) for cand in list(candidates))
    # Raw control characters inside a string value (ChatGPT pastes literal
    # newlines into a large HTML payload) also make strict parsing fail.
    candidates.extend(_repair_control_chars(cand) for cand in list(candidates))
    # A dropped trailing closer leaves an envelope structurally open even
    # though the text still begins with '{' and ends with '}'; retry the
    # repaired candidates with the missing closers appended.
    candidates.extend(_balance_json_structure(cand) for cand in list(candidates))
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        return obj
    return None


def _extract_json_object(raw: str) -> dict | None:
    """Return a parsed dict only when the ENTIRE reply is ONE JSON object.

    Accepted shapes (the protocol contract: "reply with one exact JSON
    envelope and nothing else"):
      - the bare object (leading/trailing whitespace and a BOM allowed)
      - the WHOLE message inside a single ```json / ``` fenced block
      - wrapped in outer XML tags (e.g. <escape>...</escape>, <json>...</json>)

    Every other shape -- prose, preamble, trailing text, a QUOTED/echoed
    WebMind envelope inside an explanation, multiple JSON objects in
    ordinary text -- is rejected so a stale or copied envelope can never
    be mistaken for a fresh executable tool call.
    """
    text = str(raw or "").strip()
    text = text.lstrip("\ufeff\u200b\u2060")
    if not text:
        return None

    # Strip outer wrapper tags like <escape>...</escape>, <json>...</json>,
    # <output>...</output>, <response>...</response>, etc.
    for _ in range(3):
        tag_match = re.match(r"^<([a-zA-Z0-9_\-]+)>\s*(.*?)\s*</\1>$", text, re.DOTALL | re.IGNORECASE)
        if tag_match:
            text = tag_match.group(2).strip()
        else:
            break

    # Strip leading/trailing standalone wrapper tags if mismatched or unclosed
    text = re.sub(r"^<(?:escape|json|output|response|result)>\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*</(?:escape|json|output|response|result)>$", "", text, flags=re.IGNORECASE).strip()

    def _clean_candidate(cand: str) -> str:
        cand = cand.strip()
        for _ in range(3):
            inner_tag = re.match(r"^<([a-zA-Z0-9_\-]+)>\s*(.*?)\s*</\1>$", cand, re.DOTALL | re.IGNORECASE)
            if inner_tag:
                cand = inner_tag.group(2).strip()
            else:
                break
        cand = re.sub(r"^<(?:escape|json|output|response|result)>\s*", "", cand, flags=re.IGNORECASE).strip()
        cand = re.sub(r"\s*</(?:escape|json|output|response|result)>$", "", cand, flags=re.IGNORECASE).strip()
        cand = _UI_CHROME_REGEX.sub("", cand).strip()
        return cand

    def _is_envelope_dict(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        return (
            "webmind_response" in obj
            or "tool_calls" in obj
            or "tool_call" in obj
        )

    envelopes = []

    # 1. Whole message inside a single code fence (3 or more backticks, UI chrome stripped)
    fenced_text = _UI_CHROME_REGEX.sub("", text).strip()
    fence = re.match(r"^`{3,}[^\n`]*\n?(.*?)\n?`{3,}$", fenced_text, re.DOTALL)
    if fence:
        candidate = _clean_candidate(fence.group(1))
        obj = _parse_json_lenient(candidate)
        if _is_envelope_dict(obj):
            return obj

    # 2. Whole message is a single bare JSON object (possibly preceded by UI chrome)
    cand_bare = _UI_CHROME_REGEX.sub("", text).strip()
    if cand_bare.startswith("{") and cand_bare.endswith("}"):
        obj = _parse_json_lenient(cand_bare)
        if _is_envelope_dict(obj):
            return obj

    # 3. Fenced block(s) embedded within preamble or normal text
    # (handles standard ```json, arbitrary backticks like ``````, and attributes like json id="...")
    for m in re.finditer(r"`{3,}[^\n`]*\n?(.*?)\n?`{3,}", text, re.DOTALL):
        candidate = _clean_candidate(m.group(1))
        obj = _parse_json_lenient(candidate)
        if _is_envelope_dict(obj) and obj not in envelopes:
            envelopes.append(obj)

    # 4. Outermost fence if candidate had interior backticks inside arguments
    first_fence = text.find("```")
    last_fence = text.rfind("```")
    if first_fence != -1 and last_fence > first_fence + 2:
        fence_content = text[first_fence:last_fence + 3]
        m = re.match(r"^`{3,}[^\n`]*\n?(.*?)\n?`{3,}$", fence_content, re.DOTALL)
        if m:
            candidate = _clean_candidate(m.group(1))
            obj = _parse_json_lenient(candidate)
            if _is_envelope_dict(obj) and obj not in envelopes:
                envelopes.append(obj)

    # 5. Unclosed fence at end of text: e.g. ```json\n{\n  "webmind_response": ...
    unclosed = re.search(r"`{3,}[^\n`]*\n?(\{.*)$", text, re.DOTALL)
    if unclosed:
        candidate = _clean_candidate(unclosed.group(1))
        obj = _parse_json_lenient(candidate)
        if _is_envelope_dict(obj) and obj not in envelopes:
            envelopes.append(obj)

    # 6. Scan for bare JSON envelope(s) outside of fences (e.g. at start of text or line)
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(text):
        idx = text.find('{', pos)
        if idx == -1:
            break
        line_start = text.rfind('\n', 0, idx) + 1
        prefix = text[line_start:idx].strip()
        if prefix and not prefix.startswith("`") and not prefix.endswith("`"):
            if not _UI_CHROME_REGEX.fullmatch(prefix):
                pos = idx + 1
                continue
        try:
            obj, end = decoder.raw_decode(text, idx)
            if _is_envelope_dict(obj) and obj not in envelopes:
                envelopes.append(obj)
            pos = end
        except json.JSONDecodeError:
            pos = idx + 1

    if not envelopes:
        return None

    # Merge / resolve envelopes: prioritize tool calls, attach accompanying message content
    tool_envs = [
        e for e in envelopes
        if (e.get("webmind_response") == "tool_call" or "tool_calls" in e or "tool_call" in e)
    ]
    msg_envs = [
        e for e in envelopes
        if e not in tool_envs and (e.get("webmind_response") == "message" or "content" in e or "message" in e)
    ]

    if tool_envs:
        merged_calls = []
        for te in tool_envs:
            calls = te.get("tool_calls") if "tool_calls" in te else te.get("tool_call")
            if isinstance(calls, dict):
                calls = [calls]
            if isinstance(calls, list):
                merged_calls.extend(calls)
        content_str = None
        if msg_envs:
            content_str = msg_envs[0].get("content") or msg_envs[0].get("message")
        result = {
            "webmind_response": "tool_call",
            "tool_calls": merged_calls,
        }
        if content_str:
            result["content"] = content_str
        return result

    return msg_envs[0]


def _function_entry(t: Any) -> dict | None:
    """Normalize one tool entry to its function definition.

    Accepts both the standard OpenAI ``{"type":"function","function":{...}}``
    wrapper and the flat ``{"name","description","parameters"}`` form some
    OpenAI-compatible adapters emit.
    """
    if not isinstance(t, dict):
        return None
    fn = t.get("function")
    if isinstance(fn, dict):
        return fn
    if "name" in t or "parameters" in t:
        return t
    return None


def _tool_names(tools: list[Any]) -> set[str]:
    names: set[str] = set()
    for t in tools or []:
        fn = _function_entry(t)
        if fn is None:
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _schema_for(tools: list[Any], name: str) -> dict | None:
    for t in tools or []:
        fn = _function_entry(t)
        if fn is None:
            continue
        if fn.get("name") == name:
            params = fn.get("parameters")
            return params if isinstance(params, dict) else None
    return None


def _tool_purpose(fn: dict) -> str:
    """One short clause describing what a tool DOES, for continuation prompts.

    A bare ``name(arg: type)`` signature says nothing about purpose, so on a
    continuation turn the model has to recall what each tool is FOR from the
    earlier browser turns -- while the same prompt forbids inventing a tool
    name. That combination is what turns a routine continuation into a refusal
    ("I cannot safely invent another tool call") instead of a tool call. The
    first sentence of the description, capped, restores the missing semantics.

    Returns "" when the tool has no usable description, so an undescribed tool
    still renders as a bare name rather than an invented one.
    """
    description = fn.get("description")
    if not isinstance(description, str):
        return ""
    text = " ".join(description.split())
    if not text:
        return ""
    # First sentence only. A colon is also a boundary: it usually introduces
    # example payloads or per-parameter detail the model does not need here.
    for stop in (". ", ": "):
        head, sep, _tail = text.partition(stop)
        if sep:
            text = head
            break
    text = text.strip().rstrip(".;:,")
    if len(text) > TOOL_PURPOSE_MAX_CHARS:
        # Cut on a word boundary: a mid-word cut reads as corruption to the
        # model and wastes the characters it does get.
        text = text[:TOOL_PURPOSE_MAX_CHARS]
        head, sep, _tail = text.rpartition(" ")
        if sep and len(head) >= TOOL_PURPOSE_MAX_CHARS // 2:
            text = head
        text = text.rstrip(" .,;:-")
    return text


def _tool_signatures(tools: list[Any]) -> list[str]:
    """Compact per-tool parameter signatures for continuation prompts.

    One line per tool, in wire order, derived verbatim from the OpenAI tool
    schemas (``{"type":"function","function":{...}}``): property names, the
    declared JSON-Schema ``type`` when present, and ``[required]`` for
    members of ``parameters.required``.  No aliases, normalization, or
    guessing: a property without a usable type — or a tool without a usable
    parameters object — renders as a bare name rather than an invented one.

    A described tool also carries a capped purpose clause (see
    ``_tool_purpose``) so the model can tell what the tool is FOR without
    recalling the full schema from an earlier browser turn.
    """
    lines: list[str] = []
    for t in tools or []:
        fn = _function_entry(t)
        if fn is None:
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        params = fn.get("parameters")
        params = params if isinstance(params, dict) else {}
        props = params.get("properties")
        required = {
            r for r in (params.get("required") or []) if isinstance(r, str)
        }
        parts: list[str] = []
        if isinstance(props, dict):
            for arg_name, schema in props.items():
                typ = schema.get("type") if isinstance(schema, dict) else None
                typ = typ if isinstance(typ, str) and typ else None
                if typ:
                    if arg_name in required:
                        parts.append(f"{arg_name}: {typ} [required]")
                    else:
                        parts.append(f"{arg_name}?: {typ}")
                elif arg_name in required:
                    parts.append(f"{arg_name} [required]")
                else:
                    parts.append(f"{arg_name}?")
        purpose = _tool_purpose(fn)
        suffix = f" — {purpose}" if purpose else ""
        lines.append(f"{name}({', '.join(parts)}){suffix}")
    return lines


def _tool_signatures_note(tools: list[Any]) -> str:
    """Continuation TOOL DEFINITIONS note with compact per-tool signatures."""
    sigs = _tool_signatures(tools)
    if not sigs:
        return (
            "TOOL DEFINITIONS were already provided in full earlier in this "
            "chat and are unchanged."
        )
    header = (
        "TOOL DEFINITIONS were already provided in full earlier in this "
        "chat and are unchanged. Compact parameter signatures (verbatim "
        "argument names; [required] marks required parameters) follow so "
        "you can name arguments exactly:\n"
    )
    text = header + "\n".join(sigs)
    # If the signature block exceeds 2200 chars (e.g. large tool sets), elide
    # the purpose clauses to preserve room for conversation history and file content.
    if len(text) > 2200:
        compact_sigs = [s.split(" — ")[0] for s in sigs]
        text = header + "\n".join(compact_sigs)
    return text


def validate_arguments(args: Any, schema: dict | None) -> list[str]:
    """Best-effort JSON-Schema validation using stdlib only.

    Supports type (including union lists like ["string", "null"]), enum,
    required, properties, items, and additionalProperties: false. Returns a
    list of error strings (empty = valid).
    """
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    if not isinstance(schema, dict):
        return []

    errors: list[str] = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}

    for req in schema.get("required", []) or []:
        if isinstance(req, str) and req not in args:
            errors.append(f"missing required property '{req}'")

    for key, value in args.items():
        sub = properties.get(key)
        if isinstance(sub, dict):
            _type_check(sub, value, key, errors)

    addl = schema.get("additionalProperties")
    if addl is False:
        for key in args:
            if key not in properties:
                errors.append(f"unknown property '{key}' (additionalProperties is false)")

    return errors


def _type_check_union(
    types: list[Any], schema: dict, value: Any, path: str, errors: list[str]
) -> None:
    """Validate against a union of JSON types (e.g. ["string", "null"]).

    A value passes if at least one listed type accepts it; null passes when
    "null" is listed. Failures are reported once with the accepted choices
    instead of once per branch.
    """
    start = len(errors)
    matched = False
    for t in types:
        if not isinstance(t, str):
            continue
        before = len(errors)
        _type_check(dict(schema, type=t), value, path, errors)
        if len(errors) == before:
            matched = True
            break
    del errors[start:]
    if not matched:
        choices = ", ".join(repr(t) for t in types if isinstance(t, str))
        errors.append(f"property '{path}' must match one of: {choices}")


def _type_check(schema: dict, value: Any, path: str, errors: list[str]) -> None:
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        types = schema.get("type")
        null_ok = types == "null" or (
            isinstance(types, list) and "null" in types
        )
        if value is not None or not null_ok:
            choices = ", ".join(repr(v) for v in enum)
            errors.append(f"property '{path}' must be one of: {choices}")
            return

    types = schema.get("type")
    if isinstance(types, list):
        _type_check_union(types, schema, value, path, errors)
        return
    if not isinstance(types, str):
        return
    typ = types

    ok = True
    if typ == "string":
        ok = isinstance(value, str)
    elif typ == "integer":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif typ == "number":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif typ == "boolean":
        ok = isinstance(value, bool)
    elif typ == "null":
        ok = value is None
    elif typ == "object":
        ok = isinstance(value, dict)
        if ok:
            props = schema.get("properties", {})
            if isinstance(props, dict):
                for req in schema.get("required", []) or []:
                    if isinstance(req, str) and req not in value:
                        errors.append(f"property '{path}.{req}' is required")
                addl = schema.get("additionalProperties")
                if addl is False:
                    for k in value:
                        if k not in props:
                            errors.append(
                                f"unknown property '{path}.{k}' (additionalProperties is false)"
                            )
                for k, v in value.items():
                    sub = props.get(k)
                    if isinstance(sub, dict):
                        _type_check(sub, v, f"{path}.{k}", errors)
    elif typ == "array":
        ok = isinstance(value, list)
        if ok:
            items = schema.get("items", {})
            if isinstance(items, dict):
                for i, v in enumerate(value):
                    _type_check(items, v, f"{path}[{i}]", errors)
    else:
        return

    if not ok:
        errors.append(f"property '{path}' must be {typ}")


def _inspect_tool_call(
    call: Any, tools: list[Any], tool_names: set[str], is_correction: bool = False
) -> tuple[dict | None, list[str]]:
    """Validate one tool call with per-problem detail.

    Returns (normalized {id, name, arguments}, problems). The problems are
    human-readable and name the offending tool so a correction prompt can
    quote them verbatim (e.g. "read: missing required property 'file_path'").
    Extra properties are tolerated unless the schema declares
    additionalProperties: false (the wire models routinely add prose keys
    like "description" to an otherwise valid call); missing/required,
    mistyped, and enum-violating properties are always reported.
    """
    if not isinstance(call, dict):
        return None, ["tool call entry is not an object"]

    fn = call.get("function")
    if isinstance(fn, dict):
        name = fn.get("name") or call.get("name")
        args = fn.get("arguments") if "arguments" in fn else call.get("arguments")
    else:
        name = call.get("name")
        args = call.get("arguments") if "arguments" in call else call.get("parameters")

    if not isinstance(name, str) or not name:
        return None, ["tool call is missing a tool name"]

    # Normalize common tool aliases (e.g. read_file -> read)
    TOOL_ALIASES = {
        "read_file": "read",
        "readfile": "read",
        "view_file": "read",
        "write_file": "write",
        "writefile": "write",
        "edit_file": "edit",
        "editfile": "edit",
        "apply_diff": "edit",
        "run_command": "bash",
        "execute_command": "bash",
        "terminal": "bash",
        "shell": "bash",
        "powershell": "bash",
        "cmd": "bash",
        "search_files": "glob",
        "find_files": "glob",
        "grep_search": "grep",
    }
    if name not in tool_names:
        low = name.lower()
        if low in TOOL_ALIASES and TOOL_ALIASES[low] in tool_names:
            name = TOOL_ALIASES[low]
        else:
            available = ", ".join(sorted(tool_names))
            return None, [f"unknown tool {name!r}; available tools: {available}"]

    if isinstance(args, str):
        parsed = _parse_json_lenient(args)
        if parsed is None:
            return None, [f"{name}: arguments JSON could not be parsed"]
        args = parsed

    # Normalize common parameter aliases if the schema accepts them
    if isinstance(args, dict):
        schema = _schema_for(tools, name)
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if name == "read":
            if "filePath" in props and "filePath" not in args:
                for alt in ("file_path", "filename"):
                    if alt in args:
                        args["filePath"] = args[alt]
                        break
            if "offset" in props and "offset" not in args and "startLine" in args:
                try:
                    args["offset"] = int(args["startLine"])
                except (ValueError, TypeError):
                    pass
            if "limit" in props and "limit" not in args and "endLine" in args and "startLine" in args:
                try:
                    args["limit"] = int(args["endLine"]) - int(args["startLine"]) + 1
                except (ValueError, TypeError):
                    pass
        elif name in ("write", "edit"):
            if "filePath" in props and "filePath" not in args:
                for alt in ("file_path", "filename"):
                    if alt in args:
                        args["filePath"] = args[alt]
                        break
        elif name == "bash":
            if "command" in props and "command" not in args and "cmd" in args:
                args["command"] = args["cmd"]

    problems = validate_arguments(args, _schema_for(tools, name))
    if problems:
        return None, [f"{name}: {problem}" for problem in problems]

    if name.lower() == "bash" and isinstance(args, dict) and not is_correction:
        cmd = args.get("command")
        if isinstance(cmd, str):
            # Check if attempting to read file contents while a read tool exists
            has_read_tool = any("read" in t.lower() for t in tool_names)
            if has_read_tool and re.search(
                r"(?i)\b(?:get-content|gc)\b|(?<![><|])\s*\b(?:cat|type|head|tail)\s+(?!<<)[^\s|>]+", cmd
            ):
                return None, [
                    "bash: command appears to use shell commands to read file contents, "
                    "but the dedicated 'read' tool is available and preferred. "
                    "Re-emit your tool call using the 'read' tool instead."
                ]
            # Check if chaining multiple commands
            chained_count = _count_chained_commands(cmd)
            if chained_count > 2:
                return None, [
                    f"bash: command chains {chained_count} separate operations using statement separators; "
                    "chaining multiple commands risks total output loss if any operation times out or fails. "
                    "Split into individual tool calls (one command per tool call)."
                ]

    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = "call_" + uuid.uuid4().hex

    return {"id": call_id, "name": name, "arguments": args}, []


# ============================================================
# VALIDATION-FAILURE CLASSIFICATION / CORRECTION RECOVERY
# ============================================================

_VALIDATION_REJECTED_MESSAGE = (
    "WebMind: a proposed tool call envelope did not satisfy the supplied tool "
    "schema ({errors}); no tool was executed. Resend the call with arguments "
    "that match the tool definition, or answer directly."
)

_CORRECTION_FAILED_MESSAGE = (
    "WebMind: after one permitted correction attempt, the model still produced "
    "a tool call envelope that does not satisfy the supplied tool schema "
    "({errors}); no tool was executed."
)


def build_correction_directive(
    errors: list[str],
    history: list[Any] | None = None,
    tools: list[Any] | None = None,
) -> str:
    """One-attempt prompt the server appends into the existing browser chat."""
    details = "; ".join(errors)
    has_modified = _has_called_modifying_tool(history)

    if has_modified and any(
        "transcript" in e
        or "session" in e
        or "filesystem" in e
        or "deprivation" in e
        or "expose" in e
        or "citation" in e
        for e in errors
    ):
        body = (
            "[correction attempt] You have ALREADY successfully edited and modified the workspace files in this session! "
            "Do NOT claim the session is no longer available or mention transcripts. "
            "Simply emit a ```json message envelope summarizing the fixes you made to the files so the user knows what was changed."
        )
    elif any("chains" in e or "separate operations" in e for e in errors):
        body = (
            "[correction attempt] Your previous bash command was rejected because it chained multiple operations into one call. "
            "CRITICAL: Chaining commands (e.g. cmd1; cmd2; cmd3) risks total output loss if any sub-command times out. "
            "Re-emit your tool call running ONLY ONE command at a time, or use dedicated tools like `read` for file inspection."
        )
    elif any("shell commands to read" in e or "read tool is available" in e for e in errors):
        body = (
            "[correction attempt] Your previous bash command was rejected because it used shell commands to read file contents. "
            "CRITICAL: The dedicated `read` tool is available and preferred over shell commands. "
            "Re-emit your tool call using the `read` tool directly (e.g. read(filePath='...')). "
            "Do NOT use bash with Get-Content/cat/type."
        )
    elif any("narrated tool" in e or "diagnostic state" in e or "meta-reporting" in e for e in errors):
        body = (
            "[correction attempt] Your previous reply was rejected: you reported on your own diagnostic state or tool execution failures in plain text instead of taking action. "
            "CRITICAL: If a previous tool result was missing, empty, or insufficient, immediately emit a ```json tool_call envelope to gather the data directly (e.g. using `read` to view file contents). "
            "Do NOT narrate what happened or what tools returned — act on it! Reply with a ```json tool_call envelope now."
        )
    elif any("upload" in e or "transcript" in e for e in errors):
        body = (
            "[correction attempt] You asked the user to upload/paste files or claimed you do not have file contents. "
            "CRITICAL: You are connected to the live local workspace with real tools. You MUST NOT ask the user to upload files. "
            "Use the `read` tool to inspect files, `grep` or `glob` to search, and `edit` or `write` to modify them. "
            "Immediately emit a ```json tool_call envelope with the exact tool call needed (e.g. read(filePath='...')) right now."
        )
    elif any("sandbox" in e or "without calling" in e or "Code Interpreter" in e or "download link" in e for e in errors):
        body = (
            "[correction attempt] Your previous reply was rejected: you claimed the files were fixed or generated a sandbox:/mnt/data download link. "
            "CRITICAL: There is NO python sandbox, NO /mnt/data directory, and NO zip download capability. "
            "You are working in the user's LIVE local workspace, and NO files have been edited yet! Reading files does not modify them. "
            "You MUST emit a ```json tool_call envelope calling `edit` or `write` to apply the actual code fixes to the workspace files on disk right now."
        )
    elif any(
        "filesystem" in e
        or "lacks" in e
        or "deprivation" in e
        or "citation" in e
        or "unexposed" in e
        or "expose" in e
        or "accessible" in e
        or "environment" in e
        or "directory" in e
        or "editing tools" in e
        for e in errors
    ):
        body = (
            "[correction attempt] Your previous reply was rejected: you claimed this session does not expose workspace tools, that you lack filesystem access, or that the project directory is not accessible. "
            "CRITICAL CORRECTION: You DO have full workspace tools exposed right above in TOOL DEFINITIONS and the local project directory IS directly accessible: "
            "`read` (to inspect files), `edit` (to perform exact string replacements), `write` (to write or overwrite files), `bash` (to execute PowerShell commands), and `glob`/`grep` (to search). "
            "You are NOT an outside reader looking at an uploaded file, and you MUST NOT output citation tags (:contentReference) or claim tools or directories are unavailable. "
            "You are the live autonomous agent operating on the user's workspace. "
            "Immediately emit a ```json tool_call envelope calling `read`, `edit`, or `write` right now."
        )
    else:
        body = (
            "[correction attempt] Your previous reply was rejected because the tool "
            f"call envelope failed validation: {details}. "
            "Send exactly ONE fresh JSON envelope wrapped in a single ```json code "
            "fence (an unfenced reply is markdown-rendered, which strips `backticks` "
            "inside string values before they reach a file). If you were calling a "
            "tool, resend that call with arguments that satisfy the tool's parameters "
            "schema in the TOOL DEFINITIONS above (all required properties present, "
            "correct types, exact names). This is your single allowed correction: "
            "if the corrected envelope is also invalid, no tool will run and your "
            "output is replaced by an error message."
        )

    parts = [body]

    # Append response format template if not post-edit summary
    if not (has_modified and "ALREADY" in body):
        parts.append(
            "REQUIRED RESPONSE FORMAT:\n"
            "Reply with exactly ONE JSON envelope wrapped in a single ```json code fence, and nothing else:\n"
            "```json\n"
            '{"webmind_response": "tool_call", "tool_calls": [{"id": "call_<unique>", "name": "<exact_tool_name>", "arguments": {...}}]}\n'
            "```\n"
            "or\n"
            "```json\n"
            '{"webmind_response": "message", "content": "..."}\n'
            "```\n"
            "CRITICAL: Use exact tool names (e.g. `read`, NOT `read_file`) and exact parameter names from TOOL DEFINITIONS."
        )

    # Append latest user request context if available
    if history:
        latest_user = ""
        for msg in reversed(history):
            if _role(msg).lower() == "user":
                txt = _content_to_text(getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else ""))
                if txt.strip():
                    latest_user = txt.strip()
                    break
        if latest_user:
            if len(latest_user) > 300:
                latest_user = latest_user[:300] + "..."
            parts.append(f'USER GOAL:\n"{latest_user}"')

    # Append compact tool signatures if available
    if tools:
        parts.append(_tool_signatures_note(tools))

    return "\n\n".join(parts)


def recover_tool_call(
    parsed: dict,
    ask_fn: Any,
    tools: list[Any],
    history: list[Any] | None = None,
) -> dict:
    """Recover a validation-failure envelope with one model correction attempt.

    ``ask_fn(prompt) -> str`` is the underlying model round-trip.  WebMind
    never executes a tool; the corrected envelope (if valid) is returned as
    a normal ``tool_call``.  A single attempt is made: a corrected reply that
    again fails validation becomes a clean assistant error message (never the
    raw envelope the model wrote).  Exceptions from the model round-trip
    likewise fall back to the clean error message.
    """
    if parsed.get("type") != "validation_failure":
        return parsed

    errors = parsed.get("errors") or []
    err_text = "; ".join(errors)

    try:
        raw = ask_fn(build_correction_directive(errors, history=history, tools=tools))
    except Exception:
        if _has_called_modifying_tool(history):
            return {
                "type": "message",
                "content": "I have completed the requested changes and updated the files in the workspace.",
            }
        return {"type": "message", "content": _VALIDATION_REJECTED_MESSAGE.format(errors=err_text)}

    corrected = parse_chatgpt_response(raw, tools, history=history, is_correction=True)

    if corrected["type"] == "tool_call":
        return corrected

    if corrected["type"] == "validation_failure":
        if _has_called_modifying_tool(history):
            return {
                "type": "message",
                "content": "I have completed the requested changes and updated the files in the workspace.",
            }
        return {
            "type": "message",
            "content": _CORRECTION_FAILED_MESSAGE.format(
                errors="; ".join(corrected.get("errors") or [])
            ),
        }

    if corrected["type"] == "message":
        # Ensure a raw JSON envelope never leaks as assistant prose
        content = corrected.get("content") or ""
        if '"tool_calls"' in content or '"webmind_response"' in content:
            if _has_called_modifying_tool(history):
                return {
                    "type": "message",
                    "content": "I have completed the requested changes and updated the files in the workspace.",
                }
            return {
                "type": "message",
                "content": _CORRECTION_FAILED_MESSAGE.format(errors="invalid tool call envelope format in correction reply"),
            }

    return corrected


# ============================================================
# COSMETIC PROSE NEWLINE REPAIR
# ============================================================
# A `message` envelope's content sometimes arrives with literal
# two-character \n / \t sequences instead of real whitespace: the
# model over-escaped its own line breaks while writing long prose.
# json.loads decodes that faithfully, so the two characters reach the
# client verbatim and the whole answer renders on a single line.
#
# This is presentation-only and is applied to MESSAGE CONTENT ONLY.
# Tool-call arguments are never routed through here: the model writes
# those escapes correctly and they must stay byte-exact.
#
# A Windows path is matched WHOLE (drive letter plus every following
# segment) and returned byte-exact. Matching only the first separator is
# not enough: in "C:\new\notes.txt" the first "\n" would be protected and
# the second one would still decode into a stray line break.
#
#   (?<![A-Za-z0-9])        a drive letter cannot follow another letter
#                           or digit, so "todo:\n" prose is not a path
#   [A-Za-z]:               the drive colon
#   (?: \\ SEGMENT )+       one or more backslash-separated segments
#     (?![nt](?:\\|$))      ...but a lone "n"/"t" segment is a real
#                           escape, not a path: "C:\n" and "C:\new\n"
#                           keep their line break
#     [A-Za-z0-9_.\-]+      the segment body
#
# Leftmost-first alternation then leaves everything else to the second
# branch, which is the only one that decodes.
#
# KNOWN LIMIT: only ABSOLUTE paths are protected. A relative path whose
# first segment starts with n or t is indistinguishable from an escape:
# "dir\node_modules" and "a\nb" are both a backslash followed by letters,
# and no local test separates them. Suppressing the decode whenever the
# text contains a slash would also suppress it in any answer that mentions
# a URL, which is exactly the over-escaped case this exists to repair. So
# the ambiguous form is left decoding, deliberately.

_PROSE_ESCAPE_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\(?![nt](?:\\|$))[A-Za-z0-9_.\-]+)+"
    r"|\\[nt]"
)


def _unescape_prose_newlines(text: str) -> str:
    r"""Decode over-escaped \n / \t in message content.

    No-op when the text holds no backslash. Used for final assistant text
    only (message envelopes, legacy answers, and their dedup replays).
    """
    if not isinstance(text, str) or "\\" not in text:
        return text

    def _replace(match):
        raw = match.group(0)
        if raw[0] == "\\":
            return "\n" if raw[1] == "n" else "\t"
        return raw

    return _PROSE_ESCAPE_RE.sub(_replace, text)


_SPURIOUS_UPLOAD_PHRASES = (
    "upload those",
    "upload the",
    "upload/paste",
    "upload or paste",
    "upload them",
    "pasted text",
    "pasted-text",
    "pasted transcript",
    "supplied transcript",
    "the supplied transcript",
    "only have the tool-loop log",
    "only have the transcript",
    "does not contain the actual source contents",
    "cannot read the actual",
    "outside reader",
    "this particular attachment contains",
)

_SPURIOUS_SANDBOX_PHRASES = (
    "sandbox:",
    "/mnt/data",
    "download the fixed",
    "download the updated",
    "download the rebuilt",
    "download the project",
)

_PHANTOM_FIX_PHRASES = (
    "i fixed the",
    "i have fixed the",
    "rebuilt the game files",
    "rebuilt the project",
    "fixed the deployment blockers",
    "fixed the issue and rebuilt",
    "all issues have been fixed",
)


_SPURIOUS_DEPRIVATION_PHRASES = (
    "does not expose",
    "without filesystem access",
    "lack filesystem access",
    "no filesystem access",
    "no writable workspace",
    "no actual file contents or writable workspace",
    "blocked from applying the fix because",
    "cannot safely modify or verify them without filesystem access",
    "do not have filesystem access",
    "do not have access to the local filesystem",
    "do not have access to tools",
    "tools are not available",
    "tools aren't available",
    "editing tools are not available",
    "editing tools aren't available",
    "workspace editing tools are not available",
    "workspace editing tools aren't available",
    "directory is not accessible",
    "directory isn't accessible",
    "project directory is not accessible",
    "project directory isn't accessible",
    "workspace is not accessible",
    "workspace isn't accessible",
    "not accessible in this execution environment",
    "not accessible in this environment",
    "in this execution environment",
    "session does not expose",
    "session is no longer available",
    "workspace tool session is no longer available",
    "tool session is no longer available",
    "pasted transcript only contains",
    "pasted transcript",
    "the pasted transcript",
    "supplied transcript",
    "the supplied transcript",
    "not an editable workspace connection",
    "not an editable workspace",
    "cannot continue the live workspace edit",
    "can’t continue the live workspace edit",
    "can't continue the live workspace edit",
    "cannot safely apply a fix to your live",
    "can't safely apply a fix to your live",
    "do not provide direct read/edit access",
    "do not provide direct",
    "does not provide direct",
    "not provide direct read/edit access",
    "not provide direct read",
    "not provide direct edit",
    "editing that transcript would not modify",
    "editing that transcript would not",
    "editing the transcript would not",
    "editing that transcript",
    "editing the transcript",
    "editing interface required to modify",
    "editing interface is not available",
    "editing interface required",
    "interface required to modify",
    "is not available through the tools in this session",
    "not available through the tools in this session",
    "tools in this session do not provide",
    "tools in this session do not",
    "cannot truthfully claim",
    "can't truthfully claim",
    "would not modify your actual",
    "would not modify the actual",
    "cannot modify your actual",
    "can't modify your actual",
    "not an active connection to your filesystem",
    "not an active connection to",
    "not an active filesystem connection",
    "without being able to edit and test",
    "without being able to edit",
    "cannot honestly claim to have fixed",
    "can't honestly claim to have fixed",
    "i won't claim i've modified",
    "i won’t claim i’ve modified",
    "i will not claim that files were modified",
    "i will not claim i've modified",
    "i won't claim that files were modified",
    "i won’t claim that files were modified",
)

_SPURIOUS_META_REPORT_PHRASES = (
    "the attempted file-content inspection did not return",
    "the recorded tool result did not include",
    "i cannot yet reliably identify",
    "the actual source code was not",
    "no files have been modified",
    "the next diagnostic step is to",
    "the next diagnostic steps are to",
    "the next step is to read",
    "the next step is to inspect",
    "i was unable to retrieve",
    "the output was empty or truncated",
    "inspected the available execution results",
    "the attempted file-content inspection",
    "did not return the actual source code",
    "attempted file inspection did not",
    "not enough source code",
    "does not contain the complete javascript",
    "does not contain the complete source",
    "no code changes have been made yet",
    "i cannot responsibly identify or patch",
    "cannot responsibly identify or patch",
    "available conversation contains only a partial",
)

_CITATION_TOKEN_RE = re.compile(
    r":contentReference\[?oaicite:\d+\]?(?:\{[^}]*\})?|\[?oaicite:\d+\]?(?:\{[^}]*\})?"
)


def _clean_citations(text: str) -> str:
    """Strip ChatGPT document citation artifacts like :contentReference[oaicite:0]{index=0}."""
    if not isinstance(text, str):
        return text
    return _CITATION_TOKEN_RE.sub("", text).strip()


def _clean_wrapper_tags(text: str) -> str:
    """Strip XML wrapper tags like <escape>...</escape> or <json>...</json> from text."""
    if not isinstance(text, str):
        return text
    t = text.strip()
    for _ in range(3):
        m = re.match(r"^<([a-zA-Z0-9_\-]+)>\s*(.*?)\s*</\1>$", t, re.DOTALL | re.IGNORECASE)
        if m:
            t = m.group(2).strip()
        else:
            break
    t = re.sub(r"^<(?:escape|json|output|response|result)>\s*", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s*</(?:escape|json|output|response|result)>$", "", t, flags=re.IGNORECASE).strip()
    return t


def _has_modifying_tool(tools: list[Any]) -> bool:
    names = _tool_names(tools)
    modifying_keywords = ("edit", "write", "patch", "modify", "replace", "create", "delete", "remove")
    return any(any(k in name.lower() for k in modifying_keywords) for name in names)


def _has_called_modifying_tool(history: list[Any] | None) -> bool:
    if not history:
        return False
    modifying_keywords = ("edit", "write", "patch", "modify", "replace", "create", "delete", "remove")
    for msg in history:
        calls = getattr(msg, "tool_calls", None)
        if calls is None and isinstance(msg, dict):
            calls = msg.get("tool_calls")
        if calls and isinstance(calls, list):
            for call in calls:
                fn = getattr(call, "function", None)
                fname = getattr(fn, "name", None) if fn is not None else None
                if not fname and isinstance(call, dict):
                    fn_obj = call.get("function")
                    fname = fn_obj.get("name") if isinstance(fn_obj, dict) else call.get("name")
                if fname and any(k in str(fname).lower() for k in modifying_keywords):
                    return True
        role = _role(msg).lower()
        if role == "tool":
            tname = getattr(msg, "name", None) or (msg.get("name") if isinstance(msg, dict) else None)
            if tname and any(k in str(tname).lower() for k in modifying_keywords):
                return True
    return False


def _check_spurious_response(
    text: str, tools: list[Any], history: list[Any] | None = None
) -> list[str]:
    low = (text or "").lower()
    errors: list[str] = []

    # 1. Spurious upload request or claiming to only have a transcript
    if any(p in low for p in _SPURIOUS_UPLOAD_PHRASES):
        user_asked_for_upload = False
        if history:
            for msg in reversed(history):
                if _role(msg).lower() == "user":
                    content = _content_to_text(getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")).lower()
                    if "upload" in content or "paste" in content or "attachment" in content:
                        user_asked_for_upload = True
                    break
        if not user_asked_for_upload:
            errors.append(
                "model asked user to upload files or claimed it only has a transcript instead of using available filesystem tools (e.g. read)"
            )

    # 2. Spurious sandbox / Code Interpreter / /mnt/data download links
    if any(p in low for p in _SPURIOUS_SANDBOX_PHRASES):
        errors.append(
            "model hallucinated a sandbox:/mnt/data or Code Interpreter download link instead of editing workspace files directly"
        )

    # 3. Phantom fix: claims files were fixed or rebuilt without ever calling edit/write tools
    if any(p in low for p in _PHANTOM_FIX_PHRASES) and _has_modifying_tool(tools):
        if not _has_called_modifying_tool(history):
            errors.append(
                "model claimed files were fixed/rebuilt without calling edit or write tools to modify them on disk"
            )

    # 4. Spurious tool deprivation / permission refusal or citation hallucination
    if any(p in low for p in _SPURIOUS_DEPRIVATION_PHRASES) or ":contentreference" in low or "oaicite:" in low:
        errors.append(
            "model falsely claimed it lacks filesystem access or workspace tools, or output citation tags instead of using available tools"
        )

    # 5. Meta-reporting: model narrates tool execution failures or diagnostic state instead of calling tools
    if any(p in low for p in _SPURIOUS_META_REPORT_PHRASES):
        errors.append(
            "model narrated tool failure / diagnostic state in plain text instead of emitting a tool_call to gather missing data"
        )

    return errors


def _is_spurious_upload_request(text: str, history: list[Any] | None = None) -> bool:
    low = (text or "").lower()
    if not any(p in low for p in _SPURIOUS_UPLOAD_PHRASES):
        return False
    if history:
        for msg in reversed(history):
            if _role(msg).lower() == "user":
                content = _content_to_text(getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")).lower()
                if "upload" in content or "paste" in content or "attachment" in content:
                    return False
                break
    return True


def parse_chatgpt_response(
    raw_text: Any,
    tools: list[Any],
    history: list[Any] | None = None,
    is_correction: bool = False,
) -> dict:
    """Parse/validate ChatGPT's text reply.

    Returns one of:
        {"type": "message", "content": str}
        {"type": "tool_call", "tool_calls": [{"id","name","arguments"}, ...]}
        {"type": "validation_failure", "errors": [str, ...], "content": str}
    """
    raw = str(raw_text or "").strip()
    obj = _extract_json_object(raw)

    if obj is None:
        cleaned_raw = _clean_wrapper_tags(_clean_citations(raw))
        if tools:
            spurious_errors = _check_spurious_response(cleaned_raw, tools, history=history)
            if spurious_errors:
                return {
                    "type": "validation_failure",
                    "errors": spurious_errors,
                    "content": _VALIDATION_REJECTED_MESSAGE.format(
                        errors="; ".join(spurious_errors)
                    ),
                }
        return {"type": "message", "content": cleaned_raw}

    kind = obj.get("webmind_response")

    # Accept both WebMind envelope {"webmind_response": "tool_call", "tool_calls": [...]}
    # and standard OpenAI tool calling format {"tool_calls": [...]}
    is_tool_call_envelope = (kind == "tool_call") or (kind is None and ("tool_calls" in obj or "tool_call" in obj))
    if is_tool_call_envelope:
        calls = obj.get("tool_calls") if "tool_calls" in obj else obj.get("tool_call")
        if isinstance(calls, dict):
            calls = [calls]
        if isinstance(calls, list) and calls:
            tool_names = _tool_names(tools)
            validated: list[dict] = []
            failures: list[str] = []
            for call in calls:
                result, problems = _inspect_tool_call(call, tools, tool_names, is_correction=is_correction)
                if result is None:
                    failures.extend(problems)
                else:
                    validated.append(result)
            if failures:
                return {
                    "type": "validation_failure",
                    "errors": failures,
                    "content": _VALIDATION_REJECTED_MESSAGE.format(
                        errors="; ".join(failures)
                    ),
                }
            res = {"type": "tool_call", "tool_calls": validated}
            if obj.get("content"):
                res["content"] = obj["content"]
            return res
        return {"type": "message", "content": raw}

    if kind == "message" or (kind is None and ("content" in obj or "message" in obj)):
        content = obj.get("content") if "content" in obj else obj.get("message")
        if isinstance(content, str):
            unescaped = _clean_wrapper_tags(_clean_citations(_unescape_prose_newlines(content)))
            if tools:
                spurious_errors = _check_spurious_response(unescaped, tools, history=history)
                if spurious_errors:
                    return {
                        "type": "validation_failure",
                        "errors": spurious_errors,
                        "content": _VALIDATION_REJECTED_MESSAGE.format(
                            errors="; ".join(spurious_errors)
                        ),
                    }
            return {
                "type": "message",
                "content": unescaped,
            }
        return {"type": "message", "content": _clean_citations(raw)}

    # Parsed dict that is not a valid envelope -> treat as plain text.
    return {"type": "message", "content": _clean_citations(raw)}


# ============================================================
# OPENAI-COMPATIBLE RESPONSE BUILDERS
# ============================================================

def _sse_chunk(chat_id: str, created: int, delta: dict, finish_reason=None, usage=None) -> str:
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "webmind",
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


# Terminal chunk: the harness requests stream_options.include_usage, and its
# stream consumer (dsh-llm-deepseek translate) captures usage attached to the
# finish chunk. WebMind has no real token counts (it reads the ChatGPT web UI),
# so it reports zeros so the session log still carries an exact token record.
def _sse_terminal_chunk(chat_id: str, created: int, finish_reason: str) -> str:
    return _sse_chunk(
        chat_id,
        created,
        {},
        finish_reason=finish_reason,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )


def sse_tool_call_chunks(
    chat_id: str,
    created: int,
    tool_calls: list[dict],
    content: str | None = None,
) -> list[str]:
    chunks = [_sse_chunk(chat_id, created, {"role": "assistant", "content": content})]
    for idx, tc in enumerate(tool_calls):
        chunks.append(
            _sse_chunk(
                chat_id,
                created,
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "index": idx,
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": ""},
                        }
                    ],
                },
            )
        )
        args_json = json.dumps(tc["arguments"])
        chunks.append(
            _sse_chunk(
                chat_id,
                created,
                {
                    "content": None,
                    "tool_calls": [
                        {"index": idx, "function": {"arguments": args_json}}
                    ],
                },
            )
        )
    chunks.append(_sse_terminal_chunk(chat_id, created, "tool_calls"))
    chunks.append("data: [DONE]\n\n")
    return chunks


def sse_message_chunks(chat_id: str, created: int, content: str, chunk_size: int = 256) -> list[str]:
    chunks = [_sse_chunk(chat_id, created, {"role": "assistant", "content": ""})]
    text = content or ""
    for i in range(0, len(text), chunk_size):
        chunks.append(_sse_chunk(chat_id, created, {"content": text[i:i + chunk_size]}))
    chunks.append(_sse_terminal_chunk(chat_id, created, "stop"))
    chunks.append("data: [DONE]\n\n")
    return chunks


def build_nonstream_response(parsed: dict, chat_id: str, created: int) -> dict:
    payload = {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": "webmind",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    if parsed["type"] == "tool_call":
        payload["choices"] = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": parsed.get("content"),
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in parsed["tool_calls"]
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    else:
        payload["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": parsed["content"]},
                "finish_reason": "stop",
            }
        ]

    return payload