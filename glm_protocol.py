# ============================================================
# GLM (ChatGLM / Zhipu) WebMind Protocol: Tool-call translation bridge.
#
# Pure logic only (unit-testable without a browser).
# Translates OpenAI chat completions requests into GLM prompts,
# and parses/validates GLM's returned JSON envelopes into
# standard OpenAI responses with anti-leak and multi-envelope support.
# ============================================================

import json
import os
import re
import uuid
from typing import Any


def _role(msg: Any) -> str:
    if isinstance(msg, dict):
        return str(msg.get("role", "user"))
    return str(getattr(msg, "role", "user"))


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype in ("text", "input_text") and "text" in part:
                    parts.append(str(part["text"]))
                elif ptype == "image_url":
                    parts.append("[image]")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)



_VALID_JSON_ESCAPES = set('"\\/bfnrtu')
_WINDOWS_PATH_BACKSLASH = re.compile(r"(?:(?<![A-Za-z0-9])[A-Za-z]:|\.)\\")

_UI_CHROME_REGEX = re.compile(
    r"^(?:json|python|bash|javascript|text|sh|cmd|powershell)?\s*(?:copy|download|code|复制|下载|\s)*",
    re.IGNORECASE,
)


def _wants_literal_backslashes(text: str) -> bool:
    return _WINDOWS_PATH_BACKSLASH.search(text) is not None


def _repair_json_escapes(text: str) -> str:
    return re.sub(r"\\(?![\\\\\"\/bfnrtu])", r"\\\\", text)


def _repair_windows_path_escapes(text: str) -> str:
    return re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", text)


def _repair_unescaped_quotes(text: str) -> str:
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



def _clean_candidate(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^<(?:escape|json|output|response|result)>\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*</(?:escape|json|output|response|result)>$", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^```(?:json[^\n`]*|json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    s = _UI_CHROME_REGEX.sub("", s).strip()
    return s.strip()


def _parse_json_lenient(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        snippet = text[first_brace:last_brace + 1].strip()
        if snippet != text:
            candidates.append(snippet)

    expanded = []
    for base in candidates:
        sub = [base]
        if "\\" in base:
            if _wants_literal_backslashes(base):
                sub.insert(0, _repair_windows_path_escapes(base))
            sub.append(_repair_json_escapes(base))
            sub.append(_repair_json_escapes(_repair_windows_path_escapes(base)))
        if '"' in base:
            sub.extend(_repair_unescaped_quotes(cand) for cand in list(sub))
        sub.extend(_repair_control_chars(cand) for cand in list(sub))
        sub.extend(_balance_json_structure(cand) for cand in list(sub))
        expanded.extend(sub)

    seen = set()
    for cand in expanded:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                return val
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _is_envelope_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    return (
        obj.get("webmind_response") in ("tool_call", "message")
        or "tool_calls" in obj
        or "tool_call" in obj
        or ("content" in obj and "role" not in obj)
    )


def _extract_json_object(text: str) -> dict | None:
    """Robust multi-envelope and multi-backtick extraction for GLM."""
    if not isinstance(text, str) or not text.strip():
        return None

    envelopes: list[dict] = []

    # 1. Whole message inside a single code fence (>= 3 backticks, optional attributes)
    fenced_text = _UI_CHROME_REGEX.sub("", text).strip()
    fence = re.match(r"^`{3,}[^\n`]*\n?(.*?)\n?`{3,}$", fenced_text, re.DOTALL)
    if fence:
        candidate = _clean_candidate(fence.group(1))
        obj = _parse_json_lenient(candidate)
        if _is_envelope_dict(obj):
            return obj

    # 2. Whole message is a single bare JSON object
    trimmed = _UI_CHROME_REGEX.sub("", text).strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        obj = _parse_json_lenient(trimmed)
        if _is_envelope_dict(obj):
            return obj

    # 3. Fenced block(s) embedded within preamble or normal text
    for m in re.finditer(r"`{3,}[^\n`]*\n?(.*?)\n?`{3,}", text, re.DOTALL):
        candidate = _clean_candidate(m.group(1))
        obj = _parse_json_lenient(candidate)
        if _is_envelope_dict(obj) and obj not in envelopes:
            envelopes.append(obj)

    # 4. Outermost fence if interior backticks exist
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

    # 5. Unclosed fence at end of text
    unclosed = re.search(r"`{3,}[^\n`]*\n?(\{.*)$", text, re.DOTALL)
    if unclosed:
        candidate = _clean_candidate(unclosed.group(1))
        obj = _parse_json_lenient(candidate)
        if _is_envelope_dict(obj) and obj not in envelopes:
            envelopes.append(obj)

    # 6. Scan for bare JSON envelope(s) outside fences
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
            next_close = text.rfind("}", idx)
            if next_close > idx:
                slice_cand = text[idx:next_close + 1]
                repaired = _parse_json_lenient(slice_cand)
                if _is_envelope_dict(repaired) and repaired not in envelopes:
                    envelopes.append(repaired)
                    pos = next_close + 1
                    continue
            pos = idx + 1

    if not envelopes:
        return None

    # Merge envelopes: prioritize tool_call, combine tool calls, preserve accompanying content
    tool_envs = [
        e for e in envelopes
        if (e.get("webmind_response") == "tool_call" or "tool_calls" in e or "tool_call" in e)
    ]
    msg_envs = [
        e for e in envelopes
        if e not in tool_envs and (e.get("webmind_response") == "message" or "content" in e or "message" in e)
    ]

    if tool_envs:
        merged_calls: list[dict] = []
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
            result["content"] = str(content_str)
        return result

    return msg_envs[0]


def _function_entry(t: Any) -> dict | None:
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


def validate_arguments(args: Any, schema: dict | None) -> list[str]:
    errors: list[str] = []
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    if schema is None or not isinstance(schema, dict):
        return []

    required = schema.get("required", [])
    if isinstance(required, list):
        for req in required:
            if isinstance(req, str) and req not in args:
                errors.append(f"missing required property '{req}'")

    props = schema.get("properties", {})
    if isinstance(props, dict):
        for key, val in args.items():
            prop_schema = props.get(key)
            if isinstance(prop_schema, dict):
                ptype = prop_schema.get("type")
                if ptype == "string" and not isinstance(val, str):
                    errors.append(f"property '{key}' must be string, got {type(val).__name__}")
                elif ptype == "integer" and not isinstance(val, int):
                    errors.append(f"property '{key}' must be integer, got {type(val).__name__}")
                elif ptype == "boolean" and not isinstance(val, bool):
                    errors.append(f"property '{key}' must be boolean, got {type(val).__name__}")
                elif ptype == "array" and not isinstance(val, list):
                    errors.append(f"property '{key}' must be array, got {type(val).__name__}")
                elif ptype == "object" and not isinstance(val, dict):
                    errors.append(f"property '{key}' must be object, got {type(val).__name__}")
    return errors


def _inspect_tool_call(call: Any, tools: list[Any], tool_names: set[str]) -> tuple[dict | None, list[str]]:
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

    TOOL_ALIASES = {
        "read_file": "read",
        "readfile": "read",
        "view_file": "read",
        "write_file": "write",
        "writefile": "write",
        "edit_file": "edit",
        "editfile": "edit",
        "shell": "bash",
        "terminal": "bash",
        "powershell": "bash",
        "run_command": "bash",
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

    problems = validate_arguments(args, _schema_for(tools, name))
    if problems:
        return None, [f"{name}: {p}" for p in problems]

    call_id = call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = "call_" + uuid.uuid4().hex

    return {"id": call_id, "name": name, "arguments": args}, []


def _tool_signatures_note(tools: list[Any]) -> str:
    lines = ["AVAILABLE TOOLS:"]
    for t in tools or []:
        fn = _function_entry(t)
        if not fn:
            continue
        name = fn.get("name", "unnamed")
        desc = fn.get("description", "")
        params = fn.get("parameters", {}).get("properties", {})
        param_strs = []
        for pk, pv in params.items():
            ptype = pv.get("type", "any") if isinstance(pv, dict) else "any"
            param_strs.append(f"{pk}: {ptype}")
        lines.append(f"- {name}({', '.join(param_strs)}): {desc}")
    return "\n".join(lines)


def build_glm_prompt(
    messages: list[Any],
    tools: list[Any],
    tool_choice: Any = None,
) -> str:
    """Serialize messages + tools into a GLM prompt with envelope instructions."""
    has_tools = bool(tools)
    parts: list[str] = []

    has_edit_tool = any(
        (t.get("function", {}).get("name") in ("edit", "replace", "modify") if isinstance(t, dict) else False)
        or ("oldString" in str(t) or "old_string" in str(t))
        for t in tools or []
    )

    if has_tools:
        instruction = (
            "You are an autonomous AI coding assistant connected to a live workspace with real tool capabilities.\n"
            "When executing tools, reply ONLY with a ```json code fence containing the tool_call envelope:\n"
            "```json\n"
            "{\n"
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {"id": "call_<unique_id>", "name": "<exact_tool_name>", "arguments": {...}}\n'
            "  ]\n"
            "}\n"
            "```\n"
            "When providing a final message or explanation without tools, reply with:\n"
            "```json\n"
            '{"webmind_response": "message", "content": "..."}\n'
            "```\n"
            "CRITICAL: Do not invent fake download links or claim tools are unavailable. Use the exact tool names provided."
        )
        if has_edit_tool:
            instruction += (
                "\nEXACT STRING EDITING: When calling `edit`, `oldString` MUST be copied character-for-character from a previous `read` output, matching exact whitespace, indentation, and line breaks. Never guess or fabricate `oldString`. If you have not read the file or if lines moved, call `read` first! Keep `oldString` concise (3-8 lines) with a unique anchor."
            )
        parts.append(instruction)
        parts.append(_tool_signatures_note(tools))

    parts.append("CONVERSATION HISTORY:")
    for msg in messages:
        role = _role(msg)
        content = _content_to_text(msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None))
        block = f"[{role.upper()}]:\n{content}"
        tcs = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        if tcs:
            block += f"\n[tool_calls]: {json.dumps(tcs)}"
        parts.append(block)

    last_msg = messages[-1] if messages else None
    if last_msg:
        last_role = _role(last_msg)
        last_content = _content_to_text(last_msg.get("content") if isinstance(last_msg, dict) else getattr(last_msg, "content", None))
        if last_role == "tool" and any(k in last_content.lower() for k in ("could not find oldstring", "must match exactly", "oldstring in")):
            parts.append(
                "[edit failure recovery] The previous `edit` tool call failed because `oldString` was not found in the file. "
                "CRITICAL: Do NOT guess or invent `oldString`. Call `read` on this file now to inspect the actual verbatim lines, then re-call `edit` with an exact verbatim 3-6 line snippet copied from `read`."
            )

    return "\n\n".join(parts)


def parse_glm_response(
    raw_text: Any,
    tools: list[Any],
    history: list[Any] | None = None,
) -> dict:
    """Parse/validate GLM's reply, preventing tool call JSON leaks into chat."""
    raw = str(raw_text or "").strip()
    obj = _extract_json_object(raw)

    # Emergency recovery: if obj is None but response contains tool call indicators
    if obj is None and ('"webmind_response"' in raw or '"tool_calls"' in raw or '"tool_call"' in raw):
        first_b = raw.find("{")
        last_b = raw.rfind("}")
        if first_b != -1 and last_b > first_b:
            recovered_cand = _UI_CHROME_REGEX.sub("", raw[first_b:last_b + 1].strip()).strip()
            recovered_obj = _parse_json_lenient(recovered_cand)
            if _is_envelope_dict(recovered_obj):
                obj = recovered_obj

    if obj is None:
        return {"type": "message", "content": raw}

    kind = obj.get("webmind_response")
    is_tool_call_envelope = (kind == "tool_call") or (kind is None and ("tool_calls" in obj or "tool_call" in obj))

    # If it is a message envelope, but the message content contains an unparsed tool call envelope:
    if not is_tool_call_envelope and tools:
        content_val = obj.get("content") or obj.get("message")
        if isinstance(content_val, str) and ('"webmind_response": "tool_call"' in content_val or '"webmind_response":"tool_call"' in content_val):
            inner_obj = _extract_json_object(content_val)
            if inner_obj and (inner_obj.get("webmind_response") == "tool_call" or "tool_calls" in inner_obj):
                obj = inner_obj
                kind = "tool_call"
                is_tool_call_envelope = True

    if is_tool_call_envelope:
        calls = obj.get("tool_calls") if "tool_calls" in obj else obj.get("tool_call")
        if isinstance(calls, dict):
            calls = [calls]
        if isinstance(calls, list) and calls:
            tool_names = _tool_names(tools)
            validated: list[dict] = []
            failures: list[str] = []
            for call in calls:
                result, problems = _inspect_tool_call(call, tools, tool_names)
                if result is None:
                    failures.extend(problems)
                else:
                    validated.append(result)
            if failures:
                return {
                    "type": "validation_failure",
                    "errors": failures,
                    "content": f"Tool call validation error: {'; '.join(failures)}",
                }
            res = {"type": "tool_call", "tool_calls": validated}
            if obj.get("content"):
                res["content"] = str(obj["content"])
            return res
        return {"type": "message", "content": raw}

    if kind == "message" or (kind is None and ("content" in obj or "message" in obj)):
        content = obj.get("content") if "content" in obj else obj.get("message")
        if isinstance(content, str):
            return {"type": "message", "content": content}
        return {"type": "message", "content": raw}

    return {"type": "message", "content": raw}


def _sse_chunk(chat_id: str, created: int, delta: dict, finish_reason=None, usage=None) -> str:
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "glm-4",
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload)}\n\n"


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


def build_nonstream_response(parsed: dict, chat_id: str, created: int, model: str = "glm-4") -> dict:
    payload = {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
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
