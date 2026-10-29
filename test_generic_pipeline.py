"""Generic tool-call pipeline audit tests.

A fully synthetic tool set (no read/pwsh/edit) exercises WebMind as an
arbitrary OpenAI-compatible provider the way a generic harness would use it:

    schema (nested AND flat) -> prompt (full schemas / compact signatures)
    -> model text -> parse -> validate (types, enums, arrays, nested objects,
    union type lists, additionalProperties: false)
    -> OpenAI response (message.tool_calls / delta.tool_calls, ids preserved)
    -> multi-turn loop (>= 10 consecutive agentic turns, multiple tools per
    turn, tool results as parts, user-role results with tool_call_id)

It also proves the audit invariants: executor failures never become WebMind
validation failures, a successful tool call is never emitted as a plain
assistant message, and an identical re-issued call after a completed result
is blocked while legitimate different calls proceed.

Reuses the generic harness/backend machinery from test_conformance.py.
"""

import json
import unittest

from protocol import (
    _content_to_text,
    _schema_for,
    _tool_names,
    build_prompt,
    parse_chatgpt_response,
    validate_arguments,
)

from test_conformance import (
    Harness,
    ModelBackend,
    _as_message,
    _envelope_message,
    _envelope_tool_call,
    sse_to_chat_message,
)


# ============================================================
# SYNTHETIC GENERIC TOOL SET
# ============================================================

UPLOAD_SCAN_TOOL = {
    "type": "function",
    "function": {
        "name": "upload_scan",
        "description": "Scan an uploaded data source and report rows.",
        "parameters": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source file path."},
                "format": {"type": "string", "enum": ["csv", "json", "yaml"]},
                "recurse": {"type": "boolean"},
                "timeout_ms": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["source", "format"],
        },
    },
}

MERGE_REPORT_TOOL = {
    "type": "function",
    "function": {
        "name": "merge_report",
        "description": "Merge scanned files into one report.",
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
                "target": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["files", "target"],
        },
    },
}

FILTER_ROWS_TOOL = {
    "type": "function",
    "function": {
        "name": "filter_rows",
        "description": "Filter rows from a named collection.",
        "parameters": {
            "type": "object",
            "properties": {
                "collection": {"type": "string", "enum": ["inbox", "archive"]},
                "max_results": {"type": "integer"},
                "exact": {"type": "boolean"},
            },
            "required": ["collection"],
        },
    },
}

RENDER_VIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "render_view",
        "description": "Render a view to an output file.",
        "parameters": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string"},
                "format": {"type": "string", "enum": ["html", "pdf"]},
                "data": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "rows": {"type": "array", "items": {"type": "integer"}},
                        "tone": {"type": "string", "enum": ["dark", "light"]},
                    },
                    "required": ["title", "rows"],
                },
            },
            "required": ["output_path", "data"],
        },
    },
}

GENERIC_TOOLS = [
    UPLOAD_SCAN_TOOL,
    MERGE_REPORT_TOOL,
    FILTER_ROWS_TOOL,
    RENDER_VIEW_TOOL,
]

FLAT_TOOLS = [t["function"] for t in GENERIC_TOOLS]


def _upload_call(call_id, source="a.csv", fmt="csv", extra=None):
    args = {"source": source, "format": fmt}
    if extra:
        args.update(extra)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "upload_scan", "arguments": json.dumps(args)},
    }


class GenericToolcallPipelineTest(unittest.TestCase):
    """The complete generic pipeline against a synthetic tool set."""

    # ---- schema handling -------------------------------------------------

    def test_tool_names_and_schemas_accept_flat_and_nested(self):
        expected = {"upload_scan", "merge_report", "filter_rows", "render_view"}
        self.assertEqual(_tool_names(GENERIC_TOOLS), expected)
        self.assertEqual(_tool_names(FLAT_TOOLS), expected)
        params = _schema_for(GENERIC_TOOLS, "upload_scan")
        self.assertEqual(params["properties"]["format"]["enum"], ["csv", "json", "yaml"])
        flat_params = _schema_for(FLAT_TOOLS, "merge_report")
        self.assertEqual(flat_params["properties"]["files"]["type"], "array")

    def test_continuation_signatures_cover_all_generic_tools(self):
        history = [
            {"role": "user", "content": "scan a.csv then continue"},
            {"role": "assistant", "content": None, "tool_calls": [_upload_call("call_g1")]},
            {"role": "tool", "tool_call_id": "call_g1", "name": "upload_scan",
             "content": json.dumps({"rows": 53})},
            {"role": "user", "content": "continue"},
        ]
        prompt, _ = build_prompt([_as_message(m) for m in history], GENERIC_TOOLS)
        self.assertIn(
            "upload_scan(source: string [required], format: string [required], "
            "recurse?: boolean, timeout_ms?: integer, tags?: array)",
            prompt,
        )
        self.assertIn(
            "merge_report(files: array [required], target: string [required], "
            "overwrite?: boolean)",
            prompt,
        )
        self.assertIn(
            "filter_rows(collection: string [required], max_results?: integer, "
            "exact?: boolean)",
            prompt,
        )
        self.assertIn(
            "render_view(output_path: string [required], format?: string, "
            "data: object [required])",
            prompt,
        )

    # ---- validation matrix ----------------------------------------------

    def test_validation_accepts_valid_arguments(self):
        self.assertEqual(
            validate_arguments(
                {"source": "a.csv", "format": "csv", "tags": ["x", "y"]},
                _schema_for(GENERIC_TOOLS, "upload_scan"),
            ),
            [],
        )

    def test_validation_enforces_enum(self):
        errors = validate_arguments(
            {"source": "a.csv", "format": "xml"},
            _schema_for(GENERIC_TOOLS, "upload_scan"),
        )
        self.assertEqual(
            errors, ["property 'format' must be one of: 'csv', 'json', 'yaml'"]
        )

    def test_validation_enforces_required_and_types(self):
        schema = _schema_for(GENERIC_TOOLS, "upload_scan")
        self.assertEqual(
            validate_arguments({"source": "a.csv"}, schema),
            ["missing required property 'format'"],
        )
        self.assertIn("must be string", validate_arguments(
            {"source": 1, "format": "csv"}, schema
        )[0])
        self.assertIn("must be integer", validate_arguments(
            {"source": "a.csv", "format": "csv", "timeout_ms": 1.5}, schema
        )[0])
        self.assertIn("must be boolean", validate_arguments(
            {"source": "a.csv", "format": "csv", "recurse": "yes"}, schema
        )[0])
        errors = validate_arguments(
            {"source": "a.csv", "format": "csv", "tags": [1]}, schema
        )
        self.assertEqual(errors, ["property 'tags[0]' must be string"])

    def test_validation_tolerates_unknown_arguments(self):
        schema = _schema_for(GENERIC_TOOLS, "upload_scan")
        self.assertEqual(
            validate_arguments(
                {"source": "a.csv", "format": "csv", "description": "scan this"},
                schema,
            ),
            [],
        )

    def test_validation_honors_additional_properties_false(self):
        strict = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"},
            },
            "required": ["location", "units"],
            "additionalProperties": False,
        }
        self.assertEqual(
            validate_arguments({"location": "Paris", "units": "celsius"}, strict), []
        )
        self.assertEqual(
            validate_arguments(
                {"location": "Paris", "units": "celsius", "extra": 1}, strict
            ),
            ["unknown property 'extra' (additionalProperties is false)"],
        )
        permissive = dict(strict, additionalProperties=True)
        self.assertEqual(
            validate_arguments(
                {"location": "Paris", "units": "celsius", "extra": 1}, permissive
            ),
            [],
        )
        nested = {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["files"],
        }
        self.assertEqual(
            validate_arguments({"files": [{"path": "a.csv", "junk": 1}]}, nested),
            ["unknown property 'files[0].junk' (additionalProperties is false)"],
        )

    def test_validation_honors_union_types(self):
        optional = {
            "type": "object",
            "properties": {
                "title": {"type": ["string", "null"]},
                "count": {"type": ["integer", "null"]},
            },
            "required": ["title", "count"],
            "additionalProperties": False,
        }
        self.assertEqual(validate_arguments({"title": "T", "count": 3}, optional), [])
        self.assertEqual(validate_arguments({"title": None, "count": None}, optional), [])
        self.assertEqual(
            validate_arguments({"title": 5, "count": 3}, optional),
            ["property 'title' must match one of: 'string', 'null'"],
        )
        self.assertEqual(
            validate_arguments({"title": "T", "count": "3"}, optional),
            ["property 'count' must match one of: 'integer', 'null'"],
        )
        strict_weather = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {
                    "type": ["string", "null"],
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location", "units"],
            "additionalProperties": False,
        }
        self.assertEqual(
            validate_arguments(
                {"location": "Paris", "units": "celsius"}, strict_weather
            ),
            [],
        )
        self.assertEqual(
            validate_arguments({"location": "Paris", "units": None}, strict_weather), []
        )
        self.assertEqual(
            validate_arguments({"location": "Paris", "units": "rankine"}, strict_weather),
            ["property 'units' must be one of: 'celsius', 'fahrenheit'"],
        )

    def test_validation_reaches_into_arrays_and_nested_objects(self):
        merge = _schema_for(GENERIC_TOOLS, "merge_report")
        self.assertEqual(
            validate_arguments({"files": [{"label": "A"}], "target": "out.json"}, merge),
            ["property 'files[0].path' is required"],
        )
        self.assertEqual(
            validate_arguments({"files": [{"path": "a.csv"}], "target": 5}, merge),
            ["property 'target' must be string"],
        )
        render = _schema_for(GENERIC_TOOLS, "render_view")
        self.assertIn(
            "must be integer",
            validate_arguments(
                {"output_path": "v.html", "data": {"title": "T", "rows": ["a"]}},
                render,
            )[0],
        )
        self.assertEqual(
            validate_arguments(
                {"output_path": "v.html", "data": {"rows": [1]}}, render
            ),
            ["property 'data.title' is required"],
        )
        self.assertIn(
            "must be one of",
            validate_arguments(
                {"output_path": "v.html", "data": {"title": "T", "rows": [1], "tone": "blue"}},
                render,
            )[0],
        )

    def test_similarly_named_parameters_enforced_per_schema(self):
        scan = _schema_for(GENERIC_TOOLS, "upload_scan")
        render = _schema_for(GENERIC_TOOLS, "render_view")
        self.assertIn("must be one of", validate_arguments(
            {"source": "a.csv", "format": "html"}, scan
        )[0])
        self.assertIn("must be one of", validate_arguments(
            {"output_path": "v.html", "format": "csv", "data": {"title": "T", "rows": []}},
            render,
        )[0])
        self.assertEqual(
            validate_arguments(
                {"output_path": "v.html", "format": "pdf", "data": {"title": "T", "rows": []}},
                render,
            ),
            [],
        )

    # ---- parse / unknown tool / multi-call ------------------------------

    def test_unknown_tool_is_rejected(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [{"id": "call_gx", "name": "totally_missing",
                            "arguments": {"anything": 1}}],
        })
        parsed = parse_chatgpt_response(raw, GENERIC_TOOLS)
        self.assertEqual(parsed["type"], "validation_failure")
        self.assertTrue(any("unknown tool" in e for e in parsed["errors"]))

    def test_multiple_tool_calls_preserved_in_order_with_ids(self):
        raw = json.dumps({
            "webmind_response": "tool_call",
            "tool_calls": [
                {"id": "call_g1", "name": "upload_scan",
                 "arguments": {"source": "a.csv", "format": "csv"}},
                {"id": "call_g2", "name": "filter_rows",
                 "arguments": {"collection": "inbox"}},
            ],
        })
        parsed = parse_chatgpt_response(raw, GENERIC_TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        calls = parsed["tool_calls"]
        self.assertEqual([c["id"] for c in calls], ["call_g1", "call_g2"])
        self.assertEqual([c["name"] for c in calls], ["upload_scan", "filter_rows"])
        self.assertEqual(calls[0]["arguments"], {"source": "a.csv", "format": "csv"})
        self.assertEqual(calls[1]["arguments"], {"collection": "inbox"})

    def test_flat_tool_format_parses_and_validates(self):
        raw = json.dumps(_envelope_tool_call(
            "call_gf", "upload_scan", {"source": "b.json", "format": "json"}
        ))
        parsed = parse_chatgpt_response(raw, FLAT_TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        bad = json.dumps(_envelope_tool_call(
            "call_gf2", "upload_scan", {"source": "b.json", "format": "xml"}
        ))
        rejected = parse_chatgpt_response(bad, FLAT_TOOLS)
        self.assertEqual(rejected["type"], "validation_failure")

    # ---- OpenAI response generation -------------------------------------

    def test_streaming_and_nonstreaming_parity_and_no_content_downgrade(self):
        from protocol import build_nonstream_response, sse_tool_call_chunks
        raw = json.dumps(_envelope_tool_call(
            "call_gm", "merge_report",
            {"files": [{"path": "a.csv", "label": "A"}], "target": "out.json"},
        ))
        parsed = parse_chatgpt_response(raw, GENERIC_TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        ns = build_nonstream_response(parsed, "chatcmpl-g", 1)
        msg, fin = sse_to_chat_message(
            sse_tool_call_chunks("chatcmpl-g", 1, parsed["tool_calls"])
        )
        ns_tc = ns["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(ns["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(fin, "tool_calls")
        self.assertIsNone(msg["content"])
        self.assertEqual(ns_tc["id"], "call_gm")
        self.assertEqual(msg["tool_calls"][0]["id"], "call_gm")
        self.assertEqual(ns_tc["function"]["name"], "merge_report")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "merge_report")
        self.assertEqual(
            json.loads(ns_tc["function"]["arguments"]),
            json.loads(msg["tool_calls"][0]["function"]["arguments"]),
        )

    # ---- multi-turn continuation ----------------------------------------

    def test_tool_result_parts_rendered_into_next_prompt(self):
        history = [
            {"role": "user", "content": "scan a.csv"},
            {"role": "assistant", "content": None, "tool_calls": [_upload_call("call_g1")]},
            {
                "role": "user",
                "tool_call_id": "call_g1",
                "name": "upload_scan",
                "content": [{"type": "tool-result", "toolCallId": "call_g1",
                             "content": [{"type": "text", "text": "53 rows matched csv"}]}],
            },
            {"role": "user", "content": "continue"},
        ]
        prompt, _ = build_prompt([_as_message(m) for m in history], GENERIC_TOOLS)
        self.assertIn("53 rows matched csv", prompt)
        self.assertIn("tool_call_id: call_g1", prompt)
        self.assertEqual(
            _content_to_text([
                {"type": "image_url", "image_url": {"url": "http://x"}},
                {"type": "tool-result", "toolCallId": "c",
                 "content": [{"type": "text", "text": "row output"}]},
            ]),
            "[image attached]\nrow output",
        )

    def test_user_role_result_does_not_block_reissue(self):
        # A user-role result with a matching tool_call_id still lets the
        # model re-issue the SAME call later (state may have changed):
        # WebMind forwards both the identical and the different proposal.
        history = [
            {"role": "user", "content": "scan a.csv"},
            {"role": "assistant", "content": None, "tool_calls": [_upload_call("call_g1")]},
            {"role": "user", "tool_call_id": "call_g1", "name": "upload_scan",
             "content": json.dumps({"rows": 53})},
            {"role": "user", "content": "again"},
        ]
        identical = json.dumps(_envelope_tool_call(
            "call_g_again", "upload_scan", {"source": "a.csv", "format": "csv"}
        ))
        forwarded = parse_chatgpt_response(
            identical, GENERIC_TOOLS, history=[_as_message(m) for m in history]
        )
        self.assertEqual(forwarded["type"], "tool_call")
        self.assertEqual(forwarded["tool_calls"][0]["id"], "call_g_again")
        self.assertEqual(
            forwarded["tool_calls"][0]["arguments"],
            {"source": "a.csv", "format": "csv"},
        )
        different = json.dumps(_envelope_tool_call(
            "call_g_new", "upload_scan", {"source": "b.json", "format": "json"}
        ))
        allowed = parse_chatgpt_response(
            different, GENERIC_TOOLS, history=[_as_message(m) for m in history]
        )
        self.assertEqual(allowed["type"], "tool_call")
        self.assertEqual(allowed["tool_calls"][0]["id"], "call_g_new")

    def test_executor_error_is_not_a_webmind_failure(self):
        calls = []

        def executor(name, args):
            calls.append((name, args))
            if name == "upload_scan":
                return {"isError": True, "error": "permission denied reading a.csv"}
            return {"exit_code": 0, "rows": len(args.get("files", []))}

        backup = ModelBackend([
            json.dumps(_envelope_tool_call(
                "call_e1", "upload_scan", {"source": "a.csv", "format": "csv"})),
            json.dumps(_envelope_tool_call(
                "call_e2", "merge_report",
                {"files": [{"path": "a.csv"}], "target": "out.json"})),
            json.dumps(_envelope_message("done after the error")),
        ])
        h = Harness(GENERIC_TOOLS, backup)
        h.user("scan a.csv then merge")
        resp1 = h.chat_once()
        self.assertTrue(
            resp1["choices"][0]["message"].get("tool_calls"),
            "the proposing tool call must reach the harness as tool_calls",
        )
        handled = resp1["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(handled), 1)
        h.run_tool_calls(resp1, executor)

        resp2 = h.chat_once()
        self.assertTrue(
            resp2["choices"][0]["message"].get("tool_calls"),
            "an executor error must not turn into a WebMind failure",
        )
        self.assertIn("permission denied reading a.csv", backup.prompts[1])

        h.run_tool_calls(resp2, executor)
        resp3 = h.chat_once()
        self.assertFalse(resp3["choices"][0]["message"].get("tool_calls"))
        self.assertEqual(resp3["choices"][0]["message"]["content"], "done after the error")
        self.assertEqual([c[0] for c in calls], ["upload_scan", "merge_report"])

    def test_ten_consecutive_agentic_turns_generic_tools(self):
        script = [
            _envelope_tool_call("c1", "upload_scan", {"source": "a.csv", "format": "csv"}),
            _envelope_tool_call("c2", "merge_report",
                                {"files": [{"path": "a.csv"}], "target": "out.json"}),
            _envelope_tool_call("c3", "render_view",
                                {"output_path": "v.html", "format": "html",
                                 "data": {"title": "Rows", "rows": [1, 2, 3]}}),
            _envelope_tool_call("c4", "filter_rows",
                                {"collection": "inbox", "max_results": 5}),
            _envelope_tool_call("c5", "upload_scan",
                                {"source": "b.json", "format": "json",
                                 "recurse": True, "timeout_ms": 120}),
            {
                "webmind_response": "tool_call",
                "tool_calls": [
                    {"id": "c6a", "name": "upload_scan",
                     "arguments": {"source": "c.yaml", "format": "yaml"}},
                    {"id": "c6b", "name": "filter_rows",
                     "arguments": {"collection": "archive"}},
                ],
            },
            _envelope_tool_call("c7", "merge_report",
                                {"files": [{"path": "b.json", "label": "B"}],
                                 "target": "out2.json", "overwrite": True}),
            _envelope_tool_call("c8", "render_view",
                                {"output_path": "v.pdf", "format": "pdf",
                                 "data": {"title": "T", "rows": [9]}}),
            _envelope_tool_call("c9", "filter_rows",
                                {"collection": "inbox", "exact": True}),
            _envelope_tool_call("c10", "upload_scan",
                                {"source": "d.csv", "format": "csv", "tags": ["x", "y"]}),
            _envelope_message("all done"),
        ]
        executed = []

        def executor(name, args):
            executed.append(name)
            return {"exit_code": 0}

        backend = ModelBackend([json.dumps(s) for s in script])
        h = Harness(GENERIC_TOOLS, backend)
        h.user("process all the sources end to end")
        final = None
        turns = 0
        while True:
            resp = h.chat_once()
            turns += 1
            msg = resp["choices"][0]["message"]
            if not msg.get("tool_calls"):
                final = msg["content"]
                break
            h.run_tool_calls(resp, executor)
            self.assertLess(turns, 30)
        self.assertEqual(final, "all done")
        self.assertEqual(len(executed), 11)
        self.assertEqual(
            executed,
            ["upload_scan", "merge_report", "render_view", "filter_rows",
             "upload_scan", "upload_scan", "filter_rows", "merge_report",
             "render_view", "filter_rows", "upload_scan"],
        )
        # Every continuation prompt carries the compact tool signatures.
        for i in range(1, len(backend.prompts)):
            self.assertIn("TOOL DEFINITIONS were already provided", backend.prompts[i])
            self.assertIn("upload_scan(", backend.prompts[i])


if __name__ == "__main__":
    unittest.main()
