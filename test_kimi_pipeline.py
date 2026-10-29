import json
import unittest
from fastapi.testclient import TestClient

from kimi_protocol import (
    build_kimi_prompt,
    parse_kimi_response,
    sse_message_chunks,
    sse_tool_call_chunks,
    build_nonstream_response,
    _extract_json_object,
)
import kimi_server


class TestKimiPipeline(unittest.TestCase):
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

    def test_build_kimi_prompt_with_tools(self):
        messages = [
            {"role": "user", "content": "Inspect and refactor helper.py"},
        ]
        prompt = build_kimi_prompt(messages, self.TOOLS)
        self.assertIn("webmind_response", prompt)
        self.assertIn("AVAILABLE TOOLS:", prompt)
        self.assertIn("read(", prompt)
        self.assertIn("Inspect and refactor helper.py", prompt)

    def test_multi_backtick_and_attribute_extraction(self):
        raw = (
            '`````json id="kimi_block_99"\n'
            '{\n'
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {"id": "call_kimi_1", "name": "read_file", "arguments": {"path": "helper.py"}}\n'
            '  ]\n'
            '}\n'
            '`````'
        )
        parsed = parse_kimi_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["name"], "read")

    def test_bare_json_tool_call_and_message_envelope_merging(self):
        raw = (
            '{\n'
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {"id": "call_123", "name": "bash", "arguments": {"command": "git status"}}\n'
            '  ]\n'
            '}\n'
            '```json\n'
            '{\n'
            '  "webmind_response": "message",\n'
            '  "content": "Checking git status before applying the patch."\n'
            '}\n'
            '```'
        )
        parsed = parse_kimi_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["name"], "bash")
        self.assertIn("Checking git status", parsed.get("content", ""))

    def test_nonstream_and_sse_response_formats(self):
        parsed = {
            "type": "tool_call",
            "tool_calls": [{"id": "c1", "name": "read", "arguments": {"path": "app.py"}}],
            "content": "Reading app.py",
        }
        nonstream = build_nonstream_response(parsed, "chat-kimi-1", 12345, model="kimi")
        self.assertEqual(nonstream["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(nonstream["choices"][0]["message"]["content"], "Reading app.py")

        chunks = sse_tool_call_chunks("chat-kimi-1", 12345, parsed["tool_calls"], content="Reading app.py")
        first = json.loads(chunks[0][len("data: "):])
        self.assertEqual(first["choices"][0]["delta"]["content"], "Reading app.py")
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")
        terminal = json.loads(chunks[-2][len("data: "):])
        self.assertEqual(terminal["choices"][0]["finish_reason"], "tool_calls")

    def test_server_models_endpoint(self):
        client = TestClient(kimi_server.app)
        res = client.get("/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()["data"]
        ids = [m["id"] for m in data]
        self.assertIn("kimi", ids)
        self.assertIn("moonshot-v1", ids)



    def test_json_copy_download_leak_repaired(self):
        raw = (
            'jsonCopyDownload{\n'
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {\n'
            '      "id": "call_fix_muzzle_spawn_scope_1226",\n'
            '      "name": "edit",\n'
            '      "arguments": {\n'
            '        "path": "C:\\\\Users\\\\Developer\\\\Documents\\\\Projects\\\\fps_game\\\\game.js",\n'
            '        "oldString": "player.ammo--;shootCooldown=.105;",\n'
            '        "newString": "player.ammo--;shootCooldown=.105;const scale=1;"\n'
            '      }\n'
            '    }\n'
            '  ]\n'
            '}'
        )
        parsed = parse_kimi_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["name"], "edit")
        self.assertIn("fps_game", parsed["tool_calls"][0]["arguments"]["path"])

    def test_windows_path_single_backslashes_repaired(self):
        raw = (
            'jsonCopyDownload{\n'
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {\n'
            '      "id": "call_single_bs",\n'
            '      "name": "read",\n'
            '      "arguments": {\n'
            '        "path": "C:\\Users\\Developer\\Documents\\Projects\\app.js"\n'
            '      }\n'
            '    }\n'
            '  ]\n'
            '}'
        )
        parsed = parse_kimi_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertIn("Projects", parsed["tool_calls"][0]["arguments"]["path"])

    def test_nested_tool_call_inside_message_content(self):
        raw = json.dumps({
            "webmind_response": "message",
            "content": json.dumps({
                "webmind_response": "tool_call",
                "tool_calls": [
                    {"id": "call_nested", "name": "read", "arguments": {"path": "config.json"}}
                ]
            })
        })
        parsed = parse_kimi_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "read")

if __name__ == "__main__":
    unittest.main()
