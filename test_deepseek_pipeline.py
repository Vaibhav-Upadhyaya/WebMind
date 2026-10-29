import json
import unittest
from fastapi.testclient import TestClient

from deepseek_protocol import (
    build_deepseek_prompt,
    parse_deepseek_response,
    sse_message_chunks,
    sse_tool_call_chunks,
    build_nonstream_response,
    _extract_json_object,
)
import deepseek_server


class TestDeepSeekPipeline(unittest.TestCase):
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

    def test_build_deepseek_prompt_with_tools(self):
        messages = [
            {"role": "user", "content": "Fix the muzzle flash bug in game.js"},
        ]
        prompt = build_deepseek_prompt(messages, self.TOOLS)
        self.assertIn("webmind_response", prompt)
        self.assertIn("AVAILABLE TOOLS:", prompt)
        self.assertIn("edit(", prompt)
        self.assertIn("Fix the muzzle flash bug", prompt)

    def test_multi_backtick_and_attribute_extraction(self):
        raw = (
            '``````json id="deepseek_envelope_1"\n'
            '{\n'
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {"id": "call_1", "name": "read", "arguments": {"path": "game.js"}}\n'
            '  ]\n'
            '}\n'
            '``````'
        )
        obj = _extract_json_object(raw)
        self.assertIsNotNone(obj)
        self.assertEqual(obj["webmind_response"], "tool_call")
        self.assertEqual(len(obj["tool_calls"]), 1)

    def test_bare_json_tool_call_and_message_envelope_merging(self):
        raw = (
            '{\n'
            '  "webmind_response": "tool_call",\n'
            '  "tool_calls": [\n'
            '    {"id": "call_123", "name": "shell", "arguments": {"command": "npm test"}}\n'
            '  ]\n'
            '}\n'
            '```json\n'
            '{\n'
            '  "webmind_response": "message",\n'
            '  "content": "Running the unit tests to confirm the build status."\n'
            '}\n'
            '```'
        )
        parsed = parse_deepseek_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertEqual(parsed["tool_calls"][0]["name"], "bash")
        self.assertIn("Running the unit tests", parsed.get("content", ""))

    def test_nonstream_and_sse_response_formats(self):
        parsed = {
            "type": "tool_call",
            "tool_calls": [{"id": "c1", "name": "read", "arguments": {"path": "a.txt"}}],
            "content": "Inspecting a.txt",
        }
        nonstream = build_nonstream_response(parsed, "chat-100", 12345, model="deepseek-chat")
        self.assertEqual(nonstream["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(nonstream["choices"][0]["message"]["content"], "Inspecting a.txt")

        chunks = sse_tool_call_chunks("chat-100", 12345, parsed["tool_calls"], content="Inspecting a.txt")
        first = json.loads(chunks[0][len("data: "):])
        self.assertEqual(first["choices"][0]["delta"]["content"], "Inspecting a.txt")
        self.assertEqual(first["choices"][0]["delta"]["role"], "assistant")
        terminal = json.loads(chunks[-2][len("data: "):])
        self.assertEqual(terminal["choices"][0]["finish_reason"], "tool_calls")

    def test_server_models_endpoint(self):
        client = TestClient(deepseek_server.app)
        res = client.get("/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()["data"]
        ids = [m["id"] for m in data]
        self.assertIn("deepseek-chat", ids)
        self.assertIn("deepseek-r1", ids)



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
        parsed = parse_deepseek_response(raw, self.TOOLS)
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
        parsed = parse_deepseek_response(raw, self.TOOLS)
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
        parsed = parse_deepseek_response(raw, self.TOOLS)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "read")

if __name__ == "__main__":
    unittest.main()
