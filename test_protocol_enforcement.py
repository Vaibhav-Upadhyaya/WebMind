import json
import unittest

from protocol_recovery import (
    audit_spurious_refusal,
    build_refusal_correction_directive,
    enforce_and_recover,
    extract_narrated_shell_command,
)
from protocol import build_prompt, parse_chatgpt_response
from deepseek_protocol import build_deepseek_prompt, parse_deepseek_response
from kimi_protocol import build_kimi_prompt, parse_kimi_response
from glm_protocol import build_glm_prompt, parse_glm_response


class TestProtocolEnforcement(unittest.TestCase):
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run command in terminal.",
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

    def test_in_context_assistant_history_re_enveloping(self):
        messages = [
            {"role": "user", "content": "Hello agent"},
            {"role": "assistant", "content": "I am here to help you."},
            {"role": "user", "content": "Open my report in Word"},
        ]
        # Test protocol.py
        p_chatgpt, _ = build_prompt(messages, self.TOOLS)
        self.assertIn('"webmind_response": "message"', p_chatgpt)
        self.assertIn("I am here to help you.", p_chatgpt)

        # Test deepseek_protocol.py
        p_deepseek = build_deepseek_prompt(messages, self.TOOLS)
        self.assertIn('"webmind_response": "message"', p_deepseek)
        self.assertIn("I am here to help you.", p_deepseek)

        # Test kimi_protocol.py
        p_kimi = build_kimi_prompt(messages, self.TOOLS)
        self.assertIn('"webmind_response": "message"', p_kimi)

        # Test glm_protocol.py
        p_glm = build_glm_prompt(messages, self.TOOLS)
        self.assertIn('"webmind_response": "message"', p_glm)

    def test_terminal_contract_presence(self):
        messages = [{"role": "user", "content": "test prompt"}]
        p_chatgpt, _ = build_prompt(messages, self.TOOLS)
        self.assertIn("RESPONSE PROTOCOL: reply with YOUR ONE JSON envelope", p_chatgpt)

        p_deepseek = build_deepseek_prompt(messages, self.TOOLS)
        self.assertIn("[MANDATORY RESPONSE CONTRACT]", p_deepseek)

        p_kimi = build_kimi_prompt(messages, self.TOOLS)
        self.assertIn("[MANDATORY RESPONSE CONTRACT]", p_kimi)

        p_glm = build_glm_prompt(messages, self.TOOLS)
        self.assertIn("[MANDATORY RESPONSE CONTRACT]", p_glm)

    def test_spurious_refusal_audit(self):
        refusals = [
            "I don't have access to your local machine to open Word documents.",
            "As an AI, I cannot open files directly on your desktop.",
            "You can open it by running Start-Process winword in PowerShell.",
            "I do not have access to the local filesystem or tools.",
        ]
        for r in refusals:
            errs = audit_spurious_refusal(r, self.TOOLS)
            self.assertTrue(len(errs) > 0, f"Failed to detect refusal: {r}")

    def test_auto_promotion_narrated_shell_command(self):
        raw = (
            "You can open your document with this command:\n"
            "```powershell\n"
            'Start-Process winword "C:\\Users\\test\\doc.docx"\n'
            "```"
        )
        history = [{"role": "user", "content": "open my docx in Word"}]
        res = enforce_and_recover(
            raw,
            parse_fn=parse_deepseek_response,
            tools=self.TOOLS,
            history=history,
            provider_name="deepseek",
        )
        self.assertEqual(res["type"], "tool_call")
        self.assertEqual(len(res["tool_calls"]), 1)
        self.assertEqual(res["tool_calls"][0]["name"], "bash")
        self.assertIn("Start-Process winword", res["tool_calls"][0]["arguments"]["command"])

    def test_active_correction_for_refusal(self):
        refusal_raw = "I do not have access to your computer to run commands or open files."
        history = [{"role": "user", "content": "open the project file"}]

        corrected_raw = (
            '```json\n'
            '{"webmind_response": "tool_call", "tool_calls": [{"id": "call_1", "name": "bash", "arguments": {"command": "dir"}}]}\n'
            '```'
        )

        ask_called = []
        def mock_ask(directive: str) -> str:
            ask_called.append(directive)
            return corrected_raw

        res = enforce_and_recover(
            refusal_raw,
            parse_fn=parse_deepseek_response,
            ask_fn=mock_ask,
            tools=self.TOOLS,
            history=history,
            provider_name="deepseek",
        )
        self.assertTrue(len(ask_called) == 1)
        self.assertIn("PROTOCOL CORRECTION", ask_called[0])
        self.assertEqual(res["type"], "tool_call")
        self.assertEqual(res["tool_calls"][0]["name"], "bash")

    def test_clean_message_envelope_normalization(self):
        raw = "The capital of France is Paris."
        res = enforce_and_recover(
            raw,
            parse_fn=parse_deepseek_response,
            tools=self.TOOLS,
            provider_name="deepseek",
        )
        self.assertEqual(res["type"], "message")
        self.assertEqual(res["content"], "The capital of France is Paris.")

    def test_bidirectional_tool_aliases(self):
        write_file_tools = [
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write to disk.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]
        # DeepSeek calls "write" but tool is registered as "write_file"
        raw = (
            '```json\n'
            '{"webmind_response": "tool_call", "tool_calls": [{"id": "call_w", "name": "write", "arguments": {"path": "a.txt", "content": "hi"}}]}\n'
            '```'
        )
        parsed = parse_deepseek_response(raw, write_file_tools)
        self.assertEqual(parsed["type"], "tool_call")
        self.assertEqual(parsed["tool_calls"][0]["name"], "write_file")


if __name__ == "__main__":
    unittest.main()
