"""Endpoint-level tests for tool-call validation-failure recovery (server.py).

They stub the module-global browser worker so no real browser or network is
touched, then drive the real /v1/chat/completions endpoint through both the
streaming and non-streaming tool paths.  They verify the two wiring facts that
protocol-level tests cannot:

  1. a validation-failed envelope triggers ONE correction ask;
  2. the correction ask never touches the idempotency guard (no dedup_key,
     no image re-attach) and lives in the same ChatGPT chat.
"""

import json
import queue
import threading
import time
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

import server as server_mod

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


def _valid_arguments(name, id, arguments):
    return json.dumps(
        {
            "webmind_response": "tool_call",
            "tool_calls": [{"id": id, "name": name, "arguments": arguments}],
        }
    )


class FakeWebMind:
    """Minimal stand-in for the module-global worker: scripts .text and
    records every ask so tests can assert prompt, dedup_key, chat."""

    def __init__(self, texts):
        self._texts = list(texts)
        self.consumed = []
        self.asks = []
        self.stream_items = []
        self.stream_calls = []

    def ask(self, prompt, chat_url=None, image_paths=None,
            dedup_key=None, request_id=None, conversation_id=None,
            deadline=None):
        self.asks.append(
            {
                "prompt": prompt,
                "chat_url": chat_url,
                "image_paths": image_paths,
                "dedup_key": dedup_key,
                "conversation_id": conversation_id,
                "deadline": deadline,
            }
        )
        if not self._texts:
            raise RuntimeError("no scripted response left")
        text = self._texts.pop(0)
        self.consumed.append(text)
        # Mirror the real WebMindWorker ask handler: it records the one
        # browser round trip under the dedup_key, so a retry replays it.
        if dedup_key:
            server_mod._request_guard.finish(
                dedup_key, {"kind": "success", "content": text}
            )
        return SimpleNamespace(text=text)

    def start_stream(self, prompt, chat_url=None, image_paths=None,
                     dedup_key=None, request_id=None, conversation_id=None,
                     deadline=None):
        """Mirror the real WebMindWorker: enqueue (deadline, conversation)
        onto the task contract and hand back a delta queue pre-loaded with
        the scripted items plus the STREAM_DONE sentinel."""
        self.stream_calls.append(
            {
                "prompt": prompt,
                "chat_url": chat_url,
                "image_paths": image_paths,
                "dedup_key": dedup_key,
                "conversation_id": conversation_id,
                "deadline": deadline,
            }
        )
        q = queue.Queue()
        for item in self.stream_items:
            q.put(item)
        q.put(server_mod._STREAM_DONE)
        return q


def _payload(text, tools, stream=False):
    return {
        "model": "webmind",
        "messages": [{"role": "user", "content": text}],
        "stream": stream,
        "tools": tools,
    }


class ToolCallRecoveryEndpointTest(unittest.TestCase):
    """Recovery wired into /v1/chat/completions, both tool paths."""

    def setUp(self):
        self.fake = FakeWebMind([])
        # The endpoint and _correction_ask read this module-global worker.
        server_mod.webmind = self.fake
        self.client = TestClient(server_mod.app)
        # Fresh guard state: distinct per-request keys, but keep it empty.
        server_mod._request_guard._records.clear()

    def tearDown(self):
        server_mod._request_guard._records.clear()

    def test_nontream_recovery_emits_tool_call_and_guards_original(self):
        self.fake._texts = [\
            _valid_arguments("read", "call_r1", {"path": "index.html"}),\
            _valid_arguments("read", "call_r2", {"file_path": "index.html"}),\
        ]
        resp = self.client.post(
            "/v1/chat/completions",
            json=_payload("go through index.html", [FS_READ_TOOL], stream=False),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        tc = choice["message"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "read")
        self.assertEqual(
            json.loads(tc["function"]["arguments"]), {"file_path": "index.html"}
        )

        self.assertEqual(len(self.fake.asks), 2)
        first, second = self.fake.asks
        # Original ask: dedup-guarded, carries the browser chat.
        self.assertIsNotNone(first["dedup_key"])
        self.assertEqual(first["image_paths"], [])
        # Correction ask: same chat, but guard untouched and no re-attach.
        self.assertIsNone(second["dedup_key"])
        self.assertEqual(second["image_paths"], [])
        self.assertEqual(second["conversation_id"], first["conversation_id"])
        self.assertIn(
            "read: missing required property 'file_path'", second["prompt"]
        )
        self.assertIn("single allowed correction", second["prompt"])

        # The guard recorded only the ORIGINAL (first) response text, so a
        # client retry replays that validated-failed text (safe) and never
        # the correction ask.
        records = server_mod._request_guard._records
        rec = next(r for k, r in records.items() if k == first["dedup_key"])
        self.assertEqual(rec.response["content"], self.fake.consumed[0])

    def test_untreated_second_failure_returns_clean_error(self):
        self.fake._texts = [\
            _valid_arguments("read", "call_f1", {"path": "a.txt"}),\
            _valid_arguments("read", "call_f2", {"path": "b.txt"}),\
        ]
        resp = self.client.post(
            "/v1/chat/completions",
            json=_payload("read the file", [FS_READ_TOOL], stream=False),
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        choice = data["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertNotIn("tool_calls", choice["message"])
        self.assertNotIn("webmind_response", choice["message"]["content"])
        self.assertEqual(len(self.fake.asks), 2)

    def test_completed_tool_turn_does_not_suppress_next_tool_call(self):
        # Turn 1: a fresh request with tools returns a real tool call.
        self.fake._texts = [\
            _valid_arguments("read", "call_t1", {"file_path": "a.txt"}),\
            _valid_arguments("read", "call_t2", {"file_path": "b.txt"}),\
        ]
        resp1 = self.client.post(
            "/v1/chat/completions",
            json=_payload("read index.html", [FS_READ_TOOL], stream=False),
        )
        self.assertEqual(resp1.status_code, 200)
        tc1 = resp1.json()["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(tc1["id"], "call_t1")
        self.assertEqual(
            json.loads(tc1["function"]["arguments"]), {"file_path": "a.txt"}
        )

        # Turn 2: the harness echoes the assistant tool_calls and ships the
        # result as a user-role message with tool_call_id, then asks anew.
        # This is a DIFFERENT payload: it must reach the model as a fresh,
        # guarded ask, never be replayed from the turn-1 record.
        resp2 = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "webmind",
                "messages": [\
                    {"role": "user", "content": "read index.html"},\
                    {"role": "assistant", "content": None, "tool_calls": [tc1]},\
                    {"role": "user", "tool_call_id": tc1["id"], "name": "read",\
                     "content": json.dumps(\
                         {"exit_code": 0, "stdout": "<html>...</html>"})},\
                    {"role": "user", "content": "read style.css as well"},\
                ],
                "stream": False,
                "tools": [FS_READ_TOOL],
            },
        )
        self.assertEqual(len(self.fake.asks), 2)
        choice = resp2.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        tc2 = choice["message"]["tool_calls"][0]
        self.assertEqual(tc2["id"], "call_t2")
        self.assertEqual(
            json.loads(tc2["function"]["arguments"]), {"file_path": "b.txt"}
        )
        first, second = self.fake.asks
        self.assertIsNotNone(second["dedup_key"])
        self.assertEqual(second["conversation_id"], first["conversation_id"])
        self.assertIn("style.css", second["prompt"])
        self.assertIn("tool_call_id: call_t1", second["prompt"])

    def test_streaming_recovery_emits_tool_call_delta(self):
        self.fake._texts = [\
            _valid_arguments("read", "call_s1", {"path": "a.txt"}),\
            _valid_arguments("read", "call_s2", {"file_path": "a.txt"}),\
        ]
        resp = self.client.post(
            "/v1/chat/completions",
            json=_payload("read the file", [FS_READ_TOOL], stream=True),
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["content-type"])

        tool_calls = []
        final_reason = None
        for line in resp.text.splitlines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                continue
            payload = json.loads(data)
            for choice in payload["choices"]:
                delta = choice.get("delta") or {}
                if delta.get("tool_calls"):
                    tool_calls.append(delta["tool_calls"])
                if choice.get("finish_reason"):
                    final_reason = choice["finish_reason"]

        self.assertEqual(final_reason, "tool_calls")
        deltas = [c[0] for c in tool_calls if c]
        named = [c for c in deltas if c.get("function", {}).get("name")]
        self.assertEqual(named[0]["function"]["name"], "read")
        args_delta = next(\
            c for c in deltas if c.get("function", {}).get("arguments")\
        )
        self.assertEqual(
            json.loads(args_delta["function"]["arguments"]), {"file_path": "a.txt"}
        )
        self.assertEqual(len(self.fake.asks), 2)


class _RecordingQueue:
    """Captures whatever a WebMindWorker enqueues.  When ``unblock_asks`` is
    set, also resolves the ask result's done event so ``worker.ask`` returns
    without a live worker thread."""

    def __init__(self, unblock_asks=False):
        self._items = []
        self._unblock_asks = unblock_asks

    def put(self, item):
        self._items.append(item)
        if (\
            self._unblock_asks\
            and isinstance(item, tuple)\
            and item\
            and item[0] == "ask"\
            and isinstance(item[4], dict)\
        ):
            item[4]["done"].set()

    def qsize(self):
        return len(self._items)


class WorkerTaskContractTest(unittest.TestCase):
    """Regression for worker.thread_crashed 'not enough values to unpack
    (expected 9, got 8)': the worker consumer unpacks
    (kind, prompt, chat_url, image_paths, payload, dedup_key, request_id,
    conversation_id, deadline) but the ask()/start_stream() producers were
    still emitting eight fields.  These tests pin the producer contract to
    the consumer's unpack at the source (the task tuple)."""

    def _new_worker(self, recorder):
        worker = server_mod.WebMindWorker.__new__(server_mod.WebMindWorker)
        worker.tasks = recorder
        worker.running = True
        return worker

    def _unpack(self, task):
        (kind, prompt, chat_url, image_paths, payload,\
         dedup_key, request_id, conversation_id, deadline) = task
        return locals()

    def test_ask_task_is_nine_tuple_with_deadline(self):
        recorder = _RecordingQueue(unblock_asks=True)
        worker = self._new_worker(recorder)
        worker.ask(
            "hello",
            chat_url="u",
            image_paths=[],
            dedup_key="k",
            request_id="r",
            conversation_id="c",
            deadline=1234.5,
        )
        task = recorder._items[0]
        self.assertEqual(len(task), 9)
        fields = self._unpack(task)
        self.assertEqual(fields["kind"], "ask")
        self.assertEqual(fields["prompt"], "hello")
        self.assertEqual(fields["dedup_key"], "k")
        self.assertEqual(fields["conversation_id"], "c")
        self.assertEqual(fields["deadline"], 1234.5)

    def test_start_stream_task_is_nine_tuple_with_deadline(self):
        recorder = _RecordingQueue()
        worker = self._new_worker(recorder)
        delta_queue = worker.start_stream(
            "hi",
            chat_url="u",
            image_paths=[],
            dedup_key="k",
            request_id="r",
            conversation_id="c",
            deadline=99.0,
        )
        task = recorder._items[0]
        self.assertEqual(len(task), 9)
        fields = self._unpack(task)
        self.assertEqual(fields["kind"], "stream")
        self.assertIs(fields["payload"], delta_queue)
        self.assertEqual(fields["deadline"], 99.0)


class StreamDeadlineRegressionTest(unittest.TestCase):
    """Regression for the SSE generate() NameError: the closure read
    'deadline' but the per-request deadline was never created/put in scope.
    Now chat_completions creates it once and that same value threads into
    the worker task AND is captured by generate()."""

    def setUp(self):
        self.fake = FakeWebMind([])
        server_mod.webmind = self.fake
        server_mod._request_guard._records.clear()
        self._old_deadline = server_mod._REQUEST_DEADLINE_SECONDS

    def tearDown(self):
        server_mod._request_guard._records.clear()
        server_mod._REQUEST_DEADLINE_SECONDS = self._old_deadline

    def test_streaming_generate_captures_request_deadline(self):
        # A real deadline (not None) forces generate() to evaluate the
        # non-None branch where the NameError used to fire.
        server_mod._REQUEST_DEADLINE_SECONDS = 3600.0
        self.fake.stream_items = ["hello "]
        client = TestClient(server_mod.app)
        resp = client.post(
            "/v1/chat/completions",
            json=_payload("hi there", [], stream=True),
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.text
        self.assertNotIn("NameError", body)
        self.assertIn("hello", body)
        self.assertIn("data: [DONE]", body)
        self.assertEqual(len(self.fake.stream_calls), 1)
        # The worker task got the same absolute monotonic deadline the
        # generate() closure captured, never a guessed/default value.
        self.assertIsNotNone(self.fake.stream_calls[0]["deadline"])
        self.assertGreater(
            self.fake.stream_calls[0]["deadline"], time.monotonic() - 1
        )


class WorkerStreamDispatchTest(unittest.TestCase):
    """Regression for the third smoke-test failure: the worker's streaming
    dispatch called ``webmind.ask_stream``, but ``webmind`` is the module
    global WebMindWorker instance (which has no ask_stream), so every stream
    died with 'WebMindWorker' object has no attribute 'ask_stream'.  The
    dispatch must call the module-level browser ask_stream, exactly like the
    non-streaming branch calls the module-level ask."""

    def test_worker_stream_dispatch_uses_module_ask_stream(self):
        worker = server_mod.WebMindWorker.__new__(server_mod.WebMindWorker)
        worker.tasks = queue.Queue()
        worker.started = threading.Event()
        worker.startup_error = None
        worker.running = False
        worker.thread = None

        old_start = server_mod.start
        old_stop = server_mod.stop
        old_ask_stream = server_mod.ask_stream
        old_webmind = server_mod.webmind

        def fake_start():
            pass

        def fake_stop():
            pass

        def fake_ask_stream(prompt, chat_url=None, image_paths=None,
                            conversation_id=None, deadline=None):
            assert deadline is None  # no wall-clock cap in this control run
            yield "WORKER"

        class NoAskStream:
            pass

        server_mod.start = fake_start
        server_mod.stop = fake_stop
        server_mod.ask_stream = fake_ask_stream
        server_mod.webmind = NoAskStream()
        try:
            t = threading.Thread(target=worker._run, name="wm-dispatch-test", daemon=True)
            t.start()
            self.assertTrue(worker.started.wait(timeout=5))
            self.assertIsNone(worker.startup_error)

            delta_q = queue.Queue()
            worker.tasks.put(
                ("stream", "hi", None, [], delta_q, None, "rid", "conv", None)
            )
            got = []
            while True:
                item = delta_q.get(timeout=5)
                if item is server_mod._STREAM_DONE:
                    break
                got.append(item)
            self.assertEqual(got, ["WORKER"])

            worker.tasks.put(None)
            t.join(timeout=5)
            self.assertFalse(t.is_alive())
        finally:
            server_mod.start = old_start
            server_mod.stop = old_stop
            server_mod.ask_stream = old_ask_stream
            server_mod.webmind = old_webmind


class LegacyStreamAttachmentRetryTest(unittest.TestCase):
    """The no-tools STREAM path needs the same one-shot shrink.

    Streaming is harder than non-streaming: ``start_stream`` only enqueues, so
    the refusal arrives later as an Exception *inside* the delta queue.  A
    plain try/except around the call cannot see it.  The retry therefore has
    to happen inside the generator that drains the queue, which is safe only
    because the refusal is raised at submit time -- no delta has been yielded
    yet, so nothing the client saw can be duplicated.
    """

    class _ScriptedStream:
        def __init__(self, scripts):
            self.scripts = list(scripts)
            self.stream_calls = []
            self.asks = []

        def _record(self, prompt):
            self.stream_calls.append({"prompt": prompt})
            self.asks.append(prompt)

        def start_stream(self, prompt, chat_url=None, image_paths=None,
                         dedup_key=None, request_id=None,
                         conversation_id=None, deadline=None):
            self._record(prompt)
            q = queue.Queue()
            for item in self.scripts.pop(0):
                q.put(item)
            q.put(server_mod._STREAM_DONE)
            return q

        def ask(self, prompt, **kwargs):
            self._record(prompt)
            return SimpleNamespace(text="unused")

    def setUp(self):
        server_mod._request_guard._records.clear()
        self._old_webmind = server_mod.webmind

    def tearDown(self):
        server_mod.webmind = self._old_webmind
        server_mod._request_guard._records.clear()

    def _post(self, fake, text):
        server_mod.webmind = fake
        client = TestClient(server_mod.app)
        return client.post(
            "/v1/chat/completions",
            json=_payload(text, [], stream=True),
        )

    def test_stream_attachment_is_retried_shorter_and_still_answers(self):
        import main as main_mod

        fake = self._ScriptedStream(
            [\
                [main_mod.PromptTooLargeError("file", kind="attachment")],\
                ["the real summary"],\
            ]
        )
        resp = self._post(fake, "HEAD\n" + ("transcript line\n" * 4000))

        self.assertEqual(resp.status_code, 200)
        self.assertIn("the real summary", resp.text)
        self.assertNotIn("webmind_error", resp.text)
        self.assertIn("data: [DONE]", resp.text)
        self.assertEqual(len(fake.stream_calls), 2)
        self.assertLess(
            len(fake.stream_calls[1]["prompt"]),
            len(fake.stream_calls[0]["prompt"]),
        )
        self.assertTrue(fake.stream_calls[1]["prompt"].startswith("HEAD"))

    def test_stream_retry_happens_exactly_once_then_reports_the_error(self):
        import main as main_mod

        err = main_mod.PromptTooLargeError("file", kind="attachment")
        fake = self._ScriptedStream([[err], [err]])
        resp = self._post(fake, "HEAD\n" + ("transcript line\n" * 4000))

        self.assertEqual(resp.status_code, 200)
        self.assertIn("webmind_error", resp.text)
        self.assertIn("data: [DONE]", resp.text)
        self.assertEqual(len(fake.stream_calls), 2, "must not loop")

    def test_stream_over_ceiling_is_not_retried(self):
        import main as main_mod

        fake = self._ScriptedStream(
            [[main_mod.PromptTooLargeError("ceiling", kind="over_ceiling")]]
        )
        resp = self._post(fake, "short question")

        self.assertEqual(resp.status_code, 200)
        self.assertIn("webmind_error", resp.text)
        self.assertEqual(len(fake.stream_calls), 1, "must not re-dispatch")

    def test_a_normal_stream_is_dispatched_exactly_once(self):
        fake = self._ScriptedStream([["hello"]])
        resp = self._post(fake, "short question")

        self.assertEqual(resp.status_code, 200)
        self.assertIn("hello", resp.text)
        self.assertEqual(len(fake.stream_calls), 1)


if __name__ == "__main__":
    unittest.main()
