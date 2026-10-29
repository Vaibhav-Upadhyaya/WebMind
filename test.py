import base64
import json
import os
import sys

import requests


# ============================================================
# WEBMIND IMAGE TEST (stream=false + stream=true in one file)
# ============================================================
# Usage:
#   python test.py [path/to/image.png]
#
# Requires server.py running on http://127.0.0.1:8001

API = "http://127.0.0.1:8001/v1/chat/completions"
MODEL = "webmind"
TIMEOUT = 300

DEFAULT_IMAGE = r".\assets\sample_image.png"
PROMPT_TEXT = "What is in this image?"

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def load_image_data_url(path):
    ext = os.path.splitext(path)[1].lower()
    mime = _MIME_BY_EXT.get(ext)
    if mime is None:
        raise ValueError(
            f"Unsupported image type '{ext}': use png/jpeg/webp/gif."
        )
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def build_messages(data_url):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEXT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]


def _fail(response, context):
    # Self-diagnosing failure: show status + raw body head so a
    # non-JSON reply never surfaces as a bare JSONDecodeError.
    print(f"\nFAILED [{context}]: HTTP {response.status_code}")
    print("Content-Type:", response.headers.get("content-type"))
    print("Body head:", response.text[:500])
    raise SystemExit(1)


def test_non_stream(messages):
    print("\n=== [1/2] stream=false ===")

    response = requests.post(
        API,
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=TIMEOUT,
    )

    if response.status_code != 200:
        _fail(response, "stream=false")

    if "application/json" not in response.headers.get("content-type", ""):
        _fail(response, "stream=false (expected JSON)")

    answer = response.json()["choices"][0]["message"]["content"]
    print("Answer:", answer)
    return answer


def test_stream(messages):
    print("\n=== [2/2] stream=true ===")

    response = requests.post(
        API,
        json={"model": MODEL, "messages": messages, "stream": True},
        stream=True,
        timeout=TIMEOUT,
    )

    if response.status_code != 200:
        _fail(response, "stream=true")

    if "text/event-stream" not in response.headers.get("content-type", ""):
        _fail(response, "stream=true (expected SSE)")

    full = ""
    chunks = 0

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        line = line.strip()

        if not line.startswith("data:"):
            continue

        data = line[len("data:"):].strip()

        if data == "[DONE]":
            break

        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            print(f"\nFAILED [stream=true]: non-JSON SSE line: {data[:200]}")
            raise SystemExit(1)

        if "error" in obj:
            print(f"\nFAILED [stream=true]: server error: {obj['error']}")
            raise SystemExit(1)

        piece = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")

        if piece:
            full += piece
            chunks += 1
            print(piece, end="", flush=True)

    print(f"\n(chunks: {chunks})")
    print("Full answer:", full)
    return full


def main():
    # Windows consoles (cp1252) choke on characters like →.
    # Never let printing break the test run.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    print("Image:", image_path)

    try:
        data_url = load_image_data_url(image_path)
    except (OSError, ValueError) as e:
        print(f"\nFAILED [setup]: {e}")
        raise SystemExit(1)

    messages = build_messages(data_url)

    test_non_stream(messages)
    test_stream(messages)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
