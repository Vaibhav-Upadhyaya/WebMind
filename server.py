import asyncio
import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
import threading
import queue
import urllib.request
import uuid
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request
# pyrefly: ignore [missing-import]
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import config_manager

from main import (
    ChatGPTInputError,
    PromptTooLargeError,
    _cdp_available,
    RUNTIME_LOG_PATH as _RUNTIME_LOG_PATH,
    start,
    ask,
    ask_stream,
    stop,
)
from protocol import (
    _unescape_prose_newlines,
    LEGACY_PROMPT_MAX_CHARS,
    LEGACY_RETRY_MAX_CHARS,
    build_prompt,
    build_nonstream_response,
    cap_legacy_prompt,
    parse_chatgpt_response,
    recover_tool_call,
    sse_message_chunks,
    sse_tool_call_chunks,
)


# Sentinel marking end of a browser stream.
_STREAM_DONE = object()

_BROWSER_QUEUE_MAX = int(os.environ.get("WEBMIND_BROWSER_QUEUE_MAX", "8"))
_BROWSER_COOLDOWN_SECONDS = float(
    os.environ.get("WEBMIND_BROWSER_COOLDOWN_SECONDS", "2.0")
)


class BrowserQueueFullError(RuntimeError):
    """Raised when the serialized browser queue is saturated."""


class BrowserRateLimitError(RuntimeError):
    """Raised when ChatGPT explicitly rate-limits the browser request."""

# ============================================================
# PER-REQUEST WALL-CLOCK DEADLINE
# ============================================================
# The webmind worker runs main.ask / main.ask_stream in a single
# Playwright thread. Those already enforce their own internal wait
# deadlines, but if the browser wedges, main.py can still take up to
# its ~180-600s caps -- far too long for a single OpenAI request.
# Threading an absolute, per-request deadline here makes the HTTP
# layer the backstop: the SSE drain below stops emitting at the
# deadline and the non-streaming ask() raises a clean timeout, so a
# wedged browser can never hold a client open indefinitely.
#   WEBMIND_REQUEST_DEADLINE_SECONDS=180   # per-service override
#   0 or unset -> no wall-clock cap (previous behaviour preserved).
_REQUEST_DEADLINE_SECONDS = float(
    os.environ.get("WEBMIND_REQUEST_DEADLINE_SECONDS", "0")
)


def _request_deadline():
    """Absolute monotonic deadline for one request (or None if disabled)."""
    if _REQUEST_DEADLINE_SECONDS <= 0:
        return None
    return time.monotonic() + _REQUEST_DEADLINE_SECONDS


def _deadline_remaining(deadline):
    """Seconds left until an absolute monotonic deadline (>=0)."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


# ============================================================
# PHASE 0: DEBUG REQUEST LOGGING
# ============================================================
# Enabled with:  WEBMIND_DEBUG=1 python server.py
# Logs raw /v1/chat/completions requests (model, roles,
# contents, tools, tool_choice, tool_calls) to
# ./webmind_debug.log as one JSON object per line.
# Never logs headers or credentials (we only dump the
# already-parsed Pydantic request body).

DEBUG = os.getenv("WEBMIND_DEBUG", "0") == "1"

DEBUG_LOG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "webmind_debug.log"
)


def _preview(text: Any, limit: int = 500) -> str | None:
    if text is None:
        return None
    text = str(text)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


def _debug_log(obj: dict) -> None:
    if not DEBUG:
        return
    try:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **obj
        }
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Logging must never break request handling.
        pass


def _log_event(event: str, **kw) -> None:
    """Structured lifecycle log.  One JSON object per line to stdout AND to
    the runtime log file.

    Mirrors main._log_event on purpose.  main.py only knows what happened in
    the BROWSER (a paste became an attachment); every decision about whether
    to retry smaller, and whether that retry fired, is made here.  Logging to
    stdout alone meant webmind_runtime.log held thousands of browser.* records
    and not a single request.*/guard.* one, so the attachment-retry path was
    impossible to diagnose after the fact.
    """
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kw}
    line = json.dumps(record, default=str)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(_RUNTIME_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _log_chat_request(request: "ChatRequest") -> None:
    """Log incoming request shape for OpenCode tool-call diagnosis."""
    if not DEBUG:
        return

    messages_info = []
    for m in request.messages or []:
        content = getattr(m, "content", None)
        messages_info.append(
            {
                "role": getattr(m, "role", None),
                "content_preview": _preview(content),
                "content_len": len(str(content)) if content is not None else 0,
                "tool_calls": getattr(m, "tool_calls", None),
                "tool_call_id": getattr(m, "tool_call_id", None),
                "name": getattr(m, "name", None)
            }
        )

    roles = [m.get("role") for m in messages_info]

    _debug_log(
        {
            "endpoint": "/v1/chat/completions",
            "model": request.model,
            "stream": request.stream,
            "message_count": len(messages_info),
            "roles": roles,
            "has_tool_role": "tool" in roles,
            "tools": getattr(request, "tools", None),
            "tool_choice": getattr(request, "tool_choice", None),
            "messages": messages_info
        }
    )


# ============================================================
# WEBMIND BROWSER WORKER
# ============================================================

class WebMindWorker:
    """
    Keeps the Playwright Sync API inside one dedicated thread.

    This is important because FastAPI runs on asyncio, while
    main.py uses Playwright's synchronous API.

    All browser work (streaming and non-streaming) goes through
    the single `tasks` queue, so requests are serialized:

        Request A -> browser -> stream A
        Request B -> waits
        Request C -> waits

    Only one Playwright browser instance ever exists.
    """

    def __init__(self):
        self.tasks = queue.Queue(maxsize=max(1, _BROWSER_QUEUE_MAX))
        self.thread = None
        self.started = threading.Event()
        self.startup_error = None
        self.running = False
        self._next_dispatch_at = 0.0

    def _respect_cooldown(self):
        next_dispatch_at = getattr(self, "_next_dispatch_at", 0.0)
        wait = next_dispatch_at - time.monotonic()
        if wait > 0:
            _log_event("worker.cooldown", wait_ms=int(wait * 1000))
            time.sleep(wait)

    def _mark_dispatch_finished(self):
        self._next_dispatch_at = (
            time.monotonic() + max(0.0, _BROWSER_COOLDOWN_SECONDS)
        )

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        text = str(error).lower()
        return any(
            marker in text
            for marker in (
                "too many requests",
                "rate limit",
                "rate-limit",
                "429",
            )
        )

    def _run(self):
        try:
            print("\n" + "=" * 60)
            print("STARTING WEBMIND")
            print("=" * 60)

            # Start the Playwright browser inside this thread.
            start()

            self.running = True
            self.started.set()

            print("=" * 60)
            print("WEBMIND READY")
            print("=" * 60 + "\n")

            # Keep this thread alive and process requests.
            # One task at a time -> browser access is serialized.
            while self.running:
                try:
                    task = self.tasks.get(timeout=0.5)
                except queue.Empty:
                    continue

                if task is None:
                    break

                (
                    kind,
                    prompt,
                    chat_url,
                    image_paths,
                    payload,
                    dedup_key,
                    request_id,
                    conversation_id,
                    deadline,
                ) = task

                _log_event(
                    "worker.task_started",
                    request_id=request_id,
                    kind=kind,
                    dedup_key=dedup_key,
                    queue_depth=self.tasks.qsize() if hasattr(self.tasks, "qsize") else None,
                )

                if kind == "ask":
                    try:
                        self._respect_cooldown()
                        response = ask(
                            prompt,
                            chat_url=chat_url,
                            image_paths=image_paths,
                            conversation_id=conversation_id,
                            deadline=_deadline_remaining(deadline),
                        )

                        payload["response"] = response
                        payload["error"] = None
                        if conversation_id:
                            _record_turn_completed(conversation_id)

                        if dedup_key and response is not None and hasattr(response, "text"):
                            _request_guard.finish(
                                dedup_key,
                                {"kind": "success", "content": response.text}
                            )

                    except Exception as e:
                        if self._is_rate_limit_error(e):
                            e = BrowserRateLimitError(
                                "ChatGPT rate-limited the browser request; retry later."
                            )
                        payload["response"] = None
                        payload["error"] = e

                        if dedup_key:
                            _request_guard.fail(
                                dedup_key,
                                str(e),
                                retryable=_is_retryable_error(e),
                                status_code=_error_status(e),
                            )

                    finally:
                        self._mark_dispatch_finished()
                        payload["done"].set()
                        _log_event(
                            "worker.task_completed",
                            request_id=request_id,
                            kind=kind,
                            error=type(payload["error"]).__name__ if payload["error"] else None,
                        )

                elif kind == "stream":
                    delta_queue = payload

                    stream_deltas: list[str] = []
                    stream_error: Exception | None = None

                    deadline_remaining = _deadline_remaining(deadline)

                    try:
                        self._respect_cooldown()
                        for delta in ask_stream(
                            prompt,
                            chat_url=chat_url,
                            image_paths=image_paths,
                            conversation_id=conversation_id,
                            deadline=deadline_remaining,
                        ):
                            if delta:
                                delta_queue.put(delta)
                                stream_deltas.append(delta)

                    except Exception as e:
                        # Forward worker errors to the SSE consumer.
                        if self._is_rate_limit_error(e):
                            e = BrowserRateLimitError(
                                "ChatGPT rate-limited the browser request; retry later."
                            )
                        delta_queue.put(e)
                        stream_error = e

                    finally:
                        self._mark_dispatch_finished()
                        delta_queue.put(_STREAM_DONE)
                        if stream_error is None and conversation_id:
                            _record_turn_completed(conversation_id)

                        if dedup_key:
                            if stream_error is not None:
                                _request_guard.fail(
                                    dedup_key,
                                    str(stream_error),
                                    retryable=_is_retryable_error(stream_error),
                                    status_code=_error_status(stream_error),
                                )
                            else:
                                _request_guard.finish(
                                    dedup_key,
                                    {"kind": "success", "content": "".join(stream_deltas)}
                                )

                        _log_event(
                            "worker.task_completed",
                            request_id=request_id,
                            kind=kind,
                            deltas=len(stream_deltas),
                            total_chars=sum(len(d) for d in stream_deltas),
                            error=type(stream_error).__name__ if stream_error else None,
                        )

        except Exception as e:
            self.startup_error = e
            self.started.set()
            _log_event("worker.thread_crashed", error=str(e), error_type=type(e).__name__)

        finally:
            try:
                stop()
            except Exception:
                pass

            self.running = False
            _log_event("worker.thread_stopped")

    def start(self):
        self.thread = threading.Thread(
            target=self._run,
            name="WebMind-Playwright",
            daemon=True
        )

        self.thread.start()

        # Wait until Playwright has started.
        self.started.wait()

        if self.startup_error:
            raise self.startup_error

    def ask(self, prompt, chat_url=None, image_paths=None, dedup_key=None, request_id=None, conversation_id=None, deadline=None):
        """Blocking non-streaming request (called via asyncio.to_thread)."""
        if not self.running:
            raise RuntimeError("WebMind browser is not running.")

        result = {
            "response": None,
            "error": None,
            "done": threading.Event()
        }

        _log_event(
            "request.queued",
            request_id=request_id,
            kind="ask",
            dedup_key=dedup_key,
        )

        task = (
            "ask",
            prompt,
            chat_url,
            list(image_paths) if image_paths else [],
            result,
            dedup_key,
            request_id,
            conversation_id,
            deadline,
        )
        try:
            enqueue = getattr(self.tasks, "put_nowait", self.tasks.put)
            enqueue(task)
        except queue.Full as e:
            raise BrowserQueueFullError(
                "WebMind browser queue is full; retry after a short delay."
            ) from e

        _log_event("request.waiting_for_worker", request_id=request_id, kind="ask")

        # Wait until the worker finishes the request.
        result["done"].wait()

        _log_event(
            "request.worker_completed",
            request_id=request_id,
            kind="ask",
            has_error=result["error"] is not None,
        )

        if result["error"]:
            raise result["error"]

        return result["response"]

    def start_stream(self, prompt, chat_url=None, image_paths=None, dedup_key=None, request_id=None, conversation_id=None, deadline=None):
        """
        Enqueue a streaming request and return the delta queue.

        The worker thread pushes text deltas (str), then _STREAM_DONE.
        On failure it pushes an Exception instance, then _STREAM_DONE.
        While this stream occupies the worker, other tasks wait
        in `self.tasks`, preserving single-browser serialization.
        """
        if not self.running:
            raise RuntimeError("WebMind browser is not running.")

        delta_queue: queue.Queue = queue.Queue()

        _log_event(
            "request.queued",
            request_id=request_id,
            kind="stream",
            dedup_key=dedup_key,
        )

        task = (
            "stream",
            prompt,
            chat_url,
            list(image_paths) if image_paths else [],
            delta_queue,
            dedup_key,
            request_id,
            conversation_id,
            deadline,
        )
        try:
            enqueue = getattr(self.tasks, "put_nowait", self.tasks.put)
            enqueue(task)
        except queue.Full as e:
            raise BrowserQueueFullError(
                "WebMind browser queue is full; retry after a short delay."
            ) from e

        return delta_queue

    def stop(self):
        if not self.running:
            return

        self.running = False

        # Tell worker to stop.
        self.tasks.put(None)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)


# ============================================================
# GLOBAL WEBMIND INSTANCE
# ============================================================

webmind = WebMindWorker()


# ============================================================
# REQUEST MODELS
# ============================================================

class Message(BaseModel):
    role: str
    # OpenAI multimodal: either a plain string or a list of
    # content parts like {"type": "text", "text": ...} /
    # {"type": "image_url", "image_url": {"url": ...}}.
    content: str | list[Any] | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # CoT passback on assistant history messages (harness re-sends the
    # prior turn's reasoning). Kept so build_prompt can serialize it.
    reasoning_content: str | None = None

    # Ignore unknown future fields instead of 422-ing.
    model_config = {"extra": "ignore"}


class ChatRequest(BaseModel):
    model: str = "webmind"
    messages: list[Message]
    stream: bool = False

    # Accepted for OpenAI compatibility.
    # WebMind itself does not currently use these.
    temperature: float | None = None
    max_tokens: int | None = None

    # Phase 1: preserved (not yet acted on) so OpenCode
    # tool payloads survive validation instead of being dropped.
    tools: list[Any] | None = None
    tool_choice: Any | None = None

    # WebMind-specific extension.
    chat_url: str | None = None

    # OpenAI request.id (for idempotent retries).
    request_id: str | None = None

    # OpenAI user identifier (used for session tracking if provided)
    user: str | None = None

    model_config = {"extra": "ignore"}


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Start WebMind in its own thread.
    webmind.start()

    yield

    # Shut it down when FastAPI exits.
    webmind.stop()


app = FastAPI(
    title="WebMind",
    description="OpenAI-compatible API powered by a ChatGPT browser session.",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    if request.url.path not in ("/", "/docs", "/openapi.json"):
        auth_header = request.headers.get("authorization")
        if not config_manager.is_api_key_valid(auth_header):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid WebMind API Key. Provide a valid Bearer token.",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )
    return await call_next(request)


# ============================================================
# OPENAI-COMPATIBLE ERROR ENVELOPE
# ============================================================
# The harness (dsh-llm-deepseek chat-completions adapter) reads failure
# bodies as {"error": {"message", "type", "code"}} and surfaces
# error.message verbatim. FastAPI's default {"detail": ...} body would
# discard the real message, so every HTTP/validation failure renders in
# the OpenAI error shape instead. Status -> OpenAI vocabulary mirrors
# the harness classifier (AUTH / RATE_LIMIT / INVALID_REQUEST / SERVER).

def _openai_error_payload(status: int, message: str) -> dict:
    if status == 401:
        error_type = "authentication_error"
    elif status == 403:
        error_type = "permission_error"
    elif status == 404:
        error_type = "not_found_error"
    elif status == 429:
        error_type = "rate_limit_error"
    elif status >= 500:
        error_type = "api_error"
    else:
        error_type = "invalid_request_error"
    return {
        "error": {
            "message": message,
            "type": error_type,
            "code": error_type,
        }
    }


@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, list):
        # FastAPI validation inserts a list of per-field errors; flatten
        # them into one message string so error.message stays a string.
        messages = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        ]
        message = "; ".join(messages)
    else:
        message = str(detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_openai_error_payload(exc.status_code, message),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        message = str(exc.errors())
    except Exception:
        message = "Request body validation failed."
    return JSONResponse(
        status_code=422,
        content=_openai_error_payload(422, message),
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    # webmind.running is a cached flag: it may be True while the
    # dedicated Chrome process has crashed and CDP is dead.  A real
    # probe is a short HTTP GET to the CDP /json/version endpoint.
    try:
        cdp_reachable = _cdp_available(timeout=1.0)
    except Exception:
        cdp_reachable = False
    browser_running = webmind.running and cdp_reachable
    return {
        "status": "ok",
        "service": "webmind",
        "browser_running": browser_running,
        "worker_started": webmind.running,
    }


# ============================================================
# OPENAI MODELS ENDPOINT
# ============================================================

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "webmind",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "webmind"
            }
        ]
    }


@app.get("/v1/models/{model}")
def get_model(model: str):

    if model != "webmind":
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found."
        )

    return {
        "id": "webmind",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "webmind"
    }


# ============================================================
# CHAT COMPLETIONS
# ============================================================

# ============================================================
# MULTIMODAL IMAGE INPUT (v1, chat completions only)
# ============================================================
# Accepts OpenAI-style content parts. Text behavior is unchanged.
# Images are materialized to temp files here (FastAPI side), then
# attached inside the Playwright worker thread via file upload.
# Temp paths are never exposed to the client and are always
# cleaned up after the request (success or failure).

DEFAULT_IMAGE_PROMPT = "Please analyze the attached image."
MAX_IMAGES = 4
MAX_IMAGE_BYTES = 10 * 1024 * 1024
IMAGE_DOWNLOAD_TIMEOUT = 15

_ALLOWED_IMAGE_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}

_DATA_URL_RE = re.compile(
    r"^data:(image/(?:png|jpeg|webp|gif));base64,(.+)$",
    re.IGNORECASE | re.DOTALL
)


def _sniff_image_ext(data: bytes, declared_mime: str | None = None) -> str:
    """Validate magic bytes (stdlib only) and return file suffix."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data.startswith(b"RIFF") and len(data) > 12 and data[8:12] == b"WEBP":
        return "webp"
    raise HTTPException(
        status_code=400,
        detail="Unsupported image type: file contents are not PNG/JPEG/WebP/GIF."
    )


def _write_temp_image(data: bytes, ext: str) -> str:
    fd, path = tempfile.mkstemp(prefix="webmind_", suffix=f".{ext}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise
    return path


def _data_url_to_temp(url: str) -> str:
    match = _DATA_URL_RE.match(url.strip())
    if not match:
        raise HTTPException(
            status_code=400,
            detail="Invalid data URL: expected data:image/png|jpeg|webp|gif;base64,..."
        )
    mime = match.group(1).lower()
    if mime not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {mime}."
        )
    b64_payload = match.group(2).strip()
    try:
        data = base64.b64decode(b64_payload, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid base64 in image data URL."
        )
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Invalid base64: image payload is empty."
        )
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Image too large: max 10MB."
        )
    ext = _sniff_image_ext(data, mime)
    return _write_temp_image(data, ext)


def _remote_url_to_temp(url: str) -> str:
    if not (url.startswith("http://") or url.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail="Invalid image URL: only http(s) URLs or data URLs are supported."
        )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "WebMind/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=IMAGE_DOWNLOAD_TIMEOUT) as resp:
            content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type and content_type not in _ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported image type: {content_type}."
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_IMAGE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail="Image too large: max 10MB."
                    )
                chunks.append(chunk)
            data = b"".join(chunks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed image download: {str(e)}"
        )
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Failed image download: empty response."
        )
    ext = _sniff_image_ext(data, content_type or None)
    return _write_temp_image(data, ext)


def _prepare_image_files(urls: list[str]) -> list[str]:
    """Materialize image URLs to temp files. Cleans up on failure."""
    if len(urls) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images: max {MAX_IMAGES} per request."
        )
    paths: list[str] = []
    try:
        for url in urls:
            if not isinstance(url, str) or not url.strip():
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image URL: URL must be a non-empty string."
                )
            url = url.strip()
            if url.startswith("data:"):
                paths.append(_data_url_to_temp(url))
            else:
                paths.append(_remote_url_to_temp(url))
    except Exception:
        _cleanup_files(paths)
        raise
    return paths


def _cleanup_files(paths: list[str] | None) -> None:
    if not paths:
        return
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass


def _extract_prompt_and_images(request: ChatRequest) -> tuple[str, list[str]]:
    """Validate messages; return (text prompt, image URLs).

    Preserves legacy string-content behavior. Also accepts
    OpenAI-style list content parts.
    """

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="messages cannot be empty."
        )

    user_messages = [
        message
        for message in request.messages
        if message.role == "user"
        and message.content is not None
    ]

    if not user_messages:
        raise HTTPException(
            status_code=400,
            detail="No user message found."
        )

    last = user_messages[-1].content

    texts: list[str] = []
    image_urls: list[str] = []

    if isinstance(last, str):
        if last.strip():
            texts.append(last)
    elif isinstance(last, list):
        for part in last:
            if not isinstance(part, dict):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported content part: each part must be an object."
                )
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text", "")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
            elif part_type == "image_url":
                image_obj = part.get("image_url")
                url = image_obj.get("url") if isinstance(image_obj, dict) else None
                if not isinstance(url, str) or not url.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image URL: image_url.url must be a non-empty string."
                    )
                image_urls.append(url.strip())
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported content part: {part_type}."
                )
    else:
        raise HTTPException(
            status_code=400,
            detail="No user message found."
        )

    prompt = "\n".join(texts).strip()

    if not prompt and not image_urls:
        raise HTTPException(
            status_code=400,
            detail="User message cannot be empty."
        )

    if not prompt and image_urls:
        prompt = DEFAULT_IMAGE_PROMPT

    # This path has no conversation serializer to budget against, so an
    # oversized single message would reach the browser and be rejected by
    # its hard backstop -- which for opencode's auto-compaction request means
    # compaction can never succeed.  Cap it here instead, keeping the head.
    capped = cap_legacy_prompt(prompt)
    if capped != prompt:
        _log_event(
            "request.legacy_prompt_capped",
            before=len(prompt),
            after=len(capped),
            limit=LEGACY_PROMPT_MAX_CHARS,
            reason="no-tools path has no size budget; capped so the browser "
                   "backstop does not reject the request outright",
        )
        prompt = capped

    return prompt, image_urls


def _extract_prompt(request: ChatRequest) -> str:
    """Legacy helper: validate messages and return latest text prompt."""
    prompt, _ = _extract_prompt_and_images(request)
    return prompt


def _has_tools(request: ChatRequest) -> bool:
    """True when the request supplies tool definitions (tool-call mode)."""
    return bool(request.tools)


_session_lock = threading.Lock()
_active_session_id: str | None = None
_active_opener_hash: str | None = None
_session_generation: int = 0
_session_turns_completed: dict[str, int] = {}


def _is_persistent_chat_enabled() -> bool:
    env = os.environ.get("WEBMIND_PERSISTENT_CHAT")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    try:
        return bool(config_manager.load_config().get("persistent_chat", True))
    except Exception:
        return True


def _record_turn_completed(conversation_id: str | None) -> None:
    if not conversation_id:
        return
    with _session_lock:
        _session_turns_completed[conversation_id] = (
            _session_turns_completed.get(conversation_id, 0) + 1
        )


def _conversation_id(
    request: "ChatRequest", raw_request: Request | None = None
) -> str | None:
    """Stable identity of the client conversation a request belongs to.

    Derives identity with session isolation:
    1. Explicit reset headers (e.g. x-reset-chat, x-new-chat) advance generation.
    2. Explicit headers (e.g. x-session-id, session-id, x-conversation-id, task-id)
    3. Explicit request.user from OpenAI clients
    4. Persistent chat mode keeps active session alive across turns, shielding auxiliary
       requests (e.g. title generation).
    5. Legacy isolation mode advances generation when prior turn completed or opener differs.
    """
    # 0. Explicit reset headers from client
    headers = getattr(raw_request, "headers", None) if raw_request else None
    is_explicit_reset = False
    if headers:
        for r_hdr in ("x-reset-chat", "x-new-chat", "reset-chat", "new-chat"):
            v = headers.get(r_hdr)
            if v and v.strip().lower() in ("1", "true", "yes"):
                is_explicit_reset = True
                break

    # 1. Explicit headers from client
    if headers and not is_explicit_reset:
        for header_name in (
            "x-session-id",
            "session-id",
            "x-conversation-id",
            "conversation-id",
            "x-task-id",
            "task-id",
            "x-thread-id",
            "thread-id",
            "x-client-session-id",
        ):
            val = headers.get(header_name)
            if val and val.strip():
                digest = hashlib.sha256(val.strip().encode("utf-8")).hexdigest()
                return f"conv:sess_{digest[:20]}"

    # 2. Explicit user parameter from OpenAI request
    user_val = getattr(request, "user", None)
    if (
        user_val
        and isinstance(user_val, str)
        and user_val.strip()
        and user_val.strip().lower() not in ("user", "none", "default")
        and not is_explicit_reset
    ):
        digest = hashlib.sha256(user_val.strip().encode("utf-8")).hexdigest()
        return f"conv:user_{digest[:20]}"

    # 3. Message-based session derivation
    messages = request.messages or []

    first_user = next(
        (
            m
            for m in messages
            if getattr(m, "role", None) == "user"
            and getattr(m, "content", None) is not None
        ),
        None,
    )
    if first_user is None:
        return None

    first_user_text = str(first_user.content or "")
    opener_hash = hashlib.sha256(
        first_user_text.encode("utf-8")
    ).hexdigest()[:16]

    is_auxiliary = bool(
        re.search(r"session title|generate.*title|summarize.*conversation", first_user_text, re.IGNORECASE)
    )

    is_continuation = any(
        (
            getattr(m, "role", None) in ("assistant", "tool")
            or getattr(m, "tool_calls", None)
        )
        for m in messages
    )

    persistent = _is_persistent_chat_enabled()

    global _active_session_id, _active_opener_hash, _session_generation
    with _session_lock:
        if is_explicit_reset:
            _session_generation += 1
            _active_session_id = f"conv:{opener_hash}_{_session_generation}"
            _active_opener_hash = opener_hash
            _session_turns_completed[_active_session_id] = 0
            return _active_session_id

        if persistent:
            if is_auxiliary and _active_session_id:
                return _active_session_id
            if _active_session_id is not None:
                return _active_session_id
            _active_session_id = f"conv:{opener_hash}_{_session_generation}"
            _active_opener_hash = opener_hash
            if _active_session_id not in _session_turns_completed:
                _session_turns_completed[_active_session_id] = 0
            return _active_session_id

        if is_continuation and _active_session_id and _active_opener_hash == opener_hash:
            return _active_session_id

        if is_continuation and _active_session_id:
            # Opener might have been trimmed by client-side sliding window, but it's
            # still continuing the active session.
            return _active_session_id

        if is_continuation:
            return f"conv:{opener_hash}_{_session_generation}"

        completed_turns = _session_turns_completed.get(_active_session_id or "", 0)
        if _active_session_id is not None and (
            completed_turns > 0 or _active_opener_hash != opener_hash
        ):
            _session_generation += 1

        _active_session_id = f"conv:{opener_hash}_{_session_generation}"
        _active_opener_hash = opener_hash
        if _active_session_id not in _session_turns_completed:
            _session_turns_completed[_active_session_id] = 0
        return _active_session_id


def _sse_chunk(chat_id: int | str, created: int, delta: dict, finish_reason=None, usage=None) -> str:
    """Format one OpenAI-compatible SSE chat.completion.chunk line."""
    payload = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "webmind",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }
        ]
    }

    if usage is not None:
        payload["usage"] = usage

    return f"data: {json.dumps(payload)}\n\n"


# ============================================================
# REQUEST IDEMPOTENCY GUARD
# ============================================================
# OpenCode (and other OpenAI-compatible clients) retry a request
# when their own HTTP timeout elapses before WebMind responds.
# Because the ChatGPT reasoning model can take 200-600s per turn,
# a retry would otherwise be pasted into ChatGPT as a whole new
# user message, duplicating the request in the conversation.
#
# This guard recognizes retries and short-circuits them so no
# second ChatGPT message is created. It is a bounded, expiring,
# thread-safe in-memory map:
#
#   * Key: explicit idempotency key (request.request_id or an
#     Idempotency-Key header) when provided, else a SHA-256 over
#     the canonical request payload.
#   * TTL: DEDUP_TTL_SECONDS; entries expire so memory cannot
#     grow indefinitely.
#   * Cap: DEDUP_MAX_ENTRIES (FIFO eviction of oldest when full).
#   * Concurrency: all mutations happen under one lock; the
#     browser worker queue is fully serialized, and each guard
#     mutation is a critical section, so concurrent HTTP handlers
#     are safe.
#
# Retry semantics:
#   * A duplicate whose original is STILL in flight (start() sees an
#     existing, not-yet-done record) is coalesced: it waits on the
#     original's completion event and then replays that response,
#     so the identical user message is pasted into ChatGPT exactly
#     once.
#   * A duplicate whose original already COMPLETED replays the
#     cached response immediately.
#   * A failed original marks the record as failed.  Retries of a
#     FATAL failure (retryable=False, e.g. the browser died after
#     the prompt was already pasted) receive the same error instead
#     of re-pasting, which would duplicate the request.
#   * A RETRYABLE failure (retryable=True, e.g.
#     ChatGPTInputError: the prompt was never sent) is a transient
#     flake: nothing reached ChatGPT, so a retry may safely re-run.
#     The handler drops the poisoned record and lets the retry
#     re-register as a fresh request.

class _IdempotencyRecord:
    __slots__ = ("key", "created", "event", "response", "error", "status_code", "retryable", "rerun_count")

    def __init__(self, key):
        self.key = key
        self.created = time.monotonic()
        self.event = threading.Event()
        self.response = None
        self.error = None
        self.status_code = 500
        self.retryable = False
        self.rerun_count = 0


# --------------------------------------------------------
# Retry safety budget
# --------------------------------------------------------
# A RETRYABLE failure means nothing reached ChatGPT (the prompt was
# never pasted), so an identical retry may safely re-run as a fresh
# request.  But that re-run must not loop forever: every re-run
# re-sends the prompt, and a value that keeps failing to paste could
# be bounced forever by a stubborn client.  Once a single dedup key
# has been rearmed MAX_RETRYABLE_RERUNS times the guard refuses to
# re-register it and the handler returns a definite 500 instead.
MAX_RETRYABLE_RERUNS = 3


DEDUP_TTL_SECONDS = 30 * 60         # keep completed responses 30 min
DEDUP_MAX_ENTRIES = 512             # hard cap on map size
DEDUP_WAIT_TIMEOUT = 680            # max seconds to wait on an in-flight original
                                    # (just above main.py's 600s hard cap)


# A single dedup key may be REARMED (retryable failure -> re-run as a
# fresh request) at most this many times.  Each rearm keeps the SAME key
# so the re-run is still coalesced with any further retries, but pasting
# the prompt again is not free: a stubborn client looping on a
# permanently-flaky pre-send failure must eventually get a DEFINITE 500
# instead of an unbounded paste loop.
DEDUP_MAX_REARMS = 3

# A single key may re-run a RETRYABLE failure at most this many times
# (beyond the first registration).  Every re-run pastes the prompt a
# fresh time, so this caps the worst-case extra pastes even when a
# client stubbornly retries a permanently-flaky pre-send failure.
# Once exhausted, further retries raise a definite 500 instead of
# looping (see IdempotencyGuard.rearm).
MAX_RETRYABLE_RERUNS = 3


class IdempotencyGuard:
    """Bounded, expiring, thread-safe retry detector/coalescer."""

    def __init__(self, ttl=DEDUP_TTL_SECONDS, max_entries=DEDUP_MAX_ENTRIES):
        self._ttl = ttl
        self._max = max_entries
        self._lock = threading.Lock()
        self._records = OrderedDict()

    # --------------------------------------------------------
    # Key construction
    # --------------------------------------------------------

    @staticmethod
    def _idempotency_headers(headers: Any) -> str | None:
        """Pull a client-supplied idempotency key from headers."""
        for name in ("Idempotency-Key", "X-Idempotency-Key"):
            value = headers.get(name)
            if value and str(value).strip():
                return str(value).strip()
        return None

    def _request_id(self, request: "ChatRequest") -> str | None:
        rid = getattr(request, "request_id", None)
        if rid and str(rid).strip():
            return str(rid).strip()
        return None

    def _key_for(self, request: "ChatRequest", headers: Any) -> str:
        """Build a stable dedup key.

        Priority:
          1. request.request_id (OpenAI request.id) if provided
          2. Idempotency-Key header if provided
          3. SHA-256 of the canonical request payload
        """
        rid = self._request_id(request)
        if rid:
            return f"request:{rid}"

        idem = self._idempotency_headers(headers)
        if idem:
            return f"idempotency:{idem}"

        payload = {
            "model": request.model,
            "stream": request.stream,
            "messages": [m.model_dump(mode="json") for m in request.messages] if request.messages else [],
            "tools": request.tools,
            "tool_choice": request.tool_choice,
            "chat_url": request.chat_url,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"hash:{digest}"

    # --------------------------------------------------------
    # Eviction / housekeeping
    # --------------------------------------------------------

    def _evict_expired(self):
        now = time.monotonic()
        expired = [k for k, rec in self._records.items()
                   if now - rec.created > self._ttl]
        for k in expired:
            del self._records[k]

    def _evict_if_full(self):
        # Drop oldest entries (FIFO) until under the cap.
        while len(self._records) > self._max:
            self._records.popitem(last=False)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def start(self, request: "ChatRequest", headers: Any):
        """Begin tracking a request.

        Returns (key, record, is_new):
          * is_new=True  -> caller owns the request and must eventually
                            call finish()/fail().
          * is_new=False -> a request with this key already exists. The
                            caller should wait on record.event and then
                            replay record.response (or raise record.error).
        """
        key = self._key_for(request, headers)

        with self._lock:
            self._evict_expired()
            rec = self._records.get(key)
            if rec is not None:
                _log_event(
                    "guard.result",
                    dedup_key=key,
                    result="IN_FLIGHT" if not rec.event.is_set() else
                          ("FAILED" if rec.error is not None else "COMPLETED"),
                )
                return key, rec, False

            rec = _IdempotencyRecord(key)
            self._records[key] = rec
            self._evict_if_full()
            _log_event("guard.result", dedup_key=key, result="NEW")
            return key, rec, True

    def finish(self, key: str, response: Any) -> None:
        with self._lock:
            rec = self._records.get(key)
            if rec is not None:
                rec.response = response
                rec.error = None
        if rec is not None:
            rec.event.set()
            _log_event("guard.finish", dedup_key=key)

    def fail(self, key: str, error: Any, retryable: bool = False, status_code: int | None = None) -> None:
        """Mark a request as failed.

        retryable=True means nothing reached ChatGPT (e.g. the prompt
        was never sent) so a duplicate may safely re-run as a fresh
        request instead of replaying this error.
        """
        with self._lock:
            rec = self._records.get(key)
            if rec is not None:
                rec.response = None
                rec.error = error
                rec.status_code = status_code or _error_status(error)
                rec.retryable = retryable
        if rec is not None:
            rec.event.set()
            _log_event("guard.fail", dedup_key=key, retryable=retryable, error=str(error))

    def remove(self, key: str) -> bool:
        """Drop a record so a retry can be treated as a fresh request.

        Used after a retryably-failed original: the failure was a
        transient pre-send flake, and the duplicate should re-attempt
        rather than inherit the poisoned error.  Returns True if a
        record was actually removed.
        """
        with self._lock:
            removed = self._records.pop(key, None)
        _log_event("guard.remove", dedup_key=key, removed=removed is not None)
        return removed is not None

    def rearm(self, key: str) -> bool:
        """Reset a retryably-failed record so it may run as a fresh request.

        Unlike remove(), rearm keeps the SAME dedup key and record so the
        re-run remains coalesced with any further duplicate (a second
        retry still waits for THIS re-run instead of pasting twice).  It
        is also BOUNDED: each key may be rearmed only MAX_RETRYABLE_RERUNS
        times before the guard refuses, turning an endless retry loop into
        a definite 500 (nothing more is ever pasted once the budget is
        spent).

        Returns True when the record was successfully rearmed (caller
        owns the fresh run); False when the key had no record or its
        rearm budget is exhausted (caller MUST fail the retry, never
        re-run).
        """
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return False
            if rec.rerun_count >= MAX_RETRYABLE_RERUNS:
                return False
            rec.rerun_count += 1
            rec.response = None
            rec.error = None
            rec.retryable = False
            rec.event = threading.Event()
        _log_event("guard.rearm", dedup_key=key, reruns=rec.rerun_count)
        return True


def _key_label(key: str) -> str:
    if key.startswith("request:"):
        return f"request:{key[len('request:'):]}"
    if key.startswith("idempotency:"):
        return f"idem:{key[len('idempotency:'):]}"
    return "hash:" + key[len("hash:"):][:12]


def _is_retryable_error(exc: Exception) -> bool:
    """True when a failure never reached ChatGPT, so a retry is safe.

    ChatGPTInputError is raised ONLY when the prompt was not (or not
    verified to have been) sent — nothing was pasted, so an identical
    retry cannot duplicate a message. Everything else (browser died,
    response wait timed out, image attach issues) may have already
    pasted/submitted, so retries must NOT re-run — they replay the
    error instead.
    """
    if isinstance(exc, ChatGPTInputError):
        return True
    # A too-large prompt is NOT retryable: an identical retry re-pastes an
    # identical prompt and gets converted to an attachment again.  Marking it
    # retryable would make the idempotency guard loop on a condition only the
    # caller can fix by sending less.
    if isinstance(exc, PromptTooLargeError):
        return False
    # Fatal browser/connection failures that happened while the request
    # was being processed are NOT retryable.
    return False


def _error_status(exc: Exception) -> int:
    """Map browser/upstream failures to an HTTP status.

    The attachment case deliberately does NOT answer 413.  413 ("payload
    too large") is the cue an OpenAI-compatible client uses to decide its
    context no longer fits, so it compacts and re-sends.  That re-send is
    itself still large enough for ChatGPT to convert the paste into a file,
    so the client compacts again: a 413 -> compact -> 413 loop that burns
    turns without ever advancing the work.

    By the time an attachment refusal reaches here the one-shot smaller retry
    has already been spent (see _dispatch_with_attachment_retry), so 413
    would not be actionable anyway -- the client has no smaller request to
    send, because the retry already was that.  Reporting it as a server-side
    failure instead keeps the answer honest (the request really did fail)
    while withholding the compaction trigger.

    A pre-flight "over_ceiling" refusal still answers 413: nothing was
    pasted, the prompt is genuinely larger than the configured backstop, and
    shrinking is the correct client response.
    """
    if isinstance(exc, BrowserRateLimitError):
        return 429
    if isinstance(exc, BrowserQueueFullError):
        return 503
    if isinstance(exc, PromptTooLargeError):
        if exc.is_attachment:
            return 500
        return 413
    return 504 if isinstance(exc, TimeoutError) else 500


_REPLAY_TOOLCALL_MESSAGE = (
    "WebMind: this exact completion request was already answered once with "
    "a tool call that was previously returned to the client. To prevent the "
    "same external tool from being executed twice as a result of a retry, "
    "that tool call is deliberately not being re-issued. If the original "
    "answer was lost, re-submit with the tool result appended (or with a "
    "modified request) so WebMind can produce a fresh decision."
)


def _replay_response(recorded: Any, request: "ChatRequest"):
    """Serve a stored response for a detected retry.

    The recorded value is {"kind": "success", "content": <text>}.
    The response shape (stream vs json, tool vs plain) is derived
    from the retry request itself, which is identical to the
    original (same dedup key).

    A recorded TOOL CALL is never re-issued on a retry: the original
    tool_call was already returned to the client, so forwarding it again
    would make the external client execute the same tool twice.  The
    retry instead receives a safe OpenAI-compatible message (finish_reason
    "stop") explaining the situation.  Ordinary completed message
    responses replay unchanged.
    """
    if not (isinstance(recorded, dict) and recorded.get("kind") == "success"):
        raise HTTPException(
            status_code=500,
            detail="WebMind error: cached response has an unknown format."
        )

    content = recorded.get("content") or ""
    tool_mode = _has_tools(request)
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if request.stream:
        if tool_mode:
            parsed = parse_chatgpt_response(content, request.tools, history=request.messages)
            if parsed["type"] == "tool_call":
                chunks = sse_message_chunks(chat_id, created, _REPLAY_TOOLCALL_MESSAGE)
            else:
                chunks = sse_message_chunks(chat_id, created, parsed["content"])
        else:
            chunks = sse_message_chunks(
                chat_id, created, _unescape_prose_newlines(content)
            )

        async def replay():
            for chunk in chunks:
                yield chunk

        return StreamingResponse(
            replay(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    if tool_mode:
        parsed = parse_chatgpt_response(content, request.tools, history=request.messages)
        if parsed["type"] == "tool_call":
            parsed = {"type": "message", "content": _REPLAY_TOOLCALL_MESSAGE}
        return build_nonstream_response(parsed, chat_id, created)

    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": "webmind",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _unescape_prose_newlines(content),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _correction_ask(directive: str, chat_url: Any, conversation_id: Any, deadline: Any = None) -> str:
    """One model correction round trip for a rejected tool call envelope.

    Runs through the same browser worker as the original ask but never
    touches the idempotency guard: the recorded response stays the ORIGINAL
    prompt's text, so a client retry replays that (re-parsed, safely) and
    never re-pastes a second correction message.  Images were already
    attached by the original ask in the same ChatGPT chat.  The correction
    round trip shares the parent request's wall-clock deadline.
    """
    response = webmind.ask(
        directive,
        chat_url=chat_url,
        image_paths=[],
        conversation_id=conversation_id,
        deadline=deadline,
    )
    return response.text


_request_guard = IdempotencyGuard()


# How small to re-serialize a conversation after ChatGPT rejected the first
# attempt as a pasted-text attachment.
#
# The conversion turned out to be CONTENT-dependent rather than
# length-dependent -- a real 14.3K conversation was converted while a 17.3K
# synthetic one stayed inline -- so there is no threshold that reliably
# prevents it.  Detection is therefore the dependable signal, and this
# budget is simply "comfortably below anything ChatGPT has ever refused",
# which trades old context (explicitly marked as dropped) for an answer
# that is actually about the request.
ATTACHMENT_RETRY_TARGET_CHARS = int(
    os.environ.get("WEBMIND_ATTACHMENT_RETRY_TARGET_CHARS", "5000")
)
ATTACHMENT_RETRY_TOOL_RESULT_CHARS = int(
    os.environ.get("WEBMIND_ATTACHMENT_RETRY_TOOL_RESULT_CHARS", "2000")
)
DEFAULT_MODEL = os.environ.get("WEBMIND_DEFAULT_MODEL", "gpt-5.6-luna")


async def _dispatch_with_attachment_retry(
    run,
    rebuild,
    prompt: str,
    *,
    request_id: str,
    path: str,
    target: int = ATTACHMENT_RETRY_TARGET_CHARS,
) -> Any:
    """Run `run(prompt)`, retrying ONCE smaller if it became an attachment.

    `run` is an async callable taking a prompt; `rebuild(target_chars)`
    re-serializes the same conversation under a tighter soft ceiling.

    Why a retry is safe here but not for an identical resend: the second
    attempt is a DIFFERENT, smaller prompt, so it is a new question rather
    than a duplicate paste.  The abandoned turn stays in the ChatGPT
    scrollback and is logged; that is the accepted cost of getting a real
    answer instead of a document summary.

    A refusal of kind "over_ceiling" is re-raised untouched -- the
    pre-flight already knows the prompt is too big, so re-serializing it
    more compactly is the caller's decision, not ours.
    """
    try:
        return await run(prompt)
    except PromptTooLargeError as e:
        if not e.is_attachment:
            raise
        smaller = rebuild(target)
        if len(smaller) >= len(prompt):
            # Nothing was shed, so the second attempt would be byte-identical
            # and would hit the same verdict.  Do not waste a generation.
            _log_event(
                "request.attachment_retry_skipped",
                request_id=request_id,
                path=path,
                prompt_len=len(prompt),
                smaller_len=len(smaller),
                reason="re-serialization did not shrink the prompt; "
                       "an identical resend would fail identically",
            )
            raise
        _log_event(
            "request.attachment_retry",
            request_id=request_id,
            path=path,
            first_len=len(prompt),
            retry_len=len(smaller),
            target=target,
            reason="prompt was submitted as a pasted-text attachment; "
                   "retrying once with older context dropped",
        )
        print(
            f"[attachment] {len(prompt)} char prompt landed as a pasted-text "
            f"file; retrying at {len(smaller)} chars.",
            flush=True,
        )
        return await run(smaller)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):

    request_id = uuid.uuid4().hex

    # Per-request wall-clock cap.  One absolute monotonic instant for the
    # whole request: it threads into the worker task tuple (converted to
    # remaining seconds for main.ask/ask_stream) AND is captured by the SSE
    # generate() closure so the HTTP layer is the backstop when the browser
    # wedges.  None when WEBMIND_REQUEST_DEADLINE_SECONDS is unset/0.
    deadline = _request_deadline()

    # --------------------------------------------------------
    # Model validation
    # --------------------------------------------------------

    _log_event(
        "http.request_received",
        request_id=request_id,
        model=request.model,
        stream=request.stream,
        tool_mode=bool(request.tools),
        tools_count=len(request.tools) if request.tools else 0,
        messages_count=len(request.messages) if request.messages else 0,
        roles=[m.role for m in request.messages] if request.messages else [],
        tool_choice=request.tool_choice,
        request_id_field=request.request_id,
        client_host=raw_request.client.host if raw_request.client else None,
    )

    # Model names are accepted as an opaque selector: the ChatGPT browser
    # backend is model-agnostic, and the harness wire "model" is whatever
    # the client registered for this base URL (commonly "webmind"). The
    # responses keep the "webmind" id OpenAI clients see on /v1/models.
    _log_event(
        "request.model_accepted",
        request_id=request_id,
        model=request.model,
    )

    # Phase 0: debug logging only, no behavior change.
    _log_chat_request(request)

    tool_mode = _has_tools(request)

    # Conversation isolation: a different opener means a different client
    # conversation, so the browser serves it in its own ChatGPT chat and
    # never leaks the previous conversation's turns into this answer.
    conversation_id = _conversation_id(request, raw_request=raw_request)
    is_continuation = any(
        (
            getattr(m, "role", None) in ("assistant", "tool")
            or getattr(m, "tool_calls", None)
        )
        for m in (request.messages or [])
    )
    _log_event(
        "request.conversation_identified",
        request_id=request_id,
        conversation_id=conversation_id,
        is_continuation=is_continuation,
    )

    if tool_mode:
        # Phase 2 translation: serialize tools + full message history
        # into the ChatGPT browser prompt. Images ride along so the
        # existing attach pipeline still works.
        target_model = (
            request.model
            if (request.model and request.model != "webmind")
            else DEFAULT_MODEL
        )
        try:
            prompt, image_urls = build_prompt(
                request.messages,
                request.tools,
                request.tool_choice,
                model=target_model,
                turn_count=_session_turns_completed.get(conversation_id, 0),
            )
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
    else:
        # Legacy path unchanged: latest user text (+ images).
        prompt, image_urls = _extract_prompt_and_images(request)

    _log_event(
        "request.prompt_built",
        request_id=request_id,
        tool_mode=tool_mode,
        prompt_len=len(prompt or ""),
        image_count=len(image_urls),
    )

    # --------------------------------------------------------
    # Idempotency guard: short-circuit retries.
    # --------------------------------------------------------
    # Client timeouts (the reasoning model can take minutes) make
    # OpenCode re-send the identical request. Register the key and,
    # if this is a retry, wait for the in-flight original to finish
    # and replay its response instead of pasting a duplicate
    # ChatGPT message. Validation errors above are deterministic,
    # so they are raised before any record is created.
    key, rec, is_new = _request_guard.start(request, raw_request.headers)

    _log_event(
        "request.guard_checked",
        request_id=request_id,
        dedup_key=key,
        is_new=is_new,
    )

    while not is_new:
        label = _key_label(key)
        print(f"[dedup] retry detected (key={label}); waiting for original...", flush=True)
        _log_event(
            "request.retry_waiting",
            request_id=request_id,
            dedup_key=key,
            state="IN_FLIGHT",
        )

        # Wait (blocking call offloaded so the event loop stays free)
        # for the original worker task to finish or fail.
        finished = await asyncio.to_thread(rec.event.wait, DEDUP_WAIT_TIMEOUT)

        if not finished:
            print(
                f"[dedup] retry (key={label}) timed out waiting for the original; no duplicate pasted.",
                flush=True,
            )
            _log_event(
                "request.retry_timed_out",
                request_id=request_id,
                dedup_key=key,
            )
            raise HTTPException(
                status_code=504,
                detail="WebMind error: the original request is still being processed; "
                       "no duplicate was created. Please retry the request."
            )

        if rec.error is not None:
            if rec.retryable:
                # Transient pre-send failure (e.g. "could not verify the
                # prompt text was inserted"): nothing reached ChatGPT.
                # Re-issuing the request is safe only if we keep the SAME
                # dedup key and re-register (rearm).  A remove()+start()
                # would create a SECOND record under the same key and
                # unlink this retry from further duplicates, letting an
                # attacker loop forever; so instead we rearm in place and
                # let the boundary come from the bounded rearm budget.
                if not _request_guard.rearm(key):
                    # Budget exhausted: further identical retries must
                    # NOT re-run (they would keep re-pasting a prompt
                    # that never sends).  Fail definitely instead.
                    print(
                        f"[dedup] retry (key={label}) rearm budget exhausted; "
                        f"refusing to re-run (max {MAX_RETRYABLE_RERUNS}).",
                        flush=True,
                    )
                    _log_event(
                        "request.retry_rearm_exhausted",
                        request_id=request_id,
                        dedup_key=key,
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "WebMind error: this request failed retryably more "
                            "than the allowed re-run budget; refusing to "
                            "re-register to protect ChatGPT from repeated "
                            "partial pastes. Consider a fresh request."
                        )
                    )
                print(
                    f"[dedup] retry (key={label}) rearmed; re-running the original "
                    f"on the same dedup key (current reruns={rec.rerun_count}).",
                    flush=True,
                )
                _log_event(
                    "request.retry_rearm",
                    request_id=request_id,
                    dedup_key=key,
                    reruns=rec.rerun_count,
                )
                is_new = True
                continue

            print(
                f"[dedup] retry (key={label}) replaying the original's fatal error, no duplicate pasted.",
                flush=True,
            )
            _log_event(
                "request.retry_fatal_error",
                request_id=request_id,
                dedup_key=key,
                error=str(rec.error),
            )
            raise HTTPException(
                status_code=rec.status_code,
                detail=f"WebMind error: {rec.error}"
            )

        print(
            f"[dedup] retry (key={label}) served from the completed original; no duplicate pasted.",
            flush=True,
        )
        _log_event(
            "request.retry_replay",
            request_id=request_id,
            dedup_key=key,
        )
        return _replay_response(rec.response, request)

    # --------------------------------------------------------
    # Materialize images to temp files. A NEW (non-retry) request
    # always reaches this point exactly once. If this fails, mark
    # the record failed so a retry gets the error (never waits).
    # --------------------------------------------------------
    try:
        image_paths = _prepare_image_files(image_urls) if image_urls else []
    except HTTPException as _e:
        # Nothing has been sent to ChatGPT yet, so a retry is safe.
        _request_guard.fail(key, str(_e.detail), retryable=True)
        raise

    def _is_multi_attach_error(exc: Exception) -> bool:
        msg = str(exc)
        return "MULTI_ATTACH_UNSUPPORTED" in msg or "multiple image" in msg.lower()

    # --------------------------------------------------------
    # Streaming path (OpenAI-compatible SSE)
    # --------------------------------------------------------
    # Deltas are UI-observed text slices from the ChatGPT web
    # page, NOT model tokens. Chunk sizes are arbitrary.

    if request.stream:

        if tool_mode:
            # ----------------------------------------------------
            # Phase 2 streaming: buffer ONE full browser round-trip,
            # parse/validate the envelope, then emit an OpenAI
            # compatible SSE sequence (tool_calls or plain content).
            # No incremental tool-call parsing yet.
            # ----------------------------------------------------
            _log_event(
                "request.dispatched",
                request_id=request_id,
                path="tool_stream",
                dedup_key=key,
            )
            try:
                async def _dispatch(text: str):
                    return await asyncio.to_thread(
                        webmind.ask,
                        text,
                        request.chat_url,
                        image_paths,
                        key,
                        request_id,
                        conversation_id,
                        deadline,
                    )

                def _rebuild(target_chars: int) -> str:
                    return build_prompt(
                        request.messages,
                        request.tools,
                        request.tool_choice,
                        target_chars=target_chars,
                        tool_role_max_chars=ATTACHMENT_RETRY_TOOL_RESULT_CHARS,
                        model=target_model,
                        turn_count=_session_turns_completed.get(conversation_id, 0),
                    )[0]

                response = await _dispatch_with_attachment_retry(
                    _dispatch,
                    _rebuild,
                    prompt,
                    request_id=request_id,
                    path="tool_stream",
                )
            except Exception as e:
                _cleanup_files(image_paths)
                _request_guard.fail(
                    key, str(e), retryable=_is_retryable_error(e)
                )
                _log_event(
                    "request.failed",
                    request_id=request_id,
                    path="tool_stream",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise HTTPException(
                    status_code=_error_status(e),
                    detail=f"WebMind error: {str(e)}"
                )
            _cleanup_files(image_paths)
            _log_event(
                "browser.round_trip_completed",
                request_id=request_id,
                path="tool_stream",
                chars=len(response.text),
            )

            parsed = parse_chatgpt_response(response.text, request.tools, history=request.messages)
            if parsed["type"] == "validation_failure":
                _log_event(
                    "tool_call.validation_failed",
                    request_id=request_id,
                    path="tool_stream",
                    errors=parsed.get("errors") or [],
                )
                parsed = recover_tool_call(
                    parsed,
                    lambda directive: _correction_ask(
                        directive, request.chat_url, conversation_id, deadline
                    ),
                    request.tools,
                    history=request.messages,
                )
            chat_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            if parsed["type"] == "tool_call":
                chunks = sse_tool_call_chunks(
                    chat_id, created, parsed["tool_calls"], content=parsed.get("content")
                )
            else:
                chunks = sse_message_chunks(
                    chat_id, created, parsed["content"]
                )

            async def emit_tool_mode_stream():
                for chunk in chunks:
                    yield chunk

            return StreamingResponse(
                emit_tool_mode_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )

        try:
            _log_event(
                "request.dispatched",
                request_id=request_id,
                path="stream",
                dedup_key=key,
            )
            delta_queue = webmind.start_stream(
                prompt,
                chat_url=request.chat_url,
                image_paths=image_paths,
                dedup_key=key,
                request_id=request_id,
                conversation_id=conversation_id,
                deadline=deadline,
            )
        except Exception as e:
            _cleanup_files(image_paths)
            _request_guard.fail(
                key, str(e), retryable=_is_retryable_error(e)
            )
            _log_event(
                "request.failed",
                request_id=request_id,
                path="stream",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=_error_status(e),
                detail=f"WebMind error: {str(e)}"
            )

        chat_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        async def generate():
            _log_event("stream.opened", request_id=request_id, chat_id=chat_id)
            # An attachment refusal happens before ChatGPT generates anything,
            # so no delta has been yielded yet and swapping in a shorter prompt
            # cannot duplicate anything the client has already seen.
            nonlocal delta_queue
            retried_stream = False
            try:
                # First chunk announces the assistant role.
                yield _sse_chunk(
                    chat_id,
                    created,
                    {"role": "assistant", "content": ""}
                )

                stream_error = None
                emitted_any = False

                while True:
                    # Pull from the thread-safe queue without
                    # blocking the asyncio event loop. If nothing has
                    # arrived within the window, send an SSE comment so
                    # client read timeouts never fire during long
                    # thinking phases.
                    if deadline is not None and time.monotonic() >= deadline:
                        _log_event(
                            "stream.deadline_reached",
                            request_id=request_id,
                        )
                        raise TimeoutError(
                            "WebMind stream exceeded its wall-clock deadline."
                        )
                    try:
                        item = await asyncio.to_thread(delta_queue.get, timeout=15)
                    except queue.Empty:
                        _log_event(
                            "stream.keepalive",
                            request_id=request_id,
                        )
                        yield ": keep-alive\n\n"
                        continue

                    if item is _STREAM_DONE:
                        break

                    if isinstance(item, Exception):
                        # Attachment refusal on the no-tools path: try once
                        # more with materially less text.  Only safe while
                        # nothing has been emitted, which is always true here
                        # because the refusal happens at submit time, before
                        # ChatGPT generates.  Staying in the loop keeps the
                        # swap invisible to the client.
                        if (
                            isinstance(item, PromptTooLargeError)
                            and item.is_attachment
                            and not emitted_any
                            and not retried_stream
                        ):
                            smaller = cap_legacy_prompt(
                                prompt, LEGACY_RETRY_MAX_CHARS
                            )
                            if len(smaller) < len(prompt):
                                retried_stream = True
                                _log_event(
                                    "request.attachment_retry",
                                    request_id=request_id,
                                    path="stream_legacy",
                                    first_len=len(prompt),
                                    retry_len=len(smaller),
                                    target=LEGACY_RETRY_MAX_CHARS,
                                    reason="legacy stream prompt was submitted "
                                           "as a pasted-text attachment; "
                                           "retrying shorter before any delta "
                                           "was emitted",
                                )
                                print(
                                    f"[attachment] {len(prompt)} char legacy "
                                    f"stream prompt landed as a pasted-text "
                                    f"file; retrying at {len(smaller)} chars.",
                                    flush=True,
                                )
                                delta_queue = webmind.start_stream(
                                    smaller,
                                    chat_url=request.chat_url,
                                    image_paths=image_paths,
                                    dedup_key=key,
                                    request_id=request_id,
                                    conversation_id=conversation_id,
                                    deadline=deadline,
                                )
                                continue
                        stream_error = item
                        break

                    if not item:
                        continue

                    _log_event(
                        "stream.delta",
                        request_id=request_id,
                        delta_len=len(item),
                    )
                    emitted_any = True
                    yield _sse_chunk(
                        chat_id,
                        created,
                        {"content": item}
                    )

                if stream_error is not None:
                    # Surface worker failures inside the stream so the
                    # client sees an error instead of hanging. The payload
                    # is informational; the terminal chunk below is what the
                    # harness stream consumer (dsh-llm-deepseek translate)
                    # actually reads.
                    error_payload = {
                        "error": {
                            "message": f"WebMind error: {str(stream_error)}",
                            "type": "webmind_error"
                        }
                    }
                    _log_event(
                        "stream.error",
                        request_id=request_id,
                        error=str(stream_error),
                        error_type=type(stream_error).__name__,
                    )
                    yield f"data: {json.dumps(error_payload)}\n\n"

                # OpenAI-style terminal chunk + DONE. A mid-stream failure
                # declares a non-standard finish_reason so the harness maps
                # it to an error finish (WEBMIND_ERROR) instead of an
                # EMPTY_RESPONSE misread. usage rides on the finish chunk so
                # stream_options.include_usage is honored on every stream.
                finish_reason = "webmind_error" if stream_error is not None else "stop"
                yield _sse_chunk(
                    chat_id,
                    created,
                    {},
                    finish_reason=finish_reason,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                )
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                _log_event("stream.client_disconnected", request_id=request_id, reason="GeneratorExit")
                raise
            except asyncio.CancelledError:
                _log_event("stream.client_disconnected", request_id=request_id, reason="CancelledError")
                raise
            except Exception as e:
                _log_event(
                    "stream.generator_failed",
                    request_id=request_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            finally:
                _cleanup_files(image_paths)
                _log_event(
                    "stream.closed",
                    request_id=request_id,
                    had_error=stream_error is not None,
                )

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # --------------------------------------------------------
    # Non-streaming path (unchanged behavior for text)
    # --------------------------------------------------------
    # Offload the blocking worker call so the event loop
    # stays responsive. Never call Playwright Sync API here.

    try:

        _log_event(
            "request.dispatched",
            request_id=request_id,
            path="non_stream",
            dedup_key=key,
        )

        async def _dispatch(text: str):
            return await asyncio.to_thread(
                webmind.ask,
                text,
                request.chat_url,
                image_paths,
                key,
                request_id,
                conversation_id,
                deadline,
            )

        if tool_mode:
            def _rebuild(target_chars: int) -> str:
                return build_prompt(
                    request.messages,
                    request.tools,
                    request.tool_choice,
                    target_chars=target_chars,
                    tool_role_max_chars=ATTACHMENT_RETRY_TOOL_RESULT_CHARS,
                    model=target_model,
                    turn_count=_session_turns_completed.get(conversation_id, 0),
                )[0]

            response = await _dispatch_with_attachment_retry(
                _dispatch,
                _rebuild,
                prompt,
                request_id=request_id,
                path="non_stream",
            )
        else:
            # Legacy path: the prompt is a single user message, not a
            # serialized history, so it cannot be re-serialized -- but it CAN
            # simply be sent shorter, which is what the retry does.  Without
            # this, opencode's auto-compaction hit the attachment conversion
            # at 12042 chars and failed forever.
            def _rebuild(target_chars: int) -> str:
                # `prompt` is already capped once; capping it again at a
                # lower limit just takes a shorter head, and the first
                # truncation marker sits past the new cut so it is replaced
                # rather than duplicated.
                return cap_legacy_prompt(prompt, target_chars)

            response = await _dispatch_with_attachment_retry(
                _dispatch,
                _rebuild,
                prompt,
                request_id=request_id,
                path="non_stream_legacy",
                target=LEGACY_RETRY_MAX_CHARS,
            )

    except Exception as e:
        _cleanup_files(image_paths)
        _request_guard.fail(key, str(e), retryable=_is_retryable_error(e))

        _log_event(
            "request.failed",
            request_id=request_id,
            path="non_stream",
            error=str(e),
            error_type=type(e).__name__,
        )

        if _is_multi_attach_error(e):
            raise HTTPException(
                status_code=400,
                detail=f"WebMind error: {str(e)}"
            )

        raise HTTPException(
            status_code=_error_status(e),
            detail=f"WebMind error: {str(e)}"
        )

    _cleanup_files(image_paths)
    _log_event(
        "browser.round_trip_completed",
        request_id=request_id,
        path="non_stream",
        chars=len(response.text),
    )

    # --------------------------------------------------------
    # Phase 2: tool-mode non-streaming response
    # --------------------------------------------------------

    if tool_mode:
        parsed = parse_chatgpt_response(response.text, request.tools, history=request.messages)
        if parsed["type"] == "validation_failure":
            _log_event(
                "tool_call.validation_failed",
                request_id=request_id,
                path="non_stream",
                errors=parsed.get("errors") or [],
            )
            parsed = recover_tool_call(
                parsed,
                lambda directive: _correction_ask(
                    directive, request.chat_url, conversation_id, deadline
                ),
                request.tools,
                history=request.messages,
            )
        _log_event(
            "request.completed",
            request_id=request_id,
            path="non_stream",
            parsed_type=parsed["type"],
        )
        return build_nonstream_response(
            parsed,
            f"chatcmpl-{uuid.uuid4().hex}",
            int(time.time())
        )

    # --------------------------------------------------------
    # Extract response text
    # --------------------------------------------------------

    answer = _unescape_prose_newlines(response.text)

    # --------------------------------------------------------
    # OpenAI-compatible response
    # --------------------------------------------------------

    _log_event(
        "request.completed",
        request_id=request_id,
        path="non_stream",
        parsed_type="plain",
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "webmind",

        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": "stop"
            }
        ],

        # WebMind does not have access to actual token counts
        # because it is communicating with ChatGPT through the UI.
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="WebMind ChatGPT Bridge")
    default_port = int(os.environ.get("PORT", os.environ.get("WEBMIND_PORT", "8001")))
    parser.add_argument("--port", type=int, default=default_port, help="Port to bind server")
    args, _ = parser.parse_known_args()

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=args.port,
        workers=1
    )
