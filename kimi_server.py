# ============================================================
# Kimi WebMind Server
#
# Exposes an OpenAI-compatible /v1/chat/completions API on port 8003
# backed by an authenticated Kimi browser session.
# ============================================================

import asyncio
import os
import time
import uuid
from typing import Any, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

import config_manager
import kimi_browser
from kimi_protocol import (
    build_kimi_prompt,
    parse_kimi_response,
    sse_message_chunks,
    sse_tool_call_chunks,
    build_nonstream_response,
)

KIMI_PORT = int(os.environ.get("PORT", os.environ.get("WEBMIND_KIMI_PORT", "8003")))

app = FastAPI(title="WebMind Kimi Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
    return await call_next(request)


class ChatRequest(BaseModel):
    model: Optional[str] = "kimi"
    messages: List[Any]
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    user: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print(f"WEBMIND KIMI READY on http://127.0.0.1:{KIMI_PORT}")
    print("=" * 60)
    asyncio.create_task(asyncio.to_thread(kimi_browser.start))


@app.on_event("shutdown")
def shutdown_event():
    kimi_browser.stop()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "kimi", "object": "model", "owned_by": "moonshot"},
            {"id": "kimi-chat", "object": "model", "owned_by": "moonshot"},
            {"id": "moonshot-v1", "object": "model", "owned_by": "moonshot"},
            {"id": "webmind", "object": "model", "owned_by": "webmind"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty.")

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    prompt = build_kimi_prompt(
        request.messages,
        request.tools or [],
        tool_choice=request.tool_choice,
    )

    try:
        response = await asyncio.to_thread(kimi_browser.ask, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kimi Browser Error: {str(e)}")

    parsed = parse_kimi_response(
        response.text,
        request.tools or [],
        history=request.messages,
    )

    if request.stream:
        if parsed["type"] == "tool_call":
            chunks = sse_tool_call_chunks(
                chat_id, created, parsed["tool_calls"], content=parsed.get("content")
            )
        else:
            chunks = sse_message_chunks(
                chat_id, created, parsed["content"]
            )

        async def emit():
            for c in chunks:
                yield c

        return StreamingResponse(
            emit(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return build_nonstream_response(parsed, chat_id, created, model=request.model or "kimi")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WebMind Kimi Bridge")
    parser.add_argument("--port", type=int, default=KIMI_PORT, help="Port to bind server")
    args, _ = parser.parse_known_args()
    KIMI_PORT = args.port
    os.environ["PORT"] = str(args.port)
    uvicorn.run("kimi_server:app", host="127.0.0.1", port=args.port, workers=1)