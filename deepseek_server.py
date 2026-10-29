# ============================================================
# DeepSeek WebMind Server
#
# Exposes an OpenAI-compatible /v1/chat/completions API on port 8002
# backed by an authenticated DeepSeek browser session.
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
import deepseek_browser
from deepseek_protocol import (
    build_deepseek_prompt,
    parse_deepseek_response,
    sse_message_chunks,
    sse_tool_call_chunks,
    build_nonstream_response,
)

DEEPSEEK_PORT = int(os.environ.get("PORT", os.environ.get("WEBMIND_DEEPSEEK_PORT", "8002")))

app = FastAPI(title="WebMind DeepSeek Bridge")

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
    model: Optional[str] = "deepseek-chat"
    messages: List[Any]
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    user: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print(f"WEBMIND DEEPSEEK READY on http://127.0.0.1:{DEEPSEEK_PORT}")
    print("=" * 60)
    asyncio.create_task(asyncio.to_thread(deepseek_browser.start))


@app.on_event("shutdown")
def shutdown_event():
    deepseek_browser.stop()


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-coder", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-r1", "object": "model", "owned_by": "deepseek"},
            {"id": "webmind", "object": "model", "owned_by": "webmind"},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty.")

    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    prompt = build_deepseek_prompt(
        request.messages,
        request.tools or [],
        tool_choice=request.tool_choice,
    )

    try:
        response = await asyncio.to_thread(deepseek_browser.ask, prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek Browser Error: {str(e)}")

    parsed = parse_deepseek_response(
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

    return build_nonstream_response(parsed, chat_id, created, model=request.model or "deepseek-chat")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WebMind DeepSeek Bridge")
    parser.add_argument("--port", type=int, default=DEEPSEEK_PORT, help="Port to bind server")
    args, _ = parser.parse_known_args()
    DEEPSEEK_PORT = args.port
    os.environ["PORT"] = str(args.port)
    uvicorn.run("deepseek_server:app", host="127.0.0.1", port=args.port, workers=1)
