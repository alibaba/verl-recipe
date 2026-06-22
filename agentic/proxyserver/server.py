"""LLM Proxy server — a **framework-agnostic** relay service that bridges
OpenAI-compatible HTTP requests to backend inference workers over WebSocket.

The proxy supports two operating modes:

**Relay mode** (default / standalone deployment)
    The proxy starts an :class:`InferenceRelay` that accepts WebSocket
    connections from framework-side inference workers.  Agent HTTP
    requests are forwarded to connected workers over WebSocket and the
    responses are relayed back.  No framework-specific code is needed.

**Local mode** (injected completion handler)
    A framework-specific ``completion_handler`` is injected at init time.
    The proxy routes agent requests directly to this handler without
    using WebSocket relay.  This is useful when the proxy runs in the
    same process/cluster as the inference engine.

Architecture (relay mode)::

    Third-party Agent
         |  (OpenAI /v1/chat/completions)
         v
    +----------------------------------+       WebSocket        +--------------------+
    |  LLMProxyServer (standalone)     | <-------------------> | Inference Worker   |
    |  - FastAPI HTTP + WS server      |  completion_request   | (framework-side)   |
    |  - InferenceRelay                |  completion_response  | - any LLM backend  |
    |  - SessionRecorder per trial_id  |                       +--------------------+
    +----------------------------------+

Each caller registers a *session_id* with the proxy.  The agent is
given a URL of the form ``http://host:port/{session_id}/v1`` so that
standard OpenAI SDK calls resolve to
``POST http://host:port/{session_id}/v1/chat/completions``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any, Callable, Coroutine
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from .models import TimelineSpan
from .recorder import SessionRecorder

logger = logging.getLogger(__name__)

# Type alias for the local completion handler that frameworks can inject.
# Signature: async (proxy, session_id, messages, body, is_streaming) -> Response
CompletionHandler = Callable[
    ["LLMProxyServer", str, list, dict, bool],
    Coroutine[Any, Any, Any],
]


# ---------------------------------------------------------------------------
# FastAPI application builder
# ---------------------------------------------------------------------------


def _build_app(proxy: "LLMProxyServer") -> FastAPI:
    """Build the FastAPI application that serves as the OpenAI proxy."""

    app = FastAPI(title="LLM Proxy", version="0.5.0")

    @app.get("/health")
    async def health():
        info: dict[str, Any] = {"status": "ok", "mode": proxy.mode}
        if proxy.mode == "relay" and proxy.relay is not None:
            info["connected_workers"] = proxy.relay.num_workers
        return info

    # ---- session management ----------------------------------------------

    @app.post("/sessions/{session_id}")
    async def create_session(session_id: str):
        proxy.recorder.create_session(session_id)
        return {"session_id": session_id, "status": "created"}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = proxy.recorder.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.model_dump()

    @app.post("/sessions/{session_id}/complete")
    async def complete_session(session_id: str):
        proxy.recorder.mark_completed(session_id)
        return {"session_id": session_id, "status": "completed"}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        proxy.delete_session(session_id)
        return {"session_id": session_id, "status": "deleted"}

    # ---- timeline endpoints --------------------------------------------

    @app.get("/sessions/{session_id}/timeline")
    async def get_timeline(
        session_id: str,
        since: float | None = Query(default=None),
        type: str | None = Query(default=None),
    ):
        timeline = proxy.recorder.get_timeline(session_id)
        if timeline is None:
            raise HTTPException(status_code=404, detail="Session not found")
        spans = timeline.get_spans(since=since, operation=type)
        return {
            "session_id": session_id,
            "spans": [s.model_dump() for s in spans],
        }

    @app.get("/sessions/{session_id}/stats")
    async def get_stats(session_id: str):
        timeline = proxy.recorder.get_timeline(session_id)
        if timeline is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return timeline.compute_stats().model_dump()

    @app.post("/sessions/{session_id}/events")
    async def inject_events(session_id: str, request: Request):
        timeline = proxy.recorder.get_timeline(session_id)
        if timeline is None:
            raise HTTPException(status_code=404, detail="Session not found")

        body = await request.json()
        spans_data = body.get("spans", [])
        injected: list[str] = []

        for span_data in spans_data:
            span = TimelineSpan(**span_data)
            timeline.add_standalone_span(span)
            injected.append(span.span_id)

        return {"session_id": session_id, "injected_span_ids": injected}

    @app.websocket("/sessions/{session_id}/timeline/stream")
    async def timeline_stream(session_id: str, ws: WebSocket):
        timeline = proxy.recorder.get_timeline(session_id)
        if timeline is None:
            await ws.close(code=4004, reason="Session not found")
            return

        await ws.accept()
        queue = timeline.subscribe()

        try:
            existing_spans = timeline.get_spans()
            await ws.send_json(
                {
                    "type": "initial_state",
                    "spans": [s.model_dump() for s in existing_spans],
                }
            )

            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    await ws.send_json({"type": "ping"})
                    continue

                if message.get("type") == "session_deleted":
                    await ws.send_json(message)
                    break

                await ws.send_json(message)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("Timeline stream error for %s: %s", session_id, e)
        finally:
            timeline.unsubscribe(queue)

    # ---- OpenAI-compatible chat completions ------------------------------
    # Two URL patterns so the proxy works with both:
    #   1. Agents that call a fully custom endpoint
    #   2. Standard OpenAI SDK where base_url = http://host:port/{session_id}/v1

    @app.post("/v1/chat/completions/{session_id}")
    @app.post("/{session_id}/v1/chat/completions")
    async def chat_completions(session_id: str, request: Request):
        """Generate a chat completion and record data.

        Routes to either the injected local handler or the WebSocket
        relay depending on the proxy's operating mode.
        """
        body = await request.json()
        messages: list[dict[str, Any]] = body.get("messages", [])
        is_streaming: bool = body.get("stream", False)

        if proxy.mode == "local" and proxy._completion_handler is not None:
            return await proxy._completion_handler(
                proxy, session_id, messages, body, is_streaming
            )
        else:
            return await _handle_relay_completion(
                proxy, session_id, messages, body, is_streaming
            )

    # ---- WebSocket endpoint for inference workers (relay mode) -----------

    @app.websocket("/ws/worker")
    async def worker_websocket(ws: WebSocket):
        """WebSocket endpoint for framework-side inference workers.

        Only functional in relay mode.  Workers connect here, send a
        ``worker_hello`` handshake, and then receive inference requests
        pushed by the proxy.
        """
        if proxy.relay is None:
            await ws.close(
                code=4002,
                reason="Proxy is not in relay mode",
            )
            return
        await proxy.relay.handle_worker_ws(ws)

    return app


# ---------------------------------------------------------------------------
# Relay mode completion handler (forward to worker via WebSocket)
# ---------------------------------------------------------------------------


async def _handle_relay_completion(
    proxy: "LLMProxyServer",
    session_id: str,
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    is_streaming: bool,
):
    """Handle a completion request by relaying to a connected worker."""
    span_id = proxy.recorder.start_span(
        session_id=session_id,
        operation="llm.completion",
        attributes={
            "mode": "relay",
            "streaming": is_streaming,
            "message_count": len(messages),
            "max_tokens": body.get("max_tokens", 2048),
        },
    )

    try:
        result = await proxy.relay.dispatch(
            session_id=session_id,
            messages=messages,
            temperature=body.get("temperature", 1.0),
            top_p=body.get("top_p", 1.0),
            max_tokens=body.get("max_tokens", 2048),
            stop=body.get("stop"),
            tools=body.get("tools"),
            repetition_penalty=body.get("repetition_penalty"),
            stream=is_streaming,
        )
    except RuntimeError as e:
        logger.error("Relay failed for session %s: %s", session_id, e)
        if span_id:
            proxy.recorder.end_span(
                session_id, span_id, status="error", attributes={"error": str(e)}
            )
        return JSONResponse(status_code=503, content={"error": str(e)})
    except Exception as e:
        logger.error("Relay error for session %s: %s", session_id, e)
        if span_id:
            proxy.recorder.end_span(
                session_id, span_id, status="error", attributes={"error": str(e)}
            )
        return JSONResponse(status_code=500, content={"error": str(e)})

    if result.get("error"):
        if span_id:
            proxy.recorder.end_span(
                session_id, span_id, status="error",
                attributes={"error": result["error"]},
            )
        return JSONResponse(
            status_code=500, content={"error": result["error"]}
        )

    # Record the completion from the relay result
    token_ids = result.get("token_ids", [])
    log_probs = result.get("log_probs", [])
    content_text = result.get("content_text", "")
    tool_calls = result.get("tool_calls")
    finish_reason = result.get("finish_reason", "stop")

    proxy.recorder.record_completion(
        session_id=session_id,
        messages=messages,
        completion_text=content_text,
        token_ids=token_ids,
        logprobs=log_probs,
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )

    if span_id:
        proxy.recorder.end_span(
            session_id, span_id, status="ok",
            attributes={
                "completion_tokens": len(token_ids),
                "finish_reason": finish_reason,
                "has_tool_calls": tool_calls is not None,
            },
        )

    # Build an OpenAI-compatible response
    response_body = build_openai_response(
        token_ids=token_ids,
        content_text=content_text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )
    return JSONResponse(content=response_body)


# ---------------------------------------------------------------------------
# OpenAI response builder (public — used by both relay and local handlers)
# ---------------------------------------------------------------------------


def build_openai_response(
    *,
    token_ids: list[int],
    content_text: str,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response dict.

    This is a public helper so that framework-specific local-mode
    handlers can reuse it.
    """
    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        message["content"] = (
            content_text.strip() if content_text and content_text.strip() else None
        )
        message["tool_calls"] = tool_calls
    else:
        message["content"] = content_text

    return {
        "id": f"chatcmpl-{uuid4().hex[:12]}",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": len(token_ids),
            "total_tokens": len(token_ids),
        },
    }


# ---------------------------------------------------------------------------
# LLMProxyServer
# ---------------------------------------------------------------------------


class LLMProxyServer:
    """Framework-agnostic LLM Proxy server.

    Bridges OpenAI-compatible HTTP requests to backend inference — either
    via WebSocket relay (standalone / default) or via an injected local
    completion handler (when co-located with the inference engine).

    Usage (relay / standalone)::

        proxy = LLMProxyServer()
        url = await proxy.start()
        # Workers connect to ws://host:port/ws/worker
        # Agents use base_url = f"{url}/session-1/v1"

    Usage (local mode — framework injects handler)::

        proxy = LLMProxyServer(completion_handler=my_handler)
        url = await proxy.start()
        # Agents use base_url = f"{url}/session-1/v1"
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 0,
        relay_request_timeout: float = 300.0,
        completion_handler: CompletionHandler | None = None,
        on_session_deleted: Callable[[str], None] | None = None,
    ):
        """
        Args:
            host: Bind address for the HTTP server.
            port: Port number. ``0`` means auto-select a free port.
            relay_request_timeout: Timeout (seconds) for relay-mode requests
                waiting for a worker response.
            completion_handler: Optional async callable for local-mode
                inference.  Signature:
                ``async (proxy, session_id, messages, body, is_streaming) -> Response``.
                When provided, the proxy runs in **local mode** and does not
                start the WebSocket relay.
            on_session_deleted: Optional callback invoked when a session is
                deleted.  Useful for framework-specific cleanup (e.g.
                releasing sticky-session bindings).
        """
        self.host = host
        self.port = port
        self.recorder = SessionRecorder()
        self._on_session_deleted = on_session_deleted

        if completion_handler is not None:
            # --- Local mode: framework-injected handler ---
            self._completion_handler = completion_handler
            self.relay = None
            self._mode = "local"
        else:
            # --- Relay mode: forward via WebSocket ---
            from .relay import InferenceRelay

            self.relay = InferenceRelay(
                request_timeout=relay_request_timeout,
            )
            self._completion_handler = None
            self._mode = "relay"

        self.app = _build_app(self)

        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None
        self._actual_port: int | None = None

    @property
    def mode(self) -> str:
        """Operating mode: ``"local"`` or ``"relay"``."""
        return self._mode

    # -- session management ------------------------------------------------

    def register_session(self, session_id: str) -> None:
        """Register a new session for recording."""
        self.recorder.create_session(session_id)

    def get_session_data(self, session_id: str):
        """Retrieve the recorded session data."""
        return self.recorder.get_session(session_id)

    def complete_session(self, session_id: str) -> None:
        self.recorder.mark_completed(session_id)

    def delete_session(self, session_id: str) -> None:
        if self._on_session_deleted is not None:
            try:
                self._on_session_deleted(session_id)
            except Exception as e:
                logger.warning("on_session_deleted hook error: %s", e)
        self.recorder.delete_session(session_id)

    # -- HTTP server lifecycle ---------------------------------------------

    @property
    def url(self) -> str | None:
        if self._actual_port is None:
            return None
        return f"http://{self.host}:{self._actual_port}"

    async def start(self, on_cancel_callback=None) -> str:
        """Start the HTTP server.  Returns the base URL.

        Args:
            on_cancel_callback: Optional callback function to call when the
                server task is cancelled.
        """
        if self.port == 0:
            self._actual_port = _find_free_port()
        else:
            self._actual_port = self.port

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self._actual_port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None

        async def _safe_serve():
            try:
                await self._server.serve()
            except asyncio.CancelledError:
                logger.warning("Proxy server task cancelled")
                if on_cancel_callback is not None:
                    try:
                        on_cancel_callback()
                    except Exception as e:
                        logger.error("Error in cancel callback: %s", e)
                raise

        self._serve_task = asyncio.create_task(_safe_serve())

        for _ in range(100):
            if self._server.started:
                break
            await asyncio.sleep(0.1)

        logger.info("LLM Proxy started at %s (mode=%s)", self.url, self.mode)
        return self.url

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._server is not None:
            self._server.should_exit = True
            if self._serve_task is not None:
                try:
                    await self._serve_task
                except asyncio.CancelledError:
                    logger.debug("Proxy serve task was cancelled during shutdown")
                self._serve_task = None
            self._server = None
            logger.info("LLM Proxy stopped")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
