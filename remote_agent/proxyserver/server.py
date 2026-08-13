# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
import time
from typing import Any, Callable, Coroutine
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse

from .recorder import SessionAlreadyExistsError, SessionClosedError, SessionRecorder

try:
    from .tracing import optional_span
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def optional_span(name, attributes=None, kind=None):
        yield None


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Type alias for the local completion handler that frameworks can inject.
# Signature: async (proxy, session_id, messages, body, is_streaming) -> Response
CompletionHandler = Callable[
    ["LLMProxyServer", str, list, dict, bool],
    Coroutine[Any, Any, Any],
]


# ---------------------------------------------------------------------------
# FastAPI application builder
# ---------------------------------------------------------------------------


def _build_app(proxy: LLMProxyServer) -> FastAPI:
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
        try:
            proxy.recorder.create_session(session_id)
        except SessionAlreadyExistsError:
            raise HTTPException(
                status_code=409,
                detail=f"Session {session_id} already exists",
            ) from None
        return {"session_id": session_id, "status": "created"}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = proxy.recorder.get_session(session_id)
        if session is None and proxy.session_dump_dir:
            session = proxy.recorder.load_completed_session(session_id, proxy.session_dump_dir)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.model_dump()

    @app.post("/sessions/{session_id}/complete")
    async def complete_session(session_id: str):
        proxy.recorder.mark_completed(session_id)
        return {"session_id": session_id, "status": "completed"}

    @app.post("/sessions/{session_id}/reset")
    async def reset_session(session_id: str):
        """Clear all recorded turns for an active session (used before retry)."""
        if not proxy.recorder.reset_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found or closed")
        return {"session_id": session_id, "status": "reset"}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        dumped_path = None
        if proxy.session_dump_dir:
            dumped_path = proxy.recorder.dump_session_to_file(
                session_id,
                proxy.session_dump_dir,
                remove=False,
            )
        proxy.delete_session(session_id)
        resp: dict[str, Any] = {"session_id": session_id, "status": "deleted"}
        if dumped_path:
            resp["dumped_to"] = dumped_path
        return resp

    # ---- debug-only endpoints -----------------------------------------------

    if proxy.debug:

        @app.get("/sessions")
        async def list_sessions():
            """[DEBUG] List all active session IDs with summary info.

            Also includes completed sessions from dump_dir (if configured).
            For duplicate session_ids, the dump_dir version takes priority
            as it contains the complete turns record.
            """
            # Active sessions from memory
            ids = proxy.recorder.list_sessions()
            active_items: dict[str, dict] = {}
            for sid in ids:
                sess = proxy.recorder.get_session(sid)
                if sess is not None:
                    active_items[sid] = {
                        "session_id": sid,
                        "turns": len(sess.turns),
                        "completed": sess.completed,
                    }
                else:
                    active_items[sid] = {"session_id": sid, "turns": 0, "completed": False}

            # Completed sessions from dump_dir
            completed_items: dict[str, dict] = {}
            if proxy.session_dump_dir:
                for item in proxy.recorder.list_completed_sessions(proxy.session_dump_dir):
                    completed_items[item["session_id"]] = item

            # Merge: dump_dir entries take priority for duplicates
            merged: dict[str, dict] = {}
            merged.update(active_items)
            merged.update(completed_items)

            items = list(merged.values())
            return {"count": len(items), "sessions": items}

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
        # Reject requests for sessions that have already been completed/deleted.
        if proxy.recorder.is_session_closed(session_id):
            logger.warning("[PROXY] Session %s is closed, rejecting request", session_id)
            return JSONResponse(
                status_code=410,
                content={"error": f"Session {session_id} has been closed"},
            )

        body = await request.json()
        messages: list[dict[str, Any]] = body.get("messages", [])
        is_streaming: bool = body.get("stream", False)

        logger.debug(
            "[PROXY] Received chat completion request: session=%s, mode=%s, "
            "messages_count=%d, stream=%s, tools=%s (count=%d)",
            session_id,
            proxy.mode,
            len(messages),
            is_streaming,
            "present" if body.get("tools") else "absent",
            len(body.get("tools", [])),
        )
        if body.get("tools"):
            for i, tool in enumerate(body["tools"]):
                logger.debug("[PROXY] Tool[%d]: %s", i, json.dumps(tool)[:300])

        if proxy.mode == "local" and proxy._completion_handler is not None:
            logger.debug("[PROXY] Routing to LOCAL handler")
            return await proxy._completion_handler(proxy, session_id, messages, body, is_streaming)
        else:
            logger.debug("[PROXY] Routing to RELAY handler")
            return await _handle_relay_completion(proxy, session_id, messages, body, is_streaming)

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

    # ---- training timing endpoints (always available) -------------------

    @app.post("/training/timing")
    async def record_training_timing(request: Request):
        body = await request.json()
        from .models import TrainingRoundTiming

        try:
            timing = TrainingRoundTiming.model_validate(body)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Validation error: {e}") from e
        dumped_path = None
        if proxy.session_dump_dir:
            dumped_path = proxy.recorder.record_training_timing(timing, proxy.session_dump_dir)
        resp: dict[str, Any] = {"status": "recorded", "epoch": timing.epoch, "global_step": timing.global_step}
        if dumped_path:
            resp["dumped_to"] = dumped_path
        return resp

    @app.get("/training/timing")
    async def list_training_timings():
        if not proxy.session_dump_dir:
            return {"count": 0, "timings": []}
        items = proxy.recorder.list_training_timings(proxy.session_dump_dir)
        return {"count": len(items), "timings": items}

    @app.get("/training/timing/{epoch}/{global_step}")
    async def get_training_timing(epoch: int, global_step: int):
        if not proxy.session_dump_dir:
            raise HTTPException(status_code=404, detail="No dump_dir configured")
        timing = proxy.recorder.load_training_timing(epoch, global_step, proxy.session_dump_dir)
        if timing is None:
            raise HTTPException(status_code=404, detail="Training timing not found")
        return timing.model_dump()

    return app


# ---------------------------------------------------------------------------
# Relay mode completion handler (forward to worker via WebSocket)
# ---------------------------------------------------------------------------


async def _handle_relay_completion(
    proxy: LLMProxyServer,
    session_id: str,
    messages: list[dict[str, Any]],
    body: dict[str, Any],
    is_streaming: bool,
):
    """Handle a completion request by relaying to a connected worker."""
    request_received_at = time.time()
    with optional_span(
        "proxy.chat_completions",
        attributes={"session_id": session_id, "message_count": len(messages)},
    ) as span:
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
            return JSONResponse(status_code=503, content={"error": str(e)})
        except Exception as e:
            logger.error("Relay error for session %s: %s", session_id, e)
            return JSONResponse(status_code=500, content={"error": str(e)})

        if result.get("error"):
            return JSONResponse(status_code=500, content={"error": result["error"]})

        response_received_at = time.time()

        content_text = result.get("content_text", "")
        tool_calls = result.get("tool_calls")
        finish_reason = result.get("finish_reason", "stop")

        response_body = build_openai_response(
            content_text=content_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

        timing = {
            "request_received_at": request_received_at,
            "response_received_at": response_received_at,
            "response_sent_at": time.time(),
        }
        worker_timing = result.get("timing")
        if isinstance(worker_timing, dict):
            timing.update(worker_timing)

        if span:
            span.set_attribute("finish_reason", finish_reason or "")
            rt = timing.get("relay_round_trip_ms")
            if rt is not None:
                span.set_attribute("relay_round_trip_ms", rt)

        node_id = result.get("node_id")
        gpu_id = result.get("gpu_id")
        worker_id_meta = result.get("worker_id")
        rank_info = result.get("rank_info")

        try:
            proxy.recorder.record_completion(
                session_id=session_id,
                messages=messages,
                completion_text=content_text,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                timing=timing,
                node_id=node_id,
                gpu_id=gpu_id,
                worker_id=worker_id_meta,
                rank_info=rank_info,
            )
        except SessionClosedError:
            logger.warning(
                "Session %s closed before recording (race), response still returned",
                session_id,
            )

        return JSONResponse(content=response_body)


# ---------------------------------------------------------------------------
# OpenAI response builder (public — used by both relay and local handlers)
# ---------------------------------------------------------------------------


def build_openai_response(
    *,
    content_text: str,
    tool_calls: list[dict[str, Any]] | None,
    finish_reason: str,
    token_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat completion response dict.

    This is a public helper so that framework-specific local-mode
    handlers can reuse it.
    """
    message: dict[str, Any] = {"role": "assistant"}
    if tool_calls:
        message["content"] = content_text.strip() if content_text and content_text.strip() else None
        message["tool_calls"] = tool_calls
    else:
        message["content"] = content_text
    logger.debug(
        "[OPENAI_RESP] Built response: has_tool_calls=%s, tool_calls_count=%d, content_len=%d, finish_reason=%s",
        bool(tool_calls),
        len(tool_calls) if tool_calls else 0,
        len(content_text) if content_text else 0,
        finish_reason,
    )
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            logger.debug("[OPENAI_RESP] Tool call[%d]: %s", i, json.dumps(tc)[:300])

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
            "completion_tokens": len(token_ids) if token_ids else 0,
            "total_tokens": len(token_ids) if token_ids else 0,
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
        debug: bool = False,
        session_dump_dir: str | None = None,
        session_persist_dir: str | None = None,
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
            debug: When ``True``, extra diagnostic endpoints are registered
                (e.g. ``GET /sessions`` to list all sessions).
            session_dump_dir: When set, session data is serialized to JSON
                and written to this directory before the session is deleted
                from memory.  Files are named ``{timestamp}_{session_id}.json``.
            session_persist_dir: When set, each turn is appended to a JSONL
                file in this directory as it is recorded.  On restart,
                sessions are reloaded from these files.
        """
        self.host = host
        self.port = port
        self.debug = debug
        self.session_dump_dir = session_dump_dir
        self.recorder = SessionRecorder(persist_dir=session_persist_dir)
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
