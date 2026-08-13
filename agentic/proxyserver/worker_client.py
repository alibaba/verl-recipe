"""Framework-side WebSocket client that connects to a standalone proxy server
and services inference requests.

When the proxy is deployed as a standalone service, it cannot call vLLM
directly.  Instead, the framework-side ``InferenceWorkerClient`` connects
to the proxy's ``/ws/worker`` endpoint and:

1. Receives ``completion_request`` messages from the proxy.
2. Performs inference using vLLM via Ray actor handles.
3. Sends ``completion_response`` messages back to the proxy.

The client supports **automatic reconnection** with configurable exponential
backoff, and can handle multiple concurrent inference requests over a single
WebSocket connection.

Usage::

    from recipe.agentic.proxyserver.worker_client import InferenceWorkerClient

    client = InferenceWorkerClient(
        proxy_ws_url="ws://proxy-host:8080/ws/worker",
        load_balancer=load_balancer,
        model_path="Qwen/Qwen2.5-7B-Instruct",
    )
    await client.start()   # Connects and starts serving requests
    # ...
    await client.stop()    # Graceful shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import os
from typing import Any
from uuid import uuid4

try:
    from .tracing import (
        attach_context,
        detach_context,
        extract_context,
        init_tracing,
        optional_span,
    )
except ImportError:
    from contextlib import contextmanager

    def init_tracing(service_name=""):
        return False

    @contextmanager
    def optional_span(name, attributes=None, kind=None):
        yield None

    def inject_context(carrier):
        pass

    def extract_context(carrier):
        return None

    def attach_context(ctx):
        return None

    def detach_context(token):
        pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InferenceWorkerClient:
    """WebSocket client that connects to the proxy and handles inference.

    The client manages the WebSocket lifecycle including automatic
    reconnection with exponential backoff.  Each incoming
    ``completion_request`` is processed concurrently via an asyncio task.
    """

    def __init__(
        self,
        proxy_ws_url: str,
        load_balancer: Any,
        model_path: str,
        tool_format: str | None = "hermes",
        tool_parser_factory=None,
        worker_id: str | None = None,
        reconnect_base_delay: float = 1.0,
        reconnect_max_delay: float = 60.0,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """
        Args:
            proxy_ws_url: WebSocket URL of the proxy, e.g.
                ``ws://proxy-host:8080/ws/worker``.
            load_balancer: Ray actor handle of verl's
                ``GlobalRequestLoadBalancer``. Inference requests are routed
                via ``acquire_server`` / ``release_server`` on this actor.
            model_path: HuggingFace model name/path for the tokenizer.
            tool_format: Tool-call format (e.g. ``"hermes"``).
            tool_parser_factory: Optional callable ``(format, tokenizer) -> tool_parser``.
                If not supplied, the client tries to import the default verl
                ToolParser as a convenience fallback.
            worker_id: Unique ID for this worker.  Auto-generated if None.
            reconnect_base_delay: Initial delay (seconds) before reconnecting.
            reconnect_max_delay: Maximum reconnection delay (seconds).
            heartbeat_interval: Seconds between heartbeat pings.
        """
        self.proxy_ws_url = proxy_ws_url
        self.load_balancer = load_balancer
        self.model_path = model_path
        self.tool_format = tool_format
        self.tool_parser_factory = tool_parser_factory
        self.worker_id = worker_id or f"worker-{uuid4().hex[:8]}"

        self.reconnect_base_delay = reconnect_base_delay
        self.reconnect_max_delay = reconnect_max_delay
        self.heartbeat_interval = heartbeat_interval

        self._ws = None
        self._running = False
        self._main_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None

        # Lazy-initialized inference components
        self._provider = None
        self._tokenizer = None

        # Local cache: session_id -> list of {token_ids, logprobs, completion_text}
        self._session_cache: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Lazy initialization of inference components
    # ------------------------------------------------------------------

    def _ensure_provider(self) -> None:
        """Initialize the vLLM provider and tokenizer on first use."""
        if self._provider is not None:
            return
        init_tracing(service_name="llm-proxy-worker")

        from transformers import AutoTokenizer

        from .vllm_provider import VLLMRayProvider

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        tool_parser = None
        if self.tool_format:
            if self.tool_parser_factory is not None:
                tool_parser = self.tool_parser_factory(
                    self.tool_format, self._tokenizer
                )
            else:
                try:
                    from verl.experimental.agent_loop.tool_parser import ToolParser
                    tool_parser = ToolParser.get_tool_parser(
                        self.tool_format, self._tokenizer
                    )
                except ImportError:
                    logger.warning(
                        "verl.experimental.agent_loop.tool_parser not available; "
                        "tool parsing disabled"
                    )

        self._provider = VLLMRayProvider(
            load_balancer=self.load_balancer,
            tokenizer=self._tokenizer,
            tool_parser=tool_parser,
        )
        logger.info(
            "Inference provider initialized (model=%s, tool_format=%s)",
            self.model_path, self.tool_format,
        )

    # ------------------------------------------------------------------
    # Session cache (token_ids / logprobs kept locally, not sent to proxy)
    # ------------------------------------------------------------------

    def get_session_cache(self, session_id: str) -> list[dict]:
        """Return cached turn data for *session_id* (token_ids + logprobs)."""
        return self._session_cache.get(session_id, [])

    def clear_session_cache(self, session_id: str) -> None:
        """Free cached turn data for *session_id*."""
        self._session_cache.pop(session_id, None)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def _do_inference(self, request: dict[str, Any]) -> dict[str, Any]:
        """Perform inference using vLLM and return a response dict."""
        self._ensure_provider()
        worker_received_at = time.time()

        messages = request.get("messages", [])
        session_id = request.get("session_id")
        temperature = request.get("temperature", 1.0)
        top_p = request.get("top_p", 1.0)
        max_tokens = request.get("max_tokens", 2048)
        stop = request.get("stop")
        tools = request.get("tools")
        repetition_penalty = request.get("repetition_penalty")

        logger.debug(
            "[WORKER_INFER] Request: session=%s, messages=%d, tools=%s (count=%d), "
            "max_tokens=%d",
            session_id, len(messages),
            "present" if tools else "absent",
            len(tools) if tools else 0,
            max_tokens,
        )

        try:
            inference_started_at = time.time()
            with optional_span("worker.inference") as inf_span:
                token_ids, log_probs, stop_reason, completion_text, server_id = (
                    await self._provider._generate(
                        messages=messages,
                        session_id=session_id,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop=stop,
                        tools=tools,
                        repetition_penalty=repetition_penalty,
                    )
                )
                inference_completed_at = time.time()
                if inf_span:
                    inf_span.set_attribute("token_count", len(token_ids))
                    inf_span.set_attribute("duration_ms", round((inference_completed_at - inference_started_at) * 1000, 2))
            logger.debug(
                "[WORKER_INFER] Generate returned: token_ids=%d, completion_text_len=%d, "
                "completion_preview=%r",
                len(token_ids),
                len(completion_text) if completion_text else 0,
                (completion_text[:500] if completion_text else "")[:500],
            )

            with optional_span("worker.tool_parse") as tp_span:
                content_text, tool_calls_list, finish_reason = (
                    await self._provider._parse_tool_calls(token_ids, completion_text)
                )
                tool_parse_completed_at = time.time()
                if tp_span:
                    tp_span.set_attribute("tool_calls_count", len(tool_calls_list) if tool_calls_list else 0)
            logger.debug(
                "[WORKER_INFER] Parse result: content_len=%d, tool_calls=%d, finish=%s",
                len(content_text) if content_text else 0,
                len(tool_calls_list) if tool_calls_list else 0,
                finish_reason,
            )
            if stop_reason in ("stop", "length"):
                finish_reason = stop_reason

            # Cache token_ids/logprobs locally — not sent to proxy
            if session_id:
                self._session_cache.setdefault(session_id, []).append({
                    "token_ids": token_ids,
                    "logprobs": list(log_probs),
                    "completion_text": content_text,
                })

            return {
                "completion_text": completion_text,
                "content_text": content_text,
                "tool_calls": tool_calls_list,
                "finish_reason": finish_reason,
                "error": None,
                "server_id": server_id,
                "timing": {
                    "worker_received_at": worker_received_at,
                    "worker_inference_started_at": inference_started_at,
                    "worker_inference_completed_at": inference_completed_at,
                    "worker_tool_parse_completed_at": tool_parse_completed_at,
                },
            }

        except Exception as e:
            logger.error("[WORKER_INFER] Inference failed: %s", e, exc_info=True)
            return {
                "completion_text": "",
                "content_text": "",
                "tool_calls": None,
                "finish_reason": "error",
                "error": str(e),
                "timing": {"worker_received_at": worker_received_at},
            }

    # ------------------------------------------------------------------
    # WebSocket communication
    # ------------------------------------------------------------------

    async def _handle_request(self, request: dict[str, Any]) -> None:
        """Process a single inference request and send the response."""
        request_id = request.get("request_id", "")
        logger.debug("Processing inference request %s", request_id)

        parent_ctx = extract_context(request)
        token = attach_context(parent_ctx)
        try:
            with optional_span(
                "worker.handle_request", attributes={"request_id": request_id}
            ):
                result = await self._do_inference(request)
        finally:
            detach_context(token)

        if isinstance(result.get("timing"), dict):
            result["timing"]["worker_response_sent_at"] = time.time()

        response = {
            "type": "completion_response",
            "request_id": request_id,
            "node_id": result.get("server_id"),
            "gpu_id": result.get("gpu_id"),
            "worker_id": self.worker_id,
            "rank_info": result.get("rank_info"),
            **{k: v for k, v in result.items() if k not in ("server_id", "gpu_id", "rank_info")},
        }

        try:
            if self._ws is not None:
                await self._ws.send(json.dumps(response))
        except Exception as e:
            logger.error(
                "Failed to send response for request %s: %s", request_id, e
            )

    async def _run_heartbeat(self) -> None:
        """Periodically send ping messages to keep the connection alive."""
        try:
            while self._running and self._ws is not None:
                await asyncio.sleep(self.heartbeat_interval)
                if self._ws is not None:
                    try:
                        await self._ws.send(json.dumps({"type": "ping"}))
                    except Exception:
                        break
        except asyncio.CancelledError:
            pass

    async def _connect_and_serve(self) -> None:
        """Connect to the proxy and serve inference requests.

        This method handles the full lifecycle of a single WebSocket
        connection.  It returns when the connection is lost.
        """
        import websockets

        logger.debug(
            "Connecting to proxy at %s (worker_id=%s)",
            self.proxy_ws_url, self.worker_id,
        )

        async with websockets.connect(
            self.proxy_ws_url,
            ping_interval=None,  # We handle our own heartbeat
            max_size=100 * 1024 * 1024,  # 100 MB max message size
        ) as ws:
            self._ws = ws

            # Send handshake
            await ws.send(json.dumps({
                "type": "worker_hello",
                "worker_id": self.worker_id,
            }))

            # Wait for ack
            ack_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            ack = json.loads(ack_raw)
            if ack.get("type") != "worker_hello_ack":
                raise RuntimeError(f"Unexpected handshake response: {ack}")

            logger.info("Connected to proxy, worker %s acknowledged", self.worker_id)

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._run_heartbeat())

            # Main receive loop — handle messages concurrently
            tasks: set[asyncio.Task] = set()
            try:
                async for raw_msg in ws:
                    data = json.loads(raw_msg)
                    msg_type = data.get("type")

                    if msg_type == "completion_request":
                        task = asyncio.create_task(self._handle_request(data))
                        tasks.add(task)
                        task.add_done_callback(tasks.discard)

                    elif msg_type == "pong":
                        pass  # heartbeat response

                    elif msg_type == "ping":
                        await ws.send(json.dumps({"type": "pong"}))

                    else:
                        logger.warning("Unknown message type: %s", msg_type)

            finally:
                # Cancel heartbeat
                if self._heartbeat_task and not self._heartbeat_task.done():
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
                # Wait for in-flight inference tasks
                if tasks:
                    logger.debug(
                        "Waiting for %d in-flight inference tasks...", len(tasks)
                    )
                    await asyncio.gather(*tasks, return_exceptions=True)

            self._ws = None

    async def _run_with_reconnect(self) -> None:
        """Run the worker with automatic reconnection on disconnect."""
        consecutive_failures = 0

        while self._running:
            try:
                await self._connect_and_serve()
                if self._running:
                    # Connection was lost but we should keep running
                    consecutive_failures += 1
                    delay = min(
                        self.reconnect_base_delay * (2 ** (consecutive_failures - 1)),
                        self.reconnect_max_delay,
                    )
                    logger.warning(
                        "Connection to proxy lost. Reconnecting in %.1fs "
                        "(attempt %d)...",
                        delay, consecutive_failures,
                    )
                    await asyncio.sleep(delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                consecutive_failures += 1
                delay = min(
                    self.reconnect_base_delay * (2 ** (consecutive_failures - 1)),
                    self.reconnect_max_delay,
                )
                logger.warning(
                    "Worker connection error: %s. Reconnecting in %.1fs "
                    "(attempt %d)...",
                    e, delay, consecutive_failures,
                )
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the worker client.  Connects to the proxy and begins
        serving inference requests.

        This method returns immediately — the actual work happens in a
        background asyncio task.
        """
        if self._running:
            logger.warning("Worker client is already running")
            return

        self._running = True
        self._main_task = asyncio.create_task(self._run_with_reconnect())
        logger.info("Inference worker client started (id=%s)", self.worker_id)

    async def stop(self) -> None:
        """Gracefully stop the worker client."""
        self._running = False

        # Close WebSocket
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        # Cancel main task
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass

        logger.info("Inference worker client stopped (id=%s)", self.worker_id)

    @property
    def is_connected(self) -> bool:
        """Whether the worker is currently connected to the proxy."""
        return self._ws is not None and self._running
