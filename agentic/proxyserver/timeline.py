"""Span-based timeline tracking for proxy sessions.

Each session has an associated SessionTimeline that stores TimelineSpan
objects with parent-child relationships, forming a trace tree.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any
from uuid import uuid4

from .models import SessionStats, SpanEvent, TimelineSpan

logger = logging.getLogger(__name__)


class SessionTimeline:
    """Thread-safe timeline for a single session.

    Manages spans (start/end/add-event) and maintains a list of
    WebSocket subscribers for live streaming.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._spans: list[TimelineSpan] = []
        self._active_spans: dict[str, TimelineSpan] = {}
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._sub_lock = threading.Lock()

    # -- Span lifecycle -----------------------------------------------

    def start_span(
        self,
        operation: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        span_id: str | None = None,
    ) -> str:
        """Start a new span. Returns the span_id."""
        sid = span_id or uuid4().hex
        span = TimelineSpan(
            span_id=sid,
            parent_id=parent_id,
            operation=operation,
            start_time=time.time(),
            attributes=attributes or {},
        )
        with self._lock:
            self._spans.append(span)
            self._active_spans[sid] = span
        self._notify_subscribers({"type": "span_started", "span": span.model_dump()})
        return sid

    def end_span(
        self,
        span_id: str,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End an active span, computing duration."""
        with self._lock:
            span = self._active_spans.pop(span_id, None)
            if span is None:
                return
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            span.status = status
            if attributes:
                span.attributes.update(attributes)
        self._notify_subscribers({"type": "span_ended", "span": span.model_dump()})

    def add_event(
        self,
        span_id: str,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add a sub-event to an existing span."""
        event = SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {},
        )
        with self._lock:
            span = self._active_spans.get(span_id)
            if span is not None:
                span.events.append(event)
        self._notify_subscribers(
            {
                "type": "span_event",
                "span_id": span_id,
                "event": event.model_dump(),
            }
        )

    def add_standalone_span(self, span: TimelineSpan) -> None:
        """Add a fully-formed span (for external/custom injection)."""
        if span.end_time and span.start_time and span.duration_ms is None:
            span.duration_ms = (span.end_time - span.start_time) * 1000
        with self._lock:
            self._spans.append(span)
        self._notify_subscribers({"type": "span_added", "span": span.model_dump()})

    # -- Query methods ------------------------------------------------

    def get_spans(
        self,
        since: float | None = None,
        operation: str | None = None,
    ) -> list[TimelineSpan]:
        """Return spans, optionally filtered by timestamp and operation."""
        with self._lock:
            spans = list(self._spans)
        if since is not None:
            spans = [s for s in spans if s.start_time >= since]
        if operation is not None:
            spans = [s for s in spans if s.operation == operation]
        return spans

    def compute_stats(self) -> SessionStats:
        """Compute aggregated statistics from all spans."""
        with self._lock:
            spans = list(self._spans)

        completion_spans = [
            s
            for s in spans
            if s.operation == "llm.completion" and s.duration_ms is not None
        ]
        latencies = [s.duration_ms for s in completion_spans]
        error_count = sum(1 for s in spans if s.status == "error")

        ops: dict[str, int] = {}
        for s in spans:
            ops[s.operation] = ops.get(s.operation, 0) + 1

        all_times = [s.start_time for s in spans]

        return SessionStats(
            session_id=self.session_id,
            turn_count=len(completion_spans),
            total_latency_ms=sum(latencies) if latencies else 0.0,
            avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
            min_latency_ms=min(latencies) if latencies else None,
            max_latency_ms=max(latencies) if latencies else None,
            span_count=len(spans),
            error_count=error_count,
            operations=ops,
            first_span_time=min(all_times) if all_times else None,
            last_span_time=max(all_times) if all_times else None,
        )

    # -- WebSocket subscriber management ------------------------------

    def subscribe(self) -> asyncio.Queue:
        """Register a new WebSocket subscriber. Returns a queue to read from."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._sub_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a WebSocket subscriber."""
        with self._sub_lock:
            self._subscribers.discard(q)

    def notify_session_deleted(self) -> None:
        """Push a sentinel to all subscribers so they can close gracefully."""
        self._notify_subscribers({"type": "session_deleted"})

    def _notify_subscribers(self, message: dict[str, Any]) -> None:
        """Push a message to all subscribers (non-blocking)."""
        with self._sub_lock:
            dead: list[asyncio.Queue] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subscribers.discard(q)
