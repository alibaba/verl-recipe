"""Thread-safe session data recorder for the LLM proxy."""

from __future__ import annotations

import logging
import threading
from typing import Any

from .models import CompletionRecord, SessionRecord, TimelineSpan
from .timeline import SessionTimeline

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Manages session records for the LLM proxy.

    Thread-safe: multiple proxy request handlers may record completions
    concurrently for different sessions.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._timelines: dict[str, SessionTimeline] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> None:
        """Create a new session for recording."""
        with self._lock:
            if session_id in self._sessions:
                logger.warning("Session %s already exists, resetting", session_id)
            self._sessions[session_id] = SessionRecord(session_id=session_id)
            timeline = SessionTimeline(session_id)
            self._timelines[session_id] = timeline
        timeline.start_span(
            operation="session.lifecycle",
            attributes={"event": "created"},
            span_id=f"{session_id}-lifecycle",
        )

    def record_completion(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        completion_text: str,
        token_ids: list[int],
        logprobs: list[float],
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record a single LLM completion for a session."""
        record = CompletionRecord(
            request_messages=messages,
            completion_text=completion_text,
            completion_token_ids=token_ids,
            completion_logprobs=logprobs,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(
                    "Session %s not found, auto-creating for recording", session_id
                )
                session = SessionRecord(session_id=session_id)
                self._sessions[session_id] = session
            session.turns.append(record)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve session data. Returns None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def mark_completed(self, session_id: str) -> None:
        """Mark a session as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.completed = True
            timeline = self._timelines.get(session_id)
        if timeline is not None:
            timeline.end_span(
                f"{session_id}-lifecycle",
                status="ok",
                attributes={"event": "completed"},
            )

    def delete_session(self, session_id: str) -> None:
        """Remove a session and free its memory."""
        with self._lock:
            self._sessions.pop(session_id, None)
            timeline = self._timelines.pop(session_id, None)
        if timeline is not None:
            timeline.notify_session_deleted()

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())

    # -- Timeline / span methods --------------------------------------

    def get_timeline(self, session_id: str) -> SessionTimeline | None:
        """Retrieve the timeline for a session."""
        with self._lock:
            return self._timelines.get(session_id)

    def start_span(
        self,
        session_id: str,
        operation: str,
        parent_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> str | None:
        """Start a timeline span for the given session. Returns span_id."""
        with self._lock:
            timeline = self._timelines.get(session_id)
        if timeline is None:
            return None
        return timeline.start_span(operation, parent_id, attributes)

    def end_span(
        self,
        session_id: str,
        span_id: str,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """End an active span."""
        with self._lock:
            timeline = self._timelines.get(session_id)
        if timeline is not None:
            timeline.end_span(span_id, status, attributes)

    def add_span_event(
        self,
        session_id: str,
        span_id: str,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add a sub-event to an active span."""
        with self._lock:
            timeline = self._timelines.get(session_id)
        if timeline is not None:
            timeline.add_event(span_id, name, attributes)

    def add_standalone_span(
        self,
        session_id: str,
        span: TimelineSpan,
    ) -> None:
        """Add a fully-formed span to a session's timeline."""
        with self._lock:
            timeline = self._timelines.get(session_id)
        if timeline is not None:
            timeline.add_standalone_span(span)
