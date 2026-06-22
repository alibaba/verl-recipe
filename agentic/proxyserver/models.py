"""Data models for the LLM proxy session recording."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class CompletionRecord(BaseModel):
    """Record of a single LLM completion call captured by the proxy."""

    request_messages: list[dict[str, Any]] = Field(
        description="Full messages array sent in this request"
    )
    completion_text: str = Field(description="Generated completion text")
    completion_token_ids: list[int] = Field(
        description="Token IDs of the generated completion"
    )
    completion_logprobs: list[float] = Field(
        description="Log probabilities for each generated token"
    )
    finish_reason: str | None = Field(
        default=None, description="Reason generation stopped: stop, tool_calls, length"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Parsed tool calls from the completion, if any"
    )


class SessionRecord(BaseModel):
    """All recorded data for a single agent session (one rollout)."""

    session_id: str
    turns: list[CompletionRecord] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    completed: bool = False


class SpanEvent(BaseModel):
    """A sub-event within a timeline span."""

    name: str
    timestamp: float = Field(default_factory=time.time)
    attributes: dict[str, Any] = Field(default_factory=dict)


class TimelineSpan(BaseModel):
    """A timed span representing an operation in a session's timeline."""

    span_id: str = Field(description="Unique span identifier (UUID hex)")
    parent_id: str | None = Field(
        default=None, description="Parent span ID, or None for root spans"
    )
    operation: str = Field(
        description="Operation type: 'llm.completion', 'session.lifecycle', 'custom'"
    )
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = Field(default=None)
    duration_ms: float | None = Field(
        default=None, description="Computed duration in milliseconds"
    )
    attributes: dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="ok", description="'ok' or 'error'")
    events: list[SpanEvent] = Field(default_factory=list)


class SessionStats(BaseModel):
    """Aggregated statistics for a session's timeline."""

    session_id: str
    turn_count: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float | None = None
    max_latency_ms: float | None = None
    span_count: int = 0
    error_count: int = 0
    operations: dict[str, int] = Field(
        default_factory=dict, description="Count per operation type"
    )
    first_span_time: float | None = None
    last_span_time: float | None = None
