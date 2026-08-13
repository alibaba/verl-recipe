"""Optional OpenTelemetry tracing for the LLM proxy.

All OTel imports are conditional.  When ``opentelemetry`` packages are not
installed the module exports no-op stubs so callers can use the same API
unconditionally.

Enable tracing by:

1. Installing packages::

       pip install opentelemetry-api opentelemetry-sdk \\
           opentelemetry-exporter-otlp-proto-grpc

2. Setting env vars::

       OTEL_ENABLED=1
       OTEL_SERVICE_NAME=llm-proxy          # optional, default "llm-proxy"
       OTEL_EXPORTER_OTLP_ENDPOINT=...      # optional, falls back to console
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, StatusCode
    from opentelemetry.trace.propagation import TraceContextTextMapPropagator

    _OTEL_AVAILABLE = True
except ImportError:
    SpanKind = None  # type: ignore[assignment,misc]
    StatusCode = None  # type: ignore[assignment,misc]


def init_tracing(service_name: str = "llm-proxy") -> bool:
    """Initialize OTel tracing if packages are available and enabled.

    Returns ``True`` if tracing was successfully initialized.
    """
    global _tracer
    if not _OTEL_AVAILABLE:
        return False

    enabled = os.environ.get("OTEL_ENABLED", "").lower() in ("1", "true")
    has_service = bool(os.environ.get("OTEL_SERVICE_NAME"))
    if not (enabled or has_service):
        return False

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        exporter = OTLPSpanExporter()
    except ImportError:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        exporter = ConsoleSpanExporter()

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("llm-proxy")
    logger.info("OpenTelemetry tracing initialized (service=%s)", service_name)
    return True


def get_tracer():
    """Return the tracer, or ``None`` if tracing is not initialized."""
    return _tracer


@contextmanager
def optional_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    kind: Any = None,
) -> Generator[Any, None, None]:
    """Context manager that creates a span when tracing is active, else no-ops."""
    if _tracer is None:
        yield None
        return

    span_kind = kind if kind is not None else SpanKind.INTERNAL
    with _tracer.start_as_current_span(
        name, kind=span_kind, attributes=attributes or {}
    ) as span:
        yield span


def inject_context(carrier: dict[str, Any]) -> None:
    """Inject current trace context into *carrier* (e.g. a WebSocket message dict)."""
    if not _OTEL_AVAILABLE or _tracer is None:
        return
    TraceContextTextMapPropagator().inject(carrier)


def extract_context(carrier: dict[str, Any]) -> Any:
    """Extract trace context from *carrier* and return an OTel context.

    Returns ``None`` when tracing is not available.  Callers should use
    ``attach_context`` to make the extracted context current.
    """
    if not _OTEL_AVAILABLE or _tracer is None:
        return None
    return TraceContextTextMapPropagator().extract(carrier)


def attach_context(ctx: Any) -> Any:
    """Attach an extracted context, returning a token for ``detach_context``."""
    if ctx is None or not _OTEL_AVAILABLE:
        return None
    return otel_context.attach(ctx)


def detach_context(token: Any) -> None:
    """Detach a previously attached context."""
    if token is None or not _OTEL_AVAILABLE:
        return
    otel_context.detach(token)
