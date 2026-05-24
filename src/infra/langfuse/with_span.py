from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, TypedDict, TypeVar

from langfuse import LangfuseSpan, get_client, propagate_attributes


class TraceInit(TypedDict, total=False):
    name: str | None
    user_id: str | None
    session_id: str | None
    version: str | None
    metadata: Any | None
    tags: list[str] | None
    public: bool | None


class WithSpanContext(TypedDict, total=False):
    """
    Context for starting a Langfuse span.
    Usage:
    - If you want to create a new trace, provide only `trace_init`.
    ```
    span_context={
        "trace_init": {
            "name": "my_trace",
            "user_id": "user123",
            "session_id": "session456",
            "metadata": {"key": "value"},
        },
    }
    ```
    - If you want to create a child span under an existing span, provide `parent_span`.
    ```
    with some_span.start_as_current_observation(name="parent_span") as parent_span:
        span_context={
            "parent_span": parent_span,
        }
    ```
    """

    parent_span: LangfuseSpan | None
    trace_init: TraceInit | None


T = TypeVar("T")


@dataclass(frozen=True)
class ObservationHandle:
    span: LangfuseSpan

    def set_input(self, input: Any) -> None:
        self.span.update(input=input)

    def set_output(self, output: Any) -> None:
        self.span.update(output=output)

    def finish(self, value: T) -> T:
        self.set_output(value)
        return value

    def error(self, error: Exception) -> None:
        self.span.update(level="ERROR", status_message=str(error))


@contextmanager
def with_langfuse_span(
    span_name: str,
    span_context: WithSpanContext | None = None,
) -> Iterator[ObservationHandle]:
    """Create a Langfuse span (observation) and optionally initialize the trace.

    Usage:
    - To create a new trace:
    ```
    with with_langfuse_span(
        span_name="my_span",
        span_context={
            "trace_init": {
                "name": "my_trace",
                "user_id": "user123",
                "session_id": "session456",
                "metadata": {"key": "value"},
            },
        },
    ) as obs:
        obs.set_input({"input_key": "input_value"})
        # ... do work ...
        obs.set_output({"output_key": "output_value"})
    ```
    """

    parent_span = span_context.get("parent_span") if span_context else None
    trace_init = span_context.get("trace_init") if span_context else None

    from contextlib import ExitStack

    with ExitStack() as stack:
        if trace_init is not None:
            stack.enter_context(
                propagate_attributes(
                    trace_name=trace_init.get("name"),
                    user_id=trace_init.get("user_id"),
                    session_id=trace_init.get("session_id"),
                    metadata=trace_init.get("metadata"),
                    tags=trace_init.get("tags"),
                    version=trace_init.get("version"),
                )
            )

        if parent_span is not None:
            cm = parent_span.start_as_current_observation(
                name=span_name,
                as_type="span",
            )
        else:
            langfuse = get_client()
            cm = langfuse.start_as_current_observation(
                name=span_name,
                as_type="span",
            )

        span = stack.enter_context(cm)

        if trace_init is not None and trace_init.get("public"):
            span.set_trace_as_public()

        yield ObservationHandle(span=span)
