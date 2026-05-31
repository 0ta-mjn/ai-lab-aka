import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from agents import Agent, AgentOutputSchema, Runner, Tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import ToolCallItem, ToolCallOutputItem
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from pydantic import BaseModel

from src.infra.langfuse.with_span import WithSpanContext, with_langfuse_span
from src.infra.llm.registry import ModelName, get_model

T = TypeVar("T", bound=BaseModel)

_AGENTS_INSTRUMENTED = False


@dataclass(frozen=True)
class ToolExecution:
    tool_name: str | None
    arguments: Any
    output: Any
    call_id: str | None


@dataclass(frozen=True)
class AgentRunResult(Generic[T]):
    final_output: T
    tool_executions: list[ToolExecution]


def run_agent_sync(
    model: ModelName,
    system_prompt: str,
    prompt: str,
    tools: List[Tool],
    output_schema: Type[T],
    generation_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    span_context: WithSpanContext | None = None,
) -> AgentRunResult[T]:
    """
    Run an OpenAI Agent synchronously, wrapped in a Langfuse span.
    Abstracts away the Agent SDK from the caller.

    Args:
        model: The model name (from registry).
        system_prompt: Instructions for the agent.
        prompt: Initial user prompt/input for the agent.
        tools: List of function tools available to the agent.
        output_schema: Pydantic model class defining the final structure.
        generation_name: Name of the Langfuse span.
        metadata: Optional metadata for the Langfuse span.
        span_context: Optional context for Langfuse tracing.

    Returns:
        The final output and tool execution history.
    """
    model_adapter = get_model(model)
    agent_model = model_adapter.get_litellm_model_name()

    # Instrument openai-agents with OpenTelemetry for Langfuse.
    global _AGENTS_INSTRUMENTED
    if not _AGENTS_INSTRUMENTED:
        OpenAIAgentsInstrumentor().instrument()
        _AGENTS_INSTRUMENTED = True

    with with_langfuse_span(
        span_name=generation_name,
        span_context=span_context,
    ) as obs:
        try:
            obs.set_input(
                {
                    "system": system_prompt,
                    "prompt": prompt,
                }
            )

            agent = Agent(
                name=generation_name,
                instructions=system_prompt,
                tools=tools,
                model=LitellmModel(model=agent_model),
                output_type=AgentOutputSchema(output_schema, strict_json_schema=False),
            )

            result = Runner.run_sync(agent, input=prompt)
            tool_executions = _extract_tool_executions(result.new_items)
            agent_result = AgentRunResult(
                final_output=result.final_output,
                tool_executions=tool_executions,
            )

            obs.set_output(
                {
                    "final_output": result.final_output.model_dump()
                    if isinstance(result.final_output, BaseModel)
                    else result.final_output,
                    "tool_executions": [
                        asdict(execution) for execution in tool_executions
                    ],
                }
            )
            return agent_result
        except Exception as e:
            obs.error(e)
            raise


def _extract_tool_executions(items: list[Any]) -> list[ToolExecution]:
    calls_by_id: dict[str | None, dict[str, Any]] = {}
    call_order: list[str | None] = []

    for item in items:
        if isinstance(item, ToolCallItem):
            call_id = item.call_id
            calls_by_id[call_id] = {
                "tool_name": item.tool_name,
                "arguments": _extract_tool_arguments(item.raw_item),
                "output": None,
                "call_id": call_id,
            }
            call_order.append(call_id)
            continue

        if isinstance(item, ToolCallOutputItem):
            call_id = item.call_id
            if call_id not in calls_by_id:
                calls_by_id[call_id] = {
                    "tool_name": None,
                    "arguments": None,
                    "output": item.output,
                    "call_id": call_id,
                }
                call_order.append(call_id)
            else:
                calls_by_id[call_id]["output"] = item.output

    return [
        ToolExecution(**calls_by_id[call_id])
        for call_id in call_order
        if call_id in calls_by_id
    ]


def _extract_tool_arguments(raw_item: Any) -> Any:
    if isinstance(raw_item, dict):
        arguments = raw_item.get("arguments")
    else:
        arguments = getattr(raw_item, "arguments", None)

    if not isinstance(arguments, str):
        return arguments

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
