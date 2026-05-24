from typing import Any, Dict, List, Optional, Type, TypeVar

from agents import Agent, AgentOutputSchema, Runner, Tool
from agents.extensions.models.litellm_model import LitellmModel
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from pydantic import BaseModel

from src.infra.langfuse.with_span import WithSpanContext, with_langfuse_span
from src.infra.llm.registry import ModelName, get_model

T = TypeVar("T", bound=BaseModel)


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
) -> T:
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
        The final output of the agent parsed into output_schema.
    """
    model_adapter = get_model(model)
    agent_model = model_adapter.get_litellm_model_name()

    # Instrument openai-agents with OpenTelemetry for Langfuse
    OpenAIAgentsInstrumentor().instrument()

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

            return obs.finish(result.final_output)
        except Exception as e:
            obs.error(e)
            raise
