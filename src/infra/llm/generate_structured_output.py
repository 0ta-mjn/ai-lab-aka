from typing import Any, Dict, Literal, Optional, Type, TypeVar

import litellm
from langfuse import get_client
from pydantic import BaseModel

from .registry import ModelName, get_model

T = TypeVar("T", bound=BaseModel)


def generate_structured_output(
    model: ModelName,
    system_prompt: Optional[str],
    prompt: str,
    output_schema: Type[T],
    generation_name: str,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"]
    ] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> T:
    """
    Generate structured output using LiteLLM and trace as a Langfuse Generation.

    Args:
        model: The model name (suggested from registry).
        system_prompt: Optional system prompt.
        prompt: User prompt.
        output_schema: Pydantic model class defining the structure.
        generation_name: Name of the Langfuse generation.
        metadata: Optional metadata for the Langfuse generation.
        max_tokens: The maximum number of tokens to generate.
        reasoning_effort: The reasoning effort for reasoning models.

    Returns:
        Instance of output_schema.
    """
    client = get_client()
    model_adapter = get_model(model)

    if metadata is None:
        metadata = {}
    if max_tokens is not None:
        metadata["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        metadata["reasoning_effort"] = reasoning_effort
    metadata["output_schema"] = output_schema.model_json_schema()

    # Start a generation
    with client.start_as_current_observation(
        name=generation_name,
        as_type="generation",
        model=model_adapter.get_langfuse_model_name(),
        input={
            "system": system_prompt,
            "prompt": prompt,
        },
        metadata=metadata,
    ) as generation:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = litellm.completion(
                model=model_adapter.get_litellm_model_name(),
                messages=messages,
                response_format=output_schema,
                drop_params=True,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_tokens,
            )

            # Extract usage
            if hasattr(response, "usage"):
                # litellm Usage object
                usage = response.usage
                generation.update(
                    usage_details={
                        "input": getattr(usage, "prompt_tokens", 0),
                        "output": getattr(usage, "completion_tokens", 0),
                        "total": getattr(usage, "total_tokens", 0),
                    }
                )

            # Parse output
            content = response.choices[0].message.content

            # If content is None or empty, check tool calls
            if not content and hasattr(response.choices[0].message, "tool_calls"):
                tool_calls = response.choices[0].message.tool_calls
                if tool_calls:
                    content = tool_calls[0].function.arguments

            if not content:
                raise ValueError("No content received from LLM")

            # Validate/Parse
            if isinstance(content, dict):
                parsed_output = output_schema.model_validate(content)
            else:
                parsed_output = output_schema.model_validate_json(content)

            # Update generation with output
            generation.update(output=parsed_output.model_dump())

            return parsed_output

        except Exception as e:
            generation.update(status_message=str(e), level="ERROR")
            raise
