from abc import ABC, abstractmethod
from typing import Literal

Provider = Literal["openai", "google"]


class BaseLLMModel(ABC):
    def __init__(self, model_name: str):
        self._model_name = model_name

    @abstractmethod
    def get_litellm_model_name(self) -> str:
        pass

    @abstractmethod
    def get_langfuse_model_name(self) -> str:
        pass


class OpenAILLMModel(BaseLLMModel):
    def get_litellm_model_name(self) -> str:
        return f"openai/{self._model_name}"

    def get_langfuse_model_name(self) -> str:
        return f"openai/{self._model_name}"


class GoogleLLMModel(BaseLLMModel):
    def get_litellm_model_name(self) -> str:
        # LiteLLM uses 'gemini/' prefix specifically for Google Generative AI models
        return f"gemini/{self._model_name}"

    def get_langfuse_model_name(self) -> str:
        # Langfuse typically uses 'google/' or similar standardized provider names
        return f"google/{self._model_name}"


ModelName = Literal[
    "gemini/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
]

MODELS: dict[ModelName, BaseLLMModel] = {
    "gemini/gemini-2.5-flash-lite": GoogleLLMModel("gemini-2.5-flash-lite"),
    "openai/gpt-5-mini": OpenAILLMModel("gpt-5-mini"),
}


def get_model(model_name: ModelName) -> BaseLLMModel:
    return MODELS[model_name]
