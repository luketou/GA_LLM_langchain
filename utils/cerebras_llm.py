import os
from typing import Any, List, Optional, Sequence

from cerebras.cloud.sdk import Cerebras
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "function",
}


class CerebrasChatLLM(BaseChatModel):
    """LangChain-compatible chat model backed by the Cerebras SDK."""

    model: str = "gpt-oss-120b"
    temperature: float = 0.1
    api_key: Optional[str] = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        resolved_key = self.api_key or os.environ.get("CEREBRAS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "A Cerebras API key must be provided via the constructor or the CEREBRAS_API_KEY environment variable."
            )
        self._client = Cerebras(api_key=resolved_key)

    @property
    def _llm_type(self) -> str:
        return "cerebras-chat"

    def _format_messages(self, messages: Sequence[BaseMessage]) -> List[dict[str, str]]:
        formatted: List[dict[str, str]] = []
        for message in messages:
            role = _ROLE_MAP.get(message.type, "user")
            content = message.content if hasattr(message, "content") else str(message)
            formatted.append({"role": role, "content": content})
        return formatted

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        request_payload = {
            "messages": self._format_messages(messages),
            "model": self.model,
            "temperature": self.temperature,
        }

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens
        if stop:
            request_payload["stop"] = stop

        response = self._client.chat.completions.create(**request_payload)

        first_choice = response.choices[0] if response.choices else None
        message_obj = getattr(first_choice, "message", None) if first_choice else None
        if isinstance(message_obj, dict):
            content = message_obj.get("content", "")
        else:
            content = getattr(message_obj, "content", "") if message_obj else ""
        ai_message = AIMessage(content=content)

        llm_output: dict[str, Any] = {"model": self.model}
        usage = getattr(response, "usage", None)
        if usage:
            usage_dict = {}
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, attr, None)
                if value is not None:
                    usage_dict[attr] = value
            if usage_dict:
                llm_output["usage"] = usage_dict

        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation], llm_output=llm_output)

    def invoke(self, input: Any, **kwargs: Any) -> AIMessage:
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        return super().invoke(messages, **kwargs)
