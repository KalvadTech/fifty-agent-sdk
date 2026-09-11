"""LLM contract subpackage.

Public surface: the :class:`LLMClient` protocol, an
:class:`OpenAICompatibleClient` adapter, and the Pydantic v2 message,
request, and response types.
"""

from fifty_agent_sdk.llm.openai_compat import OpenAICompatibleClient
from fifty_agent_sdk.llm.protocol import LLMClient
from fifty_agent_sdk.llm.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    FinishReason,
    Role,
    ToolCall,
    ToolChoiceFunction,
    ToolChoiceFunctionName,
    Usage,
)

__all__ = [
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "FinishReason",
    "LLMClient",
    "OpenAICompatibleClient",
    "Role",
    "ToolCall",
    "ToolChoiceFunction",
    "ToolChoiceFunctionName",
    "Usage",
]
