"""Test doubles: a chat model that replays a script, and a results list."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from helpdesk.core.results import AgentResult


class ScriptedModel(BaseChatModel):
    """Answers each call with the next scripted message and records what it was shown.

    One script covers every agent in a run, in call order, because the supervisor and
    the specialists share the model.
    """

    replies: list[AIMessage | Exception] = Field(default_factory=list)
    prompts: list[list[BaseMessage]] = Field(default_factory=list)

    def script(self, *replies: AIMessage | Exception) -> None:
        self.replies.extend(replies)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.prompts.append(messages)
        if not self.replies:
            raise AssertionError("The model was called more often than the test scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])

    @property
    def _llm_type(self) -> str:
        return "scripted"


def call(tool: str, **args: Any) -> AIMessage:
    """A model turn that requests one tool call."""
    return AIMessage("", tool_calls=[{"name": tool, "args": args, "id": f"call_{uuid4().hex[:8]}"}])


def say(text: str) -> AIMessage:
    """A model turn that answers without calling tools."""
    return AIMessage(text)


class RecordedResults(list[AgentResult]):
    """A result repository that keeps every saved result in order."""

    def save(self, result: AgentResult) -> None:
        self.append(result)

    def get(self, run_id: str) -> AgentResult | None:
        return next((r for r in self if r.run_id == run_id), None)
