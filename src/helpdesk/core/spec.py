from collections.abc import Sequence
from dataclasses import dataclass

from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph


@dataclass(frozen=True)
class AgentSpec:
    """What makes one agent different from the others.

    Shared concerns (model, persistence, standard middleware) are added by AgentFactory,
    so a spec stays short and declarative.
    """

    name: str
    description: str  # what the supervisor reads when this agent is offered as a tool
    system_prompt: str
    tools: Sequence[BaseTool]
    middleware: Sequence[AgentMiddleware] = ()  # extras on top of the standard stack


@dataclass(frozen=True)
class Agent:
    """A built agent, ready to run."""

    name: str
    description: str
    graph: CompiledStateGraph
