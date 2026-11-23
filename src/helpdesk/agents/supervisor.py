"""Supervisor: talks to the customer and delegates to the specialist agents."""

from collections.abc import Sequence
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from helpdesk.core.context import RequestContext
from helpdesk.core.memory import remember
from helpdesk.core.spec import Agent, AgentSpec
from helpdesk.prompts import load_prompt


def supervisor_agent(specialists: Sequence[Agent]) -> AgentSpec:
    return AgentSpec(
        name="supervisor",
        description="Front-line assistant. Talks to the customer and delegates to specialists.",
        system_prompt=load_prompt("supervisor"),
        tools=[*(_delegate_to(agent) for agent in specialists), remember_fact, escalate_to_human],
    )


def _delegate_to(specialist: Agent) -> BaseTool:
    """Offer a specialist as a tool. It runs its own ReAct loop and returns one answer."""

    @tool(f"ask_{specialist.name}_agent", description=specialist.description)
    def ask(
        task: Annotated[str, "A complete, self-contained request, including any ids."],
        runtime: ToolRuntime[RequestContext],
    ) -> str:
        result = specialist.graph.invoke(
            {"messages": [HumanMessage(task)]}, context=runtime.context
        )
        return result["messages"][-1].text

    return ask


@tool
def remember_fact(fact: str, runtime: ToolRuntime[RequestContext]) -> str:
    """Save a lasting fact or preference about the customer for future conversations."""
    remember(runtime.store, runtime.context.customer_id, fact)
    return "Saved."


@tool
def escalate_to_human(reason: str, runtime: ToolRuntime[RequestContext]) -> Command:
    """Hand the conversation to a human agent. Give a short reason a person can act on."""
    return Command(
        update={
            "escalation": reason,
            "messages": [
                ToolMessage(
                    "A human agent will follow up with the customer.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )
