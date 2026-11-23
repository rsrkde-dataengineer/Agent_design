from typing import NotRequired

from langchain.agents import AgentState


class HelpdeskState(AgentState):
    """Conversation state. It is checkpointed after every step of the ReAct loop.

    `messages` comes from AgentState. Add a field here only when it has to survive
    between steps or turns. Per-request facts belong in RequestContext.
    """

    # Facts recalled about the customer at the start of each turn.
    memories: NotRequired[list[str]]

    # Why the conversation was handed to a human. Once set, it stays set for the thread.
    escalation: NotRequired[str]
