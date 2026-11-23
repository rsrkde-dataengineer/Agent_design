from collections.abc import Iterable

from helpdesk.core.errors import UnknownAgentError
from helpdesk.core.spec import Agent


class AgentRegistry:
    """Finds built agents by name, so callers never import a specific agent."""

    def __init__(self, agents: Iterable[Agent]) -> None:
        self.agents: dict[str, Agent] = {}
        for agent in agents:
            if agent.name in self.agents:
                raise ValueError(f"Two agents are named {agent.name!r}")
            self.agents[agent.name] = agent

    def get(self, name: str) -> Agent:
        if name not in self.agents:
            raise UnknownAgentError(name, available=self.names())
        return self.agents[name]

    def names(self) -> list[str]:
        return sorted(self.agents)
