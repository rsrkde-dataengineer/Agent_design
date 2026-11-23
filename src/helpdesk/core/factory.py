from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from helpdesk.core.context import RequestContext
from helpdesk.core.spec import Agent, AgentSpec
from helpdesk.core.state import HelpdeskState

# Decides which middleware an agent gets. Passed in, so tests can run agents bare.
MiddlewarePolicy = Callable[[AgentSpec], Sequence[AgentMiddleware]]


@dataclass(frozen=True, kw_only=True)
class AgentFactory:
    """Builds every agent the same way.

    The model, persistence and standard middleware are chosen once, here, instead of in
    each agent module. An agent module only describes what is unique to it.
    """

    model: BaseChatModel
    checkpointer: BaseCheckpointSaver
    store: BaseStore
    middleware: MiddlewarePolicy

    def create(self, spec: AgentSpec) -> Agent:
        graph = create_agent(
            self.model,
            tools=list(spec.tools),
            system_prompt=spec.system_prompt,
            middleware=[*self.middleware(spec), *spec.middleware],
            state_schema=HelpdeskState,
            context_schema=RequestContext,
            checkpointer=self.checkpointer,
            store=self.store,
            name=spec.name,
        )
        return Agent(name=spec.name, description=spec.description, graph=graph)
