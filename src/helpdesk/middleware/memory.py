from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from helpdesk.core.context import RequestContext
from helpdesk.core.memory import recall
from helpdesk.core.state import HelpdeskState


class LoadMemoriesMiddleware(AgentMiddleware):
    """Loads what we remember about the customer into state, once per turn.

    Doing it in before_agent, not before every model call, keeps the ReAct loop from
    reading the store on each step.
    """

    state_schema = HelpdeskState

    def before_agent(
        self, state: HelpdeskState, runtime: Runtime[RequestContext]
    ) -> dict[str, Any] | None:
        if runtime.store is None:
            return None
        return {"memories": recall(runtime.store, runtime.context.customer_id)}
