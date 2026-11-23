"""Runs one turn of a conversation against a named agent.

The harness owns everything around the ReAct loop: thread and run ids, the runtime
context, tracing metadata, and saving the outcome. Agents know nothing about any of it.
"""

import logging
from dataclasses import dataclass
from functools import partial
from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from helpdesk.core.context import RequestContext
from helpdesk.core.registry import AgentRegistry
from helpdesk.core.results import AgentResult, ResultRepository, RunStatus

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentHarness:
    registry: AgentRegistry
    results: ResultRepository

    def run(
        self,
        agent_name: str,
        message: str,
        context: RequestContext,
        *,
        thread_id: str | None = None,
    ) -> AgentResult:
        """Send `message` to the agent and return its answer.

        Pass the `thread_id` from an earlier result to continue that conversation.
        """
        agent = self.registry.get(agent_name)
        thread_id = thread_id or str(uuid4())
        run_id = uuid4()
        config: RunnableConfig = {
            "run_id": run_id,  # the same id shows up in traces and in the results table
            "run_name": agent.name,
            "configurable": {"thread_id": thread_id},
            "metadata": {"customer_id": context.customer_id},
        }
        record = partial(
            AgentResult,
            run_id=str(run_id),
            thread_id=thread_id,
            agent=agent.name,
            customer_id=context.customer_id,
        )

        try:
            state = agent.graph.invoke(
                {"messages": [HumanMessage(message)]}, config=config, context=context
            )
        except Exception as exc:
            log.exception("Run %s on thread %s failed", run_id, thread_id)
            self.results.save(record(status=RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}"))
            raise

        status = RunStatus.ESCALATED if state.get("escalation") else RunStatus.COMPLETED
        result = record(status=status, answer=state["messages"][-1].text)
        self.results.save(result)
        return result
