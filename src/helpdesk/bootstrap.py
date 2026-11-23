"""Composition root: the one place that picks concrete classes and wires them together.

Everything else receives its dependencies. That is what lets tests swap in a scripted
model and in-memory storage without patching anything.
"""

from functools import partial

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from helpdesk.agents.billing import billing_agent
from helpdesk.agents.knowledge import knowledge_agent
from helpdesk.agents.orders import orders_agent
from helpdesk.agents.supervisor import supervisor_agent
from helpdesk.backends.contracts import Backends
from helpdesk.backends.demo import demo_backends
from helpdesk.core.factory import AgentFactory
from helpdesk.core.harness import AgentHarness
from helpdesk.core.registry import AgentRegistry
from helpdesk.middleware import standard_middleware
from helpdesk.settings import Settings
from helpdesk.storage import Storage
from helpdesk.storage.in_memory import in_memory_storage


def build_harness(
    settings: Settings,
    *,
    model: BaseChatModel | None = None,
    storage: Storage | None = None,
    backends: Backends | None = None,
) -> AgentHarness:
    model = model or build_model(settings)
    storage = storage or build_storage(settings)
    backends = backends or demo_backends()

    factory = AgentFactory(
        model=model,
        checkpointer=storage.checkpointer,
        store=storage.store,
        middleware=partial(
            standard_middleware,
            max_model_calls=settings.max_model_calls,
            model_retries=settings.model_retries,
        ),
    )
    orders = factory.create(orders_agent(backends.orders))
    billing = factory.create(billing_agent(backends.billing))
    knowledge = factory.create(knowledge_agent(backends.help_center))
    supervisor = factory.create(supervisor_agent([orders, billing, knowledge]))

    registry = AgentRegistry([supervisor, orders, billing, knowledge])
    return AgentHarness(registry=registry, results=storage.results)


def build_model(settings: Settings) -> BaseChatModel:
    return ChatAnthropic(
        model=settings.model,
        output_config={"effort": settings.effort},
        # If the model declines a request, the API retries it on a fallback model.
        betas=["server-side-fallback-2026-07-01"],
        model_kwargs={"fallbacks": "default"},
    )


def build_storage(settings: Settings) -> Storage:
    if settings.storage == "spanner":
        # Imported here so local runs don't need Google Cloud credentials.
        from helpdesk.storage.spanner import spanner_storage

        return spanner_storage(
            settings.spanner_project, settings.spanner_instance, settings.spanner_database
        )
    return in_memory_storage()
