import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from helpdesk.backends.contracts import Backends
from helpdesk.backends.demo import demo_backends
from helpdesk.bootstrap import build_harness
from helpdesk.core.context import RequestContext
from helpdesk.core.harness import AgentHarness
from helpdesk.settings import Settings
from helpdesk.storage import Storage
from tests.fakes import RecordedResults, ScriptedModel


@pytest.fixture
def customer() -> RequestContext:
    return RequestContext(customer_id="c-100", customer_name="Sam")


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel()


@pytest.fixture
def storage() -> Storage:
    return Storage(checkpointer=InMemorySaver(), store=InMemoryStore(), results=RecordedResults())


@pytest.fixture
def backends() -> Backends:
    return demo_backends()


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, max_model_calls=6, model_retries=0)


@pytest.fixture
def harness(settings, model, storage, backends) -> AgentHarness:
    return build_harness(settings, model=model, storage=storage, backends=backends)
