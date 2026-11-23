"""The Spanner adapters, run against the emulator.

docker run -d -p 9010:9010 gcr.io/cloud-spanner-emulator/emulator
SPANNER_EMULATOR_HOST=localhost:9010 uv run pytest -m spanner
"""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
from google.cloud import spanner

import helpdesk.storage.spanner as spanner_storage
from helpdesk.core.results import RunStatus
from helpdesk.storage import Storage
from helpdesk.storage.spanner.checkpointer import SpannerCheckpointer
from helpdesk.storage.spanner.results import SpannerResultRepository
from helpdesk.storage.spanner.store import SpannerStore
from tests.fakes import call, say

pytestmark = [
    pytest.mark.spanner,
    pytest.mark.skipif(
        not os.environ.get("SPANNER_EMULATOR_HOST"), reason="needs the Spanner emulator"
    ),
]

PROJECT = "test-project"
SCHEMA = Path(spanner_storage.__file__).with_name("schema.sql")


def schema_statements() -> list[str]:
    lines = [line for line in SCHEMA.read_text().splitlines() if not line.startswith("--")]
    return [statement.strip() for statement in "\n".join(lines).split(";") if statement.strip()]


@pytest.fixture(scope="module")
def database():
    instance = spanner.Client(project=PROJECT).instance(
        "test-instance",
        configuration_name=f"projects/{PROJECT}/instanceConfigs/emulator-config",
        display_name="Test",
        node_count=1,
    )
    if not instance.exists():
        instance.create().result(timeout=60)
    db = instance.database(f"helpdesk-{uuid4().hex[:8]}", ddl_statements=schema_statements())
    db.create().result(timeout=120)
    yield db
    db.drop()


@pytest.fixture
def storage(database) -> Storage:
    return Storage(
        checkpointer=SpannerCheckpointer(database),
        store=SpannerStore(database),
        results=SpannerResultRepository(database),
    )


def test_a_conversation_survives_in_spanner(harness, model, storage, customer):
    model.script(
        call("get_order", order_id="A-1001"),
        say("It arrives on 26 November."),
        say("You asked about order A-1001."),
    )

    first = harness.run("orders", "When does A-1001 arrive?", customer)
    second = harness.run("orders", "What did I ask?", customer, thread_id=first.thread_id)

    restored = [type(message).__name__ for message in model.prompts[-1][1:]]  # skip system
    assert restored == [
        "HumanMessage",  # first question
        "AIMessage",  # get_order call
        "ToolMessage",  # order details
        "AIMessage",  # first answer
        "HumanMessage",  # second question
    ]
    assert storage.results.get(second.run_id) == second


def test_delegation_checkpoints_nested_agents(harness, model, customer):
    model.script(
        call("ask_orders_agent", task="Status of A-1001?"),
        call("get_order", order_id="A-1001"),
        say("Shipped."),
        say("Your order has shipped."),
    )

    result = harness.run("supervisor", "Where is A-1001?", customer)

    assert result.status == RunStatus.COMPLETED
    assert result.answer == "Your order has shipped."


def test_checkpoint_history_can_be_listed_and_deleted(harness, model, storage, customer):
    model.script(say("Hello!"))
    result = harness.run("orders", "Hi", customer)
    thread = {"configurable": {"thread_id": result.thread_id}}
    checkpointer = storage.checkpointer

    history = list(checkpointer.list(thread))
    assert len(history) >= 2
    assert history == sorted(history, key=lambda c: c.checkpoint["id"], reverse=True)
    inputs = list(checkpointer.list(thread, filter={"source": "input"}))
    assert inputs and all(c.metadata["source"] == "input" for c in inputs)
    assert asyncio.run(checkpointer.aget_tuple(thread)) == checkpointer.get_tuple(thread)

    checkpointer.delete_thread(result.thread_id)
    assert checkpointer.get_tuple(thread) is None


def test_store_round_trip(storage):
    store = storage.store
    store.put(("customers", "c-1", "memories"), "m1", {"text": "Likes email", "kind": "contact"})
    store.put(("customers", "c-1", "memories"), "m2", {"text": "Vegetarian", "kind": "diet"})
    store.put(("customers", "c-10", "memories"), "m1", {"text": "Not c-1"})

    first = store.get(("customers", "c-1", "memories"), "m1")
    store.put(("customers", "c-1", "memories"), "m1", {"text": "Likes phone", "kind": "contact"})
    updated = store.get(("customers", "c-1", "memories"), "m1")
    assert updated.value["text"] == "Likes phone"
    assert updated.created_at == first.created_at
    assert updated.updated_at > first.updated_at

    found = store.search(("customers", "c-1"))
    assert sorted(item.key for item in found) == ["m1", "m2"]
    assert [i.key for i in store.search(("customers", "c-1"), filter={"kind": "diet"})] == ["m2"]
    assert store.list_namespaces(prefix=("customers",), max_depth=2) == [
        ("customers", "c-1"),
        ("customers", "c-10"),
    ]

    store.delete(("customers", "c-1", "memories"), "m2")
    assert store.get(("customers", "c-1", "memories"), "m2") is None
    with pytest.raises(NotImplementedError):
        store.search(("customers",), query="email")
