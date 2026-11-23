"""Long-term memory about a customer, kept in the LangGraph store.

The namespace layout is defined here and nowhere else.
"""

from uuid import uuid4

from langgraph.store.base import BaseStore


def _namespace(customer_id: str) -> tuple[str, ...]:
    return ("customers", customer_id, "memories")


def recall(store: BaseStore, customer_id: str, limit: int = 20) -> list[str]:
    items = store.search(_namespace(customer_id), limit=limit)
    return [item.value["text"] for item in items]


def remember(store: BaseStore, customer_id: str, text: str) -> None:
    store.put(_namespace(customer_id), key=str(uuid4()), value={"text": text})
