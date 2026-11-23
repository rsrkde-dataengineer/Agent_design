from dataclasses import dataclass, field

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from helpdesk.core.results import AgentResult
from helpdesk.storage import Storage


@dataclass(frozen=True)
class InMemoryResultRepository:
    saved: dict[str, AgentResult] = field(default_factory=dict)

    def save(self, result: AgentResult) -> None:
        self.saved[result.run_id] = result

    def get(self, run_id: str) -> AgentResult | None:
        return self.saved.get(run_id)


def in_memory_storage() -> Storage:
    """For tests and local runs. Everything is lost when the process exits."""
    return Storage(
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        results=InMemoryResultRepository(),
    )
