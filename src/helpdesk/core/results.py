from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol


class RunStatus(StrEnum):
    COMPLETED = "completed"
    ESCALATED = "escalated"
    FAILED = "failed"


@dataclass(frozen=True)
class AgentResult:
    """The outcome of one turn, kept for reporting and audit."""

    run_id: str
    thread_id: str
    agent: str
    customer_id: str
    status: RunStatus
    answer: str = ""
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ResultRepository(Protocol):
    def save(self, result: AgentResult) -> None: ...

    def get(self, run_id: str) -> AgentResult | None: ...
