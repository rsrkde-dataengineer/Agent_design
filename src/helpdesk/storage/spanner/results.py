from dataclasses import dataclass

from google.cloud.spanner_v1.database import Database
from google.cloud.spanner_v1.keyset import KeySet

from helpdesk.core.results import AgentResult, RunStatus

_TABLE = "agent_results"
_COLUMNS = (
    "run_id",
    "thread_id",
    "agent",
    "customer_id",
    "status",
    "answer",
    "error",
    "created_at",
)


@dataclass(frozen=True)
class SpannerResultRepository:
    database: Database

    def save(self, result: AgentResult) -> None:
        row = (
            result.run_id,
            result.thread_id,
            result.agent,
            result.customer_id,
            result.status.value,
            result.answer,
            result.error,
            result.created_at,
        )
        with self.database.batch() as batch:
            batch.insert_or_update(_TABLE, columns=_COLUMNS, values=[row])

    def get(self, run_id: str) -> AgentResult | None:
        with self.database.snapshot() as snapshot:
            rows = list(snapshot.read(_TABLE, _COLUMNS, KeySet(keys=[[run_id]])))
        if not rows:
            return None
        values = dict(zip(_COLUMNS, rows[0], strict=True))
        values["status"] = RunStatus(values["status"])
        return AgentResult(**values)
