"""LangGraph checkpointer on Cloud Spanner.

Each checkpoint is stored whole, as one serialized blob, the same way the SQLite saver
does it. Blocking calls; the async methods run them on a worker thread.
"""

import asyncio
import base64
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from google.cloud import spanner
from google.cloud.spanner_v1 import JsonObject
from google.cloud.spanner_v1.database import Database
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

_CHECKPOINT_COLUMNS = (
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "parent_checkpoint_id",
    "checkpoint_type",
    "checkpoint",
    "metadata",
    "created_at",
)
_WRITE_COLUMNS = (
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "task_id",
    "idx",
    "task_path",
    "channel",
    "value_type",
    "value",
)
_SELECT_CHECKPOINTS = """
    SELECT thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
           checkpoint_type, checkpoint, metadata
    FROM checkpoints
"""
# Sorted the way LangGraph applies writes within a step (see writes_sort_key).
_SELECT_WRITES = """
    SELECT task_id, channel, value_type, value
    FROM checkpoint_writes
    WHERE thread_id = @thread_id AND checkpoint_ns = @checkpoint_ns
      AND checkpoint_id = @checkpoint_id
    ORDER BY task_path, task_id, idx
"""


class SpannerCheckpointer(BaseCheckpointSaver[int]):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self.database = database

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns = _thread(config)
        sql = (
            f"{_SELECT_CHECKPOINTS} WHERE thread_id = @thread_id AND checkpoint_ns = @checkpoint_ns"
        )
        params = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if checkpoint_id := get_checkpoint_id(config):
            sql += " AND checkpoint_id = @checkpoint_id"
            params["checkpoint_id"] = checkpoint_id
        else:
            # Checkpoint ids are time-ordered, so the largest one is the latest.
            sql += " ORDER BY checkpoint_id DESC LIMIT 1"

        with self.database.snapshot(multi_use=True) as snapshot:
            row = next(iter(snapshot.execute_sql(sql, params=params)), None)
            return self._to_tuple(snapshot, row) if row else None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        conditions, params = [], {}
        if config:
            conditions.append("thread_id = @thread_id")
            params["thread_id"] = str(config["configurable"]["thread_id"])
            if (checkpoint_ns := config["configurable"].get("checkpoint_ns")) is not None:
                conditions.append("checkpoint_ns = @checkpoint_ns")
                params["checkpoint_ns"] = checkpoint_ns
            if checkpoint_id := get_checkpoint_id(config):
                conditions.append("checkpoint_id = @checkpoint_id")
                params["checkpoint_id"] = checkpoint_id
        if before and (before_id := get_checkpoint_id(before)):
            conditions.append("checkpoint_id < @before_id")
            params["before_id"] = before_id
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"{_SELECT_CHECKPOINTS} {where} ORDER BY checkpoint_id DESC"

        # The metadata filter is applied here rather than in SQL. Listing is an admin and
        # debugging path, not something the ReAct loop calls.
        found: list[CheckpointTuple] = []
        with self.database.snapshot(multi_use=True) as snapshot:
            for row in list(snapshot.execute_sql(sql, params=params)):
                checkpoint = self._to_tuple(snapshot, row)
                if filter and any(checkpoint.metadata.get(k) != v for k, v in filter.items()):
                    continue
                found.append(checkpoint)
                if limit is not None and len(found) >= limit:
                    break
        yield from found

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns = _thread(config)
        checkpoint_type, blob = self.serde.dumps_typed(checkpoint)
        row = (
            thread_id,
            checkpoint_ns,
            checkpoint["id"],
            config["configurable"].get("checkpoint_id"),  # the checkpoint this one follows
            checkpoint_type,
            _encode_blob(blob),
            JsonObject(get_checkpoint_metadata(config, metadata)),
            spanner.COMMIT_TIMESTAMP,
        )
        with self.database.batch() as batch:
            batch.insert_or_update("checkpoints", columns=_CHECKPOINT_COLUMNS, values=[row])
        return _config(thread_id, checkpoint_ns, checkpoint["id"])

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns = _thread(config)
        checkpoint_id = config["configurable"]["checkpoint_id"]
        rows = []
        for idx, (channel, value) in enumerate(writes):
            value_type, blob = self.serde.dumps_typed(value)
            rows.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    WRITES_IDX_MAP.get(channel, idx),  # special channels get fixed negative slots
                    task_path,
                    channel,
                    value_type,
                    _encode_blob(blob),
                )
            )
        with self.database.batch() as batch:
            batch.insert_or_update("checkpoint_writes", columns=_WRITE_COLUMNS, values=rows)

    def delete_thread(self, thread_id: str) -> None:
        def delete(transaction: Any) -> None:
            for table in ("checkpoint_writes", "checkpoints"):
                transaction.execute_update(
                    f"DELETE FROM {table} WHERE thread_id = @thread_id",
                    params={"thread_id": str(thread_id)},
                )

        self.database.run_in_transaction(delete)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        found = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for checkpoint in found:
            yield checkpoint

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    def _to_tuple(self, snapshot: Any, row: Sequence[Any]) -> CheckpointTuple:
        thread_id, checkpoint_ns, checkpoint_id, parent_id, checkpoint_type, blob, metadata = row
        writes = snapshot.execute_sql(
            _SELECT_WRITES,
            params={
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            },
        )
        return CheckpointTuple(
            config=_config(thread_id, checkpoint_ns, checkpoint_id),
            checkpoint=self.serde.loads_typed((checkpoint_type, _decode_blob(blob))),
            metadata=dict(metadata or {}),
            parent_config=_config(thread_id, checkpoint_ns, parent_id) if parent_id else None,
            pending_writes=[
                (task_id, channel, self.serde.loads_typed((value_type, _decode_blob(value))))
                for task_id, channel, value_type, value in writes
            ],
        )


def _thread(config: RunnableConfig) -> tuple[str, str]:
    configurable = config["configurable"]
    return str(configurable["thread_id"]), configurable.get("checkpoint_ns", "")


def _config(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }


# The Spanner client takes BYTES values base64-encoded and returns them the same way.
def _encode_blob(data: bytes) -> bytes:
    return base64.b64encode(data)


def _decode_blob(data: bytes) -> bytes:
    return base64.b64decode(data)
