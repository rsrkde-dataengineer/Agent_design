"""LangGraph store on Cloud Spanner, used for long-term memory.

Supports get, put, delete, namespace-prefix search with exact-match filters, and namespace
listing. It has no vector index, so semantic search (`query=`) is rejected rather than
silently ignored.
"""

import asyncio
from collections.abc import Iterable
from typing import Any

from google.cloud import spanner
from google.cloud.spanner_v1 import JsonObject
from google.cloud.spanner_v1.database import Database
from google.cloud.spanner_v1.keyset import KeySet
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

_TABLE = "store_items"
_COLUMNS = ("namespace", "item_key", "value", "created_at", "updated_at")


class SpannerStore(BaseStore):
    def __init__(self, database: Database) -> None:
        self.database = database

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return [self._run(op) for op in ops]

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    def _run(self, op: Op) -> Result:
        match op:
            case GetOp():
                return self._get(op)
            case PutOp():
                return self._put(op)
            case SearchOp():
                return self._search(op)
            case ListNamespacesOp():
                return self._list_namespaces(op)
        raise TypeError(f"Unsupported store operation: {type(op).__name__}")

    def _get(self, op: GetOp) -> Item | None:
        key = [_encode(op.namespace), op.key]
        with self.database.snapshot() as snapshot:
            rows = list(snapshot.read(_TABLE, _COLUMNS, KeySet(keys=[key])))
        return _item(rows[0]) if rows else None

    def _put(self, op: PutOp) -> None:
        key = [_encode(op.namespace), op.key]
        if op.value is None:
            with self.database.batch() as batch:
                batch.delete(_TABLE, KeySet(keys=[key]))
            return

        def upsert(transaction: Any) -> None:
            # Read first so an update keeps the original created_at.
            exists = list(transaction.read(_TABLE, ("item_key",), KeySet(keys=[key])))
            if exists:
                transaction.update(
                    _TABLE,
                    columns=("namespace", "item_key", "value", "updated_at"),
                    values=[(*key, JsonObject(op.value), spanner.COMMIT_TIMESTAMP)],
                )
            else:
                transaction.insert(
                    _TABLE,
                    columns=_COLUMNS,
                    values=[
                        (
                            *key,
                            JsonObject(op.value),
                            spanner.COMMIT_TIMESTAMP,
                            spanner.COMMIT_TIMESTAMP,
                        )
                    ],
                )

        self.database.run_in_transaction(upsert)

    def _search(self, op: SearchOp) -> list[SearchItem]:
        if op.query:
            raise NotImplementedError("SpannerStore has no semantic search. Search by namespace.")
        sql = f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE}"
        params = {}
        if op.namespace_prefix:
            prefix = _encode(op.namespace_prefix)
            sql += " WHERE namespace = @prefix OR STARTS_WITH(namespace, @prefix_dot)"
            params = {"prefix": prefix, "prefix_dot": f"{prefix}."}
        sql += " ORDER BY updated_at DESC"

        with self.database.snapshot() as snapshot:
            items = [_search_item(row) for row in snapshot.execute_sql(sql, params=params)]
        if op.filter:
            items = [i for i in items if all(i.value.get(k) == v for k, v in op.filter.items())]
        return items[op.offset : op.offset + op.limit]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        with self.database.snapshot() as snapshot:
            rows = snapshot.execute_sql(f"SELECT DISTINCT namespace FROM {_TABLE}")
            namespaces = [_decode(namespace) for (namespace,) in rows]

        if op.match_conditions:
            namespaces = [
                ns for ns in namespaces if all(_matches(ns, c) for c in op.match_conditions)
            ]
        if op.max_depth is not None:
            namespaces = [ns[: op.max_depth] for ns in namespaces]
        unique = sorted(set(namespaces))
        return unique[op.offset : op.offset + op.limit]


# LangGraph forbids dots inside namespace labels, so a dot is a safe separator.
def _encode(namespace: tuple[str, ...]) -> str:
    return ".".join(namespace)


def _decode(namespace: str) -> tuple[str, ...]:
    return tuple(namespace.split("."))


def _item(row: list[Any]) -> Item:
    namespace, key, value, created_at, updated_at = row
    return Item(
        namespace=_decode(namespace),
        key=key,
        value=dict(value),
        created_at=created_at,
        updated_at=updated_at,
    )


def _search_item(row: list[Any]) -> SearchItem:
    item = _item(row)
    return SearchItem(item.namespace, item.key, item.value, item.created_at, item.updated_at)


def _matches(namespace: tuple[str, ...], condition: MatchCondition) -> bool:
    path = condition.path
    if len(path) > len(namespace):
        return False
    part = namespace[: len(path)] if condition.match_type == "prefix" else namespace[-len(path) :]
    return all(wanted == "*" or wanted == actual for wanted, actual in zip(path, part, strict=True))
