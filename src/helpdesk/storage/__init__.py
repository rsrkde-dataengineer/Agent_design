from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from helpdesk.core.results import ResultRepository


@dataclass(frozen=True)
class Storage:
    """Everything we persist. The three parts are always chosen together."""

    checkpointer: BaseCheckpointSaver  # short-term: conversation state, per thread
    store: BaseStore  # long-term: memories that outlive a thread
    results: ResultRepository  # the final answer of every run
