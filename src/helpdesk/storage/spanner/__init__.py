from google.cloud import spanner

from helpdesk.storage import Storage
from helpdesk.storage.spanner.checkpointer import SpannerCheckpointer
from helpdesk.storage.spanner.results import SpannerResultRepository
from helpdesk.storage.spanner.store import SpannerStore


def spanner_storage(project: str, instance: str, database: str) -> Storage:
    """Checkpoints, memories and results in one Spanner database. Tables are in schema.sql."""
    db = spanner.Client(project=project).instance(instance).database(database)
    return Storage(
        checkpointer=SpannerCheckpointer(db),
        store=SpannerStore(db),
        results=SpannerResultRepository(db),
    )
