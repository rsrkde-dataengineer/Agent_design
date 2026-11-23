from dataclasses import dataclass


@dataclass(frozen=True)
class RequestContext:
    """Who the agents are serving on this request.

    The caller (API layer, CLI) builds it from the authenticated session. LangGraph hands
    it to every middleware and tool through the runtime. It is read-only and never
    checkpointed, which makes it the right place for identity: tools take the customer
    id from here instead of trusting an id the model typed.
    """

    customer_id: str
    customer_name: str
    plan: str = "standard"
    locale: str = "en-US"
