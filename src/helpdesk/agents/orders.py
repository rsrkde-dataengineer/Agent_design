"""Orders specialist: order status, delivery tracking and cancellations."""

from typing import Any

from langchain.tools import ToolRuntime, tool

from helpdesk.backends.contracts import Order, OrderSystem
from helpdesk.core.context import RequestContext
from helpdesk.core.spec import AgentSpec
from helpdesk.prompts import load_prompt


def orders_agent(orders: OrderSystem) -> AgentSpec:
    @tool
    def get_order(order_id: str, runtime: ToolRuntime[RequestContext]) -> dict[str, Any]:
        """Look up an order: status, items, total, carrier, tracking number and delivery date."""
        return _order_details(orders.get_order(runtime.context.customer_id, order_id))

    @tool
    def cancel_order(order_id: str, runtime: ToolRuntime[RequestContext]) -> dict[str, Any]:
        """Cancel an order that has not shipped yet."""
        return _order_details(orders.cancel_order(runtime.context.customer_id, order_id))

    return AgentSpec(
        name="orders",
        description=(
            "Answers questions about the customer's orders: status, delivery, tracking and "
            "cancellation. Include the order id when you have it."
        ),
        system_prompt=load_prompt("orders"),
        tools=[get_order, cancel_order],
    )


def _order_details(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "status": order.status,
        "items": list(order.items),
        "total": f"{order.total} {order.currency}",
        "carrier": order.carrier,
        "tracking_number": order.tracking_number,
        "estimated_delivery": order.estimated_delivery and order.estimated_delivery.isoformat(),
    }
