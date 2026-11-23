"""Billing specialist: invoices and refunds."""

from decimal import Decimal
from typing import Any

from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.tools import ToolRuntime, tool

from helpdesk.backends.contracts import BillingSystem, Invoice
from helpdesk.core.context import RequestContext
from helpdesk.core.spec import AgentSpec
from helpdesk.prompts import load_prompt


def billing_agent(billing: BillingSystem) -> AgentSpec:
    @tool
    def list_invoices(runtime: ToolRuntime[RequestContext]) -> list[dict[str, Any]]:
        """List the customer's invoices with amounts and how much was already refunded."""
        return [_invoice_details(i) for i in billing.list_invoices(runtime.context.customer_id)]

    @tool
    def refund_invoice(
        invoice_id: str, amount: Decimal, runtime: ToolRuntime[RequestContext]
    ) -> dict[str, Any]:
        """Refund part or all of an invoice. `amount` is in the invoice's currency."""
        invoice = billing.refund(runtime.context.customer_id, invoice_id, amount)
        return _invoice_details(invoice)

    return AgentSpec(
        name="billing",
        description=(
            "Handles the customer's invoices, charges and refunds. Include the order or "
            "invoice id and the refund amount when you have them."
        ),
        system_prompt=load_prompt("billing"),
        tools=[list_invoices, refund_invoice],
        # At most one refund each time this agent runs, whatever the model decides.
        middleware=[ToolCallLimitMiddleware(tool_name="refund_invoice", run_limit=1)],
    )


def _invoice_details(invoice: Invoice) -> dict[str, Any]:
    return {
        "invoice_id": invoice.invoice_id,
        "order_id": invoice.order_id,
        "amount": f"{invoice.amount} {invoice.currency}",
        "refunded": f"{invoice.refunded} {invoice.currency}",
    }
