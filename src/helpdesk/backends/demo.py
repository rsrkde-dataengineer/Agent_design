"""In-memory stand-ins for the order, billing and help-center systems."""

import re
from collections.abc import Iterable
from dataclasses import replace
from datetime import date
from decimal import Decimal

from helpdesk.backends.contracts import Article, Backends, Invoice, Order
from helpdesk.core.errors import NotFoundError, PolicyViolationError


class DemoOrderSystem:
    def __init__(self, orders: Iterable[Order]) -> None:
        self.orders = {order.order_id: order for order in orders}

    def get_order(self, customer_id: str, order_id: str) -> Order:
        order = self.orders.get(order_id)
        # Someone else's order looks exactly like a missing one, so ids can't be probed.
        if order is None or order.customer_id != customer_id:
            raise NotFoundError(f"No order {order_id} for this customer.")
        return order

    def cancel_order(self, customer_id: str, order_id: str) -> Order:
        order = self.get_order(customer_id, order_id)
        if order.status != "placed":
            raise PolicyViolationError(
                f"Order {order_id} is already {order.status} and can't be cancelled."
            )
        cancelled = replace(order, status="cancelled")
        self.orders[order_id] = cancelled
        return cancelled


class DemoBillingSystem:
    def __init__(self, invoices: Iterable[Invoice], *, approval_threshold: Decimal) -> None:
        self.invoices = {invoice.invoice_id: invoice for invoice in invoices}
        self.approval_threshold = approval_threshold

    def list_invoices(self, customer_id: str) -> list[Invoice]:
        return [i for i in self.invoices.values() if i.customer_id == customer_id]

    def refund(self, customer_id: str, invoice_id: str, amount: Decimal) -> Invoice:
        invoice = self.invoices.get(invoice_id)
        if invoice is None or invoice.customer_id != customer_id:
            raise NotFoundError(f"No invoice {invoice_id} for this customer.")
        if amount <= 0:
            raise PolicyViolationError("A refund amount must be positive.")
        remaining = invoice.amount - invoice.refunded
        if amount > remaining:
            raise PolicyViolationError(
                f"Only {remaining} {invoice.currency} is left to refund on {invoice_id}."
            )
        if amount > self.approval_threshold:
            raise PolicyViolationError(
                f"Refunds over {self.approval_threshold} {invoice.currency} need human approval."
            )
        refunded = replace(invoice, refunded=invoice.refunded + amount)
        self.invoices[invoice_id] = refunded
        return refunded


class DemoHelpCenter:
    def __init__(self, articles: Iterable[Article]) -> None:
        self.articles = list(articles)

    def search(self, query: str, limit: int = 3) -> list[Article]:
        wanted = _words(query)
        scored = [(len(wanted & _words(f"{a.title} {a.body}")), a) for a in self.articles]
        ranked = sorted(scored, key=lambda pair: pair[0], reverse=True)
        return [article for score, article in ranked if score > 0][:limit]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def demo_backends() -> Backends:
    orders = [
        Order(
            "A-1001",
            "c-100",
            "shipped",
            ("Trail running shoes",),
            Decimal("129.00"),
            "USD",
            carrier="UPS",
            tracking_number="1Z999AA10123456784",
            estimated_delivery=date(2026, 11, 26),
        ),
        Order("A-1002", "c-100", "placed", ("Rain jacket", "Wool socks"), Decimal("88.50"), "USD"),
        Order("B-2001", "c-200", "delivered", ("Camping stove",), Decimal("54.99"), "USD"),
    ]
    invoices = [
        Invoice("INV-501", "c-100", "A-1001", Decimal("129.00"), "USD"),
        Invoice("INV-502", "c-100", "A-1002", Decimal("88.50"), "USD"),
        Invoice("INV-601", "c-200", "B-2001", Decimal("54.99"), "USD"),
    ]
    articles = [
        Article(
            "Returns and refunds",
            "Unworn items can be returned within 30 days of delivery for a full refund. "
            "Refunds go back to the original payment method within 5 business days.",
        ),
        Article(
            "Shipping times",
            "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 "
            "business days. Tracking numbers are emailed when an order ships.",
        ),
        Article(
            "Cancelling an order",
            "Orders can be cancelled until they ship. After shipping, start a return instead.",
        ),
    ]
    return Backends(
        orders=DemoOrderSystem(orders),
        billing=DemoBillingSystem(invoices, approval_threshold=Decimal("100.00")),
        help_center=DemoHelpCenter(articles),
    )
