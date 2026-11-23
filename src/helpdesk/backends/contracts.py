"""What the agents need from the company's systems.

Tools depend on these protocols, not on concrete clients. Replacing the demo data with
real HTTP or gRPC clients does not touch any agent.
"""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Protocol


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    status: str  # placed, shipped, delivered or cancelled
    items: tuple[str, ...]
    total: Decimal
    currency: str
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: date | None = None


@dataclass(frozen=True)
class Invoice:
    invoice_id: str
    customer_id: str
    order_id: str
    amount: Decimal
    currency: str
    refunded: Decimal = Decimal("0")


@dataclass(frozen=True)
class Article:
    title: str
    body: str


class OrderSystem(Protocol):
    def get_order(self, customer_id: str, order_id: str) -> Order: ...

    def cancel_order(self, customer_id: str, order_id: str) -> Order: ...


class BillingSystem(Protocol):
    def list_invoices(self, customer_id: str) -> list[Invoice]: ...

    def refund(self, customer_id: str, invoice_id: str, amount: Decimal) -> Invoice: ...


class HelpCenter(Protocol):
    def search(self, query: str, limit: int = 3) -> list[Article]: ...


@dataclass(frozen=True)
class Backends:
    orders: OrderSystem
    billing: BillingSystem
    help_center: HelpCenter
