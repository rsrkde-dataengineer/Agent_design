from decimal import Decimal

import pytest
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from helpdesk.core.errors import UnknownAgentError
from helpdesk.core.results import RunStatus
from tests.fakes import call, say


def tool_messages(prompt):
    return [m for m in prompt if isinstance(m, ToolMessage)]


def test_specialist_answers_from_its_tool(harness, model, storage, customer):
    model.script(
        call("get_order", order_id="A-1001"),
        say("Order A-1001 shipped with UPS and should arrive on 26 November."),
    )

    result = harness.run("orders", "Where is order A-1001?", customer)

    assert result.status == RunStatus.COMPLETED
    assert result.answer.startswith("Order A-1001 shipped")
    assert "1Z999AA10123456784" in tool_messages(model.prompts[1])[0].text
    assert storage.results == [result]


def test_supervisor_delegates_to_a_specialist(harness, model, customer):
    model.script(
        call("ask_orders_agent", task="Status of order A-1001?"),  # supervisor
        call("get_order", order_id="A-1001"),  # orders agent
        say("A-1001 has shipped and arrives on 26 November."),  # orders agent
        say("Good news, Sam: your shoes arrive on 26 November."),  # supervisor
    )

    result = harness.run("supervisor", "Where are my shoes? Order A-1001.", customer)

    assert result.answer == "Good news, Sam: your shoes arrive on 26 November."
    specialist_reply = tool_messages(model.prompts[3])[0]
    assert specialist_reply.text == "A-1001 has shipped and arrives on 26 November."


def test_a_customer_cannot_see_another_customers_order(harness, model, customer):
    model.script(
        call("get_order", order_id="B-2001"),  # belongs to c-200
        say("I couldn't find order B-2001 on your account."),
    )

    harness.run("orders", "Where is order B-2001?", customer)

    error = tool_messages(model.prompts[1])[0]
    assert error.status == "error"
    assert error.text == "get_order failed: No order B-2001 for this customer."


def test_large_refund_is_escalated_to_a_human(harness, model, backends, customer):
    model.script(
        call("ask_billing_agent", task="Refund 129.00 USD on invoice INV-501."),
        call("refund_invoice", invoice_id="INV-501", amount="129.00"),
        say("That refund needs human approval."),
        call("escalate_to_human", reason="Full refund of INV-501 needs approval."),
        say("I've passed this to a colleague who can approve the refund."),
    )

    result = harness.run("supervisor", "Refund my shoes in full please.", customer)

    assert result.status == RunStatus.ESCALATED
    assert backends.billing.list_invoices("c-100")[0].refunded == Decimal("0")


def test_billing_refunds_at_most_once_per_run(harness, model, backends, customer):
    model.script(
        call("refund_invoice", invoice_id="INV-502", amount="10.00"),
        call("refund_invoice", invoice_id="INV-502", amount="10.00"),
        say("Refunded 10.00 USD."),
    )

    harness.run("billing", "Refund 10 dollars on INV-502, twice.", customer)

    invoice = next(i for i in backends.billing.list_invoices("c-100") if i.invoice_id == "INV-502")
    assert invoice.refunded == Decimal("10.00")
    assert tool_messages(model.prompts[2])[1].status == "error"


def test_a_remembered_fact_reaches_later_conversations(harness, model, customer):
    model.script(
        call("remember_fact", fact="Prefers to be contacted by email."),
        say("Noted, I'll use email from now on."),
        say("Hi Sam, how can I help?"),
    )

    harness.run("supervisor", "Please only contact me by email.", customer)
    harness.run("supervisor", "Hello again", customer)  # a new thread

    system_prompt = model.prompts[-1][0]
    assert isinstance(system_prompt, SystemMessage)
    assert "Name: Sam" in system_prompt.text
    assert "- Prefers to be contacted by email." in system_prompt.text


def test_a_thread_keeps_its_history_between_turns(harness, model, customer):
    model.script(say("Hi Sam!"), say("You asked me to say hello."))

    first = harness.run("supervisor", "Say hello", customer)
    harness.run("supervisor", "What did I ask you?", customer, thread_id=first.thread_id)

    human_turns = [m.text for m in model.prompts[-1] if isinstance(m, HumanMessage)]
    assert human_turns == ["Say hello", "What did I ask you?"]


def test_a_runaway_loop_fails_the_run(harness, model, storage, customer, settings):
    model.script(*[call("get_order", order_id="A-1001")] * (settings.max_model_calls + 1))

    with pytest.raises(ModelCallLimitExceededError):
        harness.run("orders", "Check A-1001 forever", customer)

    assert storage.results[-1].status == RunStatus.FAILED
    assert storage.results[-1].error.startswith("ModelCallLimitExceededError")


def test_a_model_outage_fails_the_run(harness, model, storage, customer):
    model.script(ConnectionError("model API unreachable"))

    with pytest.raises(ConnectionError):
        harness.run("orders", "Where is A-1001?", customer)

    assert storage.results[-1].status == RunStatus.FAILED
    assert storage.results[-1].error == "ConnectionError: model API unreachable"


def test_unknown_agent_is_rejected(harness, customer):
    with pytest.raises(
        UnknownAgentError, match="Available: billing, knowledge, orders, supervisor"
    ):
        harness.run("sales", "Hi", customer)
