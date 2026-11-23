"""Prompt text lives in .md files beside this module, so it can be reviewed like copy."""

from datetime import date
from functools import cache
from importlib.resources import files

from helpdesk.core.context import RequestContext


@cache
def load_prompt(name: str) -> str:
    return (files(__package__) / f"{name}.md").read_text(encoding="utf-8").strip()


def render_context_prompt(context: RequestContext, memories: list[str]) -> str:
    remembered = "\n".join(f"- {memory}" for memory in memories) or "- Nothing yet."
    return load_prompt("context").format(
        customer_name=context.customer_name,
        plan=context.plan,
        locale=context.locale,
        today=date.today().isoformat(),
        memories=remembered,
    )
