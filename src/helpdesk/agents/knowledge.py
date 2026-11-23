"""Help-center specialist: store policies and how-to questions."""

from langchain.tools import tool

from helpdesk.backends.contracts import HelpCenter
from helpdesk.core.spec import AgentSpec
from helpdesk.prompts import load_prompt


def knowledge_agent(help_center: HelpCenter) -> AgentSpec:
    @tool
    def search_help_center(query: str) -> list[dict[str, str]]:
        """Search help-center articles on store policies and how-tos. Returns the best matches."""
        return [{"title": a.title, "body": a.body} for a in help_center.search(query)]

    return AgentSpec(
        name="knowledge",
        description="Answers store policy, shipping and returns questions from the help center.",
        system_prompt=load_prompt("knowledge"),
        tools=[search_help_center],
    )
