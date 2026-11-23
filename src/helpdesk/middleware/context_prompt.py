from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

from helpdesk.prompts import render_context_prompt


class ContextPromptMiddleware(AgentMiddleware):
    """Appends the context prompt (who the customer is, what we remember) to the system prompt.

    The system prompt is identical for every customer, so the provider can cache it.
    Only this tail changes from request to request.
    """

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        context_prompt = render_context_prompt(
            request.runtime.context, request.state.get("memories", [])
        )
        system_prompt = request.system_message.text if request.system_message else ""
        system_message = SystemMessage(f"{system_prompt}\n\n{context_prompt}".strip())
        return handler(request.override(system_message=system_message))
