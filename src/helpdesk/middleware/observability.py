import logging
import time
from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

log = logging.getLogger("helpdesk.agents")


class ObservabilityMiddleware(AgentMiddleware):
    """Logs the duration of each model and tool call, and the tokens each model call used.

    LangSmith or OpenTelemetry give richer traces. This is the minimum you still want in
    plain logs when neither is wired up.
    """

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self.agent_name = agent_name

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        started = time.perf_counter()
        response = handler(request)
        usage = response.result[0].usage_metadata or {}
        log.info(
            "%s model call took %.2fs (%s input tokens, %s output tokens)",
            self.agent_name,
            time.perf_counter() - started,
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
        )
        return response

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        started = time.perf_counter()
        result = handler(request)
        status = result.status if isinstance(result, ToolMessage) else "success"
        log.info(
            "%s tool %s finished with %s in %.2fs",
            self.agent_name,
            request.tool_call["name"],
            status,
            time.perf_counter() - started,
        )
        return result
