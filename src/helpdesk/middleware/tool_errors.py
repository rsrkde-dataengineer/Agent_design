from langchain.agents.middleware.types import ToolCallRequest

from helpdesk.core.errors import RecoverableError


def report_recoverable_errors(error: Exception, request: ToolCallRequest) -> str | None:
    """Show expected failures to the model so it can recover. Anything else stops the run."""
    if isinstance(error, RecoverableError):
        return f"{request.tool_call['name']} failed: {error}"
    return None
