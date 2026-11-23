"""The middleware every agent gets.

Order matters. The first entry is the outermost layer: it wraps everything after it,
so the observability layer times each call including retries.
"""

from collections.abc import Sequence

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRefusalMiddleware,
    ModelRetryMiddleware,
    ToolErrorMiddleware,
)

from helpdesk.core.spec import AgentSpec
from helpdesk.middleware.context_prompt import ContextPromptMiddleware
from helpdesk.middleware.memory import LoadMemoriesMiddleware
from helpdesk.middleware.observability import ObservabilityMiddleware
from helpdesk.middleware.tool_errors import report_recoverable_errors


def standard_middleware(
    spec: AgentSpec, *, max_model_calls: int, model_retries: int
) -> Sequence[AgentMiddleware]:
    return [
        ObservabilityMiddleware(agent_name=spec.name),
        # A ReAct loop that never converges is a bug. Fail the run instead of burning tokens.
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error"),
        # Outside the retry layer, so a refusal is reported rather than retried.
        ModelRefusalMiddleware(),
        # When retries run out, raise. The default would turn the outage into an answer.
        ModelRetryMiddleware(max_retries=model_retries, on_failure="error"),
        ToolErrorMiddleware(on_error=report_recoverable_errors),
        LoadMemoriesMiddleware(),
        ContextPromptMiddleware(),
    ]
