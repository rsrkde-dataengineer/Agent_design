class RecoverableError(Exception):
    """A failure the agent should see and work around, like an unknown order id.

    The tool-error middleware turns it into an error message for the model, so the text
    must be safe to show. Any other exception stops the run.
    """


class NotFoundError(RecoverableError):
    pass


class PolicyViolationError(RecoverableError):
    pass


class UnknownAgentError(LookupError):
    def __init__(self, name: str, available: list[str]) -> None:
        super().__init__(f"No agent named {name!r}. Available: {', '.join(available)}")
