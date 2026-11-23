from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from HELPDESK_* environment variables or a .env file."""

    model_config = SettingsConfigDict(env_prefix="HELPDESK_", env_file=".env", extra="ignore")

    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_model_calls: int = 12
    model_retries: int = 2

    storage: Literal["memory", "spanner"] = "memory"
    spanner_project: str = ""
    spanner_instance: str = ""
    spanner_database: str = ""

    @model_validator(mode="after")
    def _spanner_needs_a_database(self) -> "Settings":
        if self.storage == "spanner" and not (
            self.spanner_project and self.spanner_instance and self.spanner_database
        ):
            raise ValueError(
                "HELPDESK_STORAGE=spanner needs HELPDESK_SPANNER_PROJECT, "
                "HELPDESK_SPANNER_INSTANCE and HELPDESK_SPANNER_DATABASE"
            )
        return self
