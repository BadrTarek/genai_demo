from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from etic_genai.tools.base import BaseAgentTool


class CurrentDateTimeInput(BaseModel):
    timezone: str = Field(default="UTC", description="IANA timezone name, e.g. 'Africa/Cairo'.")


class CurrentDateTimeTool(BaseAgentTool):
    name: str = "current_datetime"
    description: str = "Get the current date and time in a given timezone."
    args_schema: type[BaseModel] = CurrentDateTimeInput

    def _execute(self, timezone: str = "UTC") -> str:
        return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")
