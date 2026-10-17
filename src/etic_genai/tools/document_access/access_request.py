import json
from typing import Literal

from pydantic import BaseModel, Field

from etic_genai.tools.base import BaseAgentTool


class DocumentAccessRequestInput(BaseModel):
    provider: Literal["google_drive", "sharepoint"] = Field(
        description="Document provider selected by the user."
    )
    purpose: str = Field(description="Why the agent needs document access.")


class DocumentAccessRequestTool(BaseAgentTool):
    name: str = "request_document_access"
    description: str = (
        "Prepare a user approval request before accessing files in Google Drive or SharePoint. "
        "Use this before any document access tool. The assistant must wait for the user to "
        "approve or deny access."
    )
    args_schema: type[BaseModel] = DocumentAccessRequestInput

    def _execute(self, provider: str, purpose: str) -> str:
        return json.dumps(
            {
                "status": "approval_required",
                "provider": provider,
                "purpose": purpose,
                "prompt": (
                    f"Do you approve access to {provider} for this purpose: {purpose}? "
                    "Reply with approve or deny."
                ),
                "next_step": (
                    "If the user approves SharePoint access, call sharepoint_list_files with "
                    "approved=true and the required site_id or drive_id."
                ),
            },
            indent=2,
        )
