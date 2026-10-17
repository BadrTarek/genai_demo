import json

from pydantic import BaseModel, Field


from etic_genai.tools.base import BaseAgentTool
from etic_genai.tools.document_access.microsoft_auth import finish_device_login, get_user_tenant,start_device_login


class SharePointAuthStartInput(BaseModel):
    user_id: str = Field(description="Unique identifier for the user.")


class SharePointAuthStartTool(BaseAgentTool):
    name: str = "sharepoint_auth_start"
    description: str = (
        "Initiate the Microsoft device login flow for SharePoint access. "
        "This tool should be called after the user approves document access."
    )
    args_schema: type[BaseModel] = SharePointAuthStartInput

    def _execute(self, user_id: str) -> dict[str, any]:
        return start_device_login(user_id)  


class SharePointAuthFinishInput(BaseModel):
    user_id: str = Field(description="Unique identifier for the user.")

class SharePointAuthFinishTool(BaseAgentTool):
    name: str = "sharepoint_auth_finish"
    description: str = (
        "Complete the Microsoft device login flow for SharePoint access. "
        "This tool should be called after the user approves document access."
    )
    args_schema: type[BaseModel] = SharePointAuthFinishInput

    def _execute(self, user_id: str) -> dict[str, any]:
        return finish_device_login(user_id)


class SharePointTenantInfoInput(BaseModel):
    user_id: str = Field(description="Unique identifier for the user.")


class SharePointTenantInfoTool(BaseAgentTool):
    name: str = "sharepoint_tenant_info"
    description: str = (
        "Get the Microsoft tenant ID for a given user."
    )
    args_schema: type[BaseModel] = SharePointTenantInfoInput

    def _execute(self, user_id: str) -> dict[str, any]:
        # Placeholder implementation - replace with actual tenant info retrieval logic
        return json.dumps(get_user_tenant(user_id), indent=2)