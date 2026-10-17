import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, model_validator

from etic_genai.config import get_settings
from etic_genai.tools.base import BaseAgentTool
from etic_genai.tools.document_access.microsoft_auth import get_graph_access_token
def _graph_request(url: str, user_id: str,body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make a Microsoft Graph request with the given URL and user ID."""
    token = get_graph_access_token(user_id)
    if not token:
        raise ToolException(
            "No valid Microsoft Graph access token found. Please complete the device login flow first."
        )

    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    else:
        data = None

    request = Request(url, headers=headers, data=data)
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise ToolException(f"Microsoft Graph returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ToolException(f"Could not connect to Microsoft Graph: {exc.reason}") from exc


class SharePointListFilesInput(BaseModel):
    approved: bool = Field(description="True only after the user explicitly approved access.")
    user_id: str = Field(description="User ID for which to list files.")
    site_id: str | None = Field(default=None, description="Microsoft Graph SharePoint site ID.")
    drive_id: str | None = Field(default=None, description="Microsoft Graph drive ID.")
    folder_path: str = Field(default="", description="Folder path under the drive root.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of files to return.")

    @model_validator(mode="after")
    def require_site_or_drive(self) -> "SharePointListFilesInput":
        if not self.site_id and not self.drive_id:
            raise ValueError("Either site_id or drive_id is required.")
        return self


class SharePointListFilesTool(BaseAgentTool):
    name: str = "sharepoint_list_files"
    description: str = (
        "List files from SharePoint through Microsoft Graph after the user approved access. "
        "Requires MICROSOFT_GRAPH_ACCESS_TOKEN in .env."
    )
    args_schema: type[BaseModel] = SharePointListFilesInput

    def _execute(
        self,
        approved: bool,
        user_id: str,
        site_id: str | None = None,
        drive_id: str | None = None,
        folder_path: str = "",
        limit: int = 20,
    ) -> str:
        if not approved:
            return "Access denied by user. No SharePoint files were accessed."

        url= self._build_url(site_id=site_id, drive_id=drive_id, folder_path=folder_path)
        payload = _graph_request(url, user_id=user_id)
        files = [self._format_item(item) for item in payload.get("value", [])[:limit]]
        return json.dumps({"files": files}, indent=2)

    def _build_url(
        self,
        site_id: str | None,
        drive_id: str | None,
        folder_path: str,
    ) -> str:
        root = "https://graph.microsoft.com/v1.0"
        encoded_path = quote(folder_path.strip("/"))
        item_path = f"root:/{encoded_path}:/children" if encoded_path else "root/children"

        if drive_id:
            return f"{root}/drives/{quote(drive_id)}/{item_path}"
        if site_id:
            return f"{root}/sites/{quote(site_id)}/drive/{item_path}"
        raise ToolException("Either site_id or drive_id is required.")

    def _format_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": item.get("name"),
            "type": "folder" if "folder" in item else "file",
            "web_url": item.get("webUrl"),
            "size": item.get("size"),
            "last_modified": item.get("lastModifiedDateTime"),
        }
class SharePointSearchInput(BaseModel):
    approved: bool = Field(description="True only after the user explicitly approved access.")
    user_id: str = Field(description="User ID for which to perform the search.")
    query: str = Field(description="Search query string.")
    site_id: str | None = Field(default=None, description="Microsoft Graph SharePoint site ID.")
    drive_id: str | None = Field(default=None, description="Microsoft Graph drive ID.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of files to return.")

    @model_validator(mode="after")
    def require_site_or_drive(self) -> "SharePointSearchInput":
        if not self.site_id and not self.drive_id:
            raise ValueError("Either site_id or drive_id is required.")
        return self

class SharePointSearchTool(BaseAgentTool):
    name: str = "sharepoint_search"
    description: str = (
        "Search files from SharePoint through Microsoft Graph after the user approved access. "
        "Requires MICROSOFT_GRAPH_ACCESS_TOKEN in .env."
    )
    args_schema: type[BaseModel] = SharePointListFilesInput

    def _execute(
        self,
        approved: bool,
        user_id: str,
        query: str,
        site_id: str | None = None,
        drive_id: str | None = None,
        limit: int = 20,
    ) -> str:
        if not approved:
            return "Access denied by user. No SharePoint files were accessed."

        root = "https://graph.microsoft.com/v1.0"
        if not drive_id and site_id:
           body = {
               "requests": [
                   {
                       "entityTypes": ["driveItem"],
                       "query": {"queryString": query},
                       "from": 0,
                       "size": limit,
                   }
               ]
           }
           payload= _graph_request(f"{root}/search/query", user_id=user_id, body=body)
           hits= [
               hit.get("resource", {})
               for response in payload.get("value",[])
               for container in response.get("hitsContainers",[])
               for hit in container.get("hits",[])
           ]
           results= [self._format_item(hit) for hit in hits]
           return json.dumps({"files": results}, indent=2)
        if drive_id:
            base = f"{root}/drives/{quote(drive_id)}"
        else:
            base = f"{root}/sites/{quote(site_id)}/drive"
        payload = _graph_request(f"{base}/root/search(q='{quote(query)}')", user_id=user_id) 
        results = [self._format_item(item) for item in payload.get("value", [])[:limit]] 
        return json.dumps({"files": results}, indent=2)
               
