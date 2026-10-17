import json 
from typing import Any 


from langchain_core.tools import ToolException

from etic_genai.config import get_settings
from etic_genai.tools.document_access.sqllite_store import (
    clear_pending_flow,
    get_pending_flow,
    get_user_cache,
    save_pending_flow,
    upsert_user_cache,
    upsert_user_identity,
    get_user_identity
)

GRAPH_SCOPES = ["User.Read", "Files.Read.All", "Sites.Read.All"]

def _build_app(cache_json: str | None = None, tenant_id: str | None = None) -> Any:
    """Build a Microsoft Graph app with the given token cache."""
    from msal import PublicClientApplication, SerializableTokenCache

    settings = get_settings()
    client_id = settings.microsoft_client_id.strip()
    if not client_id or client_id == "id":
        raise ToolException(
            "Set MICROSOFT_CLIENT_ID to the Application (client) ID from your "
            "Microsoft Entra app registration."
        )

    tenant_id = tenant_id or settings.microsoft_tenant_id.strip()

    token_cache = SerializableTokenCache()
    if cache_json:
        token_cache.deserialize(cache_json)

    app = PublicClientApplication(
        client_id=client_id,
        authority=f"https://login.microsoftonline.com/{tenant_id}",
        token_cache=token_cache,
    )
    return app, token_cache



def start_device_login(user_id: str) -> dict[str,Any]:
    """Start the Microsoft device login flow for a given user ID."""
    app, _ = _build_app(get_user_cache(user_id))
    print(f"Starting Microsoft device login flow for user_id '{user_id}'...")
    flow = app.initiate_device_flow(scopes=GRAPH_SCOPES)
    print(f"Device login flow started for user_id '{user_id}'. User code: {flow.get('user_code')}")
    if "user_code" not in flow:
        detail = flow.get("error_description") or flow.get("error") or "Unknown Microsoft error"
        raise ToolException(f"Failed to initiate device flow: {detail}")
    print(f"Device login flow started for user_id '{user_id}'. User code: {flow['user_code']}")
    save_pending_flow(user_id, json.dumps(flow))
    print(f"Device login flow started for user_id '{user_id}'. User code: {flow['user_code']}")
    return {
        "status": "pending",
        "message": f"To sign in, use a web browser to open the page {flow['verification_uri']} and enter the code {flow['user_code']}.",
        "expires_in": flow.get("expires_in"),
        "interval": flow.get("interval"),
        "next_step":"ask the user to complete sign-in",
        "user_code": flow.get("user_code"),
        "verification_uri": flow.get("verification_uri"),
    }


def finish_device_login(user_id: str) -> dict[str,Any]:
    """Finish the Microsoft device login flow for a given user ID."""
    pending_flow_json = get_pending_flow(user_id)
    if not pending_flow_json:
        raise ToolException("No pending device flow found for this user. Start the flow first.")
    flow = json.loads(pending_flow_json)

    app, token_cache = _build_app(get_user_cache(user_id))
    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise ToolException(f"Failed to acquire token: {result.get('error_description', 'Unknown error')}")

    claims = result.get("id_token_claims", {})
    tenant_id = claims.get("tid") or result.get("tenant_id")
    account = claims.get("preferred_username", "unknown")
    if account == "unknown" or not tenant_id:
        account= app.get_accounts()
        if account:
            account = account[0].get("username", "unknown")
            if not tenant_id:
                home_account_id= app.get_accounts()[0].get("home_account_id")
                parts = home_account_id.split(".") if home_account_id else []
                if len(parts) >= 2 and parts[1]:
                    tenant_id = parts[1]

    upsert_user_cache(user_id, token_cache.serialize())
    upsert_user_identity(
        user_id,
        account=account,
        tenant_id= tenant_id)                                                   
    clear_pending_flow(user_id)
    return {
        "status": "success",
        "provider": "Microsoft",
        "user_id": user_id,
        "account": account,
        "tenant_id": tenant_id,
    }

def get_user_tenant(user_id: str) -> str | None:
    """Get the tenant ID for a given user ID."""
    identity = get_user_identity(user_id)
    if not identity:
        return{
            "tenant_id": None,
            "account": None,
            "provider": None,
            "status": "no identity found"
        }
    return {
        "tenant_id": identity.get("tenant_id") if identity else None,
        "account": identity.get("account") if identity else None,
        "provider": "Microsoft" if identity else None,
        "status": "ok"
    }


def get_graph_access_token(user_id: str) -> str | None:
    """Get a valid Microsoft Graph access token for a given user ID."""
    explicit_token = get_settings().microsoft_graph_access_token
    if explicit_token:
        return explicit_token   
    identity = get_user_identity(user_id)
    if not identity:
        raise ToolException("No user identity found. Please complete the device login flow first.")
    authority_tenant = identity.get("tenant_id")
    app, token_cache = _build_app(get_user_cache(user_id), authority_tenant)
    accounts = app.get_accounts()
    result= app.acquire_token_silent(scopes=GRAPH_SCOPES, account=accounts[0]) if accounts else None
    if not result or "access_token" not in result:
        raise ToolException("No valid access token found. Please complete the device login flow first.")
    upsert_user_cache(user_id, token_cache.serialize())
    return str(result["access_token"])


