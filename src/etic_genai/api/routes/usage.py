"""Administrator token-usage reporting endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends

from etic_genai.api.dependencies import get_admin_user, get_usage_store
from etic_genai.api.schemas import ConversationUsageResponse
from etic_genai.usage.store import UsageStore

router = APIRouter(prefix="/admin", tags=["Usage"])


@router.get(
    "/usage",
    response_model=list[ConversationUsageResponse],
    summary="Get token usage for all users and conversations",
    dependencies=[Depends(get_admin_user)],
    responses={
        401: {"description": "A valid bearer token is required."},
        403: {"description": "Administrator access is required."},
    },
)
def get_all_usage(
    usage_store: Annotated[UsageStore, Depends(get_usage_store)],
) -> list[ConversationUsageResponse]:
    return [ConversationUsageResponse(**vars(usage)) for usage in usage_store.list_all()]
