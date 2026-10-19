"""Administrator-only LangSmith evaluation endpoint."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from etic_genai.api.dependencies import get_admin_user
from etic_genai.api.schemas import EvaluationResponse
from etic_genai.auth.store import AuthUser
from etic_genai.core.logging import get_logger
from etic_genai.evaluation import run_dataset_evaluation

logger = get_logger(__name__)
router = APIRouter(prefix="/admin/evaluations", tags=["Evaluations"])


@router.post(
    "/run",
    response_model=EvaluationResponse,
    summary="Run the retrieval and answer evaluation dataset",
    description=(
        "Runs the configured LangSmith dataset against the requesting administrator's "
        "documents and returns the resulting experiment details."
    ),
    responses={
        401: {"description": "A valid bearer token is required."},
        403: {"description": "Administrator access is required."},
        502: {"description": "The evaluation service could not complete the run."},
    },
)
def run_evaluation(
    current_user: Annotated[AuthUser, Depends(get_admin_user)],
) -> EvaluationResponse:
    try:
        return EvaluationResponse(**run_dataset_evaluation(current_user.user_id))
    except Exception as exc:
        logger.exception("LangSmith evaluation failed.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Evaluation could not be completed. Check the API logs for details.",
        ) from exc
