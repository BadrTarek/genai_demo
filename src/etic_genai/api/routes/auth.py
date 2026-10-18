"""Authentication endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from etic_genai.api.dependencies import get_auth_store
from etic_genai.api.schemas import (
    RegisterRequest,
    SignInRequest,
    SignInResponse,
    UserResponse,
)
from etic_genai.auth.store import (
    ACCOUNT_LOCK_SECONDS,
    AccountLockedError,
    AuthStore,
    DuplicateEmailError,
    InvalidCredentialsError,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user account",
    description=(
        "Registers a user with a case-insensitive unique email and a password "
        "that meets the configured complexity requirements."
    ),
    responses={409: {"description": "An account with this email already exists."}},
)
def register(
    request: RegisterRequest,
    auth_store: Annotated[AuthStore, Depends(get_auth_store)],
) -> UserResponse:
    try:
        user = auth_store.create_user(request.email, request.password)
    except DuplicateEmailError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from exc
    return UserResponse(user_id=user.user_id, email=user.email)


@router.post(
    "/sign-in",
    response_model=SignInResponse,
    summary="Sign in and receive a bearer token",
    responses={
        401: {"description": "Email or password is incorrect."},
        423: {"description": "The account is temporarily locked after failed attempts."},
    },
)
def sign_in(
    request: SignInRequest,
    auth_store: Annotated[AuthStore, Depends(get_auth_store)],
) -> SignInResponse:
    try:
        session = auth_store.sign_in(request.email, request.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email or password is incorrect.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except AccountLockedError as exc:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=(
                f"Account temporarily locked. Try again in {ACCOUNT_LOCK_SECONDS // 60} minutes."
            ),
        ) from exc
    return SignInResponse(
        access_token=session.access_token,
        expires_in=session.expires_in,
        user=UserResponse(user_id=session.user.user_id, email=session.user.email),
    )
