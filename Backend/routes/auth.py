"""User registration, login, and identity endpoints."""

from __future__ import annotations

import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from Backend.App.database.connection import get_db
from Backend.App.database.crud import (
    DatabaseOperationError,
    DuplicateRecordError,
    create_user,
    get_user_by_email,
)
from Backend.App.database.models import User
from Backend.App.security import (
    AuthenticationConfigurationError,
    create_access_token,
    get_current_user,
    hash_password,
    validate_password,
    verify_password,
    verify_unknown_user_password,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegistrationRequest(BaseModel):
    """Validated information required to register an account."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=150)
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=12, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.casefold()
        if not _EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("A valid email address is required.")
        return normalized

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, value: str) -> str:
        validate_password(value)
        return value


class LoginRequest(BaseModel):
    """Email and password credentials for login."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = value.casefold()
        if not _EMAIL_PATTERN.fullmatch(normalized):
            raise ValueError("A valid email address is required.")
        return normalized


class UserResponse(BaseModel):
    """Public user information; password hashes are never returned."""

    user_id: int
    name: str
    email: str


class TokenResponse(BaseModel):
    """JWT bearer token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _auth_storage_error(exc: Exception) -> HTTPException:
    logger.error("Authentication database operation failed (%s).", type(exc).__name__)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication storage is currently unavailable.",
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a user",
    responses={
        status.HTTP_409_CONFLICT: {
            "description": "An account with this email already exists."
        }
    },
)
def register(
    request: RegistrationRequest,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    """Create an account and persist only a bcrypt password hash."""
    try:
        user = create_user(
            db,
            name=request.name,
            email=request.email,
            hashed_password=hash_password(request.password),
        )
    except DuplicateRecordError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        ) from exc
    except DatabaseOperationError as exc:
        raise _auth_storage_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    return UserResponse(user_id=user.user_id, name=user.name, email=user.email)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Log in and obtain an access token",
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid email or password."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Authentication configuration or storage is unavailable."
        },
    },
)
def login(
    request: LoginRequest,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Verify credentials and issue a short-lived signed access token."""
    try:
        user = get_user_by_email(db, request.email)
        if user is None:
            verify_unknown_user_password(request.password)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not verify_password(request.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token, expires_in = create_access_token(user.user_id)
    except HTTPException:
        raise
    except AuthenticationConfigurationError as exc:
        logger.error("JWT authentication configuration is invalid.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        ) from exc
    except DatabaseOperationError as exc:
        raise _auth_storage_error(exc) from exc
    except Exception as exc:
        logger.exception("Unexpected authentication failure.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login could not be completed.",
        ) from exc
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the authenticated user",
)
def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    """Return public information for the bearer-token owner."""
    return UserResponse(
        user_id=current_user.user_id,
        name=current_user.name,
        email=current_user.email,
    )


__all__ = ["router"]
