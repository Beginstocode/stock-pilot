"""Password hashing, JWT handling, and authenticated-user dependency."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError
from sqlalchemy.orm import Session

from Backend.App.config import PROJECT_ROOT
from Backend.App.database.connection import get_db
from Backend.App.database.models import User

load_dotenv(dotenv_path=Path(PROJECT_ROOT) / ".env")

_JWT_ALGORITHM = "HS256"
_JWT_ISSUER = "ai-stock-recommendation-system"
_JWT_SECRET_ENV = "JWT_SECRET_KEY"
_PASSWORD_MIN_LENGTH = 12
_BCRYPT_MAX_PASSWORD_BYTES = 72
_BCRYPT_ROUNDS = 12
_bearer_scheme = HTTPBearer(auto_error=False)


class AuthenticationConfigurationError(RuntimeError):
    """Raised when required authentication configuration is missing or weak."""


def _jwt_secret() -> str:
    """Read a sufficiently strong JWT secret from the environment."""
    secret = os.getenv(_JWT_SECRET_ENV)
    if secret is None or len(secret.encode("utf-8")) < 32:
        raise AuthenticationConfigurationError(
            f"{_JWT_SECRET_ENV} must be configured with at least 32 bytes."
        )
    return secret


def validate_password(password: str) -> None:
    """Enforce bcrypt-safe password length and minimum strength."""
    if len(password) < _PASSWORD_MIN_LENGTH:
        raise ValueError(
            f"Password must contain at least {_PASSWORD_MIN_LENGTH} characters."
        )
    if len(password.encode("utf-8")) > _BCRYPT_MAX_PASSWORD_BYTES:
        raise ValueError("Password must not exceed 72 UTF-8 bytes.")


def hash_password(password: str) -> str:
    """Return a bcrypt hash of a validated password."""
    validate_password(password)
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
    ).decode("ascii")


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    if not isinstance(password, str) or not isinstance(hashed_password, str):
        return False
    try:
        validate_password(password)
        return bcrypt.checkpw(
            password.encode("utf-8"),
            hashed_password.encode("ascii"),
        )
    except (ValueError, UnicodeEncodeError):
        return False


def create_access_token(user_id: int) -> tuple[str, int]:
    """Create a signed HS256 JWT and return it with its lifetime in seconds."""
    if user_id <= 0:
        raise ValueError("User ID must be positive.")
    try:
        expires_minutes = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    except ValueError as exc:
        raise AuthenticationConfigurationError(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES must be a positive integer."
        ) from exc
    if not 1 <= expires_minutes <= 1440:
        raise AuthenticationConfigurationError(
            "JWT_ACCESS_TOKEN_EXPIRE_MINUTES must be between 1 and 1440."
        )

    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(minutes=expires_minutes)
    payload = {
        "sub": str(user_id),
        "iat": issued_at,
        "exp": expires_at,
        "iss": _JWT_ISSUER,
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm=_JWT_ALGORITHM)
    return token, expires_minutes * 60


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)
    ],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """Validate the bearer JWT and return its database-backed user."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorized()

    try:
        payload = jwt.decode(
            credentials.credentials,
            _jwt_secret(),
            algorithms=[_JWT_ALGORITHM],
            issuer=_JWT_ISSUER,
            options={"require": ["exp", "iat", "iss", "sub"]},
        )
    except AuthenticationConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        ) from exc
    except InvalidTokenError as exc:
        raise _unauthorized() from exc

    subject = payload.get("sub")
    if not isinstance(subject, str) or not subject.isdecimal():
        raise _unauthorized()
    user = db.get(User, int(subject))
    if user is None:
        raise _unauthorized()
    return user


@lru_cache(maxsize=1)
def _dummy_password_hash() -> bytes:
    """Build a process-local bcrypt hash for uniform unknown-user login checks."""
    return bcrypt.hashpw(
        b"nonexistent-account-password",
        bcrypt.gensalt(rounds=_BCRYPT_ROUNDS),
    )


def verify_unknown_user_password(password: str) -> None:
    """Perform a bcrypt check for unknown accounts to reduce timing differences."""
    try:
        bcrypt.checkpw(password.encode("utf-8"), _dummy_password_hash())
    except (ValueError, UnicodeEncodeError):
        return


__all__ = [
    "AuthenticationConfigurationError",
    "create_access_token",
    "get_current_user",
    "hash_password",
    "validate_password",
    "verify_password",
    "verify_unknown_user_password",
]
