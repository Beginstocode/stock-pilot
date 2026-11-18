"""Lightweight API health-check endpoint."""

from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel

from Backend.App.config import APP_NAME

router = APIRouter()


class HealthResponse(BaseModel):
    """Health information returned by the API."""

    api_status: Literal["running"]
    service_name: str
    health_status: Literal["healthy"]


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check API health",
)
def health_check() -> HealthResponse:
    """Return the API's current health without initializing external services."""
    return HealthResponse(
        api_status="running",
        service_name=APP_NAME,
        health_status="healthy",
    )
