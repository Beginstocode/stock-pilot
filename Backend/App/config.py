"""Application settings and project paths."""

import os
from pathlib import Path


APP_NAME = os.getenv("APP_NAME", "AI Stock Recommendation System")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
)

# Resolve paths from this file so they work independently of the launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_path(environment_variable: str, default: str) -> Path:
    """Return an absolute configured path, resolving relative values at project root."""
    configured_path = Path(os.getenv(environment_variable, default)).expanduser()
    if not configured_path.is_absolute():
        configured_path = PROJECT_ROOT / configured_path
    return configured_path.resolve()


MODEL_PATH = _project_path("MODEL_PATH", "models/xgboost_model.pkl")
DATA_DIR = _project_path("DATA_DIR", "data")
PROCESSED_DATA_DIR = _project_path("PROCESSED_DATA_DIR", "data/processed")
REPORTS_DIR = _project_path("REPORTS_DIR", "reports")
