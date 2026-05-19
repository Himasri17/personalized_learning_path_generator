"""
app/config.py
-------------
Environment-based configuration classes.

All sensitive values are read from environment variables (never hardcoded).
Copy `.env.example` → `.env` and fill in your values.

Load order
----------
1. Defaults defined here
2. Values from the environment (os.getenv)
3. Per-environment overrides (DevelopmentConfig, ProductionConfig, …)
"""

from __future__ import annotations

import os
from datetime import timedelta


def _require(key: str) -> str:
    """Read an env-var that MUST be set; raise clearly if missing."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Check your .env file."
        )
    return val


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseConfig:
    # ---- Flask core --------------------------------------------------------
    SECRET_KEY        = os.getenv("SECRET_KEY", "change-me-in-production")
    DEBUG             = False
    TESTING           = False
    JSON_SORT_KEYS    = False

    # ---- Database (PostgreSQL) ---------------------------------------------
    SQLALCHEMY_DATABASE_URI          = os.getenv(
        "DATABASE_URL",
        "postgresql://plp_user:plp_pass@localhost:5432/plp_dev",
    )
    SQLALCHEMY_TRACK_MODIFICATIONS   = False
    SQLALCHEMY_ENGINE_OPTIONS        = {
        "pool_pre_ping":    True,
        "pool_recycle":     300,       # recycle connections every 5 min
        "pool_size":        10,
        "max_overflow":     20,
    }

    # ---- Redis / Celery broker ---------------------------------------------
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ---- JWT ---------------------------------------------------------------
    JWT_SECRET_KEY              = os.getenv("JWT_SECRET_KEY", "jwt-secret-change-me")
    JWT_ACCESS_TOKEN_EXPIRES    = timedelta(
        minutes=int(os.getenv("JWT_ACCESS_EXPIRE_MINUTES", "60"))
    )
    JWT_REFRESH_TOKEN_EXPIRES   = timedelta(
        days=int(os.getenv("JWT_REFRESH_EXPIRE_DAYS", "30"))
    )
    JWT_TOKEN_LOCATION          = ["headers"]
    JWT_HEADER_NAME             = "Authorization"
    JWT_HEADER_TYPE             = "Bearer"

    # ---- CORS --------------------------------------------------------------
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173"
    ).split(",")

    # ---- File storage (S3 / MinIO) -----------------------------------------
    STORAGE_BACKEND     = os.getenv("STORAGE_BACKEND", "minio")   # "s3" | "minio" | "local"
    AWS_ACCESS_KEY_ID   = os.getenv("AWS_ACCESS_KEY_ID",   "minioadmin")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    AWS_REGION          = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME      = os.getenv("S3_BUCKET_NAME", "plp-documents")
    S3_ENDPOINT_URL     = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")   # MinIO
    LOCAL_UPLOAD_DIR    = os.getenv("LOCAL_UPLOAD_DIR", "/tmp/plp_uploads")

    # ---- LLM / OpenAI ------------------------------------------------------
    OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL        = os.getenv("OPENAI_MODEL",   "gpt-4o")
    OPENAI_MAX_TOKENS   = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))
    OPENAI_TEMPERATURE  = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

    # ---- Vector store ------------------------------------------------------
    VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss")   # "faiss" | "chromadb"
    FAISS_INDEX_DIR      = os.getenv("FAISS_INDEX_DIR", "/tmp/plp_faiss")
    CHROMA_HOST          = os.getenv("CHROMA_HOST", "localhost")
    CHROMA_PORT          = int(os.getenv("CHROMA_PORT", "8001"))

    # ---- Embedding model ---------------------------------------------------
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ---- Sandbox (code execution) ------------------------------------------
    SANDBOX_DOCKER_ENABLED  = os.getenv("SANDBOX_DOCKER_ENABLED", "true").lower() == "true"
    SANDBOX_TIMEOUT_SEC     = int(os.getenv("SANDBOX_TIMEOUT_SEC", "10"))
    SANDBOX_MEMORY_LIMIT    = os.getenv("SANDBOX_MEMORY_LIMIT", "128m")

    # ---- Pagination defaults -----------------------------------------------
    DEFAULT_PAGE_SIZE = int(os.getenv("DEFAULT_PAGE_SIZE", "10"))
    MAX_PAGE_SIZE     = int(os.getenv("MAX_PAGE_SIZE", "50"))

    # ---- Celery task routing -----------------------------------------------
    CELERY_TASK_ROUTES = {
        "ml.tasks.ingest_task.*": {"queue": "ingest"},
        "ml.tasks.qgen_task.*":   {"queue": "qgen"},
        "ml.tasks.grade_task.*":  {"queue": "grade"},
    }


# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

class DevelopmentConfig(BaseConfig):
    DEBUG                          = True
    SQLALCHEMY_ECHO                = False   # set True to log every SQL statement
    JWT_ACCESS_TOKEN_EXPIRES       = timedelta(hours=8)   # longer for dev convenience
    SANDBOX_DOCKER_ENABLED         = os.getenv("SANDBOX_DOCKER_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

class TestingConfig(BaseConfig):
    TESTING                       = True
    DEBUG                         = True
    SQLALCHEMY_DATABASE_URI       = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://plp_user:plp_pass@localhost:5432/plp_test",
    )
    JWT_ACCESS_TOKEN_EXPIRES      = timedelta(minutes=5)
    WTF_CSRF_ENABLED              = False
    SANDBOX_DOCKER_ENABLED        = False   # subprocess fallback in CI
    CELERY_TASK_ALWAYS_EAGER      = True    # run Celery tasks synchronously in tests
    CELERY_TASK_EAGER_PROPAGATES  = True


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------

class ProductionConfig(BaseConfig):
    DEBUG   = False
    TESTING = False

    # In production, all critical vars must be explicitly set
    SECRET_KEY            = _require("SECRET_KEY")
    JWT_SECRET_KEY        = _require("JWT_SECRET_KEY")
    SQLALCHEMY_DATABASE_URI = _require("DATABASE_URL")
    OPENAI_API_KEY        = _require("OPENAI_API_KEY")

    SQLALCHEMY_ENGINE_OPTIONS = {
        **BaseConfig.SQLALCHEMY_ENGINE_OPTIONS,
        "pool_size":     20,
        "max_overflow":  40,
    }

    # Force HTTPS cookies / secure headers in prod (handled by Nginx, but good to flag)
    SESSION_COOKIE_SECURE   = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

config_map: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing":     TestingConfig,
    "production":  ProductionConfig,
}