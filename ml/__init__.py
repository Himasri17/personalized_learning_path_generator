# ml/__init__.py
# ML pipeline package — exposes nothing at the top level.
# Import specific modules or tasks directly:
#   from ml.tasks.ingest_task import ingest_document_task
#   from ml.pipeline.embedder import Embedder"""
ml/__init__.py
==============
Package initializer for the ML pipeline.

Exposes:
  - celery_app   : shared Celery application instance
  - get_celery   : factory function for Flask-aware Celery setup

Import pattern in Flask app factory:
    from ml import celery_app
    celery_app.conf.update(app.config)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from celery import Celery

if TYPE_CHECKING:
    from flask import Flask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery application (broker / backend wired from env-vars so tests can
# override them without touching any config file)
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app: Celery = Celery(
    "ml",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "ml.tasks.ingest_task",
        "ml.tasks.qgen_task",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,                # re-queue on worker crash
    worker_prefetch_multiplier=1,       # fair dispatch for long ML tasks
    task_routes={
        "ml.tasks.ingest_task.*": {"queue": "ingest"},
        "ml.tasks.qgen_task.*":   {"queue": "qgen"},
    },
)


def get_celery(app: "Flask") -> Celery:
    """
    Bind Celery to a Flask application context so tasks can access
    ``current_app``, ``db``, and other Flask extensions.

    Usage (in your Flask app factory)::

        from ml import get_celery
        celery = get_celery(flask_app)

    Parameters
    ----------
    app:
        The Flask application instance.

    Returns
    -------
    Celery
        The module-level ``celery_app`` after updating its config from
        ``app.config``.
    """
    celery_app.conf.update(app.config)

    class ContextTask(celery_app.Task):  # type: ignore[misc]
        """Celery task base that pushes a Flask app context before running."""

        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return super().__call__(*args, **kwargs)

    celery_app.Task = ContextTask
    logger.info("Celery configured with broker=%s", celery_app.conf.broker_url)
    return celery_app


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = ["celery_app", "get_celery"]